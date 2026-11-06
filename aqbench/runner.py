"""
PHASE 1 : one-step training, one-step validation (1-bundle).
           Early stopping on val_onestep.

PHASE 2 : pushforward training (one-step + stability, equally weighted,
           split-backward). 2-bundle validation reports val_onestep and
           val_stability. Early stopping and best-checkpoint saving on
           val_stability only.
"""

import os
import math
import time
import threading
import json
import csv
import random
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from aqbench.dataloaders.wrfchem_loader import build_dataloader
from aqbench.registry.model_registry import get_model
import aqbench.models  


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


BATCH_SIZE  = 24
BUNDLE_SIZE = 6


class NvidiaSmiLogger:
    def __init__(self, log_path, interval_s=60):
        self.log_path   = log_path
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.log_thread = threading.Thread(target=self.run_loop, daemon=True)

    def run_loop(self):
        import subprocess
        while not self.stop_event.wait(self.interval_s):
            try:
                ts  = time.strftime("%Y-%m-%d %H:%M:%S")
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=timestamp,name,index,"
                     "utilization.gpu,utilization.memory,"
                     "memory.used,memory.total,temperature.gpu,power.draw",
                     "--format=csv,noheader"], text=True).strip()
                with open(self.log_path, "a") as f:
                    f.write(f"\n[{ts}]\n{out}\n")
            except Exception:
                pass

    def start(self):
        with open(self.log_path, "a") as f:
            f.write("# nvidia-smi log\n")
        self.log_thread.start()

    def stop(self):
        self.stop_event.set()
        self.log_thread.join(timeout=5)


def setup_ddp():
    if "RANK" not in os.environ:
        return 0, False
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, True


def build_param_groups(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay,    "weight_decay": 1e-5},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def relative_l2_loss(pred, target, eps=1e-6):
    """
    Relative L2 loss averaged equally across all pollutant channels (and batch).

    pred / target shape: (B, T, H, W, C) — spatial dims are the H, W axes.
    The norm is taken over H×W, giving a (B, T, C) tensor of per-channel
    relative errors, which are then averaged with .mean() — equal weight
    across batch, time, and all pollutant channels. Works identically for
    single and multi-pollutant regimes.
    """
    spatial_dims = (-3, -2)
    diff  = torch.norm(pred - target, p=2, dim=spatial_dims)
    denom = torch.norm(target,        p=2, dim=spatial_dims)
    return (diff / (denom + eps)).mean()


def build_target_input_map(task_cfg):
    inp_feat = task_cfg["input_features"]
    mapping  = []
    for tgt_i, feat in enumerate(task_cfg["target_features"]):
        if feat in inp_feat:
            mapping.append((tgt_i, inp_feat.index(feat)))
    return mapping


def build_next_input(x_current, pred_bundle, aux_bundle, target_input_map):
    Tin  = x_current.shape[1]
    Tout = pred_bundle.shape[1]
    new_frames = aux_bundle.clone()
    for tgt_i, inp_i in target_input_map:
        new_frames[..., inp_i] = pred_bundle[..., tgt_i]
    if Tin <= Tout:
        return new_frames[:, -Tin:]
    return torch.cat([x_current[:, Tout:], new_frames], dim=1)


def rng_states():
    """Capture current state of all RNGs for checkpoint reproducibility."""
    return {
        "python": random.getstate(),
        "numpy":  np.random.get_state(),
        "torch":  torch.get_rng_state(),
        "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }



def load_rng_states(states):
    """Restore all RNG states from a checkpoint dict."""
    if states is None:
        return
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])

    torch_state = states["torch"]
    if not isinstance(torch_state, torch.ByteTensor):
        torch_state = torch_state.cpu().to(torch.uint8)
    torch.set_rng_state(torch_state)

    if states.get("cuda") is not None and torch.cuda.is_available():
        try:
            cuda_states = states["cuda"]
            cuda_states = [
                s.cpu().to(torch.uint8) if not isinstance(s, torch.ByteTensor) else s
                for s in cuda_states
            ]
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception:
            pass

def save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss,
                    phase_id, wait, cumulatives, distributed):
    state = model.module.state_dict() if distributed else model.state_dict()
    ckpt  = {
        "epoch":           epoch,
        "phase_id":        phase_id,
        "model_state":     state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "val_loss":        val_loss,
        "wait":            wait,
        "cumulatives":     cumulatives,
        "rng":             rng_states(),
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    device="cuda", distributed=False, restore_rng=True):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if distributed:
        model.module.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if restore_rng and "rng" in ckpt:
        load_rng_states(ckpt["rng"])
    return ckpt


STAGE_DEFAULTS = {
    "best_val":           float("inf"),
    "best_epoch":         0,
    "last_epoch":         0,
    "wait":               0,
    "steps_accumulated":  0,
    "time_accumulated_s": 0.0,
    "steps_to_best":      0,
    "time_to_best_s":     0.0,
    "hit_max_epochs":     True,
}


def load_training_state(exp_dir):
    path = os.path.join(exp_dir, "training_state.json")
    if not os.path.exists(path):
        return {"s1_complete": False, "s2_complete": False,
                "s1": dict(STAGE_DEFAULTS), "s2": dict(STAGE_DEFAULTS)}
    with open(path) as f:
        return json.load(f)


def save_training_state(exp_dir, state):
    path = os.path.join(exp_dir, "training_state.json")
    def clean_val(v):
        return 1e38 if isinstance(v, float) and not np.isfinite(v) else v
    with open(path, "w") as f:
        json.dump(json.loads(json.dumps(state, default=clean_val)), f, indent=2)

def csv_header():
    return [
        "epoch", "phase", "lr",
        "train_onestep", "train_stability", "train_penalty",
        "val_onestep", "val_stability",
        "wall_time_s", "cumul_steps", "is_best",
    ]


def append_csv_row(csv_path, row):
    header   = csv_header()
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow([row.get(c, "") for c in header])


def get_lr_scheduler(optimizer, warmup_epochs, max_epochs, min_lr_factor=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Phase 1 Training 
def train_epoch_phase1(model, loader, optimizer, device,
                       target_mean, target_std, target_features,
                       constraint_weight):
    """
    One-step supervised training. Loader provides 1-bundle targets.
    """
    model.train()
    total_task, total_pen, n = 0.0, 0.0, 0

    for x, y, _, t_in in loader:
        x    = x.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)
        t_in = t_in.to(device, non_blocking=True)

        pred      = model(x, t_in)
        task_loss = relative_l2_loss(pred, y)

        pen_val = 0.0
        total_loss=task_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_task += float(task_loss.item())
        total_pen  += pen_val
        n          += 1

    return {
        "onestep":   total_task / max(n, 1),
        "stability": 0.0,
        "penalty":   total_pen  / max(n, 1),
        "steps":     n,
    }


# Phase 2 Training
def train_epoch_phase2_pushforward(model, loader, optimizer, device,
                                   target_input_map,
                                   target_mean, target_std, target_features,
                                   constraint_weight):
    """
    Equally-weighted one-step + stability loss via split-backward pushforward.
    L_total = L_onestep + L_stability  (equal weight 1:1).
    Split-backward: L_onestep.backward() then L_stability.backward() in sequence.
    Multi-pollutant: relative_l2_loss averages equally across all channels.
    Returns dict with keys: onestep, stability, penalty, steps.
    """
    model.train()
    total_one, total_stab, total_pen, n = 0.0, 0.0, 0.0, 0

    for x, y, y_aux, t_in in loader:
        x     = x.to(device, non_blocking=True)
        y     = y.to(device, non_blocking=True)
        y_aux = y_aux.to(device, non_blocking=True)
        t_in  = t_in.to(device, non_blocking=True)

        y_b1 = y[:, :BUNDLE_SIZE]
        y_b2 = y[:, BUNDLE_SIZE:2 * BUNDLE_SIZE]

        optimizer.zero_grad(set_to_none=True)

        pred1     = model(x, t_in)
        L_onestep = relative_l2_loss(pred1, y_b1)
        L_onestep.backward()

        x_next    = build_next_input(x, pred1.detach(), y_aux, target_input_map)
        t_in_next = t_in + BUNDLE_SIZE

        pred2       = model(x_next, t_in_next)
        L_stability = relative_l2_loss(pred2, y_b2)
        pen_val = 0.0
        L_stability.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_one  += float(L_onestep.item())
        total_stab += float(L_stability.item())
        total_pen  += pen_val
        n          += 1

    return {
        "onestep":   total_one  / max(n, 1),
        "stability": total_stab / max(n, 1),
        "penalty":   total_pen  / max(n, 1),
        "steps":     n,
    }


@torch.no_grad()
def validate_onestep(model, loader, device):
    """
    1-bundle validation for Stage 1.
    Returns val_onestep (scalar float).
    Used as the early-stopping metric in Stage 1.
    """
    model.eval()
    total, n = 0.0, 0

    for x, y, _, t_in in loader:
        x    = x.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)
        t_in = t_in.to(device, non_blocking=True)
        pred = model(x, t_in)
        total += float(relative_l2_loss(pred, y[:, :BUNDLE_SIZE]).item())
        n += 1

    return total / max(n, 1)


@torch.no_grad()
def validate_pushforward(model, loader, device, target_input_map):
    """
    2-bundle pushforward validation for Stage 2.
    """
    model.eval()
    total_one, total_stab, n = 0.0, 0.0, 0

    for x, y, y_aux, t_in in loader:
        x     = x.to(device, non_blocking=True)
        y     = y.to(device, non_blocking=True)
        y_aux = y_aux.to(device, non_blocking=True)
        t_in  = t_in.to(device, non_blocking=True)

        y_b1 = y[:, :BUNDLE_SIZE]
        y_b2 = y[:, BUNDLE_SIZE:2 * BUNDLE_SIZE]

        pred1 = model(x, t_in)
        total_one += float(relative_l2_loss(pred1, y_b1).item())

        x_next    = build_next_input(x, pred1, y_aux, target_input_map)
        t_in_next = t_in + BUNDLE_SIZE

        pred2 = model(x_next, t_in_next)
        total_stab += float(relative_l2_loss(pred2, y_b2).item())

        n += 1

    return total_one / max(n, 1), total_stab / max(n, 1)


def run_stage(
    *, stage_name, stage_id,
    model, optimizer, device,
    train_loader, val_loader,
    target_input_map,
    patience, max_epochs,
    exp_dir, ckpt_name, csv_path,
    distributed, rank,
    target_mean, target_std, target_features,
    constraint_weight,
    warmup_epochs,
    resume_ckpt_path,
):

    scheduler = get_lr_scheduler(optimizer,
                                 warmup_epochs=warmup_epochs,
                                 max_epochs=max_epochs)

    best_val    = float("inf")
    best_epoch  = 0
    start_epoch = 1
    wait        = 0
    cumul = {
        "steps_accumulated":  0,
        "time_accumulated_s": 0.0,
        "steps_to_best":      0,
        "time_to_best_s":     0.0,
    }

    if resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
        ckpt = load_checkpoint(resume_ckpt_path, model, optimizer, scheduler,
                               device=device, distributed=distributed,
                               restore_rng=True)
        if ckpt.get("phase_id") == stage_id:
            start_epoch = int(ckpt["epoch"]) + 1
            best_val    = float(ckpt.get("val_loss", float("inf")))
            wait        = int(ckpt.get("wait", 0))
            cumul.update(ckpt.get("cumulatives", {}))
            best_epoch  = cumul.get("best_epoch_pinned", 0)
            if rank == 0:
                print(f"[{stage_id}] RESUMING epoch {start_epoch}, "
                      f"best={best_val:.6f}, wait={wait}")

    if rank == 0:
        mode_str    = "one-step" if stage_id == "s1" else "pushforward (split-backward)"
        stop_metric = "val_onestep" if stage_id == "s1" else "val_stability"
        print(f"\n===== {stage_name} | {mode_str} =====")
        print(f"  max_epochs={max_epochs}  warmup={warmup_epochs}  "
              f"patience={patience}  batch={BATCH_SIZE}  "
              f"early-stop-on={stop_metric}")
        if stage_id == "s1":
            print(f"\n{'Ep':>5} | {'LR':>9} | {'TrnOne':>9} | {'TrnPen':>9} | "
                  f"{'ValOne':>9} | {'Best':>9}")
            print("-" * 65)
        else:
            print(f"\n{'Ep':>5} | {'LR':>9} | {'TrnOne':>9} | {'TrnStab':>9} | "
                  f"{'TrnPen':>9} | {'ValOne':>9} | {'ValStab':>9} | {'Best':>9}")
            print("-" * 92)

    hit_max = True

    for epoch in tqdm(range(start_epoch, max_epochs + 1), disable=(rank != 0)):
        epoch_t0 = time.time()

        if stage_id == "s1":
            tr = train_epoch_phase1(
                model, train_loader, optimizer, device,
                target_mean=target_mean, target_std=target_std,
                target_features=target_features,
                constraint_weight=constraint_weight)
            val_onestep   = validate_onestep(model, val_loader, device)
            val_stability = float("nan")
            stop_val      = val_onestep
        else:
            tr = train_epoch_phase2_pushforward(
                model, train_loader, optimizer, device,
                target_input_map=target_input_map,
                target_mean=target_mean, target_std=target_std,
                target_features=target_features,
                constraint_weight=constraint_weight)
            val_onestep, val_stability = validate_pushforward(
                model, val_loader, device, target_input_map)
            stop_val = val_stability

        cumul["steps_accumulated"]  += tr["steps"]
        cumul["time_accumulated_s"] += time.time() - epoch_t0
        scheduler.step()

        is_best = stop_val < best_val
        if is_best:
            best_val   = stop_val
            best_epoch = epoch
            wait       = 0
            cumul["steps_to_best"]  = cumul["steps_accumulated"]
            cumul["time_to_best_s"] = cumul["time_accumulated_s"]
        else:
            wait += 1

        cumul_save = dict(cumul)
        cumul_save["best_epoch_pinned"] = best_epoch

        # using last checkpoint for resuming
        save_checkpoint(f"{exp_dir}/{ckpt_name}_last.pt",
                        model, optimizer, scheduler, epoch=epoch,
                        val_loss=best_val, phase_id=stage_id, wait=wait,
                        cumulatives=cumul_save, distributed=distributed)
        # Best checkpoint: S1 → best val_onestep; S2 → best val_stability
        if is_best:
            save_checkpoint(f"{exp_dir}/{ckpt_name}.pt",
                            model, optimizer, scheduler, epoch=epoch,
                            val_loss=best_val, phase_id=stage_id, wait=wait,
                            cumulatives=cumul_save, distributed=distributed)

        if rank == 0:
            lr     = optimizer.param_groups[0]["lr"]
            marker = " *" if is_best else ""

            if stage_id == "s1":
                print(f"{epoch:5d} | {lr:.3e} | {tr['onestep']:9.5f} | "
                      f"{tr['penalty']:9.5f} | {val_onestep:9.5f} | "
                      f"{best_val:.5f}{marker}")
            else:
                print(f"{epoch:5d} | {lr:.3e} | {tr['onestep']:9.5f} | "
                      f"{tr['stability']:9.5f} | {tr['penalty']:9.5f} | "
                      f"{val_onestep:9.5f} | {val_stability:9.5f} | "
                      f"{best_val:.5f}{marker}")

            row = {
                "epoch":           epoch,
                "phase":           stage_id,
                "lr":              f"{lr:.6e}",
                "train_onestep":   f"{tr['onestep']:.6f}",
                "train_stability": f"{tr['stability']:.6f}",
                "train_penalty":   f"{tr['penalty']:.6f}",
                "val_onestep":     f"{val_onestep:.6f}",
                "val_stability":   f"{val_stability:.6f}" if not math.isnan(val_stability) else "",
                "wall_time_s":     f"{cumul['time_accumulated_s']:.2f}",
                "cumul_steps":     cumul["steps_accumulated"],
                "is_best":         int(is_best),
            }
            append_csv_row(csv_path, row)

        if wait >= patience:
            if rank == 0:
                print(f"\n>>> Early stop at epoch {epoch} "
                      f"(best epoch {best_epoch}, patience={patience})")
            hit_max = False
            break

    best_ckpt = f"{exp_dir}/{ckpt_name}.pt"
    if os.path.exists(best_ckpt):
        load_checkpoint(best_ckpt, model, None, None, device=device,
                        distributed=distributed, restore_rng=False)

    return {
        "best_val":           best_val,
        "best_epoch":         best_epoch,
        "last_epoch":         epoch,
        "wait":               wait,
        "steps_accumulated":  cumul["steps_accumulated"],
        "time_accumulated_s": cumul["time_accumulated_s"],
        "steps_to_best":      cumul["steps_to_best"],
        "time_to_best_s":     cumul["time_to_best_s"],
        "hit_max_epochs":     hit_max,
    }


def run_persistence(task_cfg, model_cfg):
    rank, _ = setup_ddp()
    exp_name = model_cfg["exp_name"]
    exp_dir  = f"checkpoints/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    if rank == 0:
        print(f"Persistence baseline: {exp_name}")
        save_training_state(exp_dir, {
            "s1_complete": True, "s2_complete": True,
            "s1": dict(STAGE_DEFAULTS), "s2": dict(STAGE_DEFAULTS)})


def run_experiment(task_cfg, model_cfg):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    seed = model_cfg.get("seed", task_cfg.get("seed", 42))
    set_seed(seed)

    rank, distributed = setup_ddp()
    device = torch.device("cuda")

    exp_name = model_cfg["exp_name"]
    exp_dir  = f"checkpoints/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)

    smi_logger = None
    if rank == 0:
        smi_log_path = os.path.join(exp_dir, "nvidia_smi.log")
        smi_logger   = NvidiaSmiLogger(smi_log_path, interval_s=60)
        smi_logger.start()

    tr_state = load_training_state(exp_dir)

    target_features   = task_cfg["target_features"]
    target_input_map  = build_target_input_map(task_cfg)
    constraint_weight = task_cfg.get("constraint_weight", 0.0)

    model_class = get_model(model_cfg["model"])
    model = model_class(
        **model_cfg["model_args"],
        num_targets=len(target_features),
    ).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank])

    base_lr   = model_cfg["learning_rate"]
    optimizer = torch.optim.AdamW(build_param_groups(model), lr=base_lr)

    max_epochs   = model_cfg.get("epochs", 200)
    warmup_s1    = model_cfg.get("warmup_s1", 5)
    warmup_s2    = model_cfg.get("warmup_s2", 10)
    patience_s1  = model_cfg.get("patience_s1", 30)
    patience_s2  = model_cfg.get("patience_s2", 30)
    s2_lr_factor = model_cfg.get("s2_lr_factor", 0.1)    

    if model_cfg.get("use_compile", False):
        try:
            model = torch.compile(model)
            if rank == 0:
                print("  [opt] torch.compile enabled")
        except Exception as e:
            if rank == 0:
                print(f"  [opt] torch.compile skipped: {e}")

    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT : {exp_name}")
        print(f"Model      : {model_cfg['model']}  ({n_params:,} params)")
        print(f"Targets    : {target_features}")
        print(f"Batch size : {BATCH_SIZE}")
        print(f"Seed       : {seed}")
        print(f"Base LR    : {base_lr}  S2 LR: {base_lr * s2_lr_factor:.3e}  "
              f"(factor={s2_lr_factor})")
        print(f"S1 patience: {patience_s1}  S2 patience: {patience_s2}  "
              f"max_epochs: {max_epochs}")
        print(f"{'='*60}\n")

    csv_path    = os.path.join(exp_dir, "training_log.csv")
    task_cfg_bs = {**task_cfg, "batch_size": BATCH_SIZE}

    g1 = torch.Generator()
    g1.manual_seed(seed)

    train_loader_s1 = build_dataloader(
        task_cfg_bs, "train", Tout=BUNDLE_SIZE, y_aux_len=0, generator=g1)
    val_loader_s1   = build_dataloader(
        task_cfg_bs, "val",   Tout=BUNDLE_SIZE, y_aux_len=0)

    ds          = train_loader_s1.dataset
    target_mean = torch.tensor(ds.target_mean, dtype=torch.float32)
    target_std  = torch.tensor(ds.target_std,  dtype=torch.float32)

    if rank == 0:
        print(f"Train S1 : {len(train_loader_s1.dataset):,} samples  "
              f"({len(train_loader_s1)} batches)")
        print(f"Val   S1 : {len(val_loader_s1.dataset):,} samples")

    wall_start = time.time()

    stage_kwargs = dict(
        model=model, optimizer=optimizer, device=device,
        target_input_map=target_input_map,
        distributed=distributed, rank=rank,
        target_mean=target_mean, target_std=target_std,
        target_features=target_features,
        constraint_weight=constraint_weight,
        max_epochs=max_epochs, exp_dir=exp_dir, csv_path=csv_path,
    )

    if not tr_state["s1_complete"]:
        resume = f"{exp_dir}/stage1_last.pt"
        s1 = run_stage(
            stage_name="STAGE 1: one-step pretraining", stage_id="s1",
            train_loader=train_loader_s1, val_loader=val_loader_s1,
            patience=patience_s1, warmup_epochs=warmup_s1,
            ckpt_name="stage1",
            resume_ckpt_path=resume if os.path.exists(resume) else None,
            **stage_kwargs)
        tr_state["s1"]          = s1
        tr_state["s1_complete"] = True
        if rank == 0:
            save_training_state(exp_dir, tr_state)
    else:
        if rank == 0:
            print("Stage 1 complete — loading checkpoint.")
        load_checkpoint(f"{exp_dir}/stage1.pt", model, None, None,
                        device=device, distributed=distributed, restore_rng=False)
        s1 = tr_state["s1"]

    g2 = torch.Generator()
    g2.manual_seed(seed + 1)   

    train_loader_s2 = build_dataloader(
        task_cfg_bs, "train",
        Tout=2 * BUNDLE_SIZE, y_aux_len=BUNDLE_SIZE, generator=g2)
    val_loader_s2   = build_dataloader(
        task_cfg_bs, "val",
        Tout=2 * BUNDLE_SIZE, y_aux_len=BUNDLE_SIZE)

    if rank == 0:
        print(f"\nTrain S2 : {len(train_loader_s2.dataset):,} samples  "
              f"({len(train_loader_s2)} batches)")
        print(f"Val   S2 : {len(val_loader_s2.dataset):,} samples")

    stage2_lr = base_lr * s2_lr_factor
    for pg in optimizer.param_groups:
        pg["lr"]         = stage2_lr
        pg["initial_lr"] = stage2_lr
    optimizer.defaults["lr"] = stage2_lr

    if not tr_state["s2_complete"]:
        resume = f"{exp_dir}/best_model_last.pt"
        s2 = run_stage(
            stage_name="STAGE 2:  pushforward ",
            stage_id="s2",
            train_loader=train_loader_s2, val_loader=val_loader_s2,
            patience=patience_s2, warmup_epochs=warmup_s2,
            ckpt_name="best_model",
            resume_ckpt_path=resume if os.path.exists(resume) else None,
            **stage_kwargs)
        tr_state["s2"]          = s2
        tr_state["s2_complete"] = True
        if rank == 0:
            save_training_state(exp_dir, tr_state)
    else:
        if rank == 0:
            print("Stage 2 complete — nothing to do.")
            if smi_logger:
                smi_logger.stop()
        s2 = tr_state["s2"]
        if distributed:
            dist.destroy_process_group()
        return

    if rank == 0:
        efficiency = {
            "exp_name":             exp_name,
            "model":                model_cfg["model"],
            "n_params":             n_params,
            "s1_best_val":          s1["best_val"],
            "s1_best_epoch":        s1["best_epoch"],
            "s1_time_to_best_s":    s1["time_to_best_s"],
            "s1_steps_to_best":     s1["steps_to_best"],
            "s2_best_val":          s2["best_val"],
            "s2_best_epoch":        s2["best_epoch"],
            "s2_time_to_best_s":    s2["time_to_best_s"],
            "s2_steps_to_best":     s2["steps_to_best"],
            "total_time_to_best_s": s1["time_to_best_s"] + s2["time_to_best_s"],
            "total_steps_to_best":  s1["steps_to_best"]  + s2["steps_to_best"],
            "total_wall_s":         time.time() - wall_start,
            "converged":            not (s1["hit_max_epochs"] or s2["hit_max_epochs"]),
        }
        with open(os.path.join(exp_dir, "efficiency.json"), "w") as f:
            json.dump(efficiency, f, indent=2)

        print(f"\n{'='*60}")
        print(f"DONE : {exp_name}")
        print(f"S1 best val (val_onestep)   : {s1['best_val']:.6f} "
              f"@ epoch {s1['best_epoch']}")
        print(f"S2 best val (val_stability) : {s2['best_val']:.6f} "
              f"@ epoch {s2['best_epoch']}")
        print(f"Wall time   : {(time.time() - wall_start) / 3600:.2f} h")
        print(f"Checkpoint  : {exp_dir}/best_model.pt")
        print(f"{'='*60}\n")

    if distributed:
        dist.destroy_process_group()
    if rank == 0 and smi_logger:
        smi_logger.stop()