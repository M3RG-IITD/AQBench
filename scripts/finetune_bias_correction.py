import os
import math
import random
import argparse
import csv

import numpy as np
import torch
import torch.nn as nn
import yaml

from aqbench.registry.model_registry import get_model
import aqbench.models  
from aqbench.dataloaders.ft_loader import build_ft_dataloader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_configs(args):
    task_path  = (args.task_cfg
                  or os.path.join("configs", "task", "bias_correction_pm25.yaml"))
    model_path = (args.model_cfg
                  or os.path.join("aqbench", "models", args.model, "config.yaml"))

    for p in (task_path, model_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Config not found: {p}")

    task_cfg  = load_yaml(task_path)
    model_cfg = load_yaml(model_path)

    if task_cfg is None:
        raise ValueError(f"Task config is empty: {task_path}")
    if model_cfg is None:
        raise ValueError(f"Model config is empty: {model_path}")

    model_cfg["exp_name"] = args.exp_name

    task_cfg["s3_batch_size"] = 24

    print(f"Task config  : {task_path}")
    print(f"Model config : {model_path}")
    return task_cfg, model_cfg


def masked_mse(pred_pm25: torch.Tensor,
               obs:       torch.Tensor):
    """
    MSE at non-NaN cells only. Both tensors (B, Tout, H, W) normalised.
    """
    mask    = ~torch.isnan(obs)
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return torch.tensor(0.0, device=pred_pm25.device, requires_grad=True), 0

    diff = (pred_pm25 - obs).nan_to_num(nan=0.0)  
    loss = (diff.pow(2) * mask).sum() / n_valid     
    return loss, n_valid


def denorm_pm25(x_norm: torch.Tensor,
                mean: float, std: float) -> torch.Tensor:
    return x_norm * std + mean


def get_scheduler(optimizer, warmup: int, total: int, min_factor: float = 0.01):
    def lr_lambda(e):
        if e < warmup:
            return (e + 1) / max(warmup, 1)
        progress = (e - warmup) / max(total - warmup, 1)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, device,
                pm25_tgt_idx, pm25_mean, pm25_std):
    model.train()
    total_loss, n_batches = 0.0, 0

    for x, y, t_in in loader:
        x    = x.to(device, non_blocking=True)
        t_in = t_in.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)[..., 0]     

        pred      = model(x, t_in)                          
        pred_pm25 = pred[..., pm25_tgt_idx]                 

        loss, n_valid = masked_mse(pred_pm25, y)
        if n_valid == 0:
            continue                                         # no obs then skip batch

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device,
             pm25_tgt_idx, pm25_mean, pm25_std,
             eval_mask: torch.Tensor):

    model.eval()
    total_mse, sum_se, n_pts, n_batches = 0.0, 0.0, 0, 0

    mask_dev = eval_mask.unsqueeze(0).unsqueeze(0).to(device)  

    for x, y, t_in in loader:
        x    = x.to(device, non_blocking=True)
        t_in = t_in.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)[..., 0]         

        # Restrict to eval mask cells
        y = y.masked_fill(~mask_dev, float("nan"))

        pred      = model(x, t_in)
        pred_pm25 = pred[..., pm25_tgt_idx]

        mse_loss, n_valid = masked_mse(pred_pm25, y)
        if n_valid == 0:
            continue


        valid     = ~torch.isnan(y)
        pred_phys = denorm_pm25(pred_pm25, pm25_mean, pm25_std)
        y_phys    = denorm_pm25(y,         pm25_mean, pm25_std)
        se        = (pred_phys - y_phys).nan_to_num(nan=0.0).pow(2) * valid
        sum_se   += float(se.sum().item())

        n_pts     += n_valid
        total_mse += float(mse_loss.item())
        n_batches += 1

    val_rmse = math.sqrt(sum_se / max(n_pts, 1))
    val_mse  = total_mse / max(n_batches, 1)
    return val_mse, val_rmse


def run_finetune(task_cfg: dict, model_cfg: dict,
                 s2_ckpt_path: str, out_dir: str):

    set_seed(task_cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    LR         = model_cfg["learning_rate"] / 100
    MAX_EPOCHS = 100
    PATIENCE   = 10
    WARMUP     = 5
    THRESHOLD  = task_cfg.get("eval_coverage_threshold", 0.5)

    target_features = task_cfg["target_features"]
    model_class     = get_model(model_cfg["model"])
    model = model_class(
        **model_cfg["model_args"],
        num_targets=len(target_features),
    ).to(device)

    if not os.path.exists(s2_ckpt_path):
        raise FileNotFoundError(f"S2 checkpoint not found: {s2_ckpt_path}")
    ckpt = torch.load(s2_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded S2 checkpoint: {s2_ckpt_path}")

    train_loader, train_ds = build_ft_dataloader(task_cfg, "ft_train")
    val_loader,   val_ds   = build_ft_dataloader(task_cfg, "ft_val")

    pm25_tgt_idx = target_features.index("pm25")
    pm25_mean    = train_ds.pm25_mean
    pm25_std     = train_ds.pm25_std
    print(f"\nPM2.5 — mean={pm25_mean:.4f}  std={pm25_std:.4f}")

    print("\nBuilding eval mask from ft_train …")
    eval_mask = train_ds.build_eval_mask(val_ds=val_ds, threshold=THRESHOLD)
    torch.save(eval_mask, os.path.join(out_dir, "eval_mask.pt"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = get_scheduler(optimizer, WARMUP, MAX_EPOCHS)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"STAGE 3  Bias-correction finetune")
    print(f"  Model     : {model_cfg['model']}  ({n_params:,} params)")
    print(f"  S2 ckpt   : {s2_ckpt_path}")
    print(f"  Output    : {out_dir}")
    print(f"  LR={LR:.2e}  epochs={MAX_EPOCHS}  patience={PATIENCE}  warmup={WARMUP}")
    print(f"  Eval mask : coverage>={THRESHOLD:.0%}  "
          f"({eval_mask.sum().item()} cells, built from ft_train)")
    print(f"{'='*60}\n")
    print(f"{'Ep':>5} | {'LR':>9} | {'TrnLoss':>9} | "
          f"{'ValMSE':>9} | {'ValRMSE':>8} | Best")
    print("-" * 60)

    best_val, best_epoch, wait = float("inf"), 0, 0
    csv_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss     = train_epoch(model, train_loader, optimizer, device,
                                  pm25_tgt_idx, pm25_mean, pm25_std)
        vrl2, vrmse = validate(model, val_loader, device,
                               pm25_tgt_idx, pm25_mean, pm25_std, eval_mask)
        scheduler.step()

        is_best = vrl2 < best_val
        if is_best:
            best_val, best_epoch, wait = vrl2, epoch, 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_mse":     vrl2,
                "val_rmse":    vrmse,
                "pm25_mean":   pm25_mean,
                "pm25_std":    pm25_std,
            }, os.path.join(out_dir, "stage3_best.pt"))
        else:
            wait += 1

        lr     = optimizer.param_groups[0]["lr"]
        marker = " *" if is_best else ""
        print(f"{epoch:5d} | {lr:.3e} | {tr_loss:9.5f} | "
              f"{vrl2:9.5f} | {vrmse:8.3f} | {best_val:.5f}{marker}")

        csv_rows.append(dict(epoch=epoch, lr=lr, train_loss=tr_loss,
                             val_mse=vrl2, val_rmse=vrmse, is_best=int(is_best)))

        if wait >= PATIENCE:
            print(f"\n>>> Early stop at epoch {epoch}  (best={best_epoch})")
            break

    log_path = os.path.join(out_dir, "stage3_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader(); w.writerows(csv_rows)

    print(f"\nBest val RL2 : {best_val:.6f} @ epoch {best_epoch}")
    print(f"Checkpoint   : {out_dir}/stage3_best.pt")
    print(f"Eval mask    : {out_dir}/eval_mask.pt")
    print(f"Log          : {log_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     required=True)
    p.add_argument("--exp_name",  required=True)
    p.add_argument("--task_cfg",  default=None)
    p.add_argument("--model_cfg", default=None)
    p.add_argument("--s2_ckpt",   default=None)
    p.add_argument("--out_dir",   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task_cfg, model_cfg = load_configs(args)
    s2_ckpt = args.s2_ckpt or os.path.join("checkpoints", args.exp_name, "best_model.pt")
    out_dir = args.out_dir  or os.path.join("checkpoints", args.exp_name, "stage3")
    run_finetune(task_cfg, model_cfg, s2_ckpt, out_dir)