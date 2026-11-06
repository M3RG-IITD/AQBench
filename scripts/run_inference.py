import os
import argparse
import yaml
import torch
from tqdm import tqdm

from aqbench.dataloaders.wrfchem_loader import build_dataloader
from aqbench.registry.model_registry import get_model
from aqbench.utils.checkpoint import load_checkpoint
from aqbench.evaluation.inference import save_inference_outputs
from aqbench.utils.seeding import set_seed
import aqbench.models  # noqa: F401

BUNDLE_SIZE     = 6
BASELINE_MODELS = {"persistence"}



def infer_eval_tout(split: str, task_cfg: dict) -> int:
    """

    test_24 / test_epi_24   → 24
    test_72 / test_epi_72   → 72
    test_168 / test_epi_168 → 168
    else Tout
    """
    for hours in (168, 72, 24):
        if split.endswith(str(hours)):
            return hours
    return task_cfg["Tout"]


def build_target_input_map(dataset) -> list:
    """
    Return list of (tgt_i, inp_i) pairs mapping each target feature to its
    position in the input feature list (where it exists).
    """
    inp_feat = dataset.input_features
    tgt_feat = dataset.target_features
    return [
        (tgt_i, inp_feat.index(feat))
        for tgt_i, feat in enumerate(tgt_feat)
        if feat in inp_feat
    ]


def build_next_input(x_cur, pred_bundle, aux_bundle, target_input_map):
    """
    Slide the input window forward by BUNDLE_SIZE.

    x_cur        : (B, Tin,         H, W, C_in)
    pred_bundle  : (B, BUNDLE_SIZE, H, W, C_tgt)  — model output, detached
    aux_bundle   : (B, BUNDLE_SIZE, H, W, C_in)   — ground-truth met+emissions

    Returns new x of shape (B, Tin, H, W, C_in).
    """
    new_frames = aux_bundle.clone()
    for tgt_i, inp_i in target_input_map:
        new_frames[..., inp_i] = pred_bundle[..., tgt_i]

    Tin  = x_cur.shape[1]
    Tout = pred_bundle.shape[1]  
    if Tin <= Tout:
        return new_frames[:, -Tin:]
    return torch.cat([x_cur[:, Tout:], new_frames], dim=1)



def run_persistence(dataloader, device, eval_Tout, target_input_map):
    """
    The last observed frame for each target feature is read from x[:, -1]
    using target_input_map and copied across

    Returns:
        preds   : (N, H, W, eval_Tout, C_tgt)
        targets : (N, H, W, eval_Tout, C_tgt)
    """
    preds_list   = []
    targets_list = []
    C_tgt        = len(dataloader.dataset.target_features)

    with torch.no_grad():
        for x, y, _, _ in tqdm(dataloader, desc="Persistence", leave=True):
            x = x.to(device)   
            y = y.to(device)   

            B, _, H, W, _ = x.shape

            # last observed target values from the input window
            last_frame = torch.zeros(B, H, W, C_tgt, device=device)
            for tgt_i, inp_i in target_input_map:
                last_frame[..., tgt_i] = x[:, -1, ..., inp_i]

            pred = last_frame.unsqueeze(1).expand(-1, eval_Tout, -1, -1, -1)

            preds_list.append(pred.permute(0, 2, 3, 1, 4).cpu())
            targets_list.append(y.permute(0, 2, 3, 1, 4).cpu())

    preds   = torch.cat(preds_list,   dim=0)
    targets = torch.cat(targets_list, dim=0)

    assert preds.shape == targets.shape, \
        f"Shape mismatch: {preds.shape} vs {targets.shape}"
    print(f"Persistence complete — shape: {preds.shape}  "
          f"(N, H, W, T_out={eval_Tout}, C_tgt={C_tgt})")
    return preds, targets



def run_inference(model, dataloader, device, eval_Tout, target_input_map):
    """
    Bundled autoregressive rollout.

    y_aux provides ground-truth met + emissions for every future step;
    only target feature channels are overwritten with model predictions.

    Returns:
        preds   : (N, H, W, eval_Tout, C_tgt)
        targets : (N, H, W, eval_Tout, C_tgt)
    """
    model.eval()
    num_bundles  = eval_Tout // BUNDLE_SIZE
    preds_list   = []
    targets_list = []

    with torch.no_grad():
        for x, y, y_aux, t_in in tqdm(dataloader, desc="Inference", leave=True):
            x     = x.to(device)      
            y     = y.to(device)      
            y_aux = y_aux.to(device)  
            t_in  = t_in.to(device)   

            x_cur    = x
            t_in_cur = t_in
            bundle_preds = []

            for b in range(num_bundles):
                pred = model(x_cur, t_in_cur)  
                bundle_preds.append(pred)

                if b < num_bundles - 1:
                    aux_b  = y_aux[:, b * BUNDLE_SIZE : (b + 1) * BUNDLE_SIZE]
                    x_cur    = build_next_input(x_cur, pred.detach(),
                                               aux_b, target_input_map)
                    t_in_cur = t_in_cur + BUNDLE_SIZE

           
            full_pred = torch.cat(bundle_preds, dim=1)

            preds_list.append(full_pred.permute(0, 2, 3, 1, 4).cpu())
            targets_list.append(y.permute(0, 2, 3, 1, 4).cpu())

    preds   = torch.cat(preds_list,   dim=0)
    targets = torch.cat(targets_list, dim=0)

    assert preds.shape == targets.shape, \
        f"Shape mismatch: {preds.shape} vs {targets.shape}"
    print(f"Inference complete — shape: {preds.shape}  "
          f"(N, H, W, T_out={eval_Tout}, C_tgt={preds.shape[-1]})")
    return preds, targets



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.task_config) as f:
        task_cfg = yaml.safe_load(f)
    task_name = task_cfg['task_name']

    task_cfg["batch_size"] = args.batch_size
    set_seed(task_cfg.get("seed", 42))

    eval_Tout = args.eval_Tout or infer_eval_tout(args.split, task_cfg)
    num_bundles = eval_Tout // BUNDLE_SIZE

    is_baseline = args.model in BASELINE_MODELS

    if is_baseline:
        model_cfg = {
            "model":      args.model,
            "exp_name":   args.exp_name,
            "model_args": {},
        }
    else:
        config_filename = args.model_config or "config.yaml"
        config_path     = os.path.join("aqbench", "models",
                                       args.model, config_filename)
        with open(config_path) as f:
            model_cfg = yaml.safe_load(f)
        if args.exp_name is not None:
            model_cfg["exp_name"] = args.exp_name

    exp_name = args.exp_name or model_cfg["exp_name"]

    print(f"\nInference : {exp_name}")
    print(f"  task    : {args.task_config}")
    print(f"  targets : {task_cfg['target_features']}")
    print(f"  split   : {args.split}  |  eval_Tout={eval_Tout}  "
          f"({num_bundles} bundles × {BUNDLE_SIZE}h)")
    print(f"  batch   : {task_cfg['batch_size']}  |  device: {device}  "
          f"|  baseline: {is_baseline}")


    dataloader = build_dataloader(
        task_cfg,
        args.split,
        Tout      = eval_Tout,
        y_aux_len = eval_Tout - BUNDLE_SIZE,
    )
    print(f"  samples : {len(dataloader.dataset)}")

    target_input_map = build_target_input_map(dataloader.dataset)

    if is_baseline:
        preds, targets = run_persistence(
            dataloader, device, eval_Tout, target_input_map)
    else:
        model_class = get_model(model_cfg["model"])
        model       = model_class(
            **model_cfg["model_args"],
            num_targets=len(task_cfg["target_features"]),
        )
        model, _ = load_checkpoint(model, exp_name, device)
        model     = model.to(device)

        preds, targets = run_inference(
            model, dataloader, device, eval_Tout, target_input_map)

    mean = dataloader.dataset.target_mean.reshape(-1)
    std  = dataloader.dataset.target_std.reshape(-1)

    save_inference_outputs(
        preds    = preds,
        targets  = targets,
        exp_name = exp_name,
        mean     = mean,
        std      = std,
        split    = args.split,
        task_name = task_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bundled autoregressive inference for AQBench models.")
    parser.add_argument("--model",        required=True,
                        help="Model name or 'persistence'")
    parser.add_argument("--task_config",  required=True)
    parser.add_argument("--exp_name",     required=True)
    parser.add_argument("--model_config", default=None,
                        help="Model config filename (default: config.yaml)")
    parser.add_argument("--split",        default="test_168",
                        help="Data split, e.g. test_24, test_168, test_epi_72")
    parser.add_argument("--batch_size",   default=4, type=int)
    parser.add_argument("--eval_Tout",    default=None, type=int,
                        help="Override inferred eval_Tout (must be multiple of 6)")
    args = parser.parse_args()
    main(args)