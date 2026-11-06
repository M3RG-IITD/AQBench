import os
import json
import torch
import numpy as np


def prepare_stats(mean, std, tensor):
    """
    Broadcast mean/std to (B, H, W, T_out, num_targets)
    """

    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, dtype=tensor.dtype)
    if not torch.is_tensor(std):
        std = torch.tensor(std, dtype=tensor.dtype)

    mean = mean.to(tensor.device)
    std  = std.to(tensor.device)

    if mean.ndim == 0:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)

    return mean.view(1, 1, 1, 1, -1), \
           std.view(1, 1, 1, 1, -1)


def destandardize(tensor, mean, std):
    """Reverse z-score normalisation."""
    return tensor * std + mean


def save_inference_outputs(
    preds,
    targets,
    exp_name,
    mean,
    std,
    split="test",
    ckpt_dir="checkpoints",
    target_dir="reports/targets",
    task_name='pm25'
):
    """
    preds:   (N, H, W, T_out, num_targets)
    targets: (N, H, W, T_out, num_targets)
    """
    target_dir=os.path.join(target_dir,task_name)

    mean, std = prepare_stats(mean, std, preds)

    preds   = destandardize(preds,   mean, std)
    targets = destandardize(targets, mean, std)

    preds_np   = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()

    pred_dir = os.path.join(ckpt_dir, exp_name, "predictions", split)
    os.makedirs(pred_dir, exist_ok=True)
    np.save(os.path.join(pred_dir, "preds.npy"), preds_np)

    meta = {
        "exp_name": exp_name,
        "split":    split,
        "shape":    list(preds_np.shape),   
        "dtype":    str(preds_np.dtype),
        "axes":     "N, H, W, T_out, num_targets",
    }
    with open(os.path.join(pred_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"{split}_targets.npy")
    np.save(target_path, targets_np)

    target_meta = {
        "split":  split,
        "shape":  list(targets_np.shape),
        "dtype":  str(targets_np.dtype),
        "axes":   "N, H, W, T_out, num_targets",
    }
    with open(os.path.join(target_dir, f"{split}_meta.json"), "w") as f:
        json.dump(target_meta, f, indent=2)

    print(f"Saved predictions: {pred_dir}/preds.npy  shape={preds_np.shape}")