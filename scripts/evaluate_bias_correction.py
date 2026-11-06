import os
import argparse
import numpy as np
import torch
import yaml

from aqbench.registry.model_registry   import get_model
from aqbench.datasets.wrfchem_dataset  import WRFchemDataset
from aqbench.dataloaders.ft_loader     import build_ft_dataloader, FT_INDEX_FILES
from aqbench.evaluation.metrics import (
    compute_rmse, compute_mae, compute_smape,
    compute_bias, compute_correlation, compute_mfb,
)
import aqbench.models  

from aqbench.evaluation.evaluate import upsert_scalar_csv, update_leaderboard


BUNDLE_SIZE = 6


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def lb(out_dir, filename):
    return os.path.join(out_dir, filename)


def load_model(model_cfg, ckpt_path, device):
    model_class = get_model(model_cfg["model"])
    model = model_class(
        **model_cfg["model_args"],
        num_targets=1,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded: {ckpt_path}")
    return model


@torch.no_grad()
def run_model_inference(model, loader, device, pm25_tgt_idx, pm25_mean, pm25_std):
    """
    Single 6h forward pass per sample.
    Returns:
        preds   : (N, H, W, Tout)  float32  physical µg/m³
        targets : (N, H, W, Tout)  float32  physical µg/m³, NaN at non-station cells
    """
    all_preds, all_targets = [], []
    for x, y_gs, t_in in loader:
        x    = x.to(device, non_blocking=True)
        t_in = t_in.to(device, non_blocking=True)

        pred      = model(x, t_in)
        pred_pm25 = pred[..., pm25_tgt_idx]
        pred_phys = (pred_pm25 * pm25_std + pm25_mean).cpu().numpy()
        gs_phys   = (y_gs[..., 0] * pm25_std + pm25_mean).numpy()

        all_preds.append(pred_phys.transpose(0, 2, 3, 1))   
        all_targets.append(gs_phys.transpose(0, 2, 3, 1))

    return (np.concatenate(all_preds,   axis=0),
            np.concatenate(all_targets, axis=0))


def extract_wrfchem_raw(ft_test_ds, task_cfg, pm25_mean, pm25_std):
    """
    Extracts raw WRF-Chem PM2.5 for ft_test windows aligned with
    ground-station observations.
    Returns (wrf_raw, gs_obs) both (N, H, W, Tout) physical µg/m³.
    """
    wrf_ds = WRFchemDataset(
        index_file      = FT_INDEX_FILES["ft_test"],
        raw_dir         = task_cfg["raw_dir"],
        Tin             = task_cfg["Tin"],
        Tout            = task_cfg["Tout"],
        input_features  = task_cfg["input_features"],
        target_features = task_cfg["target_features"],
        y_aux_len       = 0,
    )
    print("  Prewarming WRFchemDataset for ft_test …")
    wrf_ds.prewarm()

    ft_keys  = {(int(ft_test_ds.years[i]),
                 int(ft_test_ds.months[i]),
                 int(ft_test_ds.starts[i])): i
                for i in range(len(ft_test_ds))}
    wrf_keys = {(int(wrf_ds.years[i]),
                 int(wrf_ds.months[i]),
                 int(wrf_ds.starts[i])): i
                for i in range(len(wrf_ds))}
    common = sorted(set(ft_keys) & set(wrf_keys))
    print(f"  Aligned samples: {len(common)}")

    all_wrf, all_gs = [], []
    for key in common:
        _, y_wrf, _, _ = wrf_ds[wrf_keys[key]]
        wrf_phys = y_wrf[..., 0].numpy() * pm25_std + pm25_mean   

        _, y_gs, _ = ft_test_ds[ft_keys[key]]
        gs_phys    = y_gs[..., 0].numpy() * pm25_std + pm25_mean   

        all_wrf.append(wrf_phys.transpose(1, 2, 0))  
        all_gs.append(gs_phys.transpose(1, 2, 0))

    return (np.stack(all_wrf, axis=0),
            np.stack(all_gs,  axis=0))


def build_test_eval_mask(ckpt_dir, exp_name, gs_obs, threshold=0.5):
    """
    Returns a per-sample per-cell boolean mask of shape (N, H, W).

    A cell is valid for a specific sample only if:
      1. The cell is in the train_mask — was supervised during S3 training
         (≥ threshold fraction of ft_train samples had obs there)
      2. That specific sample has non-NaN obs at ALL Tout timesteps
         at that cell — complete ground truth for the evaluation window

    This means different samples may be evaluated on different sets of cells.
    """
    import torch

    N, H, W, Tout = gs_obs.shape

    eval_mask_path = os.path.join(ckpt_dir, exp_name, "stage3", "eval_mask.pt")
    if not os.path.exists(eval_mask_path):
        print(f"  [WARN] eval_mask.pt not found: {eval_mask_path}")
        print(f"  [WARN] Falling back to all cells — "
              f"may include cells not supervised during S3")
        train_mask = np.ones((H, W), dtype=bool)
    else:
        train_mask = torch.load(eval_mask_path).numpy()   
        print(f"  Train coverage mask (eval_mask.pt): {train_mask.sum()} cells")


    sample_complete = np.all(~np.isnan(gs_obs), axis=3)   

    mask = sample_complete & train_mask[np.newaxis, :, :]  

    total_valid   = int(mask.sum())
    cells_ever    = int(mask.any(axis=0).sum())
    print(f"  Valid (sample, cell) pairs : {total_valid:,}  "
          f"({total_valid / (N * H * W) * 100:.4f}% of grid)")
    print(f"  Cells valid in ≥1 sample   : {cells_ever}")
    print(f"  Mean valid cells per sample: {mask.sum(axis=(1,2)).mean():.1f}")
    return mask


def compute_station_metrics(preds, targets, station_mask):
    """
    station_mask : (N, H, W) bool — per-sample per-cell validity.
                   Cell (h,w) is valid for sample n only if:
                     1. in train_mask (supervised during S3)
                     2. targets[n,h,w,:] has no NaN across all Tout steps

    Masks invalid entries in both preds and targets to NaN so metrics
    only aggregate over genuinely valid (sample, cell, timestep) entries.

    Returns:
        scalar_dict : {metric_name: float}   nanmean over T, for summary CSV
        arrays_dict : {metric_name: (T,)}    per-timestep, for leaderboard CSVs
    """
    mask_4d = station_mask[:, :, :, np.newaxis]          # (N, H, W, 1)
    p = preds.copy();   p[~np.broadcast_to(mask_4d, preds.shape)]   = np.nan
    t = targets.copy(); t[~np.broadcast_to(mask_4d, targets.shape)] = np.nan

    rmse_m,  rmse_s  = compute_rmse(p, t)
    mae_m,   mae_s   = compute_mae(p, t)
    smape_m, smape_s = compute_smape(p, t)
    bias_m,  bias_s  = compute_bias(p, t)
    mfb_m,   mfb_s   = compute_mfb(p, t)
    corr_m,  corr_s  = compute_correlation(p, t)

    arrays = {
        "rmse":      rmse_m,   "rmse_std":  rmse_s,
        "mae":       mae_m,    "mae_std":   mae_s,
        "smape":     smape_m,  "smape_std": smape_s,
        "bias":      bias_m,   "bias_std":  bias_s,
        "mfb":       mfb_m,    "mfb_std":   mfb_s,
        "corr":      corr_m,   "corr_std":  corr_s,
    }
    scalars = {k: float(np.nanmean(v)) for k, v in arrays.items()}
    scalars["n_station_cells"]      = int(station_mask.any(axis=0).sum())
    scalars["n_valid_sample_cells"] = int(station_mask.sum())

    return scalars, arrays


def save_leaderboard_csvs(out_dir, exp_name, split_label, arrays_dict):
    """Saves one CSV per metric, prefixed bc_ to avoid collision with main evals."""
    metric_to_file = {
        "rmse":      "leaderboard_bc_pm25_rmse.csv",
        "rmse_std":  "leaderboard_bc_pm25_rmse_std.csv",
        "mae":       "leaderboard_bc_pm25_mae.csv",
        "mae_std":   "leaderboard_bc_pm25_mae_std.csv",
        "smape":     "leaderboard_bc_pm25_smape.csv",
        "smape_std": "leaderboard_bc_pm25_smape_std.csv",
        "bias":      "leaderboard_bc_pm25_bias.csv",
        "bias_std":  "leaderboard_bc_pm25_bias_std.csv",
        "mfb":       "leaderboard_bc_pm25_mfb.csv",
        "mfb_std":   "leaderboard_bc_pm25_mfb_std.csv",
        "corr":      "leaderboard_bc_pm25_corr.csv",
        "corr_std":  "leaderboard_bc_pm25_corr_std.csv",
    }
    for metric, fname in metric_to_file.items():
        update_leaderboard(
            lb(out_dir, fname),
            exp_name,
            split_label,
            arrays_dict[metric],
        )


def print_metrics(label, m):
    print(f"  {label:<12}  RMSE={m['rmse']:7.3f}  MAE={m['mae']:7.3f}  "
          f"SMAPE={m['smape']:6.2f}  Corr={m['corr']:6.3f}  "
          f"Bias={m['bias']:+7.3f}  MFB={m['mfb']:+6.3f}")


def evaluate_bias_correction(
    model_name:  str,
    exp_name:    str,
    task_cfg:    dict,
    model_cfg:   dict,
    skip_s3:     bool = False,
    ckpt_dir:    str  = "checkpoints",
    report_dir:  str  = "reports/bias_correction",
):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(report_dir, "ft_test")
    os.makedirs(out_dir, exist_ok=True)

    task_cfg["s3_batch_size"] = 32
    target_features = task_cfg["target_features"]
    pm25_tgt_idx    = target_features.index("pm25")

    print("\nLoading ft_test (ground station) …")
    test_loader, test_ds = build_ft_dataloader(task_cfg, "ft_test")
    pm25_mean = test_ds.pm25_mean
    pm25_std  = test_ds.pm25_std
    print(f"  PM2.5  mean={pm25_mean:.4f}  std={pm25_std:.4f}")
    print(f"  ft_test samples: {len(test_ds)}")

    results_scalars = {}

    print(f"\n{'='*60}")
    print("WRF-Chem raw (pre-NN) vs ground station")
    wrf_raw, gs_obs = extract_wrfchem_raw(test_ds, task_cfg, pm25_mean, pm25_std)

    station_mask = build_test_eval_mask(ckpt_dir, exp_name, gs_obs)
    np.save(os.path.join(out_dir, "station_mask.npy"), station_mask)
    np.save(os.path.join(out_dir, "targets_gs.npy"),   gs_obs)
    np.save(os.path.join(out_dir, "preds_wrfraw.npy"), wrf_raw)

    sc, arr = compute_station_metrics(wrf_raw, gs_obs, station_mask)
    results_scalars["wrf_raw"] = sc
    save_leaderboard_csvs(out_dir, exp_name, "ft_test_wrf_raw", arr)
    print_metrics("WRF-Chem raw", sc)

    s2_ckpt = os.path.join(ckpt_dir, exp_name, "best_model.pt")
    if not os.path.exists(s2_ckpt):
        raise FileNotFoundError(f"S2 checkpoint not found: {s2_ckpt}")

    print(f"\n{'='*60}")
    print(f"S2 model ({exp_name})")
    model_s2 = load_model(model_cfg, s2_ckpt, device)
    preds_s2, _ = run_model_inference(
        model_s2, test_loader, device, pm25_tgt_idx, pm25_mean, pm25_std)
    del model_s2

    np.save(os.path.join(out_dir, "preds_s2.npy"), preds_s2)
    sc, arr = compute_station_metrics(preds_s2, gs_obs, station_mask)
    results_scalars["s2"] = sc
    save_leaderboard_csvs(out_dir, exp_name, "ft_test_s2", arr)
    print_metrics("S2", sc)

    s3_ckpt = os.path.join(ckpt_dir, exp_name, "stage3", "stage3_best.pt")
    if skip_s3:
        print("\nSkipping S3 (--skip_s3)")
    elif not os.path.exists(s3_ckpt):
        print(f"\n[WARN] S3 checkpoint not found: {s3_ckpt} — skipping")
    else:
        print(f"\n{'='*60}")
        print(f"S3 model ({exp_name}/stage3)")
        model_s3 = load_model(model_cfg, s3_ckpt, device)
        preds_s3, _ = run_model_inference(
            model_s3, test_loader, device, pm25_tgt_idx, pm25_mean, pm25_std)
        del model_s3

        np.save(os.path.join(out_dir, "preds_s3.npy"), preds_s3)
        sc, arr = compute_station_metrics(preds_s3, gs_obs, station_mask)
        results_scalars["s3"] = sc
        save_leaderboard_csvs(out_dir, exp_name, "ft_test_s3", arr)
        print_metrics("S3", sc)

    print(f"\n{'='*60}")
    print(f"{'':12}  {'RMSE':>7}  {'MAE':>7}  {'SMAPE':>6}  "
          f"{'Corr':>6}  {'Bias':>7}  {'MFB':>6}")
    print("-" * 60)
    for label, m in results_scalars.items():
        print(f"  {label:<12}  {m['rmse']:7.3f}  {m['mae']:7.3f}  "
              f"{m['smape']:6.2f}  {m['corr']:6.3f}  "
              f"{m['bias']:+7.3f}  {m['mfb']:+6.3f}")

    if "s2" in results_scalars and "s3" in results_scalars:
        print(f"\n  Δ S2→S3 (negative = S3 better):")
        for metric in ("rmse", "mae", "smape", "bias", "mfb"):
            delta = results_scalars["s3"][metric] - results_scalars["s2"][metric]
            sign  = "↓ better" if delta < 0 else "↑ worse"
            print(f"    {metric:8s}: {delta:+.4f}  {sign}")

    csv_path = os.path.join(out_dir, "bias_correction_eval.csv")
    for label, m in results_scalars.items():
        upsert_scalar_csv(csv_path, exp_name, f"ft_test_{label}", m)
    print(f"\n  Summary  → {csv_path}")
    print(f"  Per-step → {out_dir}/leaderboard_bc_pm25_*.csv")

    return results_scalars


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True)
    p.add_argument("--exp_name",   required=True)
    p.add_argument("--task_cfg",   default="configs/task/bias_correction_pm25.yaml")
    p.add_argument("--model_cfg",  default=None)
    p.add_argument("--ckpt_dir",   default="checkpoints")
    p.add_argument("--report_dir", default="reports/bias_correction")
    p.add_argument("--skip_s3",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task_cfg  = load_yaml(args.task_cfg)
    model_path = (args.model_cfg
                  or os.path.join("aqbench", "models", args.model, "config.yaml"))
    model_cfg  = load_yaml(model_path)
    model_cfg["exp_name"] = args.exp_name
    task_cfg["s3_batch_size"] = 32

    evaluate_bias_correction(
        model_name  = args.model,
        exp_name    = args.exp_name,
        task_cfg    = task_cfg,
        model_cfg   = model_cfg,
        skip_s3     = args.skip_s3,
        ckpt_dir    = args.ckpt_dir,
        report_dir  = args.report_dir,
    )