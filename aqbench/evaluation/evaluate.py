import os
import numpy as np
import pandas as pd
import h5py
from aqbench.evaluation.metrics import (
    compute_rmse, compute_mae, compute_smape,
    compute_bias, compute_correlation,
    compute_r2, compute_fac2,
    compute_episode_metrics, compute_exceedance_metrics,compute_frmse
)
from aqbench.evaluation.constraint_violations import compute_constraint_violations



def compute_and_save_standard_metrics(preds_4d, targets_4d, feat,
                                       exp_name, split, report_dir):
    """
    preds_4d / targets_4d : (N, H, W, T_out) in physical units.

    Saves per-timestep leaderboard CSVs for every metric and its std,
    and fRMSE in low / mid / high frequency bands.
    Returns a scalar summary dict (nanmean over T).
    """
    rmse_mean,  rmse_std  = compute_rmse(preds_4d, targets_4d)
    mae_mean,   mae_std   = compute_mae(preds_4d, targets_4d)
    smape_mean, smape_std = compute_smape(preds_4d, targets_4d)
    bias_mean,  bias_std  = compute_bias(preds_4d, targets_4d)
    corr_mean,  corr_std  = compute_correlation(preds_4d, targets_4d)

    frmse_low, frmse_mid, frmse_high = compute_frmse(preds_4d, targets_4d)

    for name, arr in [
        (f"leaderboard_{feat}_rmse.csv",        rmse_mean),
        (f"leaderboard_{feat}_rmse_std.csv",    rmse_std),
        (f"leaderboard_{feat}_mae.csv",         mae_mean),
        (f"leaderboard_{feat}_mae_std.csv",     mae_std),
        (f"leaderboard_{feat}_smape.csv",       smape_mean),
        (f"leaderboard_{feat}_smape_std.csv",   smape_std),
        (f"leaderboard_{feat}_bias.csv",        bias_mean),
        (f"leaderboard_{feat}_bias_std.csv",    bias_std),
        (f"leaderboard_{feat}_corr.csv",        corr_mean),
        (f"leaderboard_{feat}_corr_std.csv",    corr_std),
        (f"leaderboard_{feat}_frmse_low.csv",   frmse_low),
        (f"leaderboard_{feat}_frmse_mid.csv",   frmse_mid),
        (f"leaderboard_{feat}_frmse_high.csv",  frmse_high),
    ]:
        update_leaderboard(lb(report_dir, name), exp_name, split, arr)

    print(f"  [{feat}] RMSE={np.nanmean(rmse_mean):.4f}  "
          f"MAE={np.nanmean(mae_mean):.4f}  "
          f"SMAPE={np.nanmean(smape_mean):.4f}  "
          f"Corr={np.nanmean(corr_mean):.4f}  "
          f"Bias={np.nanmean(bias_mean):.4f}  "
          f"fRMSE[L/M/H]={np.nanmean(frmse_low):.4f}/"
          f"{np.nanmean(frmse_mid):.4f}/"
          f"{np.nanmean(frmse_high):.4f}")

    return {
        f"{feat}_rmse":       float(np.nanmean(rmse_mean)),
        f"{feat}_rmse_std":   float(np.nanmean(rmse_std)),
        f"{feat}_mae":        float(np.nanmean(mae_mean)),
        f"{feat}_mae_std":    float(np.nanmean(mae_std)),
        f"{feat}_smape":      float(np.nanmean(smape_mean)),
        f"{feat}_smape_std":  float(np.nanmean(smape_std)),
        f"{feat}_bias":       float(np.nanmean(bias_mean)),
        f"{feat}_bias_std":   float(np.nanmean(bias_std)),
        f"{feat}_corr":       float(np.nanmean(corr_mean)),
        f"{feat}_corr_std":   float(np.nanmean(corr_std)),
        f"{feat}_frmse_low":  float(np.nanmean(frmse_low)),
        f"{feat}_frmse_mid":  float(np.nanmean(frmse_mid)),
        f"{feat}_frmse_high": float(np.nanmean(frmse_high)),
    }


def update_leaderboard(csv_path, exp_name, split, values, col_prefix="t"):
    index_name = f"{exp_name}_{split}"
    cols = ["experiment"] + [f"{col_prefix}{i+1}" for i in range(len(values))]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=cols)
    for col in cols:
        if col not in df.columns:
            df[col] = None
    if index_name in df["experiment"].values:
        idx = df.index[df["experiment"] == index_name][0]
        df.loc[idx, cols[1:]] = values
    else:
        vals = list(values)
        expected = len(df.columns) - 1
        if len(vals) < expected:
            vals += [""] * (expected - len(vals))
        df.loc[len(df)] = [index_name] + vals
    df.to_csv(csv_path, index=False)


def upsert_scalar_csv(csv_path, exp_name, split, row_dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    index_name = f"{exp_name}_{split}"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["experiment"] + list(row_dict.keys()))
    for col in row_dict:
        if col not in df.columns:
            df[col] = None
    if index_name in df["experiment"].values:
        idx = df.index[df["experiment"] == index_name][0]
        for col, val in row_dict.items():
            df.loc[idx, col] = val
    else:
        df.loc[len(df)] = {"experiment": index_name, **row_dict}
    df.to_csv(csv_path, index=False)


def lb(report_dir, filename):
    return os.path.join(report_dir, filename)




def get_report_dir(base_report_dir, target_features, split):
    subdir = target_features[0] if len(target_features) == 1 else "mul"
    return os.path.join(base_report_dir, subdir, split)




def filter_index_by_horizon(index_df, T_in, T_out, raw_dir):
    h5_cache = {}
    valid = []
    for i, row in index_df.iterrows():
        year  = int(row["year"])
        month = int(row["month"])
        start = int(row["start"])
        if year not in h5_cache:
            h5_cache[year] = {}
        if month not in h5_cache[year]:
            path = os.path.join(raw_dir, f"wrfchem_{year}.h5")
            with h5py.File(path, "r") as f:
                key = f"month_{month:02d}"
                h5_cache[year][month] = f[key]["data"].shape[0] if key in f else 0
        if start + T_in + T_out <= h5_cache[year][month]:
            valid.append(i)
    return index_df.loc[valid].reset_index(drop=True)



def evaluate_bias_correction_split(
    model_name:  str,
    exp_name:    str,
    model:       str  = None,           
    task_cfg:    dict = None,
    model_cfg:   dict = None,
    skip_s3:     bool = False,
    ckpt_dir:    str  = "checkpoints",
    report_dir:  str  = "reports/bias_correction",
    task_cfg_path: str = "configs/task/bias_correction_pm25.yaml",
):

    import yaml
    from evaluate_bias_correction import evaluate_bias_correction

    model_name = model_name or model
    if model_name is None:
        raise ValueError("model_name is required")

    if task_cfg is None:
        with open(task_cfg_path) as f:
            task_cfg = yaml.safe_load(f)
        task_cfg["s3_batch_size"] = 32

    if model_cfg is None:
        cfg_path = os.path.join("aqbench", "models", model_name, "config.yaml")
        with open(cfg_path) as f:
            model_cfg = yaml.safe_load(f)
        model_cfg["exp_name"] = exp_name

    return evaluate_bias_correction(
        model_name = model_name,
        exp_name   = exp_name,
        task_cfg   = task_cfg,
        model_cfg  = model_cfg,
        skip_s3    = skip_s3,
        ckpt_dir   = ckpt_dir,
        report_dir = report_dir,
    )



def load_episode_data(index_df, N, H, W, T_out, T_in, epi_mask_dir):
    epi_masks    = np.zeros((N, H, W, T_out), dtype=np.uint8)
    trend        = np.zeros((N, H, W, T_out), dtype=np.float32)
    seasonal     = np.zeros((N, H, W, T_out), dtype=np.float32)
    residual_std = np.zeros((N, H, W),        dtype=np.float32)
    epi_files    = {}
    found_any    = False

    try:
        for i, row in index_df.iterrows():
            year  = int(row["year"])
            month = int(row["month"])
            start = int(row["start"])
            sl    = slice(start + T_in, start + T_in + T_out)

            epi_path = os.path.join(epi_mask_dir, f"{year}_epi.h5")
            if not os.path.exists(epi_path):
                continue
            if year not in epi_files:
                epi_files[year] = h5py.File(epi_path, "r")
                found_any = True

            grp = epi_files[year].get(f"month_{month:02d}")
            if grp is None:
                continue

            mask_raw = grp["episode_mask"][sl].transpose(1, 2, 0)
            if mask_raw.shape[2] < T_out:
                continue

            epi_masks[i]    = mask_raw
            trend[i]        = grp["trend"][sl].transpose(1, 2, 0)
            seasonal[i]     = grp["seasonal"][sl].transpose(1, 2, 0)
            residual_std[i] = grp["residual_std"][:]
    finally:
        for f in epi_files.values():
            f.close()

    if not found_any:
        print(f"  [warn] No epi mask files in '{epi_mask_dir}' — masks all zero.")
    return epi_masks, trend, seasonal, residual_std




def compute_and_save_episode_metrics(
    p25, t25, exp_name, split, report_dir,
    N, H, W, T_out, T_in,
    episode_index, epi_mask_dir,
    raw_dir="data/raw",
):
    """
    p25 / t25 : (N, H, W, T_out) PM2.5 in physical units.

    Saves:
      - episode-masked rmse, smape, mfb  (grids with epi_mask=1 only)
      - exceedance classification metrics (>90 µg/m³, independent masks)
    """
    idx_path = episode_index or f"data/processed/{split}_index.parquet"
    if not os.path.exists(idx_path):
        print(f"  [skip] episode_eval=True but index not found: {idx_path}")
        return {}

    print(f"\n  Episode metrics on PM2.5 (index: {idx_path}) ...")
    index_df = pd.read_parquet(idx_path).reset_index(drop=True)
    index_df = filter_index_by_horizon(index_df, T_in, T_out, raw_dir)

    assert len(index_df) == N, (
        f"Episode index has {len(index_df)} samples after filtering "
        f"but predictions have {N}. "
        f"Check T_in={T_in}, T_out={T_out} match the inference run."
    )

    epi_masks, trend, seasonal, residual_std = load_episode_data(
        index_df, N, H, W, T_out, T_in, epi_mask_dir
    )
    print(f"  GT episode coverage: {epi_masks.sum() / epi_masks.size * 100:.2f}%")

    summary = {}

    epi_reg = compute_episode_metrics(p25, t25, epi_masks)
    for name, arr in epi_reg.items():
        update_leaderboard(
            lb(report_dir, f"leaderboard_{name}.csv"),
            exp_name, split, arr)
        summary[name]          = float(np.nanmean(arr))
        summary[f"{name}_std"] = float(np.nanstd(arr))

    print(f"  [epi] RMSE={summary['epi_rmse']:.4f}  "
          f"SMAPE={summary['epi_smape']:.4f}  "
          f"MFB={summary['epi_mfb']:.4f}")

    exc = compute_exceedance_metrics(p25, t25, threshold=90.0)
    for name, arr in exc.items():
        update_leaderboard(
            lb(report_dir, f"leaderboard_{name}.csv"),
            exp_name, split, arr)
        summary[name] = float(arr.mean())

    print(f"  [exc>90] Recall={summary['exc_recall']:.4f}  "
          f"Precision={summary['exc_precision']:.4f}  "
          f"F1={summary['exc_f1']:.4f}  "
          f"CSI={summary['exc_csi']:.4f}")

    upsert_scalar_csv(
        lb(report_dir, "leaderboard_episode_summary.csv"),
        exp_name, split, summary)

    return summary




def compute_and_save_pde_metrics(
    exp_name, split, report_dir, ckpt_dir,
    raw_dir, grid_height_path, T_in, T_out, pm25_channel,
):
    """
    Computes PDE residual metrics and saves:
      - leaderboard_pde_residual.csv  — per-hour spatial-mean RMSE (existing)
      - pde_grid_rmse/<exp>_<split>_pde_grid_rmse.npy
            Shape: (N, T_out-1, H, W), dtype float32
            Pointwise RMSE of the advection-diffusion residual at every
            grid cell, timestep, and sample. 
    """
    from aqbench.evaluation.pde_residual import compute_pde_residual_metrics

    grid_save_dir  = os.path.join(report_dir, "pde_grid_rmse")
    grid_rmse_path = os.path.join(grid_save_dir,
                                  f"{exp_name}_{split}_pde_grid_rmse.npy")

    pde_result = compute_pde_residual_metrics(
        exp_name         = exp_name,
        split            = split,
        ckpt_dir         = ckpt_dir,
        raw_dir          = raw_dir,
        grid_height_path = grid_height_path,
        T_in             = T_in,
        T_out            = T_out,
        pm25_channel     = pm25_channel,
        grid_rmse_path   = grid_rmse_path,   
    )

    update_leaderboard(
        lb(report_dir, "leaderboard_pde_residual.csv"),
        exp_name, split, pde_result["per_hour"], col_prefix="dt")

    print(f"  PDE RMSE residual : {pde_result['mean_residual']:.6e}")
    print(f"  Grid RMSE saved   → {grid_rmse_path}  "
          f"shape={pde_result['grid_rmse'].shape}")

    return {
        "pde_rmse_residual":    float(pde_result["mean_residual"]),
        "pde_grid_rmse_path":   grid_rmse_path,   
    }




def evaluate_experiment(
    exp_name,
    split            = "test",
    ckpt_dir         = "checkpoints",
    target_dir       = "reports/targets",
    compute_pde      = True,
    raw_dir          = "data/raw",
    grid_height_path = "data/stats/emission_height.npy",
    T_in             = 10,
    T_out            = 24,
    base_report_dir  = "reports",
    report_dir       = None,
    episode_eval     = False,
    episode_index    = None,
    epi_mask_dir     = "data/processed",
    target_features  = None,
):
    if target_features is None:
        target_features = ["pm25"]

    if report_dir is not None:
        base_report_dir = report_dir

    report_dir = get_report_dir(base_report_dir, target_features, split)
    os.makedirs(report_dir, exist_ok=True)

    preds_path   = os.path.join(ckpt_dir, exp_name, "predictions", split, "preds.npy")
    targets_path = os.path.join(target_dir, f"{split}_targets.npy")

    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Missing predictions: {preds_path}")
    if not os.path.exists(targets_path):
        raise FileNotFoundError(f"Missing targets: {targets_path}")

    preds_all   = np.load(preds_path)   
    targets_all = np.load(targets_path)
    assert preds_all.shape == targets_all.shape
    N, H, W, T_out_loaded, num_targets = preds_all.shape

    assert num_targets == len(target_features), (
        f"Saved preds have {num_targets} channels but "
        f"target_features={target_features}"
    )

    print(f"\nEvaluating [{exp_name} | {split}]")
    print(f"  Targets    : {target_features}")
    print(f"  Shape      : {preds_all.shape}")
    print(f"  Report dir : {report_dir}")

    summary = {}

    for tgt_i, feat in enumerate(target_features):
        p = preds_all[..., tgt_i]      # (N, H, W, T_out)
        t = targets_all[..., tgt_i]
        s = compute_and_save_standard_metrics(
            p, t, feat, exp_name, split, report_dir)
        summary.update(s)


    if num_targets > 1:
        for metric in ["rmse", "rmse_std",
                        "mae",  "mae_std",
                        "smape","smape_std",
                        "bias", "bias_std",
                        "corr", "corr_std"]:
            vals = [summary[f"{f}_{metric}"] for f in target_features
                    if f"{f}_{metric}" in summary]
            if vals:
                summary[f"mean_{metric}"] = float(np.mean(vals))

    if num_targets > 1:
        import json
        with open("data/stats/global_mean_std.json") as f:
            stats = json.load(f)
        stats_idx   = {feat: i for i, feat in enumerate(stats["feature_names"])}
        target_mean = np.array(
            [stats["mean"][stats_idx[f]] for f in target_features], dtype=np.float32)
        target_std  = np.array(
            [stats["std"][stats_idx[f]]  for f in target_features], dtype=np.float32)
        preds_phys  = preds_all * target_std + target_mean
        viol = compute_constraint_violations(preds_phys, target_features)
        print("  Constraint violations:")
        for k, v in viol.items():
            print(f"    {k}: {v * 100:.3f}%")
        summary.update(viol)

    pm25_tgt_i = (target_features.index("pm25")
                  if "pm25" in target_features else None)

    if episode_eval and pm25_tgt_i is not None:
        epi = compute_and_save_episode_metrics(
            p25           = preds_all[..., pm25_tgt_i],
            t25           = targets_all[..., pm25_tgt_i],
            exp_name      = exp_name,
            split         = split,
            report_dir    = report_dir,
            N=N, H=H, W=W, T_out=T_out_loaded, T_in=T_in,
            episode_index = episode_index,
            epi_mask_dir  = epi_mask_dir,
            raw_dir       = raw_dir,
        )
        summary.update(epi)

    if compute_pde and pm25_tgt_i is not None:
        pde = compute_and_save_pde_metrics(
            exp_name         = exp_name,
            split            = split,
            report_dir       = report_dir,
            ckpt_dir         = ckpt_dir,
            raw_dir          = raw_dir,
            grid_height_path = grid_height_path,
            T_in             = T_in,
            T_out            = T_out,
            pm25_channel     = pm25_tgt_i,
        )
        summary.update(pde)

    upsert_scalar_csv(
        lb(report_dir, "leaderboard_summary.csv"),
        exp_name, split, summary)
    print(f"  Summary saved → {report_dir}/leaderboard_summary.csv")

    return preds_all, targets_all



def evaluate_test_episodes(
    exp_name,
    ckpt_dir        = "checkpoints",
    split           = "test_epi",
    target_dir      = "reports/targets",
    index_path      = "data/processed/test_epi_index.parquet",
    raw_dir         = "data/raw",
    T_in            = 10,
    base_report_dir = "reports",
    epi_mask_dir    = "data/processed",
    target_features = None,
):
    evaluate_experiment(
        exp_name        = exp_name,
        split           = split,
        ckpt_dir        = ckpt_dir,
        target_dir      = target_dir,
        compute_pde     = False,
        raw_dir         = raw_dir,
        T_in            = T_in,
        base_report_dir = base_report_dir,
        episode_eval    = True,
        episode_index   = index_path,
        epi_mask_dir    = epi_mask_dir,
        target_features = target_features,
    )