import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm


SPACING = 25000.0
DT      = 3600.0
PHYS_FEATURES = ["u10", "v10", "Kyy", "PM25_e", "PM25_finn"]


def load_feature_names(h5_path):
    with h5py.File(h5_path, "r") as f:
        return [x.decode() for x in f["feature_names"][:]]


def get_phys_indices(feature_names):
    idx = {}
    for feat in PHYS_FEATURES:
        if feat not in feature_names:
            raise ValueError(f"Feature '{feat}' not in HDF5. Available: {feature_names}")
        idx[feat] = feature_names.index(feat)
    return idx


def read_phys_fields(h5_path, month, start, T_in, T_out, feat_idx):
    key = f"month_{month:02d}"
    with h5py.File(h5_path, "r") as f:
        raw = f[key]["data"][start + T_in : start + T_in + T_out, :, :, :]
    return {feat: raw[..., col] for feat, col in feat_idx.items()}


def compute_pde_residual_sample(pred_sample, phys, grid_height):
    """
    pred_sample: (T_out, H, W) — PM2.5 only
    Returns: residual**2  shape (T_out-1, H, W)
    """
    C2 = pred_sample[1:];   C1 = pred_sample[:-1]
    dC_dt = (C2 - C1) / DT

    dC_dy  = np.gradient(C2, SPACING, axis=1)
    dC_dx  = np.gradient(C2, SPACING, axis=2)
    dC_dy2 = np.gradient(dC_dy, SPACING, axis=1)
    dC_dx2 = np.gradient(dC_dx, SPACING, axis=2)
    laplacian = dC_dx2 + dC_dy2

    u   = phys["u10"][:-1];   v   = phys["v10"][:-1]
    K   = phys["Kyy"][:-1];   E_e = phys["PM25_e"][:-1]
    E_f = phys["PM25_finn"][:-1]
    source = (E_e + E_f) * 1e9 / grid_height[None, :, :]

    residual = dC_dt + u * dC_dx + v * dC_dy - K * laplacian - source
    return residual ** 2


def _filter_index(index_df, T_in, T_out, raw_dir):

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
                if key in f:
                    h5_cache[year][month] = f[key]["data"].shape[0]
                else:
                    h5_cache[year][month] = 0

        T_month = h5_cache[year][month]
        if start + T_in + T_out <= T_month:
            valid.append(i)

    return index_df.loc[valid].reset_index(drop=True)


def compute_pde_residual_metrics(
    exp_name,
    split            = "test",
    ckpt_dir         = "checkpoints",
    raw_dir          = "data/raw",
    grid_height_path = "data/stats/emission_height.npy",
    T_in             = 10,
    T_out            = 24,  
    pm25_channel     = 0,
    grid_rmse_path   = None, 
):

    preds_path = os.path.join(ckpt_dir, exp_name, "predictions", split, "preds.npy")
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Predictions not found: {preds_path}")
    preds = np.load(preds_path)[..., pm25_channel]
    N, H, W, eval_Tout = preds.shape
    preds = preds.transpose(0, 3, 1, 2)   

    index_path = f"data/processed/{split}.parquet"
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")

    index_df = pd.read_parquet(index_path).reset_index(drop=True)

    index_df = _filter_index(index_df, T_in, eval_Tout, raw_dir)

    assert len(index_df) == N, (
        f"After horizon filtering, index has {len(index_df)} samples "
        f"but predictions have {N}. "
        f"Check T_in={T_in}, eval_Tout={eval_Tout} match the inference run."
    )

    grid_height = np.load(grid_height_path).astype(np.float32)
    grid_height = np.mean(grid_height, axis=0)
    assert grid_height.shape == (H, W)

    first_year    = int(index_df["year"].iloc[0])
    feature_names = load_feature_names(os.path.join(raw_dir, f"wrfchem_{first_year}.h5"))
    feat_idx      = get_phys_indices(feature_names)

    h5_paths = {int(y): os.path.join(raw_dir, f"wrfchem_{y}.h5")
                for y in index_df["year"].unique()}


    all_per_sample = np.zeros((N, eval_Tout - 1),       dtype=np.float32)
    all_grid_rmse  = np.zeros((N, eval_Tout - 1, H, W), dtype=np.float32)

    for i, row in tqdm(index_df.iterrows(), total=N,
                       desc=f"PDE residual [{exp_name}|{split}]"):
        phys   = read_phys_fields(h5_paths[int(row["year"])],
                                  int(row["month"]), int(row["start"]),
                                  T_in, eval_Tout, feat_idx)
        res_sq = compute_pde_residual_sample(preds[i], phys, grid_height)
        grid_rmse         = np.sqrt(res_sq)              
        all_grid_rmse[i]  = grid_rmse
        all_per_sample[i] = grid_rmse.mean(axis=(1, 2)) 

    per_hour      = all_per_sample.mean(axis=0)
    mean_residual = per_hour.mean()

    print(f"\nPDE Residual [{exp_name} | {split}]  (PM2.5 channel {pm25_channel})")
    print(f"  eval_Tout : {eval_Tout}h  (read from predictions file)")
    print(f"  Mean RMSE : {mean_residual:.6e}")
    print(f"  Range     : {per_hour.min():.4e} – {per_hour.max():.4e}")

    if grid_rmse_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(grid_rmse_path)), exist_ok=True)
        np.save(grid_rmse_path, all_grid_rmse)
        print(f"  Grid RMSE saved  → {grid_rmse_path}")
        print(f"  Shape            : {all_grid_rmse.shape}")

    return {"per_hour":      per_hour,
            "per_sample":    all_per_sample,
            "mean_residual": mean_residual,
            "grid_rmse":     all_grid_rmse}   