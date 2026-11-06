import numpy as np

BACKGROUND_THRESHOLD = 1.0 




def compute_frmse(preds_4d, targets_4d, iLow=4, iHigh=12):
    import math
    N, H, W, T = preds_4d.shape
    nx, ny = H, W

    pred_F   = np.fft.fftn(preds_4d,   axes=(1, 2))   # (N, H, W, T)
    target_F = np.fft.fftn(targets_4d, axes=(1, 2))
    _err_F   = np.abs(pred_F - target_F) ** 2          # (N, H, W, T)

    K_max = min(nx // 2, ny // 2)                      
    err_F = np.zeros((N, K_max, T), dtype=np.float64)

    for i in range(nx // 2):
        for j in range(ny // 2):
            it = math.floor(math.sqrt(i**2 + j**2))
            if it > K_max - 1:
                continue
            err_F[:, it, :] += _err_F[:, i, j, :]

    _err_F = np.sqrt(err_F.mean(axis=0)) / (nx * ny)   # (K_max, T)

    frmse_low  = _err_F[:iLow,      :].mean(axis=0)    # (T,)
    frmse_mid  = _err_F[iLow:iHigh, :].mean(axis=0)
    frmse_high = _err_F[iHigh:,     :].mean(axis=0)

    return frmse_low, frmse_mid, frmse_high


def compute_rmse(preds, targets):
    """
    Args:
        preds, targets: (N, H, W, T)  — NaN-safe: NaN cells are excluded.
    Returns:
        mean_rmse: (T,), std_rmse: (T,)
    """
    err = preds - targets
    per_sample = np.sqrt(np.nanmean(err ** 2, axis=(1, 2)))   # (N, T)
    return np.nanmean(per_sample, axis=0), np.nanstd(per_sample, axis=0)


def compute_mae(preds, targets):
    """
    Args:
        preds, targets: (N, H, W, T)  — NaN-safe: NaN cells are excluded.
    Returns:
        mean_mae: (T,), std_mae: (T,)
    """
    per_sample = np.nanmean(np.abs(preds - targets), axis=(1, 2))   
    return np.nanmean(per_sample, axis=0), np.nanstd(per_sample, axis=0)


def compute_smape(preds, targets, eps=1e-6, threshold=BACKGROUND_THRESHOLD):
    """
    Returns:
        mean_smape: (T,) in percent [0, 200], std_smape: (T,)
    """
    mask = targets > threshold
    denom = (np.abs(preds) + np.abs(targets)) / 2.0 + eps
    pct   = np.abs(preds - targets) / denom * 100.0

    combined   = mask & ~np.isnan(pct)
    masked_sum = np.where(combined, pct,  0.0).sum(axis=(1, 2))  
    count      = combined.sum(axis=(1, 2)).clip(min=1)
    per_sample = masked_sum / count
    return per_sample.mean(axis=0), per_sample.std(axis=0)


def compute_bias(preds, targets, threshold=BACKGROUND_THRESHOLD):
    """
    Returns:
        mean_bias: (T,), std_bias: (T,)  — positive = over-prediction
    """
    diff = preds - targets
    mask = (targets > threshold) & ~np.isnan(diff)

    masked_sum = np.where(mask, diff, 0.0).sum(axis=(1, 2))
    count      = mask.sum(axis=(1, 2)).clip(min=1)
    per_sample = masked_sum / count
    return per_sample.mean(axis=0), per_sample.std(axis=0)


def compute_correlation(preds, targets, threshold=BACKGROUND_THRESHOLD):
    """
    Pearson correlation over masked (H, W) cells per sample, then averaged
    across samples

    Args:
        preds, targets: (N, H, W, T)
    Returns:
        mean_corr: (T,), std_corr: (T,)
    """
    N, H, W, T = preds.shape
    per_sample  = np.full((N, T), np.nan)

    for n in range(N):
        for t in range(T):
            p    = preds[n, :, :, t]
            obs  = targets[n, :, :, t]
            mask = (obs > threshold) & ~np.isnan(p) & ~np.isnan(obs)

            if mask.sum() < 2:
                continue

            pe = p[mask]
            oe = obs[mask]

            if np.std(oe) < 1e-8 or np.std(pe) < 1e-8:
                continue

            per_sample[n, t] = float(np.corrcoef(pe, oe)[0, 1])

    return np.nanmean(per_sample, axis=0), np.nanstd(per_sample, axis=0)


def compute_r2(preds, targets):
    """
    Returns:
        r2: (T,)
    """
    T = preds.shape[-1]
    r2 = np.zeros(T)
    for t in range(T):
        p = preds[..., t].flatten()
        g = targets[..., t].flatten()
        # NaN-safe
        valid  = ~(np.isnan(p) | np.isnan(g))
        p, g   = p[valid], g[valid]
        ss_res = np.sum((g - p) ** 2)
        ss_tot = np.sum((g - g.mean()) ** 2)
        r2[t]  = 1.0 - ss_res / (ss_tot + 1e-8)
    return r2


def compute_fac2(preds, targets, threshold=BACKGROUND_THRESHOLD):
    """
    Fraction of predictions within a factor of 2 of observations,
    restricted to cells where target > threshold.

    Returns:
        fac2: (T,) in [0, 1], higher is better
    """
    mask  = targets > threshold
    ratio = np.where(mask, preds / (targets + 1e-8), np.nan)

    T    = preds.shape[-1]
    fac2 = np.zeros(T)
    for t in range(T):
        r = ratio[..., t].flatten()
        valid = r[~np.isnan(r)]
        if len(valid) > 0:
            fac2[t] = np.mean((valid >= 0.5) & (valid <= 2.0))
    return fac2

def compute_mfb(preds, targets, threshold=BACKGROUND_THRESHOLD, eps=1e-8):
    """
    Mean Fractional Bias, restricted to cells where target > threshold.
    MFB = mean( 2*(pred-obs) / (pred+obs+eps) )
    Range [-2, 2]. 0 is best. 

    Returns:
        mean_mfb: (T,), std_mfb: (T,)
    """
    denom    = np.abs(preds) + np.abs(targets) + eps
    mfb_vals = 2.0 * (preds - targets) / denom

    mask       = (targets > threshold) & ~np.isnan(mfb_vals)
    masked_sum = np.where(mask, mfb_vals, 0.0).sum(axis=(1, 2))   
    count      = mask.sum(axis=(1, 2)).clip(min=1)
    per_sample = masked_sum / count
    return per_sample.mean(axis=0), per_sample.std(axis=0)


def compute_episode_metrics(preds_4d, targets_4d, epi_mask):
    """
    Regression metrics restricted to episode grid points only.

    preds_4d / targets_4d : (N, H, W, T_out) in physical units.
    epi_mask              : (N, H, W, T_out) uint8 — 1 where episode is occurring.
    Returns dict of {name: np.ndarray(T_out,)}.
    """
    N, H, W, T = preds_4d.shape
    rmse_per  = np.full((N, T), np.nan)
    smape_per = np.full((N, T), np.nan)
    mfb_per   = np.full((N, T), np.nan)

    for n in range(N):
        for t in range(T):
            p   = preds_4d[n, :, :, t]
            obs = targets_4d[n, :, :, t]
            m   = epi_mask[n, :, :, t].astype(bool)

            if m.sum() < 2:
                continue

            pe  = p[m]
            oe  = obs[m]

            rmse_per[n, t]  = float(np.sqrt(np.mean((pe - oe) ** 2)))
            denom            = (np.abs(oe) + np.abs(pe)) / 2.0
            smape_per[n, t] = float(np.mean(np.abs(pe - oe) / (denom + 1e-8)))
            mfb_per[n, t]   = float(np.mean(2.0 * (pe - oe) / (pe + oe + 1e-8)))

    return {
        "epi_rmse":  np.nanmean(rmse_per, axis=0),
        "epi_smape": np.nanmean(smape_per, axis=0),
        "epi_mfb":   np.nanmean(mfb_per,  axis=0),
    }


def compute_exceedance_metrics(preds_4d, targets_4d, threshold=90.0):
    """
    Classification metrics for pollution exceedance events.

    Per-timestep metrics (T_out,):
        exc_recall    — TP / (TP + FN)
        exc_precision — TP / (TP + FP)
        exc_f1        — harmonic mean of precision and recall
        exc_csi       — Critical Success Index = TP / (TP + FP + FN)

    Returns dict of {name: np.ndarray(T_out,)}.
    """
    T         = preds_4d.shape[-1]
    recall_t    = np.zeros(T)
    precision_t = np.zeros(T)
    f1_t        = np.zeros(T)
    csi_t       = np.zeros(T)

    pred_mask = preds_4d   > threshold
    true_mask = targets_4d > threshold

    for t in range(T):
        p = pred_mask[..., t]
        g = true_mask[..., t]

        tp = float(np.sum(p & g))
        fp = float(np.sum(p & ~g))
        fn = float(np.sum(~p & g))

        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)

        recall_t[t]    = rec
        precision_t[t] = prec
        f1_t[t]        = 2 * prec * rec / (prec + rec + 1e-8)
        csi_t[t]       = tp / (tp + fp + fn + 1e-8)

    return {
        "exc_recall":    recall_t,
        "exc_precision": precision_t,
        "exc_f1":        f1_t,
        "exc_csi":       csi_t,
    }