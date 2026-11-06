"""
ft_dataset.py
─────────────
Dataset for Stage-3 bias-correction finetuning.

Identical to WRFchemDataset in terms of what the MODEL sees (x, t_in),
but replaces y with ground-station PM2.5 observations in physical units
(µg/m³), NaN where no station data exists for that grid cell / timestep.

Returns per sample:
    x    : (Tin,  H, W, C_in)    float32   normalised WRF-Chem input
    y    : (Tout, H, W, 1)       float32   physical PM2.5, NaN = no obs
    t_in : (Tin,)                int64     hours since 2000-01-01

y is in physical units so the model prediction must be denormalised
before computing the masked loss.  NaN cells are excluded from the loss.

prewarm() must be called in the main process before DataLoader forks.
"""

import os
import json
import h5py
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

_EPOCH = np.datetime64("2000-01-01", "h")


class FTDataset(Dataset):
    def __init__(
        self,
        index_file: str,       # ft_train / ft_val / ft_test parquet
        raw_dir:    str,        # wrfchem_{year}.h5
        gs_dir:     str,        # ground_station_{year}.h5
        Tin:        int,
        Tout:       int,
        input_features:  list,
        target_features: list,  # must contain 'pm25'
    ):
        df = pd.read_parquet(index_file).reset_index(drop=True)

        self.raw_dir         = raw_dir
        self.gs_dir          = gs_dir
        self.Tin             = Tin
        self.Tout            = Tout
        self.input_features  = input_features
        self.target_features = target_features

        # ── Normalisation stats (input only — y stays physical) ───────────
        with open("data/stats/global_mean_std.json") as f:
            stats = json.load(f)

        global_means      = np.array(stats["mean"], dtype=np.float32)
        global_stds       = np.array(stats["std"],  dtype=np.float32)
        stats_feat_to_idx = {feat: i for i, feat in enumerate(stats["feature_names"])}

        # Feature indices in the WRF-Chem HDF5
        first_year = int(df["year"].iloc[0])
        with h5py.File(os.path.join(raw_dir, f"wrfchem_{first_year}.h5"), "r") as f:
            feature_names = [x.decode() for x in f["feature_names"][:]]
        data_feat_to_idx = {feat: i for i, feat in enumerate(feature_names)}
        self.input_idx   = [data_feat_to_idx[feat] for feat in input_features]

        # Input normalisation
        self._inp_mean = np.array(
            [global_means[stats_feat_to_idx[f]] for f in input_features],
            dtype=np.float32).reshape(1, 1, 1, -1)
        self._inp_std = np.array(
            [global_stds[stats_feat_to_idx[f]] for f in input_features],
            dtype=np.float32).reshape(1, 1, 1, -1) + 1e-6

        # PM2.5 denorm constants (for external use in loss / eval)
        pm25_s_idx       = stats_feat_to_idx["pm25"]
        self.pm25_mean   = float(global_means[pm25_s_idx])
        self.pm25_std    = float(global_stds[pm25_s_idx])

        self.years  = df["year"].values
        self.months = df["month"].values
        self.starts = df["start"].values

        # Caches: {year: {month: array}}
        self._inp_cache:  dict = {}   # float32 (T, H, W, C_in)  normalised
        self._gs_cache:   dict = {}   # float32 (T, H, W, 1)     normalised (same space as model output)
        self._time_cache: dict = {}   # int64   (T,)

    # ── Cache loading ─────────────────────────────────────────────────────────
    def _get_year(self, year: int):
        if year in self._inp_cache:
            return

        self._inp_cache[year]  = {}
        self._gs_cache[year]   = {}
        self._time_cache[year] = {}

        wrf_path = os.path.join(self.raw_dir, f"wrfchem_{year}.h5")
        gs_path  = os.path.join(self.gs_dir,  f"ground_station_{year}.h5")
        gs_exists = os.path.exists(gs_path)

        if not gs_exists:
            print(f"  [FTDataset] WARNING: {gs_path} not found — "
                  f"y will be all-NaN for year {year}")

        with h5py.File(wrf_path, "r") as wf:
            gs_f = h5py.File(gs_path, "r") if gs_exists else None

            for m in range(1, 13):
                key = f"month_{m:02d}"
                if key not in wf:
                    continue
                if gs_exists and key not in gs_f:
                    # WRF month exists but no ground station data → skip entirely
                    # (these months won't be in the ft parquets anyway)
                    continue

                # ── WRF-Chem input (normalised) ───────────────────────────
                raw = wf[key]["data"][:]                        # (T, H, W, 26)
                inp = raw[..., self.input_idx].astype(np.float32)
                inp -= self._inp_mean
                inp /= self._inp_std
                inp.flags.writeable = False
                self._inp_cache[year][m] = inp

                # ── Ground station y — normalised, NaN-masked ─────────────
                # Normalise with same PM2.5 mean/std as WRF-Chem so loss is
                # in normalised space, identical to S1/S2 supervision.
                # NaN cells (no station) propagate through arithmetic → stay NaN.
                if gs_exists:
                    gs = gs_f[key]["data"][:].astype(np.float32)  # (T,H,W,1)
                else:
                    gs = np.full((*inp.shape[:3], 1), np.nan, dtype=np.float32)
                gs = (gs - self.pm25_mean) / self.pm25_std        # normalise
                gs.flags.writeable = False
                self._gs_cache[year][m] = gs

                # ── Timestamps ────────────────────────────────────────────
                raw_times = wf[key]["time"][:]
                decoded   = [t.decode() if isinstance(t, bytes) else t
                             for t in raw_times]
                times_dt  = np.array(decoded, dtype="datetime64[h]")
                t_int     = (times_dt - _EPOCH).astype(np.int64)
                t_int.flags.writeable = False
                self._time_cache[year][m] = t_int

            if gs_f is not None:
                gs_f.close()

    # ── Prewarm ───────────────────────────────────────────────────────────────
    def prewarm(self):
        unique_years = sorted(set(int(y) for y in self.years))
        print(f"  [FTDataset prewarm] {len(unique_years)} year(s): {unique_years}")

        for year in unique_years:
            self._get_year(year)
            months = list(self._inp_cache[year].keys())
            mb_inp = sum(self._inp_cache[year][m].nbytes for m in months) / 1024**2
            mb_gs  = sum(self._gs_cache[year][m].nbytes  for m in months) / 1024**2
            print(f"    {year} — {len(months)} months  "
                  f"inp={mb_inp:.0f} MB  gs={mb_gs:.0f} MB")

        # Filter samples whose full window fits within the cached month
        valid = []
        for i in range(len(self.years)):
            year  = int(self.years[i])
            month = int(self.months[i])
            start = int(self.starts[i])
            if (year not in self._inp_cache
                    or month not in self._inp_cache[year]):
                continue
            # T from gs cache — correctly aligned to WRF timestamps.
            # Do NOT use inp_cache shape; WRF month_07/2018 has data/time inconsistency.
            T = self._gs_cache[year][month].shape[0]
            if start + self.Tin + self.Tout <= T:
                valid.append(i)

        dropped = len(self.years) - len(valid)
        if dropped:
            print(f"  [FTDataset] filtered {dropped} samples "
                  f"(window exceeds month boundary or missing gs data)")

        self.years  = self.years[valid]
        self.months = self.months[valid]
        self.starts = self.starts[valid]

        total_mb = sum(
            self._inp_cache[y][m].nbytes + self._gs_cache[y][m].nbytes
            for y in self._inp_cache for m in self._inp_cache[y]
        ) / 1024**2
        print(f"  [FTDataset] {len(self.years)} samples  {total_mb:.0f} MB in RAM")

    # ── Eval mask (built from this dataset's samples) ─────────────────────────
    def build_eval_mask(self, val_ds: "FTDataset",
                        threshold: float = 0.5) -> torch.BoolTensor:
        """
        Builds eval mask as the INTERSECTION of:
          1. Cells where >= threshold of TRAIN samples have a non-NaN obs
             (ensures the model was actually supervised there)
          2. Cells that have at least one non-NaN obs anywhere in the VAL set
             (ensures validation has actual ground truth to compare against)

        Evaluating on cells with no val observations produces pure noise.

        Parameters
        ----------
        val_ds    : the ft_val FTDataset, after prewarm()
        threshold : fraction of train samples that must have obs at a cell
        """
        H, W = 140, 124

        # ── Condition 1: train coverage ───────────────────────────────────
        n_train = len(self.years)
        train_counts = np.zeros((H, W), dtype=np.int32)
        for i in range(n_train):
            year  = int(self.years[i])
            month = int(self.months[i])
            t1    = int(self.starts[i]) + self.Tin
            t2    = t1 + self.Tout
            obs   = self._gs_cache[year][month][t1:t2, :, :, 0]
            train_counts += np.any(~np.isnan(obs), axis=0).astype(np.int32)
        train_mask = (train_counts / max(n_train, 1)) >= threshold

        # ── Condition 2: val has at least one obs at that cell ────────────
        n_val = len(val_ds.years)
        val_any = np.zeros((H, W), dtype=bool)
        for i in range(n_val):
            year  = int(val_ds.years[i])
            month = int(val_ds.months[i])
            t1    = int(val_ds.starts[i]) + val_ds.Tin
            t2    = t1 + val_ds.Tout
            if (year not in val_ds._gs_cache
                    or month not in val_ds._gs_cache[year]):
                continue
            obs = val_ds._gs_cache[year][month][t1:t2, :, :, 0]
            val_any |= np.any(~np.isnan(obs), axis=0)

        # ── Intersection ──────────────────────────────────────────────────
        mask    = train_mask & val_any
        n_cells = int(mask.sum())
        print(f"  [EvalMask] train_coverage>={threshold:.0%}: {train_mask.sum()} cells")
        print(f"  [EvalMask] val has obs          : {val_any.sum()} cells")
        print(f"  [EvalMask] intersection (final) : {n_cells} cells")
        return torch.from_numpy(mask)

    # ── Dataset interface ─────────────────────────────────────────────────────
    def __len__(self):
        return len(self.years)

    def __getitem__(self, idx):
        year  = int(self.years[idx])
        month = int(self.months[idx])
        start = int(self.starts[idx])
        t0, t1, t2 = start, start + self.Tin, start + self.Tin + self.Tout

        x    = torch.from_numpy(
                   self._inp_cache[year][month][t0:t1].copy()).contiguous()
        y    = torch.from_numpy(
                   self._gs_cache[year][month][t1:t2].copy()).contiguous()
        t_in = torch.from_numpy(
                   self._time_cache[year][month][t0:t1].copy())

        return x, y, t_in   # (Tin,H,W,C_in), (Tout,H,W,1), (Tin,)