import os
import json
import h5py
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

_EPOCH = np.datetime64("2000-01-01", "h")


class WRFchemDataset(Dataset):
    """
    Dataset for WRF-Chem spatiotemporal forecasting.

    prewarm() must be called in the main process before DataLoader forks workers.
    It loads, feature-selects and normalises entire year files into read-only
    float32 arrays.  The time array (hours since 2000-01-01) is also
    cached from the HDF5 'time' dataset stored alongside the data.

    Returns per sample:
        x     : (Tin,  H, W, C_in)           float32
        y     : (Tout, H, W, C_tgt)           float32
        y_aux : (y_aux_len, H, W, C_in)       float32  (or torch.empty(0))
        t_in  : (Tin,)                         int64
                wall-clock hours since 2000-01-01 for each input timestep;
                fed directly to AbsoluteTimeEmbedding inside the model.
    """

    def __init__(
        self,
        index_file,
        raw_dir,
        Tin,
        Tout,
        input_features,
        target_features,
        y_aux_len=None,
    ):
        df = pd.read_parquet(index_file).reset_index(drop=True)

        self.raw_dir         = raw_dir
        self.Tin             = Tin
        self.Tout            = Tout
        self.y_aux_len       = y_aux_len if y_aux_len is not None else Tout
        self.input_features  = input_features
        self.target_features = target_features

        with open("data/stats/global_mean_std.json") as f:
            stats = json.load(f)

        global_means      = np.array(stats["mean"], dtype=np.float32)
        global_stds       = np.array(stats["std"],  dtype=np.float32)
        stats_feat_to_idx = {feat: i for i, feat in enumerate(stats["feature_names"])}

        first_year = int(df["year"].iloc[0])
        with h5py.File(os.path.join(raw_dir, f"wrfchem_{first_year}.h5"), "r") as f:
            feature_names = [x.decode() for x in f["feature_names"][:]]

        data_feat_to_idx = {feat: i for i, feat in enumerate(feature_names)}
        self.input_idx   = [data_feat_to_idx[feat] for feat in input_features]
        self.target_idx  = [data_feat_to_idx[feat] for feat in target_features]

        self._inp_mean = np.array(
            [global_means[stats_feat_to_idx[f]] for f in input_features],
            dtype=np.float32).reshape(1, 1, 1, -1)
        self._inp_std  = np.array(
            [global_stds[stats_feat_to_idx[f]] for f in input_features],
            dtype=np.float32).reshape(1, 1, 1, -1) + 1e-6
        self._tgt_mean = np.array(
            [global_means[stats_feat_to_idx[f]] for f in target_features],
            dtype=np.float32).reshape(1, 1, 1, -1)
        self._tgt_std  = np.array(
            [global_stds[stats_feat_to_idx[f]] for f in target_features],
            dtype=np.float32).reshape(1, 1, 1, -1) + 1e-6

        self.target_mean = np.array(
            [global_means[stats_feat_to_idx[f]] for f in target_features],
            dtype=np.float32)
        self.target_std  = np.array(
            [global_stds[stats_feat_to_idx[f]] for f in target_features],
            dtype=np.float32)

        self.years  = df["year"].values
        self.months = df["month"].values
        self.starts = df["start"].values

        self._inp_cache:  dict = {}
        self._tgt_cache:  dict = {}
        self._time_cache: dict = {}   

    def __len__(self):
        return len(self.years)

    def _get_year(self, year: int):
        if year in self._inp_cache:
            return

        self._inp_cache[year]  = {}
        self._tgt_cache[year]  = {}
        self._time_cache[year] = {}

        path = os.path.join(self.raw_dir, f"wrfchem_{year}.h5")
        with h5py.File(path, "r") as f:
            for m in range(1, 13):
                key = f"month_{m:02d}"
                if key not in f:
                    continue

                raw = f[key]["data"][:]
                inp = raw[..., self.input_idx].astype(np.float32)
                tgt = raw[..., self.target_idx].astype(np.float32)
                del raw

                inp -= self._inp_mean
                inp /= self._inp_std
                tgt -= self._tgt_mean
                tgt /= self._tgt_std

                inp.flags.writeable = False
                tgt.flags.writeable = False

                self._inp_cache[year][m] = inp
                self._tgt_cache[year][m] = tgt

                raw_times = f[key]["time"][:]
                decoded   = [t.decode() if isinstance(t, bytes) else t
                             for t in raw_times]
                times_dt  = np.array(decoded, dtype="datetime64[h]")
                t_int     = (times_dt - _EPOCH).astype(np.int64)
                t_int.flags.writeable = False
                self._time_cache[year][m] = t_int

    def prewarm(self):
        unique_years = sorted(set(int(y) for y in self.years))
        print(f"  [prewarm] {len(unique_years)} year(s): {unique_years}")

        for year in unique_years:
            self._get_year(year)
            months = list(self._inp_cache[year].keys())
            mb = sum(
                self._inp_cache[year][m].nbytes +
                self._tgt_cache[year][m].nbytes
                for m in months
            ) / 1024 ** 2
            print(f"    {year} — {len(months)} months  {mb:.0f} MB")

        valid = []
        for i in range(len(self.years)):
            year  = int(self.years[i])
            month = int(self.months[i])
            start = int(self.starts[i])
            T     = self._inp_cache[year][month].shape[0]
            if start + self.Tin + self.Tout <= T:
                valid.append(i)

        total_before = len(self.years)
        if len(valid) < total_before:
            dropped = total_before - len(valid)
            print(f"  [prewarm] filtered {dropped} samples "
                  f"(window exceeds month boundary)")
            self.years  = self.years[valid]
            self.months = self.months[valid]
            self.starts = self.starts[valid]

        total_mb = sum(
            self._inp_cache[y][m].nbytes
            for y in self._inp_cache for m in self._inp_cache[y]
        ) / 1024 ** 2 + sum(
            self._tgt_cache[y][m].nbytes
            for y in self._tgt_cache for m in self._tgt_cache[y]
        ) / 1024 ** 2
        print(f"  [prewarm] done — {len(self.years)} samples  "
              f"{total_mb:.0f} MB in RAM")

    def __getitem__(self, idx):
        year  = int(self.years[idx])
        month = int(self.months[idx])
        start = int(self.starts[idx])

        inp    = self._inp_cache[year][month]
        tgt    = self._tgt_cache[year][month]
        tcache = self._time_cache[year][month]

        t0 = start
        t1 = start + self.Tin
        t2 = start + self.Tin + self.Tout

        x    = torch.from_numpy(inp[t0:t1].copy()).contiguous()
        y    = torch.from_numpy(tgt[t1:t2].copy()).contiguous()
        t_in = torch.from_numpy(tcache[t0:t1].copy())   # (Tin,) int64

        if self.y_aux_len > 0:
            y_aux = torch.from_numpy(
                inp[t1 : t1 + self.y_aux_len].copy()
            ).contiguous()
        else:
            y_aux = torch.empty(0)

        return x, y, y_aux, t_in