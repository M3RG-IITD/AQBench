import os
import argparse

import numpy as np
import pandas as pd
import h5py
import yaml


DEFAULT_CFG = "configs/data/prepare_ft_data.yaml"
OUT_DIR     = "data/processed"


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_index(split_months: dict,
                raw_dir: str,
                gs_dir:  str,
                Tin:     int,
                Tout:    int,
                stride:  int) -> pd.DataFrame:
    """
    Parameters
    ----------
    split_months : {year: [month, ...]}
    raw_dir      : directory with wrfchem_{year}.h5  (existence check only)
    gs_dir       : directory with ground_station_{year}.h5  (T and existence)
    Tin, Tout    : window sizes
    stride       : step between consecutive start indices

    T is read from the ground_station HDF5 data array — it is always
    correctly aligned to WRF timestamps by the converter.
    """
    window  = Tin + Tout
    records = []

    for year, months in sorted(split_months.items()):
        wrf_path = os.path.join(raw_dir, f"wrfchem_{year}.h5")
        gs_path  = os.path.join(gs_dir,  f"ground_station_{year}.h5")

        if not os.path.exists(wrf_path):
            print(f"  [WARN] WRF file missing: {wrf_path} — skipping year {year}")
            continue
        if not os.path.exists(gs_path):
            print(f"  [WARN] Ground-station file missing: {gs_path} — skipping year {year}")
            continue

        with h5py.File(wrf_path, "r") as wrf_f, \
             h5py.File(gs_path,  "r") as gs_f:

            for month in sorted(months):
                wrf_key = f"month_{month:02d}"
                gs_key  = f"month_{month:02d}"

                if wrf_key not in wrf_f:
                    print(f"  [WARN] {wrf_key} missing in WRF {year} — skipping")
                    continue
                if gs_key not in gs_f:
                    print(f"  [WARN] {gs_key} missing in ground_station_{year}.h5 — skipping")
                    continue

                T = gs_f[gs_key]["data"].shape[0]

                if T < window:
                    print(f"  [WARN] {year}/{gs_key} T={T} < window={window} — skipping")
                    continue

                n_windows = len(range(0, T - window + 1, stride))
                for s in range(0, T - window + 1, stride):
                    records.append((year, month, s))

                print(f"    {year} month {month:02d}: T={T}  windows={n_windows}  stride={stride}")

    return pd.DataFrame(records, columns=["year", "month", "start"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=DEFAULT_CFG)
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)

    raw_dir          = cfg["raw_dir"]
    gs_dir           = cfg.get("gs_dir", raw_dir)
    Tin              = cfg["Tin"]
    Tout             = cfg["Tout"]
    train_stride     = cfg.get("train_stride",    1)
    val_test_stride  = cfg.get("val_test_stride", Tin + Tout)

    os.makedirs(OUT_DIR, exist_ok=True)

    splits = {
        "ft_train": ("ft_train_index.parquet", cfg.get("ft_train", {}), train_stride),
        "ft_val":   ("ft_val_index.parquet",   cfg.get("ft_val",   {}), val_test_stride),
        "ft_test":  ("ft_test_index.parquet",  cfg.get("ft_test",  {}), val_test_stride),
    }

    for split_name, (fname, split_months, stride) in splits.items():
        print(f"\n{'='*50}")
        print(f"Building {split_name}  (stride={stride}) …")

        if not split_months:
            print(f"  [WARN] No months defined for {split_name} — skipping")
            continue

        df       = build_index(split_months, raw_dir, gs_dir, Tin, Tout, stride)
        out_path = os.path.join(OUT_DIR, fname)
        df.to_parquet(out_path, index=False)

        print(f"\n  {split_name}: {len(df):,} samples  → {out_path}")
        for (y, m), grp in df.groupby(["year", "month"]):
            print(f"    {y} month {m:02d}: {len(grp):,} windows")

    print("\nDone.")


if __name__ == "__main__":
    main()