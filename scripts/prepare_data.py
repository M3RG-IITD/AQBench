import os
import argparse

import numpy as np
import pandas as pd
import h5py
import yaml

try:
    from aqbench.utils.seeding import set_seed
except ImportError:
    def set_seed(seed: int):
        np.random.seed(seed)


def _load_times(h5group) -> np.ndarray:
    """
    Load the 'time' dataset from an open h5py group and return a
    1-D datetime64[h] array. 
    """
    raw = h5group["time"][:]
    decoded = [t.decode() if isinstance(t, bytes) else t for t in raw]
    return np.array(decoded, dtype="datetime64[h]")


def _find_valid_starts(times: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    Return an array of all valid start indices within ``times``.

    A start index ``s`` is valid iff:
      - s + window <= len(times)          (window fits within the month)
      - times[s : s + window] are hourly contiguous

    Cross-month safety is a *caller* guarantee: ``times`` belongs to one
    month only, and the ``s + window <= T`` bound is enforced by the range.

    Parameters
    ----------
    times  : 1-D datetime64[h] array for a single calendar month
    window : Tin + Tout
    stride : step between candidate positions (hours)

    Returns
    -------
    np.ndarray of valid start indices (int64), sorted ascending.
    """
    T     = len(times)
    valid = []
    for s in range(0, T - window + 1, stride):
        ts = times[s : s + window]
        # all consecutive differences must be exactly 1 hour
        if np.all(np.diff(ts) == np.timedelta64(1, "h")):
            valid.append(s)
    return np.array(valid, dtype=np.int64)




def generate_windows(
    raw_dir:         str,
    mapping:         dict,
    Tin:             int,
    Tout:            int,
    stride:          int,
    max_per_month,           
    rng:             np.random.Generator,
    split_label:     str,
) -> pd.DataFrame:
    """
    Build an index DataFrame for one data split.

    Parameters
    ----------
    raw_dir       : directory containing wrfchem_YYYY.h5
    mapping       : {year: [month, ...]} — years iterated in sorted order
    Tin           : input window length (hours)
    Tout          : output window length (hours)
    stride        : step between consecutive windows (hours)
    max_per_month : if int, sub-sample to at most this many windows per month.
                    Sampling is done without replacement using ``rng``.
                    None means use all valid windows.
    rng           : seeded numpy Generator — all randomness goes through here
    split_label   : label for printed progress

    Returns
    -------
    DataFrame[year, month, start] sorted by (year, month, start).
    """
    window = Tin + Tout
    rows   = []
    n_months          = 0
    n_windows_raw     = 0
    n_windows_dropped = 0  

    for year in sorted(mapping.keys()):
        months = mapping[year]
        path   = os.path.join(raw_dir, f"wrfchem_{year}.h5")

        if not os.path.exists(path):
            print(f"  [WARN] {split_label}: raw file not found: {path}")
            continue

        with h5py.File(path, "r") as f:
            for m in months:
                key = f"month_{m:02d}"
                if key not in f:
                    print(f"  [WARN] {split_label}: {key} missing in {path}")
                    continue

                times  = _load_times(f[key])
                T      = len(times)

                n_candidates = max(0, (T - window) // stride + 1)
                n_windows_raw += n_candidates

                valid = _find_valid_starts(times, window, stride)
                n_windows_dropped += n_candidates - len(valid)
                n_months          += 1

                if len(valid) == 0:
                    print(f"  [WARN] {split_label}: no valid windows in "
                          f"year={year} month={m:02d} (T={T}, window={window})")
                    continue

                # subsampling if needed
                if max_per_month is not None and len(valid) > max_per_month:
                    chosen_idx = rng.choice(len(valid), size=max_per_month,
                                            replace=False)
                    chosen_idx.sort()       
                    valid = valid[chosen_idx]

                for s in valid:
                    rows.append({"year": year, "month": m, "start": int(s)})

    df = (pd.DataFrame(rows, columns=["year", "month", "start"])
            .sort_values(["year", "month", "start"])
            .reset_index(drop=True))

    print(f"  {split_label:<8} | {n_months} month(s) | "
          f"{n_windows_raw} candidates | "
          f"{n_windows_dropped} dropped (non-contiguous) | "
          f"{len(df)} rows")

    return df



def _run_sanity_checks(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Warn if any (year, month, start) tuples appear in both splits.
    Also report year ranges for a quick visual check.
    """
    print(f"\n{'─'*60}")
    print("Sanity checks")
    print(f"{'─'*60}")

    if train_df.empty or val_df.empty:
        print("  [SKIP] one or both DataFrames are empty.")
        return

    train_keys = set(zip(train_df.year, train_df.month, train_df.start))
    val_keys   = set(zip(val_df.year,   val_df.month,   val_df.start))
    overlap    = train_keys & val_keys

    if overlap:
        print(f"  [WARN] {len(overlap)} identical (year, month, start) "
              f"rows in both train and val.")
    else:
        print("  No (year, month, start) overlap between train and val.  ✓")

    print(f"  train year range : "
          f"{train_df.year.min()} – {train_df.year.max()}")
    print(f"  val   year range : "
          f"{val_df.year.min()} – {val_df.year.max()}")

    shared_ym = (set(zip(train_df.year, train_df.month)) &
                 set(zip(val_df.year,   val_df.month)))
    if shared_ym:
        print(f"  [NOTE] {len(shared_ym)} (year, month) pairs appear in both "
              f"splits — expected if data periods overlap.")

    print(f"{'─'*60}")



def main(args):
    with open(args.dataset_config) as fh:
        cfg = yaml.safe_load(fh)

    seed = int(cfg["seed"])
    set_seed(seed)
    rng  = np.random.default_rng(seed)  

    raw_dir = cfg["raw_dir"]
    out_dir = cfg.get("out_dir", "data/processed")
    os.makedirs(out_dir, exist_ok=True)

    Tin  = int(cfg["Tin"])
    Tout = int(cfg["Tout"])

    train_stride = int(cfg["train_stride"])
    val_stride   = int(cfg.get("val_stride", 6))
    train_max    = cfg.get("train_max_samples_per_month", None)
    val_max      = cfg.get("val_max_samples_per_month",   None)

    if train_max is not None:
        train_max = int(train_max)
    if val_max is not None:
        val_max = int(val_max)

    print(f"\n{'='*60}")
    print("prepare_data.py")
    print(f"  raw_dir      : {raw_dir}")
    print(f"  out_dir      : {out_dir}")
    print(f"  seed         : {seed}")
    print(f"  Tin={Tin}  Tout={Tout}  window={Tin + Tout}h")
    print(f"  train stride : {train_stride}h  "
          f"max_per_month: {train_max}")
    print(f"  val   stride : {val_stride}h  "
          f"max_per_month: {val_max}")
    print(f"{'='*60}\n")

    train_df = generate_windows(
        raw_dir=raw_dir, mapping=cfg["train"],
        Tin=Tin, Tout=Tout, stride=train_stride,
        max_per_month=train_max, rng=rng, split_label="train",
    )
    val_df = generate_windows(
        raw_dir=raw_dir, mapping=cfg["val"],
        Tin=Tin, Tout=Tout, stride=val_stride,
        max_per_month=val_max, rng=rng, split_label="val",
    )

    train_path = os.path.join(out_dir, "train_index.parquet")
    val_path   = os.path.join(out_dir, "val_index.parquet")

    train_df.to_parquet(train_path, engine="fastparquet", index=False)
    val_df.to_parquet(  val_path,   engine="fastparquet", index=False)

    print(f"\n  Wrote {len(train_df):>8,} train rows → {train_path}")
    print(f"  Wrote {len(val_df):>8,} val rows   → {val_path}")

    _run_sanity_checks(train_df, val_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate train and val index parquets."
    )
    parser.add_argument("--dataset_config", required=True,
                        help="Path to prepare_data.yaml")
    main(parser.parse_args())