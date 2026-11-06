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
    raw = h5group["time"][:]
    decoded = [t.decode() if isinstance(t, bytes) else t for t in raw]
    return np.array(decoded, dtype="datetime64[h]")


def _find_valid_starts(times: np.ndarray, window: int, stride: int) -> np.ndarray:
    T, valid = len(times), []
    for s in range(0, T - window + 1, stride):
        if np.all(np.diff(times[s : s + window]) == np.timedelta64(1, "h")):
            valid.append(s)
    return np.array(valid, dtype=np.int64)


#
def generate_test_windows(raw_dir, mapping, Tin, horizon):
    window, stride = Tin + horizon, horizon
    rows, n_months, n_dropped = [], 0, 0

    for year in sorted(mapping.keys()):
        path = os.path.join(raw_dir, f"wrfchem_{year}.h5")
        if not os.path.exists(path):
            print(f"  [WARN] not found: {path}")
            continue
        with h5py.File(path, "r") as f:
            for m in mapping[year]:
                key = f"month_{m:02d}"
                if key not in f:
                    print(f"  [WARN] {key} missing in {path}")
                    continue
                times    = _load_times(f[key])
                T        = len(times)
                n_cands  = max(0, (T - window) // stride + 1)
                valid    = _find_valid_starts(times, window, stride)
                n_dropped += n_cands - len(valid)
                n_months  += 1
                for s in valid:
                    rows.append({"year": year, "month": m, "start": int(s)})

    df = (pd.DataFrame(rows, columns=["year", "month", "start"])
            .sort_values(["year", "month", "start"])
            .reset_index(drop=True))
    print(f"    test_{horizon:<4}: {n_months} month(s)  "
          f"{n_dropped} non-contiguous dropped  -> {len(df)} windows")
    return df


def generate_event_windows(raw_dir, epi_mask_dir, mapping,
                            Tin, horizon, episode_thresh):
    """
    Retain only windows where episode_mask at the last target timestep
    has > episode_thresh active cells.

    target_idx = start + Tin + horizon - 1
    """
    window, stride = Tin + horizon, horizon
    rows = []
    n_months = n_drop_cont = n_drop_epi = 0

    for year in sorted(mapping.keys()):
        raw_path = os.path.join(raw_dir,      f"wrfchem_{year}.h5")
        epi_path = os.path.join(epi_mask_dir, f"{year}_epi.h5")

        if not os.path.exists(raw_path):
            print(f"  [WARN] raw not found : {raw_path}")
            continue
        if not os.path.exists(epi_path):
            print(f"  [WARN] epi not found : {epi_path}")
            continue

        with h5py.File(raw_path, "r") as f, h5py.File(epi_path, "r") as ef:
            for m in mapping[year]:
                key = f"month_{m:02d}"
                if key not in f:
                    print(f"  [WARN] {key} missing in {raw_path}")
                    continue
                if key not in ef:
                    print(f"  [WARN] {key} missing in {epi_path}")
                    continue

                times = _load_times(f[key])
                T     = len(times)
                mask  = ef[key]["episode_mask"][:]   

                if mask.shape[0] != T:
                    print(f"  [WARN] mask T={mask.shape[0]} != data T={T} "
                          f"({key}) — skipping month")
                    continue

                n_months += 1

                for s in range(0, T - window + 1, stride):
                    if not np.all(np.diff(times[s : s + window])
                                  == np.timedelta64(1, "h")):
                        n_drop_cont += 1
                        continue

                 
                    target_idx = s + Tin + horizon - 1  
                    if int(np.sum(mask[target_idx])) > episode_thresh:
                        rows.append({"year": year, "month": m, "start": int(s)})
                    else:
                        n_drop_epi += 1

    df = (pd.DataFrame(rows, columns=["year", "month", "start"])
            .sort_values(["year", "month", "start"])
            .reset_index(drop=True))
    print(f"    event_{horizon:<3}: {n_months} month(s)  "
          f"{n_drop_cont} non-contiguous  "
          f"{n_drop_epi} below threshold  "
          f"-> {len(df)} windows")
    return df



def main(args):
    with open(args.dataset_config) as fh:
        cfg = yaml.safe_load(fh)

    seed = int(cfg["seed"])
    set_seed(seed)

    raw_dir       = cfg["raw_dir"]
    epi_mask_dir  = cfg.get("epi_mask_dir", "data/processed")
    out_dir       = cfg.get("out_dir",       "data/processed")
    os.makedirs(out_dir, exist_ok=True)

    Tin            = int(cfg["Tin"])          
    horizons       = cfg.get("horizons",       [24, 72, 168])
    episode_thresh = int(cfg.get("episode_thresh", 20))
    test_mapping   = cfg["test"]

    print(f"\n{'='*60}")
    print("prepare_event_tests.py")
    print(f"  raw_dir       : {raw_dir}")
    print(f"  epi_mask_dir  : {epi_mask_dir}")
    print(f"  out_dir       : {out_dir}")
    print(f"  seed          : {seed}")
    print(f"  Tin           : {Tin}h  (from config)")
    print(f"  horizons      : {horizons}h")
    print(f"  episode_thresh: >{episode_thresh} active cells")
    print(f"  stride        : = horizon (non-overlapping)")
    print(f"  target_idx    : start + Tin + horizon - 1  (last forecast hour)")
    print(f"{'='*60}\n")

    summary = []

    for horizon in horizons:
        print(f"{'─'*55}")
        print(f"  Horizon {horizon}h  |  window={Tin+horizon}h  "
              f"|  target at month-idx = start + {Tin + horizon - 1}")

        df_test = generate_test_windows(
            raw_dir, test_mapping, Tin, horizon)
        df_test.to_parquet(
            os.path.join(out_dir, f"test_{horizon}.parquet"),
            engine="fastparquet", index=False)

        df_event = generate_event_windows(
            raw_dir, epi_mask_dir, test_mapping,
            Tin, horizon, episode_thresh)
        df_event.to_parquet(
            os.path.join(out_dir, f"test_{horizon}_event.parquet"),
            engine="fastparquet", index=False)

        ratio = (100 * len(df_event) / len(df_test)
                 if len(df_test) else 0.0)
        summary.append({
            "horizon_h":     horizon,
            "window_h":      Tin + horizon,
            "test_n":        len(df_test),
            "event_n":       len(df_event),
            "event_%":       round(ratio, 1),
        })

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nAll files written to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", required=True)
    main(parser.parse_args())