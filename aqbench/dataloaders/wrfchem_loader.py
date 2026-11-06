from torch.utils.data import DataLoader
import torch
from aqbench.datasets.wrfchem_dataset import WRFchemDataset

INDEX_FILES = {
    "train":        "data/processed/train_index.parquet",
    "val":          "data/processed/val_index.parquet",
    "test_24":      "data/processed/test_24.parquet",
    "test_72":      "data/processed/test_72.parquet",
    "test_168":     "data/processed/test_168.parquet",
    "test_epi_24":  "data/processed/test_24_event.parquet",
    "test_epi_72":  "data/processed/test_72_event.parquet",
    "test_epi_168": "data/processed/test_168_event.parquet",
}


def _safe_collate(batch):
    """Collate 4-tuples (x, y, y_aux, t_in) into batched tensors."""
    xs, ys, y_auxs, t_ins = zip(*batch)
    return (
        torch.stack(xs),
        torch.stack(ys),
        torch.stack(y_auxs) if y_auxs[0].numel() > 0 else torch.empty(0),
        torch.stack(t_ins),  
    )


def build_dataloader(task_cfg: dict,
                     split: str,
                     *,
                     Tout: int,
                     y_aux_len: int,
                     generator: torch.Generator = None) -> DataLoader:
    """
    Build a DataLoader for the given split.

    Parameters
    ----------
    task_cfg  : must contain Tin, raw_dir, input_features, target_features,
                batch_size, and optionally num_workers.
    split     : one of INDEX_FILES keys
    Tout      : number of output hours (target)
    y_aux_len : number of future input-feature hours in y_aux
    """
    if split not in INDEX_FILES:
        raise ValueError(f"Unknown split '{split}'. Valid: {list(INDEX_FILES)}")

    dataset = WRFchemDataset(
        index_file      = INDEX_FILES[split],
        raw_dir         = task_cfg["raw_dir"],
        Tin             = task_cfg["Tin"],
        Tout            = Tout,
        input_features  = task_cfg["input_features"],
        target_features = task_cfg["target_features"],
        y_aux_len       = y_aux_len,
    )

    num_workers = task_cfg.get("num_workers", 2)
    print(f"[DATALOADER] split={split:<12} Tout={Tout:<4} "
          f"y_aux_len={y_aux_len:<4} batch={task_cfg['batch_size']} "
          f"workers={num_workers}")

    dataset.prewarm()

    return DataLoader(
        dataset,
        batch_size         = task_cfg["batch_size"],
        shuffle            = (split == "train"),
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = (num_workers > 0),
        prefetch_factor    = 2 if num_workers > 0 else None,
        drop_last          = (split == "train"),
        collate_fn         = _safe_collate,
        generator          = generator,  
    )