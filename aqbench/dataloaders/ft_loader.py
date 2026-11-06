import torch
from torch.utils.data import DataLoader
from aqbench.datasets.ft_dataset import FTDataset

FT_INDEX_FILES = {
    "ft_train": "data/processed/ft_train_index.parquet",
    "ft_val":   "data/processed/ft_val_index.parquet",
    "ft_test":  "data/processed/ft_test_index.parquet",
}


def _collate(batch):
    """Stack (x, y, t_in) tuples into batched tensors."""
    xs, ys, t_ins = zip(*batch)
    return (
        torch.stack(xs),    
        torch.stack(ys),    
        torch.stack(t_ins), 
    )


def build_ft_dataloader(task_cfg: dict, split: str) -> tuple[DataLoader, FTDataset]:
    """
    Parameters
    ----------
    task_cfg : bias_correction_pm25.yaml contents
    split    : 'ft_train' | 'ft_val' | 'ft_test'

    Returns
    -------
    (DataLoader, FTDataset)
    FTDataset is returned so callers can access pm25_mean/pm25_std
    and call build_eval_mask() on the train dataset.
    """
    if split not in FT_INDEX_FILES:
        raise ValueError(f"Unknown split '{split}'. Valid: {list(FT_INDEX_FILES)}")

    s3_batch = task_cfg["s3_batch_size"]   

    dataset = FTDataset(
        index_file      = FT_INDEX_FILES[split],
        raw_dir         = task_cfg["raw_dir"],
        gs_dir          = task_cfg["gs_dir"],
        Tin             = task_cfg["Tin"],
        Tout            = task_cfg["Tout"],
        input_features  = task_cfg["input_features"],
        target_features = task_cfg["target_features"],
    )

    nw = task_cfg.get("num_workers", 1)
    print(f"[FT DATALOADER] split={split:<12} batch={s3_batch} workers={nw}")
    dataset.prewarm()

    loader = DataLoader(
        dataset,
        batch_size         = s3_batch,
        shuffle            = (split == "ft_train"),
        num_workers        = nw,
        pin_memory         = True,
        persistent_workers = (nw > 0),
        prefetch_factor    = 2 if nw > 0 else None,
        drop_last          = (split == "ft_train"),
        collate_fn         = _collate,
    )
    return loader, dataset