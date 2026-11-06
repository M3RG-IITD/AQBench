# AQBench — Air Quality Forecasting Benchmark

AQBench is an open benchmarking framework for evaluating deep learning models on air quality forecasting tasks using WRF-Chem simulation data. It provides a standardised pipeline covering data preparation, multi-stage training, autoregressive inference, and a rich evaluation suite — including exceedance detection, episode-focused metrics, spectral error decomposition, and ground-station bias correction.

All model configs exactly what was used for the benchmarking.

---

## Table of Contents

- [AQBench — Air Quality Forecasting Benchmark](#aqbench--air-quality-forecasting-benchmark)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Repository Layout](#repository-layout)
  - [Installation](#installation)
  - [Data Format](#data-format)
  - [Tasks](#tasks)
  - [Supported Models](#supported-models)
  - [Step-by-Step Workflow](#step-by-step-workflow)
    - [1. Prepare Training / Validation Indices](#1-prepare-training--validation-indices)
    - [2. Prepare Test Indices and Event Tests](#2-prepare-test-indices-and-event-tests)
    - [3. Train a Model](#3-train-a-model)
    - [4. Run Inference](#4-run-inference)
    - [5. Evaluate Results](#5-evaluate-results)
  - [Configuration Reference](#configuration-reference)
    - [Task Config](#task-config)
    - [Dataset Config](#dataset-config)
    - [Model Config](#model-config)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Dataset Features](#dataset-features)
    - [Feature Descriptions](#feature-descriptions)
    - [Channel Index by Task](#channel-index-by-task)
    - [Normalisation](#normalisation)
    - [Special HDF5 Fields: `time` and `Kyy`](#special-hdf5-fields-time-and-kyy)
    - [Autoregressive Input Tensor (`y_aux`)](#autoregressive-input-tensor-y_aux)
  - [Extending AQBench](#extending-aqbench)
    - [Adding a New Model](#adding-a-new-model)
    - [Swapping the Loss Function](#swapping-the-loss-function)
    - [Changing the Optimiser or LR Schedule](#changing-the-optimiser-or-lr-schedule)
    - [Other Configurable Levers](#other-configurable-levers)
    - [Adding a New Evaluation Metric](#adding-a-new-evaluation-metric)
  - [Output Files and Directory Structure](#output-files-and-directory-structure)
  - [Makefile Commands](#makefile-commands)
  - [License](#license)

---

## Overview

AQBench is built around a **bundled autoregressive** architecture. Models predict a fixed-length output bundle (default: 6 hours) and are evaluated at multiple forecast horizons (24 h, 72 h, 168 h) via iterative rollout. Training uses a two-stage curriculum:

- **Stage 1** — Standard one-step supervised pre-training; early stopping on validation one-step loss.
- **Stage 2** — Pushforward training with split-backward gradients; early stopping on validation stability loss.
- **Stage 3** *(optional)* — Bias-correction fine-tuning against sparse ground-station observations.

---

## Repository Layout

```
AQBench/
├── aqbench/
│   ├── dataloaders/          # DataLoader construction (WRF-Chem + ground-station)
│   ├── datasets/             # PyTorch Dataset classes
│   ├── evaluation/           # Inference runner, metrics, evaluate scripts
│   ├── layers/               # Shared layers (e.g., time embeddings)
│   ├── models/               # One sub-directory per model architecture
│   │   ├── afno/             #   Adaptive Fourier Neural Operator
│   │   ├── cono/             #   CoNO
│   │   ├── convlstm/         #   ConvLSTM
│   │   ├── fno2d/            #   Fourier Neural Operator 2D
│   │   ├── persistence/      #   Persistence (last-observation) baseline
│   │   ├── regular_lamo/     #   RegularLAMO
│   │   ├── simvp/            #   SimVP
│   │   ├── swin/             #   Swin Transformer U-Net
│   │   ├── transolver/       #   Transolver
│   │   ├── unet/             #   Standard U-Net
│   │   └── uno2d/            #   U-shaped Neural Operator 2D
│   ├── registry/             # Model, dataset, and task registries
│   ├── utils/                # Seeding, checkpointing, normalisation, logging
│   └── runner.py             # Core two-stage training loop
│
├── configs/
│   ├── dataset/              # Data preparation configs
│   │   ├── prepare_data.yaml
│   │   └── prepare_ft_data.yaml
│   └── task/                 # Task configs used during training and evaluation
│       ├── pm25.yaml
│       ├── mul.yaml
│       └── bias_correction_pm25.yaml
│
├── scripts/
│   ├── prepare_data.py           # Build train/val index parquets
│   ├── prepare_ft_data.py        # Build fine-tuning (ground-station) indices
│   ├── prepare_event_tests.py    # Build episode-filtered test sets
│   ├── run_experiment.py         # Train a model (Stages 1 + 2)
│   ├── run_inference.py          # Autoregressive rollout -> save predictions
│   ├── run_eval.py               # Evaluate saved predictions
│   ├── finetune_bias_correction.py   # Stage 3 fine-tuning against observations
│   └── evaluate_bias_correction.py   # Evaluate bias-corrected model
│
├── docs/                     # MkDocs documentation source
├── pyproject.toml
└── Makefile
```

---

## Installation

**Requirements:** Python 3.11, a CUDA-capable GPU (required for training).

```bash
# 1. Clone the repository
cd AQBench

# 2. Create and activate a virtual environment (uv recommended)
make create_environment
source .venv/bin/activate   # Linux / macOS
# .\.venv\Scripts\activate  # Windows

# 3. Install PyTorch with the correct CUDA wheel FIRST
#    (replace cu121 with your CUDA version, e.g. cu118 or cpu)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install all core dependencies
pip install -e .
# Or via uv: make requirements

# 5. (Optional) Install dependencies for specific models
pip install -e ".[lamo]"        # RegularLAMO  (needs mamba-ssm + Triton)
pip install -e ".[cono]"        # CoNO         (needs torch-frft + torchcomplex)
pip install -e ".[all-models]"  # All of the above

# 6. (Optional) Install dev tools (linting, docs)
pip install -e ".[dev]"
```

> **PyTorch note:** `pyproject.toml` lists `torch>=2.0` as a dependency so that
> editable installs resolve correctly, but the CUDA wheel **must** be installed
> via the PyTorch index URL before running `pip install -e .`, otherwise you get
> a CPU-only build.

---

## Data Format

Dataset: [AQBench on Hugging Face](https://huggingface.co/datasets/m3rg-iitd/AQBench).

AQBench expects WRF-Chem output stored as **HDF5 files**, one file per year, each containing monthly groups:

```
wrfchem_YYYY.h5
├── feature_names        # shape (26,) — ordered list of all 26 features
├── month_XX/
│   ├── data             # shape (timesteps, 140, 124, 26), float32
│   └── time             # shape (timesteps,) — hourly timestamps

```

Place all raw files under `data/raw/` (or change `raw_dir` in the config).

For the bias-correction task, matching ground-station files are required:

```
ground_station_YYYY.h5
├── feature_names        # shape (1,) — ['pm25']
├── station_mask         # shape (140, 124), bool — precomputed mask for stations with high temporal density of data
├── month_XX/
│   ├── data             # shape (timesteps, 140, 124, 1), float32
│   └── time             # shape (timesteps,) — hourly timestamps

```

<!-- For episode/event test sets, you also need annual episode-mask files:

```
2017_epi.h5
├── month_01/
│   └── episode_mask    # (T, H, W) uint8, 1 where an episode is active
└── ...
``` -->

---

## Tasks

| Task config | Target variables | Description |
|---|---|---|
| `configs/task/pm25.yaml` | `pm25` | Single-pollutant PM2.5 forecasting |
| `configs/task/mul.yaml` | `pm25`, `no2`, `co` | Multi-pollutant forecasting |
| `configs/task/bias_correction_pm25.yaml` | `pm25` | Stage 3 bias correction against observations |

---

## Supported Models

| Model key | Architecture |
|---|---|
| `swin` | Swin Transformer U-Net |
| `unet` | Standard convolutional U-Net |
| `fno2d` | Fourier Neural Operator 2D |
| `uno2d` | U-shaped Neural Operator 2D |
| `afno` | Adaptive Fourier Neural Operator |
| `cono` | CoNO |
| `convlstm` | ConvLSTM |
| `simvp` | SimVP |
| `transolver` | Transolver |
| `regular_lamo` | RegularLAMO |
| `persistence` | Persistence baseline (no training required) |

Each model lives in `aqbench/models/<key>/` and exposes:
- `model.py` — PyTorch `nn.Module` registered via `@register_model("<key>")`
- `config.yaml` — default hyper-parameters for the single-pollutant task
- `config_mul.yaml` — hyper-parameters for the multi-pollutant task

---

## Step-by-Step Workflow

All scripts are run from the **repository root**.

---

### 1. Prepare Training / Validation Indices

This script scans the raw HDF5 files and writes two Parquet index files listing valid (year, month, start-hour) windows for training and validation.

```bash
python scripts/prepare_data.py --dataset_config configs/dataset/prepare_data.yaml
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--dataset_config` | Yes | Path to the dataset preparation YAML |

**Key options in `prepare_data.yaml`:**

| Key | Description |
|---|---|
| `raw_dir` | Path to directory containing `wrfchem_YYYY.h5` files |
| `out_dir` | Output directory (default: `data/processed`) |
| `Tin` | Input context length in hours (e.g. `6`) |
| `Tout` | Output forecast horizon in hours (e.g. `6`) |
| `train_stride` | Step between training windows in hours |
| `val_stride` | Step between validation windows in hours |
| `train` | Dict mapping year → list of months to use for training |
| `val` | Dict mapping year → list of months to use for validation |
| `train_max_samples_per_month` | Cap on windows per month (`null` = no cap) |
| `seed` | Random seed for reproducible sub-sampling |

**Outputs** written to `data/processed/`:
- `train_index.parquet`
- `val_index.parquet`

---

### 2. Prepare Test Indices and Event Tests

This script generates non-overlapping test windows at multiple forecast horizons (24 h, 72 h, 168 h) and, if episode-mask files are present, filters them to windows where a pollution episode is active.

```bash
python scripts/prepare_event_tests.py --dataset_config configs/dataset/prepare_data.yaml
```

**Additional keys in the dataset config used by this script:**

| Key | Default | Description |
|---|---|---|
| `horizons` | `[24, 72, 168]` | Forecast horizons to generate test sets for |
| `episode_thresh` | `20` | Minimum number of active episode cells at the last forecast hour to qualify as an event window |
| `epi_mask_dir` | `data/processed` | Directory containing `{year}_epi.h5` files |
| `test` | — | Dict mapping year → list of months for the test set |

**Outputs** written to `data/processed/`:
- `test_24.parquet`, `test_72.parquet`, `test_168.parquet` — full test sets
- `test_24_event.parquet`, `test_72_event.parquet`, `test_168_event.parquet` — episode-filtered subsets

For the bias-correction fine-tuning splits, run:

```bash
python scripts/prepare_ft_data.py --cfg configs/dataset/prepare_ft_data.yaml
```

This produces `ft_train_index.parquet`, `ft_val_index.parquet`, and `ft_test_index.parquet`.

---

### 3. Train a Model

```bash
python scripts/run_experiment.py \
    --model swin \
    --task_config configs/task/pm25.yaml
```

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model` | Yes | — | Model key (e.g. `swin`, `unet`, `fno2d`) |
| `--task_config` | No | `configs/task/pm25.yaml` | Path to task YAML |
| `--exp_name` | No | From model config | Override the `exp_name` from the model config |
| `--config_name` | No | `config` | Model config file basename to load |
| `--time_embed_dim` | No | From model config | Override `time_embed_d` in model args |

**Persistence baseline** (no training):

```bash
python scripts/run_experiment.py \
    --model persistence \
    --task_config configs/task/pm25.yaml
```

**Multi-pollutant task:**

```bash
python scripts/run_experiment.py \
    --model swin \
    --task_config configs/task/mul.yaml \
    --config_name config_mul
```

**Checkpoints and logs** are saved to `checkpoints/<exp_name>/`:

| File | Contents |
|---|---|
| `stage1.pt` | Best Stage 1 checkpoint (lowest `val_onestep`) |
| `stage1_last.pt` | Latest Stage 1 checkpoint (for resuming) |
| `best_model.pt` | Best Stage 2 checkpoint (lowest `val_stability`) |
| `best_model_last.pt` | Latest Stage 2 checkpoint (for resuming) |
| `training_log.csv` | Per-epoch loss and learning rate history |
| `training_state.json` | Stage completion flags; allows safe resumption |
| `efficiency.json` | Timing and convergence summary |
| `nvidia_smi.log` | GPU utilisation log (sampled every 60 s) |

If a run is interrupted, re-running the same command will **resume automatically** from the last saved checkpoint.

---

### 4. Run Inference

Inference performs a **bundled autoregressive rollout**: the model is called repeatedly in 6-hour bundles, with ground-truth meteorology and emissions fed in at each step while only the predicted pollutant fields are carried forward.

```bash
python scripts/run_inference.py \
    --model swin \
    --task_config configs/task/pm25.yaml \
    --exp_name swin_pm25 \
    --split test_168
```

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model` | Yes | — | Model key or `persistence` |
| `--task_config` | Yes | — | Path to task YAML |
| `--exp_name` | Yes | — | Experiment name (must match training) |
| `--split` | No | `test_168` | Data split: `test_24`, `test_72`, `test_168`, `test_epi_24`, `test_epi_72`, `test_epi_168` |
| `--batch_size` | No | `4` | Inference batch size |
| `--model_config` | No | `config.yaml` | Model config filename |
| `--eval_Tout` | No | Inferred from split name | Override forecast horizon (must be a multiple of 6) |

The `eval_Tout` is inferred from the split name automatically:

| Split | `eval_Tout` |
|---|---|
| `test_24` / `test_epi_24` | 24 |
| `test_72` / `test_epi_72` | 72 |
| `test_168` / `test_epi_168` | 168 |
| Any other | Value of `Tout` from task config |

**Outputs** written to `data/outputs/<task_name>/<exp_name>/`:

| File | Shape | Contents |
|---|---|---|
| `preds_<split>.npy` | `(N, H, W, T_out, C_tgt)` | Predictions in physical units |
| `targets_<split>.npy` | `(N, H, W, T_out, C_tgt)` | Ground truth in physical units |

---

### 5. Evaluate Results

```bash
python scripts/run_eval.py \
    --exp_name swin_pm25 \
    --task_config configs/task/pm25.yaml \
    --split test_168
```

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--exp_name` | Yes | — | Experiment name |
| `--task_config` | Yes | — | Path to task YAML |
| `--split` | No | `test` | Split to evaluate |
| `--no_pde` | No | False | Skip PDE residual computation |
| `--grid_height_path` | No | `data/stats/emission_height.npy` | Path to grid height array (for PDE residual) |
| `--episode_eval` | No | False | Enable episode-specific metric evaluation |
| `--episode_index` | No | None | Path to episode index file |
| `--epi_mask_dir` | No | `data/processed` | Directory containing episode masks |

**Outputs** written to `reports/<task_name>/`:
- `leaderboard_<metric>.csv` — per-experiment, per-timestep metric tables
- `summary.csv` — scalar summary across all experiments

---

### 6. Bias Correction (Optional Stage 3)

Stage 3 fine-tunes a trained Stage 2 checkpoint against sparse ground-station PM2.5 observations using a masked MSE loss.

**Fine-tune:**

```bash
python scripts/finetune_bias_correction.py \
    --model swin \
    --exp_name swin_pm25 \
    --task_cfg configs/task/bias_correction_pm25.yaml
```

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model` | Yes | — | Model key |
| `--exp_name` | Yes | — | Experiment name (loads Stage 2 checkpoint from `checkpoints/<exp_name>/best_model.pt`) |
| `--task_cfg` | No | `configs/task/bias_correction_pm25.yaml` | Task config for fine-tuning |
| `--model_cfg` | No | `aqbench/models/<model>/config.yaml` | Model config path |
| `--s2_ckpt` | No | `checkpoints/<exp_name>/best_model.pt` | Override Stage 2 checkpoint path |
| `--out_dir` | No | `checkpoints/<exp_name>/stage3/` | Directory for Stage 3 outputs |

Stage 3 runs for up to 100 epochs with patience 10, using a learning rate of `1/100` of the Stage 2 learning rate. The best checkpoint is saved as `stage3_best.pt`.

**Evaluate:**

```bash
python scripts/evaluate_bias_correction.py \
    --model swin \
    --exp_name swin_pm25 \
    --task_cfg configs/task/bias_correction_pm25.yaml
```

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model` | Yes | — | Model key |
| `--exp_name` | Yes | — | Experiment name |
| `--task_cfg` | No | `configs/task/bias_correction_pm25.yaml` | Task config |
| `--model_cfg` | No | Model default | Model config path |
| `--ckpt_dir` | No | `checkpoints` | Root checkpoint directory |
| `--report_dir` | No | `reports/bias_correction` | Output directory |
| `--skip_s3` | No | False | Evaluate only Stage 2 (skip Stage 3 checkpoint) |

This produces a comparison table of **WRF-Chem raw → Stage 2 → Stage 3** performance at ground-station locations, along with per-timestep leaderboard CSVs.

---

## Configuration Reference

### Task Config

Located under `configs/task/`. Passed to both training and evaluation.

```yaml
raw_dir: data/raw          # Path to wrfchem_YYYY.h5 files

task_name: pm25            # Identifier used in output paths

Tin: 6                     # Input context window (hours)
Tout: 6                    # Training output bundle length (hours)

num_workers: 1             # DataLoader worker count
seed: 42

constraint_weight: 0       # Weight for optional physics penalty (0 = disabled)

input_features:             # Add no2, co for multi-pollutant
  - pm25
  - q2
  - t2
  - u10
  - v10
  - rainc
  - rainnc
  - swdown
  - pblh
  - psfc
  - NOx_e
  - NOx_finn
  - NH3_e
  - NH3_finn
  - PM25_e
  - PM25_finn
  - NMVOC_e
  - NMVOC_finn
  - SO2_e
  - SO2_finn
  - bio

target_features:
  - pm25                   # Add no2, co for multi-pollutant
```

### Dataset Config

Located under `configs/dataset/`. Used only by data-preparation scripts.

```yaml
raw_dir: data/raw
out_dir: data/processed

Tin: 6
Tout: 6
seed: 42

train_stride: 1            # Hours between consecutive training windows
val_stride: 6

train_max_samples_per_month: null   # null = use all windows

train:
  2016: [2, 4, 6, 7, 9, 10, 12]
  2017: [2, 3, 4, 6, 7, 10, 12]
  2018: [1, 2, 3, 4, 5, 6]

val:
  2018: [7, 8, 9, 10]

test:
  2018: [12]
  2019: [2, 3, 4, 5]
```

### Model Config

Located under `aqbench/models/<key>/config.yaml`. Passed to training scripts.

```yaml
model: swin                # Must match the @register_model key
exp_name: swin_pm25        # Checkpoint and report directory name

learning_rate: 1.0e-3

use_compile: false         # Enable torch.compile (requires PyTorch >= 2.0)

model_args:
  in_channels: 21          # Must match len(task_config.input_features)
  height: 140
  width: 124
  Tin: 6
  Tout: 6
  embed_dim: 72
  # ... model-specific hyper-parameters
```

---

## Evaluation Metrics

All metrics are computed per-timestep and reported as `(T_out,)` arrays in the leaderboard CSVs, plus scalar summaries.

| Metric | Description |
|---|---|
| `rmse` | Root Mean Squared Error |
| `mae` | Mean Absolute Error |
| `smape` | Symmetric Mean Absolute Percentage Error (background-masked, %) |
| `bias` | Mean bias — positive = over-prediction |
| `mfb` | Mean Fractional Bias, range [−2, 2] |
| `corr` | Pearson correlation coefficient |
| `r2` | Coefficient of determination |
| `fac2` | Fraction of predictions within a factor of 2 of observations |
| `frmse_low/mid/high` | Spectral RMSE decomposed by wavenumber band |
| `epi_rmse` | RMSE restricted to episode grid cells |
| `epi_smape` | SMAPE restricted to episode grid cells |
| `epi_mfb` | MFB restricted to episode grid cells |
| `exc_recall` | Exceedance recall: TP / (TP + FN) at threshold 90 µg/m³ |
| `exc_precision` | Exceedance precision |
| `exc_f1` | Exceedance F1 score |
| `exc_csi` | Critical Success Index for exceedance detection |

Background-masked metrics exclude grid cells where the observed concentration is ≤ 1.0 µg/m³.

---

## Dataset Features

The dataset reads features **by name** from the HDF5 `feature_names` array, then selects the subset listed in `input_features` and `target_features` in the task config. The order of features in your task config determines the channel index seen by the model — `input_features[i]` is channel `i` in the input tensor `x[..., i]`.

`in_channels` in the model config must equal `len(input_features)`.

---

### Feature Descriptions

**Pollutant state variables** — can appear as both inputs and targets.

| Feature | Full name | Units |
|---|---|---|
| `pm25` | PM2.5 particulate concentration | µg/m³ |
| `no2` | Nitrogen dioxide concentration | µg/m³ |
| `co` | Carbon monoxide concentration | µg/m³ |
| `pm10` | PM10 particulate concentration | µg/m³ |

**Meteorological fields** — WRF output variables, input-only.

| Feature | Full name | Units |
|---|---|---|
| `q2` | 2 m specific humidity | kg/kg |
| `t2` | 2 m temperature | K |
| `u10` | 10 m U-component of wind | m/s |
| `v10` | 10 m V-component of wind | m/s |
| `rainc` | Accumulated convective rainfall | mm |
| `rainnc` | Accumulated non-convective rainfall | mm |
| `swdown` | Downward shortwave radiation at surface | W/m² |
| `pblh` | Planetary boundary layer height | m |
| `psfc` | Surface pressure | Pa |

**Anthropogenic emissions** — input-only.

| Feature | Species | Source |
|---|---|---|
| `NOx_e` | NOx | Anthropogenic inventory |
| `NH3_e` | NH3 | Anthropogenic inventory |
| `PM25_e` | PM2.5 | Anthropogenic inventory |
| `NMVOC_e` | NMVOC | Anthropogenic inventory |
| `SO2_e` | SO2 | Anthropogenic inventory |

**Fire emissions (FINN)** — input-only.

| Feature | Species | Source |
|---|---|---|
| `NOx_finn` | NOx | FINN fire emissions |
| `NH3_finn` | NH3 | FINN fire emissions |
| `PM25_finn` | PM2.5 | FINN fire emissions |
| `NMVOC_finn` | NMVOC | FINN fire emissions |
| `SO2_finn` | SO2 | FINN fire emissions |

**Biogenic** — input-only.

| Feature | Description |
|---|---|
| `bio` | Biogenic VOC emissions |

---

### Channel Index by Task

The channel index is the position of a feature in `input_features` as listed in the task config. This directly maps to the last dimension of `x` and `y_aux` tensors seen by the model.

**`pm25` task — 21 input channels, 1 target**

| Index | Feature | Group |
|---|---|---|
| 0 | `pm25` | Pollutant |
| 1 | `q2` | Meteorology |
| 2 | `t2` | Meteorology |
| 3 | `u10` | Meteorology |
| 4 | `v10` | Meteorology |
| 5 | `rainc` | Meteorology |
| 6 | `rainnc` | Meteorology |
| 7 | `swdown` | Meteorology |
| 8 | `pblh` | Meteorology |
| 9 | `psfc` | Meteorology |
| 10 | `NOx_e` | Anthropogenic |
| 11 | `NOx_finn` | Fire |
| 12 | `NH3_e` | Anthropogenic |
| 13 | `NH3_finn` | Fire |
| 14 | `PM25_e` | Anthropogenic |
| 15 | `PM25_finn` | Fire |
| 16 | `NMVOC_e` | Anthropogenic |
| 17 | `NMVOC_finn` | Fire |
| 18 | `SO2_e` | Anthropogenic |
| 19 | `SO2_finn` | Fire |
| 20 | `bio` | Biogenic |

Target channel 0: `pm25`

**`mul` task — 23 input channels, 3 targets**

| Index | Feature | Group |
|---|---|---|
| 0 | `pm25` | Pollutant |
| 1 | `q2` | Meteorology |
| 2 | `t2` | Meteorology |
| 3 | `u10` | Meteorology |
| 4 | `v10` | Meteorology |
| 5 | `rainc` | Meteorology |
| 6 | `rainnc` | Meteorology |
| 7 | `swdown` | Meteorology |
| 8 | `pblh` | Meteorology |
| 9 | `no2` | Pollutant |
| 10 | `co` | Pollutant |
| 11 | `psfc` | Meteorology |
| 12 | `NOx_e` | Anthropogenic |
| 13 | `NOx_finn` | Fire |
| 14 | `NH3_e` | Anthropogenic |
| 15 | `NH3_finn` | Fire |
| 16 | `PM25_e` | Anthropogenic |
| 17 | `PM25_finn` | Fire |
| 18 | `NMVOC_e` | Anthropogenic |
| 19 | `NMVOC_finn` | Fire |
| 20 | `SO2_e` | Anthropogenic |
| 21 | `SO2_finn` | Fire |
| 22 | `bio` | Biogenic |

Target channel 0: `pm25` — Target channel 1: `no2` — Target channel 2: `co`

---

### Normalisation

All features are normalised to zero mean and unit variance using **global statistics** stored in `data/stats/global_mean_std.json`. This file must contain:

```json
{
  "feature_names": ["pm25", "q2", "t2", ...],
  "mean": [<float>, ...],
  "std":  [<float>, ...]
}
```

The dataset looks up each feature by name in this file, so the order in `global_mean_std.json` does not need to match the order in the task config. Statistics are computed once across all training data and fixed for all splits — test data is normalised with training statistics.

Model predictions are in normalised space. `run_inference.py` denormalises them to physical units before saving, using `target_mean` and `target_std` from the dataset object.

---

### Special HDF5 Fields: `time` and `Kyy`

These two fields exist in the HDF5 files but are **not** feature channels in the data array and do not appear in `input_features` in any task config.

---

**`time`**

Stored as a separate dataset at `f["month_MM"]["time"][:]` — a 1-D array of ISO-8601 hourly timestamp strings (bytes or str), one entry per timestep `T`. It is **not** part of the `data` array and is never passed through feature selection or normalisation.

The dataset class reads it, decodes it to `datetime64[h]`, then converts it to **integer hours since 2000-01-01 00:00 UTC**:

```python
_EPOCH = np.datetime64("2000-01-01", "h")
t_int  = (times_dt - _EPOCH).astype(np.int64)
```

This integer array is returned as `t_in` — the second argument to every model's `forward(x, t_in)`. It has shape `(Tin,)` and gives the model absolute wall-clock position in the year, which is important for capturing seasonal and diurnal cycles.

Your HDF5 files must include this dataset. If it is missing the dataset will raise a `KeyError` at load time.

The `time` array is also used during data preparation (`prepare_data.py`) to verify that all timesteps within a candidate window are **contiguous** — gaps caused by missing hours are detected via `np.diff(times_dt) == 1h` and those windows are silently dropped.

---

**`Kyy`**

`Kyy` is the **horizontal turbulent diffusion coefficient** (eddy diffusivity, units m²/s). It is stored as a regular feature channel inside the HDF5 `data` array and must appear in `feature_names`, but it is **not listed in any task config's `input_features`** — it is never fed to the model.

It is used exclusively by the PDE residual evaluator (`aqbench/evaluation/pde_residual.py`), which implements a finite-difference advection-diffusion-source check:

```
dC/dt + u·∇C − Kyy·∇²C = (PM25_e + PM25_finn) × 10⁹ / grid_height
```

where `u`, `v` are the 10 m wind components, `PM25_e` and `PM25_finn` are anthropogenic and fire emission rates, and `grid_height` is loaded from `data/stats/emission_height.npy`. The evaluator reads `Kyy`, `u10`, `v10`, `PM25_e`, and `PM25_finn` directly from the raw HDF5 (bypassing the dataset class), using a grid spacing of 25 km and a time step of 3600 s.

If `Kyy` is absent from your HDF5 `feature_names`, the PDE residual computation will raise a `ValueError`. Pass `--no_pde` to `run_eval.py` to skip it:

```bash
python scripts/run_eval.py --exp_name swin_pm25 --task_config configs/task/pm25.yaml \
    --split test_168 --no_pde
```

---

### Autoregressive Input Tensor (`y_aux`)

During the autoregressive rollout, meteorological and emission channels are taken from ground truth for each future bundle while only the predicted pollutant channels are carried forward. This is handled by `build_next_input` in both `runner.py` and `run_inference.py`.

The `y_aux` tensor returned by the dataset provides the ground-truth input features for future timesteps. It has shape `(y_aux_len, H, W, C_in)` and covers the auxiliary (non-target) channels for the future window. During training Stage 2, `y_aux_len = BUNDLE_SIZE` (6 hours). During inference at horizon `T`, `y_aux_len = T - BUNDLE_SIZE`.

---

## Extending AQBench

---

### Adding a New Model

A model needs four things: a registered `nn.Module`, a config file, an `__init__.py`, and one import line. Nothing else in the framework needs to change.

**1. Create the directory**

```
aqbench/models/mymodel/
├── __init__.py      (empty)
├── model.py
└── config.yaml
```

**2. Implement the model**

The only hard contract is the `forward` signature and the `num_targets` constructor argument. Everything else is up to you.

```python
# aqbench/models/mymodel/model.py

import torch.nn as nn
from aqbench.registry.model_registry import register_model

@register_model("mymodel")
class MyModel(nn.Module):
    def __init__(self, in_channels, height, width, Tin, Tout,
                 num_targets, **kwargs):
        super().__init__()
        # Build your architecture here.
        # num_targets is 1 for pm25-only, 3 for mul (pm25+no2+co).

    def forward(self, x, t_in):
        # x     : (B, Tin, H, W, C_in)  — normalised input stack
        # t_in  : (B,)                  — integer hour-of-year timestamp
        #                                 use aqbench.layers.embeddings for
        #                                 a ready-made time embedding layer
        # return: (B, Tout, H, W, num_targets)  — normalised predictions
        ...
```

The time embedding helper is available if you want it:

```python
from aqbench.layers.embeddings import AbsoluteTimeEmbedding
self.time_embed = AbsoluteTimeEmbedding(embed_dim=16, max_hours=8760)
# then in forward:
t_emb = self.time_embed(t_in)   # (B, embed_dim)
```

**3. Write the config**

```yaml
# aqbench/models/mymodel/config.yaml
model: mymodel
exp_name: mymodel_pm25
learning_rate: 1.0e-3

# Training schedule — all optional, these are the defaults
epochs: 200
warmup_s1: 5
warmup_s2: 10
patience_s1: 30
patience_s2: 30
s2_lr_factor: 0.1    # Stage 2 LR = learning_rate × s2_lr_factor
use_compile: false   # set true for torch.compile speedup (PyTorch >= 2.0)

model_args:
  in_channels: 21    # must equal len(task_config.input_features)
  height: 140
  width: 124
  Tin: 6
  Tout: 6
  # ... your own hyper-parameters
```

For the multi-pollutant task, add a `config_mul.yaml` with `in_channels: 23` and any architecture changes needed.

**4. Register the import**

Add one line to `aqbench/models/__init__.py`:

```python
from . import mymodel  # noqa: F401
```

**5. Train**

```bash
python scripts/run_experiment.py --model mymodel --task_config configs/task/pm25.yaml
```

That is everything. The two-stage training loop, checkpointing, early stopping, DDP, and evaluation all work automatically.

---

### Swapping the Loss Function

The loss function lives in one place in `aqbench/runner.py`:

```python
def relative_l2_loss(pred, target, eps=1e-6):
    spatial_dims = (-3, -2)
    diff  = torch.norm(pred - target, p=2, dim=spatial_dims)
    denom = torch.norm(target,        p=2, dim=spatial_dims)
    return (diff / (denom + eps)).mean()
```

It is called in four spots — `train_epoch_phase1`, `train_epoch_phase2_pushforward`, `validate_onestep`, and `validate_pushforward` — always as `relative_l2_loss(pred, y)`.

To replace it, rewrite the function or swap in a different one. Some drop-in examples:

```python
# MAE
def loss_fn(pred, target, **_):
    return torch.mean(torch.abs(pred - target))
```

The function signature only needs to accept `(pred, target)` — nothing else is passed.

---

### Changing the Optimiser or LR Schedule

**Optimiser** — find this block in `run_experiment` in `runner.py`:

```python
optimizer = torch.optim.AdamW(build_param_groups(model), lr=base_lr)
```

Swap in any PyTorch optimiser:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
```

`build_param_groups` separates weight-decay and no-decay parameters (biases and norms get zero weight decay). You can keep using it with any optimiser that accepts param groups, or replace it with `model.parameters()` for simplicity.

**LR schedule** — `get_lr_scheduler` in `runner.py` implements cosine annealing with linear warmup:

```python
def get_lr_scheduler(optimizer, warmup_epochs, max_epochs, min_lr_factor=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

To use a different schedule, replace the return value with any PyTorch scheduler:

```python
return torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=base_lr, total_steps=max_epochs)
```

The `min_lr_factor` floor (default `0.01`) means the LR never drops below 1% of its peak — increase it if you want a higher floor.

---

### Other Configurable Levers

These are constants at the top of `runner.py` and `run_inference.py` that you can change directly:

| Constant | File | Default | Effect |
|---|---|---|---|
| `BATCH_SIZE` | `runner.py` | `24` | Training batch size |
| `BUNDLE_SIZE` | `runner.py` | `6` | Hours per autoregressive bundle — must divide evenly into your forecast horizons |
| `BUNDLE_SIZE` | `run_inference.py` | `6` | Must match the value used during training |

And these are set in the **model config YAML** with no code changes needed:

| Key | Default | Effect |
|---|---|---|
| `epochs` | `200` | Maximum training epochs per stage |
| `warmup_s1` | `5` | Linear LR warmup epochs for Stage 1 |
| `warmup_s2` | `10` | Linear LR warmup epochs for Stage 2 |
| `patience_s1` | `30` | Early stopping patience for Stage 1 |
| `patience_s2` | `30` | Early stopping patience for Stage 2 |
| `s2_lr_factor` | `0.1` | Stage 2 LR multiplier relative to Stage 1 LR |
| `use_compile` | `false` | Enable `torch.compile` for faster training |

And in the **task config YAML**:

| Key | Default | Effect |
|---|---|---|
| `constraint_weight` | `0` | Weight applied to any physics penalty term in the loss |
| `Tin` | `6` | Input context window length (hours) |
| `Tout` | `6` | Bundle size used during training |
| `input_features` | (21 features) | Which WRF-Chem channels are fed as input |
| `target_features` | `[pm25]` | Which channels the model predicts |
| `num_workers` | `1` | DataLoader worker count |

---

### Adding a New Evaluation Metric

All metrics live in `aqbench/evaluation/metrics.py`. Each function takes `(preds, targets)` as `(N, H, W, T)` arrays and returns `(mean_T, std_T)` arrays of shape `(T_out,)`.

To add a new metric:

```python
# aqbench/evaluation/metrics.py

def compute_mymetric(preds, targets):
    """
    Args:
        preds, targets: (N, H, W, T) — NaN-safe
    Returns:
        mean_metric: (T,), std_metric: (T,)
    """
    per_sample = ...  # (N, T)
    return np.nanmean(per_sample, axis=0), np.nanstd(per_sample, axis=0)
```

Then import and call it in `aqbench/evaluation/evaluate.py` alongside the existing metrics, and add it to the `update_leaderboard` calls to get it into the output CSVs automatically.

---

## Output Files and Directory Structure

After a full run the project directory will contain:

```
data/
├── raw/                          # Input HDF5 files (user-provided)
└── processed/
    ├── train_index.parquet
    ├── val_index.parquet
    ├── test_24.parquet
    ├── test_72.parquet
    ├── test_168.parquet
    ├── test_24_event.parquet     # Episode-filtered test windows
    ├── test_72_event.parquet
    ├── test_168_event.parquet
    ├── ft_train_index.parquet    # Ground-station splits (Stage 3)
    ├── ft_val_index.parquet
    └── ft_test_index.parquet

checkpoints/
└── <exp_name>/
    ├── stage1.pt                 # Best Stage 1 checkpoint
    ├── stage1_last.pt
    ├── best_model.pt             # Best Stage 2 checkpoint
    ├── best_model_last.pt
    ├── training_log.csv
    ├── training_state.json
    ├── efficiency.json
    ├── nvidia_smi.log
    └── stage3/                   # Created if bias correction is run
        ├── stage3_best.pt
        ├── stage3_log.csv
        └── eval_mask.pt

data/outputs/
└── <task_name>/
    └── <exp_name>/
        ├── preds_test_168.npy
        └── targets_test_168.npy

reports/
└── <task_name>/
    ├── leaderboard_rmse.csv
    ├── leaderboard_mae.csv
    └── summary.csv
```

---

## Makefile Commands

```bash
make help                # List all available commands
make create_environment  # Create .venv with uv
make requirements        # Install dependencies (uv sync)
make data                # Build training and validation indices
make lint                # Check code style with ruff
make format              # Auto-format code with ruff
make clean               # Remove compiled Python files and __pycache__
```

---


## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
