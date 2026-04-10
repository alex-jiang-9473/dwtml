# DWT-SIREN Train + Reconstruction Guide

This project trains SIREN networks on wavelet (DWT) sub-bands and reconstructs images from the trained checkpoints.

This README is focused on running:
- `train_dwt_siren.py`
- `reconstruct_dwt_siren.py`

## 1. Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Current minimal dependencies are listed in `requirements.txt`.

## 2. Input Data

Training and reconstruction expect Kodak PNG images in:

- `kodak-dataset/`

The selected image is controlled by `IMAGEID` in `experiment_config.py`.
Example: `IMAGEID = "kodim08"` means the script reads:

- `kodak-dataset/kodim08.png`

## 3. Quick Start

Run from the project root (`dwt-ml`):

```bash
python train_dwt_siren.py
python reconstruct_dwt_siren.py
```

Order matters:
1. Run training first (generates checkpoints + manifest).
2. Run reconstruction second (consumes training outputs).

## 4. What Each Script Produces

### Training output (`train_dwt_siren.py`)

Base folder:

- `results/dwt_siren_models/<IMAGEID>/`

Important files:
- `manifest.json`: full manifest used by reconstruction.
- `manifest.csv`: tabular view of all band candidates.
- Per-channel and per-band subfolders with:
  - candidate checkpoints (`*.pt`)
  - `best_model.pt`
  - `band_metadata.pt`
  - `comparison.json`
  - `comparison.csv`

### Reconstruction output (`reconstruct_dwt_siren.py`)

Base folder:

- `results/reconstructed_images/`

Important files:
- `<IMAGEID>_combination_log.json`
- `<IMAGEID>_combination_metrics.csv`
- `<IMAGEID>_combination_selections.csv`
- `<IMAGEID>_best_per_band_training.csv`
- `<IMAGEID>_best_combination.png`
- `<IMAGEID>_worst_combination.png`
- `<IMAGEID>_reconstructed_best.png`
- Optional combination images folder: `<IMAGEID>_combo_images/`

## 5. Configuration Guide

Most settings are controlled in `experiment_config.py`.

### Core pipeline settings

Edit these first:

- `IMAGEID`: input image basename in `kodak-dataset/`.
- `LEVELS`: DWT decomposition depth.
- `WAVELET`: wavelet name passed to PyWavelets (example: `"db4"`).
- `MODEL_DIR`: where training outputs and manifest are written.
- `OUTPUT_DIR`: where reconstruction outputs are written.

### Training behavior switches

- `TRAIN_HF_BANDS`:
  - `True`: train LL + high-frequency bands (`cH/cV/cD`).
  - `False`: only LL bands are trained.
- `COMPARE_CONFIGS`:
  - `True`: evaluate all candidate network/iteration/hyperparameter combinations.
  - `False`: train only the first candidate per band.
- `SKIP_HF_TRAINING`:
  - `True`: skip HF training in legacy parameter-allocation path.
  - In current main flow, prefer controlling HF behavior with `TRAIN_HF_BANDS`.

### Candidate config spaces

You can tune search spaces in these dictionaries:

- `NETWORK_CONFIG_GROUPS`
- `ITERATION_CONFIG_GROUPS`
- `HYPERPARAM_CONFIG_GROUPS`

Role keys used by the pipeline:
- `"y_ll"` for Y channel LL band
- `"uv_ll"` for U/V channel LL bands
- `"hf"` for all high-frequency bands

The Cartesian product of these groups defines candidate configurations for each role.

### Threshold / sparsity controls

Sparse sampling thresholds are controlled by:

- `ROLE_FILTER_THRESHOLDS`: default threshold by role.
- `BAND_FILTER_THRESHOLDS`: optional per-band override (example key: `"Y_cH_L1"`).

Resolution order:
1. exact `BAND_FILTER_THRESHOLDS` override
2. role default in `ROLE_FILTER_THRESHOLDS`
3. fallback default argument

## 6. Reconstruction Sampling Controls

`reconstruct_dwt_siren.py` has additional top-level constants:

- `MAX_COMBINATIONS`: cap number of checkpoint combinations to evaluate.
- `SAMPLE_RANDOM_COMBINATIONS`:
  - `True`: random unique combination sampling.
  - `False`: deterministic first-N/exhaustive order.
- `RANDOM_SAMPLE_SEED`: random seed for reproducibility.
- `SAVE_COMBINATION_IMAGES`: whether to save each sampled combo image.

## 7. Typical Configuration Recipes

### Fast sanity run

In `experiment_config.py`:
- set `COMPARE_CONFIGS = False`
- set `TRAIN_HF_BANDS = False`
- reduce iteration list(s), e.g. keep only `[100]`

In `reconstruct_dwt_siren.py`:
- set `MAX_COMBINATIONS = 5`
- set `SAVE_COMBINATION_IMAGES = False`

### Full comparison run

In `experiment_config.py`:
- set `COMPARE_CONFIGS = True`
- set `TRAIN_HF_BANDS = True`
- keep broader network/iteration/hyperparameter groups

In `reconstruct_dwt_siren.py`:
- increase `MAX_COMBINATIONS` or set to `None` for exhaustive

## 8. Troubleshooting

- Error: `No manifest.json found ... Run train_dwt_siren.py first.`
  - Run training first and verify `MODEL_DIR` points to the same image ID.

- Error: missing image file
  - Confirm `kodak-dataset/<IMAGEID>.png` exists.

- CUDA out-of-memory / slow runs
  - Reduce candidate count (`COMPARE_CONFIGS=False`), reduce iterations, or run on CPU.

## 9. Reproducibility Notes

- Reconstruction combination sampling is reproducible when:
  - `SAMPLE_RANDOM_COMBINATIONS = True`
  - fixed `RANDOM_SAMPLE_SEED`
- Training outcomes can still vary slightly due to hardware/runtime nondeterminism unless full deterministic settings are enforced in PyTorch.
