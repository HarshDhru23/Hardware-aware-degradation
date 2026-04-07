# Hardware-aware Degradation Pipeline

Comprehensive pipeline for generating realistic low-resolution (LR) training data from high-resolution (HR) satellite imagery for multi-frame super-resolution (MFSR).

This repository supports both:
- Single-band (PAN) workflows
- Multi-band (MS) workflows

and both:
- On-the-fly degradation (during training)
- Pre-generated datasets (saved to disk)

## Quick Command Cheat Sheet

Use these first if you just need the core flow.

```bash
# 1) Environment
conda env create -f environment.yml
conda activate hardware-degradation
pip install torch torchvision torchaudio

# 2) Compute global stats for normalization
python compute_global_stats.py --input-dir data/input_pan --output configs/pan_stats.yaml

# 3) (Optional) Combine many stats into one global YAML
python combine_histograms.py --histograms configs/a_hist.npz configs/b_hist.npz --output configs/combined_stats.yaml

# 4) Generate full PAN flat dataset (v2)
python generate_training_dataset_v2.py \
  --input_dir data/input_pan \
  --output_dir data/training_dataset_pan_v2 \
  --config configs/default_config.yaml \
  --global_stats configs/combined_stats.yaml \
  --verify

# 5) Generate full MS flat dataset (v2)
python generate_training_dataset_ms_v2.py \
  --input_dir data/input_ms \
  --output_dir data/training_dataset_ms_v2 \
  --config configs/default_config.yaml \
  --global_stats configs/combined_stats.yaml \
  --verify

# 6) Quick smoke tests
python test_downsampling_mode.py
python test.py --create-sample
pytest tests/test_ms_dataset.py -v
```

## 1. What This Repository Does

The implemented observation model is:

$$
y_k = D B_k M_k x + n_k
$$

Where:
- $x$: HR image
- $M_k$: sub-pixel warping (deterministic or stochastic)
- $B_k$: anisotropic Gaussian PSF blur
- $D$: downsampling
- $n_k$: Poisson + Gaussian noise and ADC quantization
- $y_k$: $k$-th LR frame

Frame generation is controlled by downsampling_mode:
- 2: 2 LR frames with nominal shifts [(0,0), (0.5,0.5)]
- 4: 4 LR frames with nominal shifts [(0,0), (0.25,0.25), (0.5,0.5), (0.75,0.75)]

## 2. Repository Map

```text
Hardware-aware-degradation/
├── src/
│   ├── dataset.py                     # DegradationDataset (PAN), MSDataset (MS), dataloader helpers
│   ├── pregenerated_dataset.py        # PreGeneratedDataset for sample_XXXXXX datasets
│   ├── config.py                      # ConfigManager
│   ├── degradation/
│   │   ├── pipeline.py                # DegradationPipeline
│   │   └── operators.py               # Warping, Blur, Downsampling, Noise operators
│   └── utils/
│       ├── data_io.py                 # GeoTIFFLoader, PatchExtractor, patch save helpers
│       ├── bicubic_core.py
│       └── validation.py
├── configs/
│   ├── default_config.yaml            # Main actively used config
│   ├── fast_config.yaml               # Legacy/older-key config
│   ├── high_quality_config.yaml       # Legacy/older-key config
│   └── debug_config.yaml              # Legacy/older-key config
├── examples/
│   ├── train_pan_minimal.py           # Minimal PAN training loop template
│   └── train_ms_minimal.py            # Minimal MS training loop template
├── generate_training_dataset.py       # PAN pregen, sample_XXXXXX format (PreGeneratedDataset compatible)
├── generate_training_dataset_v2.py    # PAN pregen, flat ISRO naming
├── generate_training_dataset_ms_v2.py # MS pregen, flat ISRO naming
├── test_dataset_generation.py         # PAN small test dataset + PNG previews
├── test_dataset_generation_ms.py      # MS small test dataset + RGB previews
├── compute_global_stats.py            # Per-folder percentile/hist stats
├── combine_histograms.py              # Merge multiple histogram files
├── analyze_degradation.py             # Step-wise pipeline visualization
├── test.py                            # Single-image interactive test
├── test_downsampling_mode.py          # Mode-2 vs mode-4 behavior check
├── convert_npy_to_png.py              # Utility converter
format
├── tests/                             # Pytest tests (mix of current and legacy assumptions)
└── scripts/process_images.py          # Older patch-extraction pipeline entrypoint
```

## 3. Setup From Scratch

### 3.1 System Requirements

- macOS/Linux (tested mostly on Linux clusters and macOS local dev)
- Python 3.9 recommended
- Conda or venv
- gdal/rasterio compatible build environment (Conda usually easiest)

### 3.2 Option A: Conda (Recommended)

```bash
git clone <your-repo-url>
cd Hardware-aware-degradation

conda env create -f environment.yml
conda activate hardware-degradation
pip install torch torchvision torchaudio
```

### 3.3 Option B: venv + pip

```bash
git clone <your-repo-url>
cd Hardware-aware-degradation

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install torch torchvision torchaudio
```

### 3.4 Quick Environment Validation

```bash
python -c "import torch, numpy, rasterio, cv2, yaml; print('env ok')"
```

If this fails, fix missing package(s) before dataset generation.

## 4. Data Expectations

- Input images are GeoTIFF/TIFF files (default glob *.tif)
- PAN workflow expects single-band imagery
- MS workflow expects multiband imagery ([C,H,W] or [H,W,C], with small C, e.g. 4 or 8)

Suggested structure:

```text
data/
├── input_pan/
│   ├── img_001.tif
│   └── ...
└── input_ms/
    ├── ms_001.tif
    └── ...
```

## 5. PAN vs MS: Which Classes to Use

### 5.1 Single-band (PAN)

Use:
- DegradationDataset from src/dataset.py for on-the-fly generation
- create_dataloader helper for batching

Returns per sample:
- hr: [1,H,W]
- lr: list of frames, each [1,H',W']
- flow_vectors: [num_frames,2,H',W']
- psf_kernels, psf_params, shift_values, metadata

### 5.2 Multi-band (MS)

Use:
- MSDataset from src/dataset.py
- create_ms_dataloader helper for batching

Returns per sample:
- hr: [C,H,W]
- lr: list of frames, each [C,H',W']
- psf_kernels, psf_params, shift_values, metadata

Note: MSDataset does not currently return flow_vectors.

### 5.3 Pre-generated Dataset Loader

Use PreGeneratedDataset (src/pregenerated_dataset.py) only with datasets generated by:
- generate_training_dataset.py (the samples/sample_000000.npz style)

It is not meant for flat v2 naming outputs directly.

## 6. Configuration Guide

Primary active config: configs/default_config.yaml

Important note:
- fast_config.yaml, high_quality_config.yaml, and debug_config.yaml contain older keys (optical_sigma, motion_kernel_size, etc.) that do not fully align with the current anisotropic PSF operator requirements.
- For reliable current workflows, start from configs/default_config.yaml.

### 6.1 Strict Config Glossary (Code-Verified)

The table below is code-verified against the current implementation.

Legend for Support:
- Full: actively used and meaningful in current core flow
- Partial: accepted/used but weakly validated or may silently fallback
- Legacy/No-op: present for compatibility or old paths, not driving current core behavior

| Key | Type | Default | Enforced/Actual Range in Code | Support | Notes |
|---|---|---:|---|---|---|
| downsampling_factor | int | 4 | validated 2 to 8, operator clamps 2 to 8 | Full | Must also satisfy patch ratio rules |
| downsampling_mode | int | 4 | only 2 or 4 | Full | Invalid values raise error |
| shift_mode | str | stochastic | only stochastic has special behavior, others act deterministic | Partial | Use deterministic or stochastic only |
| shift_variance_2x | float | 0.08 | no explicit validation | Partial | Should be > 0 to avoid invalid sqrt |
| shift_variance_4x | float | 0.03 | no explicit validation | Partial | Should be > 0 to avoid invalid sqrt |
| psf_sigma_x | float | 0.6 | clipped to 0.5 to 2.0 | Full | Required by BlurOperator |
| psf_sigma_y | float | 0.8 | clipped to 0.5 to 2.0 | Full | Required by BlurOperator |
| psf_theta | float | 0.0 | clipped to 0 to 180 | Full | |
| psf_kernel_size | int | 9 | clipped to 3 to 15 and forced odd | Full | Required by BlurOperator |
| enable_gaussian | bool | true | must exist for NoiseOperator | Full | Missing key raises error |
| enable_poisson | bool | true | must exist for NoiseOperator | Full | Missing key raises error |
| gaussian_mean | float | 0.0 | clipped to -10 to 10 | Full | |
| gaussian_std | float | 0.0005 | clipped to 0.0001 to 0.1 | Full | Must exist for NoiseOperator |
| poisson_lambda | float | 1.0 | clipped to 0.1 to 5.0 | Legacy/No-op | Loaded/clipped but not used in current noise math |
| photon_gain | float | 30000.0 | clipped to 10 to 10000 | Full | Must exist for NoiseOperator |
| enable_quantization | bool | true | bool toggle | Full | Controls ADC simulation |
| quantization_bits | int | 11 | no strict validation in code | Partial | Typically use 8 to 16 |
| normalize | bool | true | boolean validation present | Full | Used by GeoTIFF loader path |
| target_dtype | str | float32 | validated in {float32,float64,uint8,uint16} | Full | |
| hr_patch_size | int | 256 | validated 64 to 1024 | Full | |
| lr_patch_size | int | 64 | validated 16 to 256 | Full | |
| patch_stride | int | 256 | no strict validation | Partial | Used mainly in older patch extraction script |
| min_valid_pixels | float | 0.95 | no strict validation in current core path | Partial | Used in patch extraction filtering |
| input_format | str | tiff | no strict runtime enforcement | Legacy/No-op | Mostly metadata/older scripts |
| output_format | str | npy | consumed by older patch-processing script | Partial | Not central to v2 generators |
| save_visualization | bool | true | consumed by scripts | Partial | Utility-level behavior |
| log_level | str | INFO | consumed by scripts | Partial | Utility-level behavior |
| verbose | bool | true | not central in current core path | Legacy/No-op | Mostly informational |

Practical constraints:
- Keep hr_patch_size / lr_patch_size == downsampling_factor; this is validated.
- For stable current behavior, rely on psf_* keys, not optical_* keys.
- Use downsampling_mode = 2 or 4 only.

## 7. End-to-End Workflows

### 7.1 Workflow A: Compute Global Stats (Recommended First)

Per-folder stats:

```bash
python compute_global_stats.py \
  --input-dir data/input_pan \
  --output configs/pan_stats.yaml \
  --pattern "*.tif"
```

This produces:
- configs/pan_stats.yaml
- histogram plot (.png)
- histogram data (_hist.npz)

Combine multiple histogram files:

```bash
python combine_histograms.py \
  --histograms configs/a_hist.npz configs/b_hist.npz \
  --output configs/combined_stats.yaml
```

Use the resulting p2/p98 YAML as --global_stats input in dataset scripts.

### 7.2 Workflow B: PAN Pre-generated Dataset (Flat v2, ISRO naming)

```bash
python generate_training_dataset_v2.py \
  --input_dir data/input_pan \
  --output_dir data/training_dataset_pan_v2 \
  --config configs/default_config.yaml \
  --global_stats configs/combined_stats.yaml \
  --verify
```

Output naming:
- {basename}_aug1.npz ... {basename}_aug8.npz
- {basename}_aug1_lr1.npz ...

### 7.3 Workflow C: MS Pre-generated Dataset (Flat v2)

```bash
python generate_training_dataset_ms_v2.py \
  --input_dir data/input_ms \
  --output_dir data/training_dataset_ms_v2 \
  --config configs/default_config.yaml \
  --global_stats configs/combined_stats.yaml \
  --verify
```

Optional flags:
- --no_copy_originals
- --file_pattern "*.tif"

### 7.4 Workflow D: PAN Pre-generated Dataset (sample_XXXXXX format)

Use this if you want PreGeneratedDataset compatibility:

```bash
python generate_training_dataset.py \
  --input_dir data/input_pan \
  --output_dir data/training_dataset_pan \
  --config configs/default_config.yaml \
  --global_stats configs/combined_stats.yaml \
  --format npz \
  --verify
```

### 7.5 Workflow E: Small Test Dataset With PNG Preview

PAN:

```bash
python test_dataset_generation.py \
  --input_dir data/input_pan \
  --output_dir data/test_pan \
  --config configs/default_config.yaml \
  --global_stats configs/combined_stats.yaml \
  --num_images 10
```

MS:

```bash
python test_dataset_generation_ms.py \
  --input_dir data/input_ms \
  --output_dir data/test_ms \
  --config configs/default_config.yaml \
  --global_stats configs/combined_stats.yaml \
  --num_images 10 \
  --rgb_bands 0,1,2
```

### 7.6 Workflow F: Single Image Analysis and Debugging

Step-wise pipeline analysis:

```bash
python analyze_degradation.py \
  --image data/input_pan/sample.tif \
  --config configs/default_config.yaml \
  --output analysis_output
```

General test runner (quick visual outputs):

```bash
python test.py --image data/input_pan/sample.tif
# or
python test.py --create-sample
```

Downsampling-mode behavior test:

```bash
python test_downsampling_mode.py
```

### 7.7 Workflow G: Convert NPY to PNG

```bash
python convert_npy_to_png.py \
  --input-dir test_output \
  --output-dir test_output/png \
  --bit-depth 16
```

## 8. On-the-fly Training Usage (No Pre-generation)

### 8.1 PAN DataLoader

```python
from src.dataset import create_dataloader

loader = create_dataloader(
    hr_image_dir="data/input_pan",
    config_path="configs/default_config.yaml",
    global_stats_path="configs/combined_stats.yaml",
    batch_size=8,
    num_workers=4,
    shuffle=True,
    augment=True,
)

batch = next(iter(loader))
print(batch["hr"].shape)          # [B, 1, H, W]
print(len(batch["lr"]))           # 2 or 4
print(batch["lr"][0].shape)       # [B, 1, H', W']
print(batch["shift_values"].shape)
```

### 8.2 MS DataLoader

```python
from src.dataset import create_ms_dataloader

loader = create_ms_dataloader(
    hr_image_dir="data/input_ms",
    config_path="configs/default_config.yaml",
    global_stats_path="configs/combined_stats.yaml",
    batch_size=4,
    num_workers=4,
)

batch = next(iter(loader))
print(batch["hr"].shape)    # [B, C, H, W]
print(batch["lr"][0].shape) # [B, C, H', W']
```

### 8.3 Minimal Training Loop Templates

Use these ready-to-edit templates:
- examples/train_pan_minimal.py
- examples/train_ms_minimal.py

Run:

```bash
python examples/train_pan_minimal.py
python examples/train_ms_minimal.py
```

## 9. Script Reference (How to Run Each)

Main generation and processing:
- generate_training_dataset.py
  - Required: --input_dir --output_dir
  - Optional: --config --global_stats --file_pattern --format [npz|pt] --verify
- generate_training_dataset_v2.py
  - Required: --input_dir --output_dir
  - Optional: --config --global_stats --file_pattern --no_copy_originals --verify
- generate_training_dataset_ms_v2.py
  - Required: --input_dir --output_dir
  - Optional: --config --global_stats --file_pattern --no_copy_originals --verify

Stats and normalization:
- compute_global_stats.py
  - Required: --input-dir
  - Optional: --output --pattern --bins --percentiles --no-histogram --histogram-output --histogram-data
- combine_histograms.py
  - Required: --histograms <many files>
  - Optional: --output --plot-output --percentiles

Inspection/test utilities:
- test_dataset_generation.py
- test_dataset_generation_ms.py
- test.py
- test_downsampling_mode.py
- analyze_degradation.py
- convert_npy_to_png.py

Cluster helper:
- run_combine_histograms.sbatch (Slurm submission helper for combining many histogram NPZ files)

Legacy script:
- scripts/process_images.py exists for older patch-extraction flow (hr/lr1/lr2 patch triplets).

## 10. Training Templates

This repo now includes minimal, runnable starting points:
- examples/train_pan_minimal.py: tiny PAN model + one-epoch training loop
- examples/train_ms_minimal.py: tiny MS model + one-epoch training loop

Both templates show:
- dataloader wiring
- simple model/loss/optimizer
- how to consume multi-frame LR inputs

## 11. Testing

Run tests:

```bash
pytest -v
```

Run MS-specific tests only:

```bash
pytest tests/test_ms_dataset.py -v
```

Note:
- The tests folder currently includes a mix of up-to-date and legacy expectations (some tests assume older operator signatures/keys).
- If you encounter failures, prioritize tests/test_ms_dataset.py and direct script-level validation (test.py, dataset generation scripts) for current behavior.

## 12. Common Issues and Fixes

1. ModuleNotFoundError: torch
- Install PyTorch explicitly (pip install torch torchvision torchaudio) matching your compute target.

2. rasterio install/build issues
- Prefer Conda installation via environment.yml.

3. No files found
- Check --input_dir and --file_pattern.
- Verify extensions (.tif vs .tiff).

4. Config mismatch errors
- Use configs/default_config.yaml as baseline.
- Ensure hr_patch_size / lr_patch_size == downsampling_factor.

5. Memory pressure
- Reduce batch_size, cache_size, or num_workers.
- Prefer pre-generated dataset workflows when training throughput matters.

