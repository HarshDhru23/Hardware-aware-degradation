# Hardware-aware Degradation Pipeline

A comprehensive degradation pipeline for generating synthetic training data for satellite Multi-Frame Super-Resolution (MFSR). This pipeline simulates hardware characteristics of satellite sensors to create realistic low-resolution image frames from high-resolution ground truth data.

## Project Overview

### Primary Goal
Generate synthetic training data for deep learning-based 4× super-resolution from multiple low-resolution (LR) input frames.

### Hardware Context  
We simulate data from satellite sensors with multiple panchromatic (PAN) imagers:
- **Mode 2**: 2 LR frames with shifts [(0,0), (0.5,0.5)]
- **Mode 4**: 4 LR frames with shifts [(0,0), (0.25,0.25), (0.5,0.5), (0.75,0.75)]
- **WorldView-3 data**: From SpaceNet 2 dataset (AOI_3_Paris/PAN/)

### Key Features
- **PyTorch Dataset Integration**: `DegradationDataset` class for on-the-fly degradation during training
- **Multi-frame LR Generation**: 2 or 4 frames based on downsampling mode
- **Anisotropic Gaussian PSF**: Realistic sensor blur modeling
- **Stochastic Sub-pixel Shifts**: Simulates sensor jitter with configurable variance
- **Poisson-Gaussian Noise**: Combined photon shot noise and read noise
- **Global Percentile Normalization**: Consistent normalization across datasets
- **8-way Data Augmentation**: 4 rotations × 2 flips

## Theoretical Model

The pipeline implements the **Observation Model**:

$$y_k = D B_k M_k x + n_k$$

Where:
- **x**: High-resolution ground truth image
- **y_k**: k-th observed LR image
- **M_k**: Warping operator (sub-pixel shift)
- **B_k**: Blur operator (anisotropic Gaussian PSF)
- **D**: Downsampling operator (spatial integration)
- **n_k**: Noise (Poisson shot noise + Gaussian read noise + ADC quantization)

## Repository Structure

```
Hardware-aware-degradation/
├── src/
│   ├── __init__.py
│   ├── config.py                    # ConfigManager for YAML configuration
│   ├── dataset.py                   # DegradationDataset (PyTorch Dataset)
│   ├── degradation/
│   │   ├── __init__.py
│   │   ├── pipeline.py              # DegradationPipeline (main pipeline class)
│   │   └── operators.py             # WarpingOperator, BlurOperator, 
│   │                                # DownsamplingOperator, NoiseOperator
│   └── utils/
│       ├── __init__.py
│       ├── bicubic_core.py          # Bicubic interpolation with antialiasing
│       ├── data_io.py               # GeoTIFFLoader, PatchExtractor
│       ├── validation.py            # Input/config validation utilities
│       └── visualization.py         # Visualization tools
│
├── configs/
│   ├── default_config.yaml          # Default parameters (stochastic, 4-frame)
│   ├── high_quality_config.yaml     # High-quality processing
│   ├── fast_config.yaml             # Fast processing for testing
│   └── debug_config.yaml            # Debug configuration
│
├── scripts/
│   └── process_images.py            # Batch processing script
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Test fixtures
│   ├── test_operators.py            # Operator unit tests
│   └── test_pipeline.py             # Pipeline integration tests
│
├── data/
│   ├── input/                       # Place HR GeoTIFF images here
│   └── output/                      # Generated outputs
│
├── test_output/                     # Sample outputs for verification
│   ├── hr_image.npy
│   ├── lr1_image.npy
│   └── lr2_image.npy
│
├── analyze_degradation.py           # Step-by-step pipeline visualization
├── compute_global_stats.py          # Compute dataset-wide percentile stats
├── combine_histograms.py            # Combine histogram stats from multiple dirs
├── convert_npy_to_png.py            # Convert NPY outputs to PNG
├── test.py                          # Quick test script
├── test_downsampling_mode.py        # Test 2-frame vs 4-frame modes
│
├── requirements.txt                 # Python dependencies (pip)
├── environment.yml                  # Conda environment
├── setup.py                         # Package installation
└── README.md                        # This file
```

## Quick Start

### 1. Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/HarshDhru23/Hardware-aware-degradation.git
cd Hardware-aware-degradation

# Create conda environment
conda env create -f environment.yml
conda activate hardware-degradation
```

#### Option B: Using pip
```bash
# Clone the repository
git clone https://github.com/HarshDhru23/Hardware-aware-degradation.git
cd Hardware-aware-degradation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option C: Install as Package
```bash
pip install -e .
```

### 2. Compute Global Statistics (Recommended)

For consistent normalization across your dataset:
```bash
python compute_global_stats.py \
    --input-dir /path/to/your/images \
    --output configs/global_stats.yaml
```

### 3. Data Preparation

Place your HR GeoTIFF files in `data/input/`:
```bash
data/input/
├── image_001.tif
├── image_002.tif
└── ...
```

### 4. Configuration

Edit `configs/default_config.yaml` or create your own. Key parameters:
- `downsampling_mode`: 2 or 4 (number of LR frames)
- `shift_mode`: 'deterministic' or 'stochastic'
- `psf_sigma_x`, `psf_sigma_y`: PSF blur parameters
- `photon_gain`: Sensor gain for Poisson noise

### 5. Run Processing

```bash
python scripts/process_images.py \
    --input_dir data/input \
    --output_dir data/output \
    --config configs/default_config.yaml
```

## Usage Examples

### Using DegradationDataset (Recommended for Training)

The `DegradationDataset` class is a PyTorch Dataset that generates degraded LR images on-the-fly during training:

```python
import torch
from torch.utils.data import DataLoader
from src.dataset import DegradationDataset, create_dataloader, collate_fn

# Create dataset
dataset = DegradationDataset(
    hr_image_dir='data/input',
    config_path='configs/default_config.yaml',
    global_stats_path='configs/global_stats.yaml',  # Optional
    augment=True,                                    # 8-way augmentation
    cache_size=100,                                  # LRU cache for HR images
    file_pattern='*.tif',
    seed=42
)

print(f"Dataset size: {len(dataset)}")

# Get a single sample
sample = dataset[0]
print(f"HR shape: {sample['hr'].shape}")        # [1, H, W]
print(f"LR frames: {len(sample['lr'])}")        # 2 or 4 frames
print(f"LR[0] shape: {sample['lr'][0].shape}")  # [1, H/4, W/4]
print(f"PSF sigma_x: {sample['psf_params']['sigma_x']}")
print(f"Shift values: {sample['shift_values']}")

# Create DataLoader using helper function
dataloader = create_dataloader(
    hr_image_dir='data/input',
    config_path='configs/default_config.yaml',
    global_stats_path='configs/global_stats.yaml',
    batch_size=8,
    num_workers=4,
    shuffle=True,
    augment=True
)

# Training loop
for batch in dataloader:
    hr = batch['hr']                    # [B, 1, H, W]
    lr_frames = batch['lr']             # List of [B, 1, H', W']
    psf_kernels = batch['psf_kernels']  # List of [B, Kh, Kw]
    shift_values = batch['shift_values'] # [B, num_frames, 2]
    
    # Your training code here
    break
```

### Dataset Output Structure

Each sample from `DegradationDataset` returns:
```python
{
    'hr': torch.Tensor,           # [1, H, W] - HR image
    'lr': List[torch.Tensor],     # List of [1, H', W'] - LR frames
    'psf_kernels': List[Tensor],  # List of [Kh, Kw] - PSF kernels
    'psf_params': {
        'sigma_x': List[float],   # PSF sigma_x per frame
        'sigma_y': List[float],   # PSF sigma_y per frame
        'theta': List[float]      # PSF rotation per frame
    },
    'shift_values': List,         # [[dx, dy], ...] per frame
    'metadata': {
        'filename': str,
        'file_idx': int,
        'aug_idx': int,
        'rotation': int,          # 0, 90, 180, or 270 degrees
        'flip': bool,
        'num_lr_frames': int,
        'downsampling_factor': int,
        'hr_shape': tuple,
        'lr_shape': tuple
    }
}
```

### Using DegradationPipeline Directly

For batch processing or custom workflows:

```python
from src.degradation.pipeline import DegradationPipeline
from src.config import ConfigManager
from src.utils.data_io import GeoTIFFLoader

# Load configuration
config = ConfigManager('configs/default_config.yaml')

# Initialize pipeline
pipeline = DegradationPipeline(config.get_all())

# Load HR image
loader = GeoTIFFLoader(normalize=True)
hr_image = loader.load_image('data/input/sample.tif')

# Generate LR frames (2 or 4 based on config)
lr_frames = pipeline.process_image(hr_image, seed=42)

print(f"HR shape: {hr_image.shape}")
print(f"Number of LR frames: {len(lr_frames)}")
print(f"LR frame shape: {lr_frames[0].shape}")
print(f"Shift values: {pipeline.shift_values}")
```

### Analyzing the Degradation Pipeline

Visualize each step of the pipeline:

```bash
python analyze_degradation.py --image data/input/sample.tif
```

This creates visualizations showing:
1. Original HR image
2. After warping (geometric shift)
3. After blur (PSF convolution)
4. After downsampling (spatial integration)
5. After noise (Poisson + Gaussian + ADC)

## Configuration Parameters

### Core Pipeline Parameters
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `downsampling_factor` | 2-8 | 4 | Super-resolution factor |
| `downsampling_mode` | 2, 4 | 4 | Number of LR frames to generate |
| `shift_mode` | deterministic/stochastic | stochastic | Shift sampling method |
| `shift_variance_2x` | 0.01-0.15 | 0.08 | Shift variance for 2-frame mode |
| `shift_variance_4x` | 0.01-0.1 | 0.03 | Shift variance for 4-frame mode |

### PSF Blur Parameters (Anisotropic Gaussian)
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `psf_sigma_x` | 0.5-2.0 | 0.6 | Horizontal PSF sigma |
| `psf_sigma_y` | 0.5-2.0 | 0.8 | Vertical PSF sigma |
| `psf_theta` | 0-180 | 0.0 | PSF rotation angle (degrees) |
| `psf_kernel_size` | 3-15 (odd) | 9 | PSF kernel size |

### Noise Parameters
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `enable_gaussian` | true/false | true | Enable Gaussian read noise |
| `enable_poisson` | true/false | true | Enable Poisson shot noise |
| `gaussian_std` | 0.0001-0.1 | 0.0005 | Gaussian noise std dev |
| `photon_gain` | 20000-40000 | 30000 | Photon gain for Poisson noise |
| `enable_quantization` | true/false | true | Enable ADC quantization |
| `quantization_bits` | 8-16 | 11 | Bit depth (11-bit for WV-3) |

### Patch Parameters
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `hr_patch_size` | 64-1024 | 256 | HR patch size (pixels) |
| `lr_patch_size` | 16-256 | 64 | LR patch size (auto: hr/factor) |
| `min_valid_pixels` | 0.5-1.0 | 0.95 | Min valid pixel fraction |

## Testing

Run the test suite to verify installation:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py -v

# Quick functionality test
python test.py

# Test downsampling modes (2-frame vs 4-frame)
python test_downsampling_mode.py
```

## Output Format

### When Using DegradationDataset

The dataset returns PyTorch tensors directly, ready for training:
- HR image: `[1, H, W]` tensor
- LR frames: List of `[1, H/4, W/4]` tensors
- PSF kernels: List of `[Kh, Kw]` tensors
- Metadata: Dictionary with augmentation info

### When Using Batch Processing Script

```
data/output/
├── image_001/
│   ├── image_001_000000_hr.npy    # HR patch (256×256)
│   ├── image_001_000000_lr0.npy   # LR frame 0 (64×64)
│   ├── image_001_000000_lr1.npy   # LR frame 1 (64×64)
│   ├── image_001_000000_lr2.npy   # LR frame 2 (if mode=4)
│   ├── image_001_000000_lr3.npy   # LR frame 3 (if mode=4)
│   └── ...
├── visualizations/
│   └── image_001_degradation.png
└── processing_summary.txt
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   # Or with conda
   conda env update -f environment.yml
   ```

2. **GeoTIFF Loading Issues**
   ```bash
   # Install rasterio for better GeoTIFF support
   conda install -c conda-forge rasterio
   ```

3. **CUDA/GPU Issues**
   The pipeline uses NumPy/OpenCV (CPU). PyTorch is only needed for `DegradationDataset`.

4. **Memory Issues**
   - Reduce `cache_size` in DegradationDataset
   - Use smaller `batch_size` in DataLoader
   - Process images in chunks

## Dependencies

Core dependencies:
- Python ≥ 3.8
- NumPy, SciPy, OpenCV
- PyTorch (for DegradationDataset)
- rasterio or Pillow (for GeoTIFF support)
- PyYAML (for configuration)
- matplotlib (for visualization)

See [requirements.txt](requirements.txt) for full list.

## License

MIT License

## Citation

If you use this pipeline in your research, please cite:
```bibtex
@software{hardware_aware_degradation,
  title={Hardware-aware Degradation Pipeline for Satellite MFSR},
  author={ISRO MFSR Team},
  year={2025},
  url={https://github.com/HarshDhru23/Hardware-aware-degradation}
}
```
