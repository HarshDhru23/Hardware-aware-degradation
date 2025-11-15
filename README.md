# Hardware-aware Degradation Pipeline

## ISRO Multi-Frame Super-Resolution (MFSR) Project - Contribution 1

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

A comprehensive degradation pipeline that implements the observation model for generating synthetic training data for satellite super-resolution. This pipeline simulates the hardware characteristics of satellite sensors to create realistic low-resolution image pairs from high-resolution ground truth data.

## Project Overview

### Primary Goal
Build a deep learning model (Contribution-2) that can create a 4x super-resolution image from two low-resolution (LR) input frames.

### Hardware Context  
We simulate data from a specific satellite sensor with two panchromatic (PAN) imagers:
- **P1 and P2**: Physically offset by (0.5, 0.5) pixels (diagonal shift)
- **WorldView-3 data**: From SpaceNet 2 dataset (AOI_3_Paris/PAN/)

### This Repository's Goal (Contribution-1)
Create synthetic training data by implementing the 'Observation Model' from super-resolution research. The pipeline takes one High-Resolution (HR) 'ground truth' image and generates two degraded, low-resolution images (LR1 and LR2) that simulate the P1 and P2 sensors.

## Theoretical Model

The pipeline implements the following **Observation Model**:

$$y_k = D B_k M_k x + n_k$$

Where:
- **x**: The "Desired HR Image" (ground truth)  
- **y_k**: The k-th observed LR image (we create y_1 and y_2)
- **M_k**: The "Warping" operator (0.5-pixel shift simulation)
- **B_k**: The "Blur" operator (Optical + Motion blur)
- **D**: The "Downsampling" operator (includes Sensor PSF blur)
- **n_k**: Additive noise (Gaussian + Poisson)

## Pipeline Implementation

### A. Generate y_1 (LR1 - Reference Frame)
1. **Warping (M_1)**: Identity operation (no shift)
2. **Blurring (B_1)**: 
   - Optical Blur: 2D Gaussian kernel
   - Motion Blur: 1D vertical kernel (TDI velocity mismatch)
3. **Downsampling & Sensor PSF (D)**: Average pooling (4x4, stride=4)
4. **Noise (n_1)**: Gaussian + Poisson noise

### B. Generate y_2 (LR2 - Shifted Frame)  
1. **Warping (M_2)**: (0.5, 0.5) LR pixel shift = (2, 2) HR pixel shift
2. **Blurring (B_2)**: Same as LR1
3. **Downsampling & Sensor PSF (D)**: Same as LR1  
4. **Noise (n_2)**: Independent Gaussian + Poisson noise

### C. Patch Extraction
- Extract corresponding patches: (HR_patch, LR1_patch, LR2_patch)
- **HR patches**: 256×256 pixels
- **LR patches**: 64×64 pixels (4x downsampling)

## Repository Structure

```
Hardware-aware-degradation/
├── src/
│   ├── degradation/
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Main DegradationPipeline class
│   │   └── operators.py         # Individual operators (M_k, B_k, D, n_k)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_io.py          # GeoTIFF loading and patch extraction
│   │   ├── validation.py       # Input validation utilities
│   │   └── visualization.py    # Result visualization tools
│   └── config.py               # Configuration management
├── configs/
│   ├── default_config.yaml     # Default parameters
│   ├── high_quality_config.yaml # High-quality processing
│   └── fast_config.yaml        # Fast processing for testing
├── scripts/
│   └── process_images.py       # Main batch processing script
├── data/
│   ├── input/                  # Input HR GeoTIFF files
│   └── output/                 # Generated LR patches
├── tests/
│   └── (unit and integration tests)
├── docs/
│   └── (documentation files)
├── requirements.txt            # Python dependencies
├── environment.yml            # Conda environment
├── setup.py                   # Package installation
└── README.md                  # This file
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
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your HR GeoTIFF files in the `data/input/` directory:
```bash
data/input/
├── image_001.tif
├── image_002.tif
└── ...
```

### 3. Configuration

Choose or modify a configuration file:
- `configs/default_config.yaml` - Balanced processing
- `configs/high_quality_config.yaml` - High-quality, slower processing  
- `configs/fast_config.yaml` - Fast processing for testing

### 4. Run Processing

```bash
# Basic usage
python scripts/process_images.py \\
    --input_dir data/input \\
    --output_dir data/output \\
    --config configs/default_config.yaml

# Advanced usage with options
python scripts/process_images.py \\
    --input_dir data/input \\
    --output_dir data/output \\
    --config configs/high_quality_config.yaml \\
    --pattern "*.tif" \\
    --max_images 100 \\
    --log_file processing.log
```

## Configuration Parameters

### Core Pipeline Parameters
| Parameter | Range | Default | Description |
|-----------|--------|---------|-------------|
| `downsampling_factor` | 2-8 | 4 | Super-resolution factor |
| `optical_sigma` | 0.5-3.0 | 1.0 | Gaussian blur standard deviation |
| `optical_kernel_size` | 3-15 (odd) | 5 | Gaussian kernel size |
| `motion_kernel_size` | 1-9 (odd) | 3 | Motion blur kernel size |

### Noise Parameters
| Parameter | Range | Default | Description |
|-----------|--------|---------|-------------|
| `gaussian_mean` | -10.0 to 10.0 | 0.0 | Gaussian noise mean |
| `gaussian_std` | 0.001-20.0 | 0.01 | Gaussian noise std (0.01-0.03 for normalized, 3-10 for [0,255]) |
| `poisson_lambda` | 0.1-5.0 | 1.0 | Poisson noise scaling factor |
| `enable_gaussian` | true/false | true | Enable Gaussian noise |
| `enable_poisson` | true/false | false | Enable Poisson noise (recommended: false for normalized) |

### Patch Extraction Parameters  
| Parameter | Range | Default | Description |
|-----------|--------|---------|-------------|
| `hr_patch_size` | 64-1024 | 256 | HR patch size (pixels) |
| `lr_patch_size` | 16-256 | 64 | LR patch size (pixels) |
| `patch_stride` | - | 256 | Patch extraction stride |
| `min_valid_pixels` | 0.5-1.0 | 0.95 | Min fraction of valid pixels |

## Usage Examples

### Basic Python API Usage

```python
from degradation import DegradationPipeline
from config import ConfigManager
from utils.data_io import GeoTIFFLoader

# Load configuration
config = ConfigManager('configs/default_config.yaml')

# Initialize pipeline
pipeline = DegradationPipeline(config.get_all())

# Load and process an image
loader = GeoTIFFLoader()
hr_image = loader.load_image('data/input/sample.tif')

# Generate LR image pair
lr1, lr2 = pipeline.process_image(hr_image, seed=42)

print(f"HR shape: {hr_image.shape}")
print(f"LR1 shape: {lr1.shape}")  
print(f"LR2 shape: {lr2.shape}")
```

### Custom Configuration

```python
# Create custom configuration
custom_config = {
    'downsampling_factor': 4,
    'optical_sigma': 1.5,
    'gaussian_std': 8.0,
    'hr_patch_size': 512,
    'lr_patch_size': 128
}

pipeline = DegradationPipeline(custom_config)
```

### Batch Processing with Progress Tracking

```python
from tqdm import tqdm
from utils.data_io import GeoTIFFLoader, PatchExtractor

loader = GeoTIFFLoader()
extractor = PatchExtractor(hr_patch_size=256, lr_patch_size=64)

# Process multiple images
image_files = loader.find_geotiff_files('data/input')

for image_path in tqdm(image_files):
    hr_image = loader.load_image(image_path)
    lr1, lr2 = pipeline.process_image(hr_image)
    
    # Extract patches
    patches = extractor.extract_patches(hr_image, lr1, lr2)
    
    # Save patches
    # ... (save logic)
```

## Testing

Run the test suite to verify installation:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py -v
```

## Output Format

The pipeline generates the following outputs:

### Patch Files (NumPy format)
```
data/output/
├── image_001/
│   ├── image_001_000000_hr.npy    # HR patch (256x256)
│   ├── image_001_000000_lr1.npy   # LR1 patch (64x64)
│   ├── image_001_000000_lr2.npy   # LR2 patch (64x64)
│   └── ...
├── visualizations/
│   ├── image_001_degradation.png  # Visualization
│   └── ...
└── processing_summary.txt         # Processing statistics
```

### Patch Loading Example
```python
import numpy as np

# Load patches
hr_patch = np.load('data/output/image_001/image_001_000000_hr.npy')
lr1_patch = np.load('data/output/image_001/image_001_000000_lr1.npy')  
lr2_patch = np.load('data/output/image_001/image_001_000000_lr2.npy')

# Verify spatial correspondence
assert hr_patch.shape == (256, 256)
assert lr1_patch.shape == (64, 64)
assert lr2_patch.shape == (64, 64)
```

## Validation and Quality Control

The pipeline includes comprehensive validation:

### Image Validation
- Dimension compatibility with downsampling factor
- Data type and value range checks  
- NaN/infinity detection
- Minimum valid pixel requirements

### Configuration Validation
- Parameter range verification
- Compatibility checks between related parameters
- Required parameter presence

### Processing Validation
- Patch extraction quality assessment
- SNR analysis for noise validation
- Edge preservation metrics for blur validation

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **GeoTIFF Loading Issues**
   ```bash
   # Install GDAL for better GeoTIFF support
   conda install gdal
   ```

3. **Memory Issues with Large Images**
   ```yaml
   # Use smaller patch sizes in config
   hr_patch_size: 128
   lr_patch_size: 32
   ```

4. **Dimension Compatibility Errors**
   - Ensure HR image dimensions are divisible by downsampling factor
   - The pipeline automatically crops images if needed

### Debug Mode
```bash
# Run with debug logging
python scripts/process_images.py \\
    --input_dir data/input \\
    --output_dir data/output \\
    --config configs/default_config.yaml \\
    --log_file debug.log
```
