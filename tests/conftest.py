"""
Test configuration for Hardware-aware Degradation Pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_hr_image():
    """Create a sample HR image for testing."""
    # Create a 256x256 test image with some patterns
    image = np.zeros((256, 256), dtype=np.float32)
    
    # Add some gradient patterns
    y, x = np.ogrid[:256, :256]
    image += 0.3 * np.sin(2 * np.pi * x / 64) * np.cos(2 * np.pi * y / 64)
    image += 0.2 * np.sin(2 * np.pi * x / 32) 
    image += 0.1 * np.random.random((256, 256))
    
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    
    return image


@pytest.fixture
def sample_lr_image():
    """Create a sample LR image for testing."""
    # Create a 64x64 test image
    image = np.random.random((64, 64)).astype(np.float32)
    return image


@pytest.fixture
def test_config():
    """Create a test configuration dictionary."""
    return {
        'downsampling_factor': 4,
        'optical_sigma': 1.0,
        'optical_kernel_size': 5,
        'motion_kernel_size': 3,
        'enable_gaussian': True,
        'enable_poisson': True,
        'gaussian_mean': 0.0,
        'gaussian_std': 5.0,
        'poisson_lambda': 1.0,
        'normalize': True,
        'target_dtype': 'float32',
        'hr_patch_size': 128,
        'lr_patch_size': 32,
        'patch_stride': 128,
        'min_valid_pixels': 0.9
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config_file(temp_dir):
    """Create a sample YAML config file."""
    config_content = """
downsampling_factor: 4
optical_sigma: 1.0
optical_kernel_size: 5
motion_kernel_size: 3
enable_gaussian: true
enable_poisson: true
gaussian_mean: 0.0
gaussian_std: 5.0
poisson_lambda: 1.0
normalize: true
target_dtype: "float32"
hr_patch_size: 128
lr_patch_size: 32
"""
    
    config_file = temp_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    return config_file