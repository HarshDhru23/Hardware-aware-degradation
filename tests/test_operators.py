"""
Test suite for the degradation pipeline operators.

Tests each operator individually: WarpingOperator, BlurOperator, 
DownsamplingOperator, and NoiseOperator.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from degradation.operators import WarpingOperator, BlurOperator, DownsamplingOperator, NoiseOperator


class TestWarpingOperator:
    """Test the WarpingOperator class."""
    
    def test_identity_warp(self, sample_hr_image):
        """Test identity warping (no shift)."""
        operator = WarpingOperator(shift_x=0.0, shift_y=0.0)
        result = operator.apply(sample_hr_image)
        
        # Should be identical to input
        np.testing.assert_array_equal(result, sample_hr_image)
    
    def test_shift_warp(self, sample_hr_image):
        """Test warping with shift."""
        operator = WarpingOperator(shift_x=2.0, shift_y=2.0)
        result = operator.apply(sample_hr_image)
        
        # Result should have same shape but different content
        assert result.shape == sample_hr_image.shape
        assert not np.array_equal(result, sample_hr_image)
    
    def test_shift_clipping(self):
        """Test that shifts are clipped to valid range."""
        operator = WarpingOperator(shift_x=10.0, shift_y=-5.0)
        
        # Should be clipped to [0, 4] range
        assert operator.shift_x == 4.0
        assert operator.shift_y == 0.0
    
    def test_3d_image(self):
        """Test warping with 3D image."""
        image_3d = np.random.random((64, 64, 3)).astype(np.float32)
        operator = WarpingOperator(shift_x=1.0, shift_y=1.0)
        result = operator.apply(image_3d)
        
        assert result.shape == image_3d.shape
        assert len(result.shape) == 3


class TestBlurOperator:
    """Test the BlurOperator class."""
    
    def test_optical_blur_only(self, sample_hr_image):
        """Test optical blur only."""
        operator = BlurOperator(
            optical_sigma=1.0,
            optical_kernel_size=5, 
            motion_kernel_size=1  # No motion blur
        )
        result = operator.apply(sample_hr_image)
        
        assert result.shape == sample_hr_image.shape
        # Blurred image should be smoother (lower high-frequency content)
        # Simple test: check that max gradient is reduced
        orig_grad = np.max(np.abs(np.gradient(sample_hr_image)))
        blur_grad = np.max(np.abs(np.gradient(result)))
        assert blur_grad < orig_grad
    
    def test_motion_blur(self, sample_hr_image):
        """Test motion blur."""
        operator = BlurOperator(
            optical_sigma=0.5,  # Minimal optical blur
            optical_kernel_size=3,
            motion_kernel_size=5
        )
        result = operator.apply(sample_hr_image)
        
        assert result.shape == sample_hr_image.shape
        assert not np.array_equal(result, sample_hr_image)
    
    def test_parameter_clipping(self):
        """Test parameter clipping to valid ranges."""
        operator = BlurOperator(
            optical_sigma=10.0,     # Should be clipped to 3.0
            optical_kernel_size=2,  # Should be made odd (3)
            motion_kernel_size=20   # Should be clipped and made odd
        )
        
        assert operator.optical_sigma == 3.0
        assert operator.optical_kernel_size == 3
        assert operator.motion_kernel_size == 9  # Max clipped to 9
    
    def test_3d_image_blur(self):
        """Test blurring with 3D image."""
        image_3d = np.random.random((64, 64, 3)).astype(np.float32)
        operator = BlurOperator(optical_sigma=1.0, motion_kernel_size=3)
        result = operator.apply(image_3d)
        
        assert result.shape == image_3d.shape


class TestDownsamplingOperator:
    """Test the DownsamplingOperator class."""
    
    def test_4x_downsampling(self, sample_hr_image):
        """Test 4x downsampling."""
        operator = DownsamplingOperator(downsampling_factor=4)
        result = operator.apply(sample_hr_image)
        
        expected_shape = (64, 64)  # 256/4 = 64
        assert result.shape == expected_shape
    
    def test_2x_downsampling(self):
        """Test 2x downsampling."""
        image = np.random.random((128, 128)).astype(np.float32)
        operator = DownsamplingOperator(downsampling_factor=2)
        result = operator.apply(image)
        
        expected_shape = (64, 64)  # 128/2 = 64
        assert result.shape == expected_shape
    
    def test_factor_clipping(self):
        """Test downsampling factor clipping."""
        operator = DownsamplingOperator(downsampling_factor=10)
        assert operator.factor == 8  # Should be clipped to max 8
        
        operator = DownsamplingOperator(downsampling_factor=1)
        assert operator.factor == 2  # Should be clipped to min 2
    
    def test_3d_image_downsampling(self):
        """Test downsampling with 3D image."""
        image_3d = np.random.random((128, 128, 3)).astype(np.float32)
        operator = DownsamplingOperator(downsampling_factor=4)
        result = operator.apply(image_3d)
        
        expected_shape = (32, 32, 3)  # 128/4 = 32, channels preserved
        assert result.shape == expected_shape
    
    def test_average_pooling_property(self):
        """Test that downsampling actually performs average pooling."""
        # Create a simple pattern where we can verify averaging
        image = np.ones((4, 4), dtype=np.float32)
        image[0:2, 0:2] = 0.0  # Top-left quadrant = 0
        image[0:2, 2:4] = 1.0  # Top-right quadrant = 1
        image[2:4, 0:2] = 2.0  # Bottom-left quadrant = 2
        image[2:4, 2:4] = 3.0  # Bottom-right quadrant = 3
        
        operator = DownsamplingOperator(downsampling_factor=2)
        result = operator.apply(image)
        
        # Result should be 2x2 with averages of each quadrant
        expected = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


class TestNoiseOperator:
    """Test the NoiseOperator class."""
    
    def test_gaussian_noise_only(self, sample_lr_image):
        """Test Gaussian noise only."""
        operator = NoiseOperator(
            gaussian_mean=0.0,
            gaussian_std=0.1,
            enable_gaussian=True,
            enable_poisson=False
        )
        result = operator.apply(sample_lr_image, seed=42)
        
        assert result.shape == sample_lr_image.shape
        assert result.dtype == sample_lr_image.dtype
        # Should be different due to noise
        assert not np.array_equal(result, sample_lr_image)
    
    def test_poisson_noise_only(self, sample_lr_image):
        """Test Poisson noise only."""
        operator = NoiseOperator(
            poisson_lambda=1.0,
            enable_gaussian=False,
            enable_poisson=True
        )
        result = operator.apply(sample_lr_image, seed=42)
        
        assert result.shape == sample_lr_image.shape
        assert result.dtype == sample_lr_image.dtype
    
    def test_both_noise_types(self, sample_lr_image):
        """Test both Gaussian and Poisson noise."""
        operator = NoiseOperator(
            gaussian_std=0.05,
            poisson_lambda=1.0,
            enable_gaussian=True,
            enable_poisson=True
        )
        result = operator.apply(sample_lr_image, seed=42)
        
        assert result.shape == sample_lr_image.shape
        assert not np.array_equal(result, sample_lr_image)
    
    def test_no_noise(self, sample_lr_image):
        """Test with noise disabled."""
        operator = NoiseOperator(
            enable_gaussian=False,
            enable_poisson=False
        )
        result = operator.apply(sample_lr_image, seed=42)
        
        # Should be identical to input when no noise is applied
        np.testing.assert_array_equal(result, sample_lr_image)
    
    def test_parameter_clipping(self):
        """Test parameter clipping to valid ranges."""
        operator = NoiseOperator(
            gaussian_mean=100.0,    # Should be clipped to 10.0
            gaussian_std=100.0,     # Should be clipped to 20.0
            poisson_lambda=100.0    # Should be clipped to 5.0
        )
        
        assert operator.gaussian_mean == 10.0
        assert operator.gaussian_std == 20.0
        assert operator.poisson_lambda == 5.0
    
    def test_seed_reproducibility(self, sample_lr_image):
        """Test that same seed produces same noise."""
        operator = NoiseOperator(gaussian_std=0.1, enable_poisson=False)
        
        result1 = operator.apply(sample_lr_image, seed=123)
        result2 = operator.apply(sample_lr_image, seed=123)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_different_seeds(self, sample_lr_image):
        """Test that different seeds produce different noise."""
        operator = NoiseOperator(gaussian_std=0.1, enable_poisson=False)
        
        result1 = operator.apply(sample_lr_image, seed=123)
        result2 = operator.apply(sample_lr_image, seed=456)
        
        assert not np.array_equal(result1, result2)
    
    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        # Test with different input dtypes
        dtypes = [np.uint8, np.uint16, np.float32, np.float64]
        
        for dtype in dtypes:
            image = (np.random.random((32, 32)) * 100).astype(dtype)
            operator = NoiseOperator(gaussian_std=1.0)
            result = operator.apply(image, seed=42)
            
            assert result.dtype == dtype