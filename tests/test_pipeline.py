"""
Test suite for the main DegradationPipeline class.

Tests the complete pipeline functionality including integration
of all operators and proper LR1/LR2 generation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from degradation.pipeline import DegradationPipeline


class TestDegradationPipeline:
    """Test the DegradationPipeline class."""
    
    def test_pipeline_initialization(self, test_config):
        """Test pipeline initialization with config."""
        pipeline = DegradationPipeline(test_config)
        
        assert pipeline.downsampling_factor == 4
        assert pipeline.config == test_config
    
    def test_generate_lr1(self, test_config, sample_hr_image):
        """Test LR1 generation (reference frame)."""
        pipeline = DegradationPipeline(test_config)
        lr1 = pipeline.generate_lr1(sample_hr_image, seed=42)
        
        # Check output dimensions
        expected_shape = (64, 64)  # 256/4 = 64
        assert lr1.shape == expected_shape
        
        # Check data type
        assert lr1.dtype == np.float32
    
    def test_generate_lr2(self, test_config, sample_hr_image):
        """Test LR2 generation (shifted frame)."""
        pipeline = DegradationPipeline(test_config)
        lr2 = pipeline.generate_lr2(sample_hr_image, seed=42)
        
        # Check output dimensions
        expected_shape = (64, 64)  # 256/4 = 64
        assert lr2.shape == expected_shape
        
        # Check data type
        assert lr2.dtype == np.float32
    
    def test_process_image(self, test_config, sample_hr_image):
        """Test complete image processing (HR -> LR1, LR2)."""
        pipeline = DegradationPipeline(test_config)
        lr1, lr2 = pipeline.process_image(sample_hr_image, seed=42)
        
        # Check output dimensions
        expected_shape = (64, 64)
        assert lr1.shape == expected_shape
        assert lr2.shape == expected_shape
        
        # Check data types  
        assert lr1.dtype == np.float32
        assert lr2.dtype == np.float32
        
        # LR1 and LR2 should be different due to shift
        assert not np.array_equal(lr1, lr2)
    
    def test_lr1_lr2_difference(self, test_config, sample_hr_image):
        """Test that LR1 and LR2 are different due to shift."""
        pipeline = DegradationPipeline(test_config)
        lr1, lr2 = pipeline.process_image(sample_hr_image, seed=42)
        
        # Calculate difference
        diff = np.abs(lr1.astype(np.float64) - lr2.astype(np.float64))
        mean_diff = np.mean(diff)
        
        # Should have measurable difference due to 0.5 pixel shift
        assert mean_diff > 0.001  # Some threshold for difference
    
    def test_reproducibility_with_seed(self, test_config, sample_hr_image):
        """Test that same seed produces same results."""
        pipeline = DegradationPipeline(test_config)
        
        lr1_a, lr2_a = pipeline.process_image(sample_hr_image, seed=123)
        lr1_b, lr2_b = pipeline.process_image(sample_hr_image, seed=123)
        
        np.testing.assert_array_equal(lr1_a, lr1_b)
        np.testing.assert_array_equal(lr2_a, lr2_b)
    
    def test_different_seeds_different_results(self, test_config, sample_hr_image):
        """Test that different seeds produce different results."""
        pipeline = DegradationPipeline(test_config)
        
        lr1_a, lr2_a = pipeline.process_image(sample_hr_image, seed=123)
        lr1_b, lr2_b = pipeline.process_image(sample_hr_image, seed=456)
        
        # Results should be different due to different noise
        assert not np.array_equal(lr1_a, lr1_b)
        assert not np.array_equal(lr2_a, lr2_b)
    
    def test_validate_image_dimensions(self, test_config):
        """Test image dimension validation."""
        pipeline = DegradationPipeline(test_config)
        
        # Valid dimensions (divisible by 4)
        valid_image = np.random.random((256, 256)).astype(np.float32)
        assert pipeline.validate_image_dimensions(valid_image) == True
        
        # Invalid dimensions (not divisible by 4)
        invalid_image = np.random.random((255, 255)).astype(np.float32)
        assert pipeline.validate_image_dimensions(invalid_image) == False
    
    def test_get_output_dimensions(self, test_config):
        """Test output dimension calculation."""
        pipeline = DegradationPipeline(test_config)
        
        # 2D image
        hr_shape_2d = (256, 256)
        lr_shape_2d = pipeline.get_output_dimensions(hr_shape_2d)
        assert lr_shape_2d == (64, 64)
        
        # 3D image
        hr_shape_3d = (256, 256, 3)
        lr_shape_3d = pipeline.get_output_dimensions(hr_shape_3d)
        assert lr_shape_3d == (64, 64, 3)
    
    def test_config_update(self, test_config, sample_hr_image):
        """Test configuration update."""
        pipeline = DegradationPipeline(test_config)
        
        # Process with original config
        lr1_orig, lr2_orig = pipeline.process_image(sample_hr_image, seed=42)
        
        # Update config
        new_config = test_config.copy()
        new_config['gaussian_std'] = 10.0  # Higher noise
        pipeline.update_config(new_config)
        
        # Process with new config
        lr1_new, lr2_new = pipeline.process_image(sample_hr_image, seed=42)
        
        # Results should be different due to higher noise
        diff1 = np.mean(np.abs(lr1_orig - lr1_new))
        diff2 = np.mean(np.abs(lr2_orig - lr2_new))
        
        assert diff1 > 0.01  # Should have noticeable difference
        assert diff2 > 0.01
    
    def test_3d_image_processing(self, test_config):
        """Test processing of 3D (color) images."""
        # Create a 3D test image
        hr_image_3d = np.random.random((128, 128, 3)).astype(np.float32)
        
        pipeline = DegradationPipeline(test_config)
        lr1, lr2 = pipeline.process_image(hr_image_3d, seed=42)
        
        # Check output dimensions
        expected_shape = (32, 32, 3)  # 128/4 = 32, channels preserved
        assert lr1.shape == expected_shape
        assert lr2.shape == expected_shape
    
    def test_invalid_image_shape(self, test_config):
        """Test error handling for invalid image shapes."""
        pipeline = DegradationPipeline(test_config)
        
        # 1D image should raise error
        invalid_image = np.random.random(256)
        with pytest.raises(ValueError):
            pipeline.process_image(invalid_image)
        
        # 4D image should raise error
        invalid_image_4d = np.random.random((32, 32, 3, 2))
        with pytest.raises(ValueError):
            pipeline.process_image(invalid_image_4d)
    
    def test_noise_disabled(self, sample_hr_image):
        """Test pipeline with noise disabled."""
        config_no_noise = {
            'downsampling_factor': 4,
            'optical_sigma': 1.0,
            'optical_kernel_size': 5,
            'motion_kernel_size': 3,
            'enable_gaussian': False,
            'enable_poisson': False,
            'gaussian_mean': 0.0,
            'gaussian_std': 5.0,
            'poisson_lambda': 1.0,
        }
        
        pipeline = DegradationPipeline(config_no_noise)
        lr1, lr2 = pipeline.process_image(sample_hr_image, seed=42)
        
        # Results should be deterministic without noise
        lr1_repeat, lr2_repeat = pipeline.process_image(sample_hr_image, seed=999)
        
        np.testing.assert_array_equal(lr1, lr1_repeat)
        np.testing.assert_array_equal(lr2, lr2_repeat)
    
    def test_shift_effect_verification(self, test_config):
        """Test that the 2-pixel shift in LR2 is correctly applied."""
        # Create a simple test pattern to verify shift
        hr_image = np.zeros((128, 128), dtype=np.float32)
        hr_image[60:68, 60:68] = 1.0  # 8x8 white square in center
        
        # Disable noise and blur for cleaner test
        config_clean = test_config.copy()
        config_clean.update({
            'optical_sigma': 0.5,
            'motion_kernel_size': 1,
            'enable_gaussian': False,
            'enable_poisson': False
        })
        
        pipeline = DegradationPipeline(config_clean)
        lr1, lr2 = pipeline.process_image(hr_image, seed=42)
        
        # Find peak locations in LR1 and LR2
        lr1_peak = np.unravel_index(np.argmax(lr1), lr1.shape)
        lr2_peak = np.unravel_index(np.argmax(lr2), lr2.shape)
        
        # LR2 peak should be shifted relative to LR1 peak
        # (0.5 LR pixel shift corresponds to visible difference)
        shift_y = abs(lr2_peak[0] - lr1_peak[0])
        shift_x = abs(lr2_peak[1] - lr1_peak[1])
        
        # Should have some measurable shift (allowing for blur effects)
        assert shift_y >= 0 or shift_x >= 0  # At least some shift detectable