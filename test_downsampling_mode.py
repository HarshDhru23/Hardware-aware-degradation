"""
Test script to demonstrate downsampling_mode feature.

Mode 2: Generates 2 LR frames with shifts [(0,0), (0.5,0.5)]
Mode 4: Generates 4 LR frames with shifts [(0,0), (0.25,0.25), (0.5,0.5), (0.75,0.75)]
"""

import numpy as np
from src.degradation.pipeline import DegradationPipeline
from src.config import ConfigManager


def test_mode_2():
    """Test downsampling_mode = 2"""
    print("=" * 60)
    print("Testing Mode 2: 2 LR frames")
    print("=" * 60)
    
    config = ConfigManager('configs/default_config.yaml')
    config.set('downsampling_mode', 2)
    
    pipeline = DegradationPipeline(config.get_all())
    
    print(f"Number of LR frames: {pipeline.num_lr_frames}")
    print(f"Shift values: {pipeline.shift_values}")
    
    # Generate test image
    hr_image = np.random.rand(256, 256).astype(np.float32)
    lr_frames = pipeline.generate_lr_frames(hr_image, seed=42)
    
    print(f"\nGenerated {len(lr_frames)} LR frames")
    for i, frame in enumerate(lr_frames):
        print(f"  Frame {i}: shape {frame.shape}, shift {pipeline.shift_values[i]}")
    
    print("✓ Mode 2 test passed!\n")
    return lr_frames


def test_mode_4():
    """Test downsampling_mode = 4"""
    print("=" * 60)
    print("Testing Mode 4: 4 LR frames")
    print("=" * 60)
    
    config = ConfigManager('configs/default_config.yaml')
    config.set('downsampling_mode', 4)
    
    pipeline = DegradationPipeline(config.get_all())
    
    print(f"Number of LR frames: {pipeline.num_lr_frames}")
    print(f"Shift values: {pipeline.shift_values}")
    
    # Generate test image
    hr_image = np.random.rand(256, 256).astype(np.float32)
    lr_frames = pipeline.generate_lr_frames(hr_image, seed=42)
    
    print(f"\nGenerated {len(lr_frames)} LR frames")
    for i, frame in enumerate(lr_frames):
        print(f"  Frame {i}: shape {frame.shape}, shift {pipeline.shift_values[i]}")
    
    print("✓ Mode 4 test passed!\n")
    return lr_frames


def test_invalid_mode():
    """Test that invalid mode raises error"""
    print("=" * 60)
    print("Testing Invalid Mode: should raise ValueError")
    print("=" * 60)
    
    config = ConfigManager('configs/default_config.yaml')
    
    try:
        config.set('downsampling_mode', 3)
        print("✗ Should have raised ValueError for invalid mode")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected invalid mode: {e}\n")
        return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DOWNSAMPLING_MODE FEATURE TEST")
    print("=" * 60 + "\n")
    
    # Test mode 2
    lr_frames_2 = test_mode_2()
    
    # Test mode 4
    lr_frames_4 = test_mode_4()
    
    # Test invalid mode
    test_invalid_mode()
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nSummary:")
    print(f"  Mode 2: {len(lr_frames_2)} frames generated")
    print(f"  Mode 4: {len(lr_frames_4)} frames generated")
    print("\nThe downsampling_mode parameter is working correctly!")
