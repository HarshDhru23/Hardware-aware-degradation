#!/usr/bin/env python3
"""
Test script for Hardware-aware Degradation Pipeline.

This script allows you to quickly test the pipeline on a single image
and visualize the results (HR, LR1, LR2).

Usage:
    python test.py --image path/to/image.tif
    python test.py --create-sample  # Creates and tests a sample image
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from degradation import DegradationPipeline
from config import ConfigManager
from utils.data_io import GeoTIFFLoader
from utils.visualization import visualize_degradation_results


def create_sample_image(output_path: str = "sample_test_image.tif", size: int = 512):
    """
    Create a sample grayscale test image with patterns.
    
    Args:
        output_path: Path to save the test image
        size: Image size (default: 512x512)
    """
    print(f"Creating sample test image ({size}x{size})...")
    
    # Create a test image with various patterns
    y, x = np.ogrid[:size, :size]
    
    # Combine multiple patterns for visual interest
    image = np.zeros((size, size), dtype=np.float32)
    
    # Add sinusoidal patterns
    image += 0.3 * np.sin(2 * np.pi * x / 64) * np.cos(2 * np.pi * y / 64)
    image += 0.2 * np.sin(2 * np.pi * x / 32)
    image += 0.15 * np.cos(2 * np.pi * y / 48)
    
    # Add some geometric shapes
    # Square in the center
    center = size // 2
    square_size = size // 8
    image[center-square_size:center+square_size, center-square_size:center+square_size] += 0.5
    
    # Circle in top-left quadrant
    circle_center_y, circle_center_x = size // 4, size // 4
    circle_radius = size // 10
    circle_mask = (x - circle_center_x)**2 + (y - circle_center_y)**2 <= circle_radius**2
    image[circle_mask] += 0.4
    
    # Add some random noise for texture
    image += 0.05 * np.random.random((size, size))
    
    # Normalize to [0, 1] range
    image = (image - image.min()) / (image.max() - image.min())
    
    # Convert to uint8 for saving
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Save as TIFF
    try:
        from PIL import Image
        Image.fromarray(image_uint8).save(output_path)
        print(f"Sample image saved to: {output_path}")
        return output_path
    except ImportError:
        # Fallback: save as numpy array
        np.save(output_path.replace('.tif', '.npy'), image)
        print(f"PIL not available. Sample image saved as numpy array: {output_path.replace('.tif', '.npy')}")
        return output_path.replace('.tif', '.npy')


def test_pipeline(image_path: str, config_path: str = "configs/default_config.yaml", 
                 output_dir: str = "test_output", seed: int = 42):
    """
    Test the degradation pipeline on a single image.
    
    Args:
        image_path: Path to input image
        config_path: Path to configuration file
        output_dir: Directory to save output visualization
        seed: Random seed for reproducibility
    """
    print("\n" + "="*60)
    print("Testing Hardware-aware Degradation Pipeline")
    print("="*60)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load configuration
        print(f"\n1. Loading configuration from: {config_path}")
        config = ConfigManager(config_path)
        print(f"   Downsampling factor: {config.get('downsampling_factor')}")
        print(f"   Optical blur sigma: {config.get('optical_sigma')}")
        print(f"   Motion blur kernel: {config.get('motion_kernel_size')}")
        print(f"   Gaussian noise std: {config.get('gaussian_std')}")
        print(f"   Poisson noise enabled: {config.get('enable_poisson')}")
        
        # 2. Initialize pipeline
        print("\n2. Initializing degradation pipeline...")
        pipeline = DegradationPipeline(config.get_all())
        
        # 3. Load image
        print(f"\n3. Loading test image: {image_path}")
        loader = GeoTIFFLoader(normalize=True, target_dtype='float32')
        hr_image = loader.load_image(image_path)
        print(f"   HR image shape: {hr_image.shape}")
        print(f"   HR image dtype: {hr_image.dtype}")
        print(f"   HR image range: [{hr_image.min():.3f}, {hr_image.max():.3f}]")
        
        # Check dimension compatibility
        if not pipeline.validate_image_dimensions(hr_image):
            print("\n   WARNING: Image dimensions not divisible by downsampling factor!")
            print(f"   Cropping image to make it compatible...")
            h, w = hr_image.shape[:2]
            factor = pipeline.downsampling_factor
            new_h = (h // factor) * factor
            new_w = (w // factor) * factor
            hr_image = hr_image[:new_h, :new_w]
            print(f"   New HR image shape: {hr_image.shape}")
        
        # 4. Generate LR1 and LR2
        print(f"\n4. Generating LR1 and LR2 images (seed={seed})...")
        lr1, lr2 = pipeline.process_image(hr_image, seed=seed)
        print(f"   LR1 shape: {lr1.shape}")
        print(f"   LR2 shape: {lr2.shape}")
        print(f"   LR1 range: [{lr1.min():.3f}, {lr1.max():.3f}]")
        print(f"   LR2 range: [{lr2.min():.3f}, {lr2.max():.3f}]")

        # # After generating lr1, lr2, add this before visualization:
        # print(f"Before clipping - LR1 range: [{lr1.min():.3f}, {lr1.max():.3f}]")
        # lr1 = np.clip(lr1, 0, 1)
        # lr2 = np.clip(lr2, 0, 1)
        # print(f"After clipping - LR1 range: [{lr1.min():.3f}, {lr1.max():.3f}]")  

        # Calculate difference between LR1 and LR2
        diff = np.abs(lr1.astype(np.float64) - lr2.astype(np.float64))
        print(f"   Mean difference between LR1 and LR2: {diff.mean():.6f}")
        print(f"   Max difference between LR1 and LR2: {diff.max():.6f}")
        
        # 5. Save outputs
        print(f"\n5. Saving outputs to: {output_dir}")
        
        # Save as numpy arrays
        np.save(output_dir / "hr_image.npy", hr_image)
        np.save(output_dir / "lr1_image.npy", lr1)
        np.save(output_dir / "lr2_image.npy", lr2)
        print(f"   Saved numpy arrays: hr_image.npy, lr1_image.npy, lr2_image.npy")
        
        # 6. Visualize results
        print("\n6. Creating visualization...")
        
        # Main visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(hr_image, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'HR Image\nShape: {hr_image.shape}', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(lr1, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'LR1 (P1 Sensor - Reference)\nShape: {lr1.shape}', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(lr2, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'LR2 (P2 Sensor - Shifted)\nShape: {lr2.shape}', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle('Hardware-aware Degradation Pipeline - Test Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = output_dir / "degradation_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization: {output_path}")
        plt.show()
        
        # Create detailed comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # LR1
        axes[0, 0].imshow(lr1, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('LR1 (P1 Sensor)', fontweight='bold')
        axes[0, 0].axis('off')
        
        # LR2
        axes[0, 1].imshow(lr2, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title('LR2 (P2 Sensor)', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Difference map
        axes[1, 0].imshow(diff, cmap='RdBu_r', vmin=0, vmax=diff.max())
        axes[1, 0].set_title(f'Absolute Difference\nMean: {diff.mean():.6f}', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Histogram comparison
        axes[1, 1].hist(lr1.flatten(), bins=50, alpha=0.6, label='LR1', density=True)
        axes[1, 1].hist(lr2.flatten(), bins=50, alpha=0.6, label='LR2', density=True)
        axes[1, 1].set_title('Pixel Value Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Pixel Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('LR1 vs LR2 Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        comparison_path = output_dir / "lr1_lr2_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"   Saved comparison: {comparison_path}")
        plt.show()
        
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)
        print(f"\nOutput files saved to: {output_dir.absolute()}")
        print(f"  - hr_image.npy")
        print(f"  - lr1_image.npy")
        print(f"  - lr2_image.npy")
        print(f"  - degradation_results.png")
        print(f"  - lr1_lr2_comparison.png")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test the Hardware-aware Degradation Pipeline on a single image"
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image (GeoTIFF, TIFF, or PNG)'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create and test a sample image'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file (default: configs/default_config.yaml)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='test_output',
        help='Output directory for results (default: test_output)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=512,
        help='Size of sample image if --create-sample is used (default: 512)'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"ERROR: Configuration file not found: {args.config}")
        print("\nAvailable configs:")
        config_dir = Path("configs")
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                print(f"  - {config_file}")
        sys.exit(1)
    
    # Create or use provided image
    if args.create_sample:
        print("Creating sample test image...")
        image_path = create_sample_image(size=args.size)
        print(f"\nTesting with sample image: {image_path}")
    elif args.image:
        image_path = args.image
        if not Path(image_path).exists():
            print(f"ERROR: Image file not found: {image_path}")
            sys.exit(1)
    else:
        print("ERROR: Either --image or --create-sample must be specified")
        print("\nUsage examples:")
        print("  python test.py --create-sample")
        print("  python test.py --image path/to/image.tif")
        print("  python test.py --image path/to/image.tif --config configs/high_quality_config.yaml")
        sys.exit(1)
    
    # Run test
    success = test_pipeline(
        image_path=image_path,
        config_path=args.config,
        output_dir=args.output,
        seed=args.seed
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
