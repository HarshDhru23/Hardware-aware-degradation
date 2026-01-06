#!/usr/bin/env python3
"""
Comprehensive degradation analysis script.

This script performs detailed analysis of the degradation pipeline including:
- HR/LR1/LR2 comparison with zoom regions
- Noise characterization and SNR analysis
- Blur analysis with edge preservation metrics
- Step-by-step intermediate visualization

Usage:
    python analyze_degradation.py --image path/to/image.tif
    python analyze_degradation.py --create-sample  # Test with synthetic image
    python analyze_degradation.py --config configs/custom_config.yaml
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from degradation import DegradationPipeline
from config import ConfigManager
from utils.data_io import GeoTIFFLoader
from utils.visualization import (
    visualize_degradation_results,
    plot_degradation_comparison,
    plot_noise_analysis,
    plot_blur_analysis
)


def setup_logging(verbose: bool = True):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_sample_image(size: int = 512) -> np.ndarray:
    """
    Create a sample test image with various patterns.
    
    Args:
        size: Image size (default: 512x512)
        
    Returns:
        Normalized test image in [0, 1] range
    """
    print(f"Creating sample test image ({size}x{size})...")
    
    y, x = np.ogrid[:size, :size]
    
    # Combine multiple patterns
    image = np.zeros((size, size), dtype=np.float32)
    
    # Sinusoidal patterns
    image += 0.3 * np.sin(2 * np.pi * x / 64) * np.cos(2 * np.pi * y / 64)
    image += 0.2 * np.sin(2 * np.pi * x / 32)
    image += 0.15 * np.cos(2 * np.pi * y / 48)
    
    # Geometric shapes
    # Center square
    center = size // 2
    square_size = size // 6
    image[center-square_size:center+square_size, 
          center-square_size:center+square_size] += 0.4
    
    # Circle
    circle_radius = size // 8
    circle_x, circle_y = size // 4, size // 4
    dist = np.sqrt((x - circle_x)**2 + (y - circle_y)**2)
    image[dist < circle_radius] += 0.35
    
    # Diagonal lines
    for offset in range(-size//2, size//2, size//10):
        mask = np.abs(y - x - offset) < 2
        image[mask] += 0.25
    
    # Edge patterns (high frequency)
    edge_region = (x > size//4) & (x < size*3//4) & (y > size*3//4)
    image[edge_region] += 0.2 * np.sin(4 * np.pi * x[edge_region] / size)
    
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    
    print(f"   Sample image created: shape={image.shape}, range=[{image.min():.3f}, {image.max():.3f}]")
    return image


def extract_center_patch(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Extract a center patch from image for detailed analysis.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        patch_size: Size of square patch to extract
        
    Returns:
        Center patch of size (patch_size, patch_size)
    """
    h, w = image.shape[:2]
    center_y, center_x = h // 2, w // 2
    half_size = patch_size // 2
    
    y1 = max(0, center_y - half_size)
    y2 = min(h, center_y + half_size)
    x1 = max(0, center_x - half_size)
    x2 = min(w, center_x + half_size)
    
    if len(image.shape) == 2:
        return image[y1:y2, x1:x2]
    else:
        return image[y1:y2, x1:x2, :]


def analyze_pipeline_step_by_step(pipeline: DegradationPipeline, 
                                   hr_image: np.ndarray,
                                   output_dir: Path,
                                   seed: int = 42):
    """
    Perform step-by-step analysis of the degradation pipeline.
    
    Args:
        pipeline: Degradation pipeline instance
        hr_image: High-resolution input image
        output_dir: Directory to save analysis results
        seed: Random seed for reproducibility
    """
    print("\n" + "="*70)
    print("STEP-BY-STEP DEGRADATION ANALYSIS")
    print("="*70)
    
    # Extract 256x256 HR patch for analysis
    print("\nExtracting 256x256 center patch from HR image for analysis...")
    hr_patch = extract_center_patch(hr_image, 256)
    print(f"   HR patch shape: {hr_patch.shape}")
    print(f"   HR patch range: [{hr_patch.min():.3f}, {hr_patch.max():.3f}]")
    
    # Create intermediate outputs directory
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== LR1 BRANCH ANALYSIS ==========
    print("\n[LR1 Branch - Reference Frame]")
    print("-" * 70)
    
    # Step 1: Warping (LR1 uses identity - no shift)
    print("Step 1: Warping (identity - no shift)...")
    hr_warped_lr1 = pipeline.warp_lr1.apply(hr_patch)
    print(f"   Output shape: {hr_warped_lr1.shape}")
    print(f"   Value range: [{hr_warped_lr1.min():.3f}, {hr_warped_lr1.max():.3f}]")
    
    # Step 2: Blurring
    print("Step 2: Blurring (optical + motion)...")
    hr_blurred_lr1 = pipeline.blur_lr1.apply(hr_warped_lr1)
    print(f"   Output shape: {hr_blurred_lr1.shape}")
    print(f"   Value range: [{hr_blurred_lr1.min():.3f}, {hr_blurred_lr1.max():.3f}]")
    
    # Blur analysis
    print("   Analyzing blur effects...")
    plot_blur_analysis(
        original=hr_warped_lr1,
        blurred=hr_blurred_lr1,
        title="LR1 Blur Analysis (256x256 HR)",
        save_path=str(intermediate_dir / "lr1_blur_analysis.png")
    )
    print(f"   Saved: lr1_blur_analysis.png")
    
    # Step 3: Poisson noise (on HR before downsampling)
    print("Step 3: Adding Poisson noise (photon shot noise on HR)...")
    hr_poisson_lr1 = pipeline.noise_lr1.apply_poisson_only(hr_blurred_lr1, seed=seed)
    print(f"   Output shape: {hr_poisson_lr1.shape}")
    print(f"   Value range: [{hr_poisson_lr1.min():.3f}, {hr_poisson_lr1.max():.3f}]")
    
    # Poisson noise analysis
    print("   Analyzing Poisson noise effects...")
    plot_noise_analysis(
        clean_image=hr_blurred_lr1,
        noisy_image=hr_poisson_lr1,
        title="LR1 Poisson Noise Analysis (256x256 HR)",
        save_path=str(intermediate_dir / "lr1_poisson_analysis.png")
    )
    print(f"   Saved: lr1_poisson_analysis.png")
    
    # Step 4: Downsampling
    print("Step 4: Downsampling (sensor integration)...")
    lr1_downsampled = pipeline.downsample.apply(hr_poisson_lr1)
    print(f"   Output shape: {lr1_downsampled.shape}")
    print(f"   Value range: [{lr1_downsampled.min():.3f}, {lr1_downsampled.max():.3f}]")
    print(f"   Note: Downsampling smooths HR Poisson noise by factor of √16 = 4x")
    
    # Step 5: Gaussian noise + quantization (on LR after downsampling)
    print("Step 5: Adding Gaussian noise + 11-bit quantization (read noise on LR)...")
    lr1_final = pipeline.noise_lr1.apply_gaussian_only(lr1_downsampled, seed=seed)
    print(f"   Output shape: {lr1_final.shape}")
    print(f"   Value range: [{lr1_final.min():.3f}, {lr1_final.max():.3f}]")
    
    # Gaussian noise + quantization analysis (full LR image)
    print("   Analyzing Gaussian noise + quantization on full LR...")
    plot_noise_analysis(
        clean_image=lr1_downsampled,
        noisy_image=lr1_final,
        title="LR1 Gaussian Noise + Quantization (64x64 LR)",
        save_path=str(intermediate_dir / "lr1_gaussian_quantization_analysis.png")
    )
    print(f"   Saved: lr1_gaussian_quantization_analysis.png")
    
    # ========== LR2 BRANCH ANALYSIS ==========
    print("\n[LR2 Branch - Shifted Frame]")
    print("-" * 70)
    
    # Step 1: Warping (LR2 uses shift)
    print("Step 1: Warping (with stochastic shift)...")
    hr_warped_lr2 = pipeline.warp_lr2.apply(hr_patch)
    print(f"   Output shape: {hr_warped_lr2.shape}")
    print(f"   Value range: [{hr_warped_lr2.min():.3f}, {hr_warped_lr2.max():.3f}]")
    
    # Warp comparison
    warp_diff = np.abs(hr_warped_lr1.astype(np.float64) - hr_warped_lr2.astype(np.float64))
    print(f"   Warp difference: mean={warp_diff.mean():.6f}, max={warp_diff.max():.6f}")
    
    # Step 2: Blurring
    print("Step 2: Blurring (optical + motion)...")
    hr_blurred_lr2 = pipeline.blur_lr2.apply(hr_warped_lr2)
    print(f"   Output shape: {hr_blurred_lr2.shape}")
    print(f"   Value range: [{hr_blurred_lr2.min():.3f}, {hr_blurred_lr2.max():.3f}]")
    
    # Blur analysis
    print("   Analyzing blur effects...")
    plot_blur_analysis(
        original=hr_warped_lr2,
        blurred=hr_blurred_lr2,
        title="LR2 Blur Analysis (256x256 HR)",
        save_path=str(intermediate_dir / "lr2_blur_analysis.png")
    )
    print(f"   Saved: lr2_blur_analysis.png")
    
    # Step 3: Poisson noise (on HR before downsampling)
    print("Step 3: Adding Poisson noise (photon shot noise on HR)...")
    hr_poisson_lr2 = pipeline.noise_lr2.apply_poisson_only(hr_blurred_lr2, seed=seed+1)
    print(f"   Output shape: {hr_poisson_lr2.shape}")
    print(f"   Value range: [{hr_poisson_lr2.min():.3f}, {hr_poisson_lr2.max():.3f}]")
    
    # Poisson noise analysis
    print("   Analyzing Poisson noise effects...")
    plot_noise_analysis(
        clean_image=hr_blurred_lr2,
        noisy_image=hr_poisson_lr2,
        title="LR2 Poisson Noise Analysis (256x256 HR)",
        save_path=str(intermediate_dir / "lr2_poisson_analysis.png")
    )
    print(f"   Saved: lr2_poisson_analysis.png")
    
    # Step 4: Downsampling
    print("Step 4: Downsampling (sensor integration)...")
    lr2_downsampled = pipeline.downsample.apply(hr_poisson_lr2)
    print(f"   Output shape: {lr2_downsampled.shape}")
    print(f"   Value range: [{lr2_downsampled.min():.3f}, {lr2_downsampled.max():.3f}]")
    print(f"   Note: Downsampling smooths HR Poisson noise by factor of √16 = 4x")
    
    # Step 5: Gaussian noise + quantization (on LR after downsampling)
    print("Step 5: Adding Gaussian noise + 11-bit quantization (read noise on LR)...")
    lr2_final = pipeline.noise_lr2.apply_gaussian_only(lr2_downsampled, seed=seed+1)
    print(f"   Output shape: {lr2_final.shape}")
    print(f"   Value range: [{lr2_final.min():.3f}, {lr2_final.max():.3f}]")
    
    # Gaussian noise + quantization analysis (full LR image)
    print("   Analyzing Gaussian noise + quantization on full LR...")
    plot_noise_analysis(
        clean_image=lr2_downsampled,
        noisy_image=lr2_final,
        title="LR2 Gaussian Noise + Quantization (64x64 LR)",
        save_path=str(intermediate_dir / "lr2_gaussian_quantization_analysis.png")
    )
    print(f"   Saved: lr2_gaussian_quantization_analysis.png")
    
    # ========== FINAL COMPARISON ==========
    print("\n[Final LR1 vs LR2 Comparison]")
    print("-" * 70)
    
    # Calculate differences
    lr_diff = np.abs(lr1_final.astype(np.float64) - lr2_final.astype(np.float64))
    print(f"LR1 vs LR2 difference: mean={lr_diff.mean():.6f}, max={lr_diff.max():.6f}")
    
    # Overall visualization
    print("\nGenerating comprehensive visualizations...")
    
    # Basic 3-panel view
    visualize_degradation_results(
        hr_image=hr_patch,
        lr1_image=lr1_final,
        lr2_image=lr2_final,
        title="Degradation Pipeline Results - Overview",
        save_path=str(output_dir / "degradation_overview.png")
    )
    print(f"   Saved: degradation_overview.png")
    
    # Detailed comparison with zoom
    # Select center region for zoom (on HR patch)
    h, w = hr_patch.shape
    zoom_size = min(h, w) // 4
    center_y, center_x = h // 2, w // 2
    zoom_region = (
        center_y - zoom_size // 2,
        center_x - zoom_size // 2,
        center_y + zoom_size // 2,
        center_x + zoom_size // 2
    )
    
    plot_degradation_comparison(
        original=hr_patch,
        degraded_images={'LR1 (Reference)': lr1_final, 'LR2 (Shifted)': lr2_final},
        region=zoom_region,
        save_path=str(output_dir / "degradation_comparison_zoom.png")
    )
    print(f"   Saved: degradation_comparison_zoom.png")
    
    return {
        'hr': hr_patch,
        'lr1': lr1_final,
        'lr2': lr2_final,
        'lr1_downsampled': lr1_downsampled,
        'lr2_downsampled': lr2_downsampled,
        'hr_blurred_lr1': hr_blurred_lr1,
        'hr_blurred_lr2': hr_blurred_lr2,
        'hr_poisson_lr1': hr_poisson_lr1,
        'hr_poisson_lr2': hr_poisson_lr2
    }


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive degradation pipeline analysis"
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to input GeoTIFF image'
    )
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create and analyze a synthetic test image'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file (default: configs/default_config.yaml)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='analysis_output',
        help='Output directory for analysis results (default: analysis_output)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    print("="*70)
    print("HARDWARE-AWARE DEGRADATION PIPELINE - COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    try:
        # Load configuration
        print(f"\n1. Loading configuration from: {args.config}")
        config_manager = ConfigManager(args.config)
        config = config_manager.config
        print(f"   Configuration loaded successfully")
        print(f"\n   Key Parameters:")
        print(f"   - Downsampling factor: {config.get('downsampling_factor', 4)}")
        print(f"   - Optical sigma: {config.get('optical_sigma', 0.8)}")
        print(f"   - Motion kernel size: {config.get('motion_kernel_size', 3)}")
        print(f"   - Shift mean: {config.get('shift_mean', 0.5)}")
        print(f"   - Shift std: {config.get('shift_std', 0.1)}")
        print(f"   - Gaussian noise std: {config.get('gaussian_std', 0.01)}")
        print(f"   - Poisson noise enabled: {config.get('enable_poisson', False)}")
        print(f"   - Photon gain: {config.get('photon_gain', 100.0)}")
        print(f"   - Enable quantization: {config.get('enable_quantization', False)}")
        print(f"   - Quantization bits: {config.get('quantization_bits', 8)}")
        
        # Initialize pipeline
        print(f"\n2. Initializing degradation pipeline...")
        pipeline = DegradationPipeline(config)
        print(f"   Pipeline initialized successfully")
        
        # Load or create image
        print(f"\n3. Loading input image...")
        if args.create_sample:
            print("   Creating synthetic test image...")
            hr_image = create_sample_image(size=512)
        elif args.image:
            print(f"   Loading from: {args.image}")
            loader = GeoTIFFLoader(normalize=True, target_dtype='float32')
            hr_image = loader.load_image(args.image)
            print(f"   Image loaded: shape={hr_image.shape}, dtype={hr_image.dtype}")
            print(f"   Value range: [{hr_image.min():.3f}, {hr_image.max():.3f}]")
        else:
            print("\nERROR: Must specify --image or --create-sample")
            parser.print_help()
            return 1
        
        # Perform comprehensive analysis
        print(f"\n4. Performing comprehensive analysis...")
        results = analyze_pipeline_step_by_step(
            pipeline=pipeline,
            hr_image=hr_image,
            output_dir=output_dir,
            seed=args.seed
        )
        
        # Save numpy arrays
        print(f"\n5. Saving intermediate results...")
        np.save(output_dir / "hr_image.npy", results['hr'])
        np.save(output_dir / "lr1_final.npy", results['lr1'])
        np.save(output_dir / "lr2_final.npy", results['lr2'])
        np.save(output_dir / "lr1_downsampled.npy", results['lr1_downsampled'])
        np.save(output_dir / "lr2_downsampled.npy", results['lr2_downsampled'])
        np.save(output_dir / "hr_poisson_lr1.npy", results['hr_poisson_lr1'])
        np.save(output_dir / "hr_poisson_lr2.npy", results['hr_poisson_lr2'])
        print(f"   Saved numpy arrays to {output_dir}")
        
        # Print summary
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nOutput files saved to: {output_dir.absolute()}")
        print("\nGenerated files:")
        print("  Main visualizations:")
        print("    - degradation_overview.png")
        print("    - degradation_comparison_zoom.png")
        print("  Intermediate analysis (LR1):")
        print("    - intermediate/lr1_blur_analysis.png")
        print("    - intermediate/lr1_poisson_analysis.png")
        print("    - intermediate/lr1_gaussian_quantization_analysis.png")
        print("  Intermediate analysis (LR2):")
        print("    - intermediate/lr2_blur_analysis.png")
        print("    - intermediate/lr2_poisson_analysis.png")
        print("    - intermediate/lr2_gaussian_quantization_analysis.png")
        print("  Numpy arrays:")
        print("    - hr_image.npy, lr1_final.npy, lr2_final.npy")
        print("    - lr1_downsampled.npy, lr2_downsampled.npy")
        print("    - hr_poisson_lr1.npy, hr_poisson_lr2.npy")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Analysis failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
