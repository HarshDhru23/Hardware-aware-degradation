#!/usr/bin/env python3
"""
Step-by-step Degradation Analysis Script

This script visualizes each step of the degradation pipeline:
1. Original HR Image
2. After Warping (geometric shift)
3. After Blur (PSF convolution)
4. After Downsampling (spatial integration)
5. After Noise (Poisson + Gaussian + ADC)

Usage:
    python analyze_degradation.py --image path/to/image.tiff
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.degradation.operators import WarpingOperator, BlurOperator, DownsamplingOperator, NoiseOperator
from src.config import ConfigManager
from src.utils.data_io import GeoTIFFLoader


def visualize_step(image, title, ax, percentile_clip=True):
    """
    Visualize a single step image.
    
    Args:
        image: Image array
        title: Title for the plot
        ax: Matplotlib axis
        percentile_clip: Whether to clip to 2-98 percentile for better visualization
    """
    if percentile_clip:
        vmin, vmax = np.percentile(image, 2), np.percentile(image, 98)
    else:
        vmin, vmax = image.min(), image.max()
    
    ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Add shape and range info
    info_text = f'Shape: {image.shape}\nRange: [{image.min():.3f}, {image.max():.3f}]'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def visualize_difference(img1, img2, title, ax):
    """
    Visualize the difference between two images.
    
    Args:
        img1: First image
        img2: Second image
        title: Title for the plot
        ax: Matplotlib axis
    """
    # Compute absolute difference
    diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    
    ax.imshow(diff, cmap='RdBu_r', vmin=0, vmax=diff.max())
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Add statistics
    stats_text = f'Mean: {diff.mean():.6f}\nMax: {diff.max():.6f}\nStd: {diff.std():.6f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def analyze_degradation_pipeline(image_path, config_path="configs/default_config.yaml", 
                                 output_dir="analysis_output", seed=42):
    """
    Analyze the degradation pipeline step by step.
    
    Args:
        image_path: Path to input HR image
        config_path: Path to configuration file
        output_dir: Directory to save output visualizations
        seed: Random seed for reproducibility
    """
    print("\n" + "="*80)
    print("STEP-BY-STEP DEGRADATION PIPELINE ANALYSIS")
    print("="*80)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load configuration
    print(f"\n[CONFIG] Loading configuration from: {config_path}")
    config = ConfigManager(config_path)
    config_dict = config.get_all()
    
    print(f"  - Downsampling factor: {config.get('downsampling_factor')}")
    print(f"  - Downsampling mode: {config.get('downsampling_mode')}")
    print(f"  - PSF sigma_x: {config.get('psf_sigma_x')}")
    print(f"  - PSF sigma_y: {config.get('psf_sigma_y')}")
    print(f"  - Gaussian noise std: {config.get('gaussian_std')}")
    print(f"  - Poisson noise enabled: {config.get('enable_poisson')}")
    print(f"  - Photon gain: {config.get('photon_gain')}")
    
    # 2. Initialize operators
    print(f"\n[OPERATORS] Initializing degradation operators...")
    warp_op = WarpingOperator(shift_x=0.0, shift_y=0.0, stochastic=False)  # LR0 has shift (0,0)
    blur_op = BlurOperator(config_dict)
    downsample_op = DownsamplingOperator(config_dict)
    noise_op = NoiseOperator(config_dict)
    
    downsampling_factor = config.get('downsampling_factor')
    print(f"  ✓ Warping operator (shift: [0.0, 0.0])")
    print(f"  ✓ Blur operator (PSF)")
    print(f"  ✓ Downsampling operator ({downsampling_factor}x)")
    print(f"  ✓ Noise operator (Poisson + Gaussian + ADC)")
    
    # 3. Load HR image
    print(f"\n[STEP 0] Loading HR image: {image_path}")
    loader = GeoTIFFLoader(normalize=True, target_dtype='float32')
    hr_original = loader.load_image(image_path)
    
    # Crop to be divisible by downsampling factor
    h, w = hr_original.shape[:2]
    new_h = (h // downsampling_factor) * downsampling_factor
    new_w = (w // downsampling_factor) * downsampling_factor
    hr_original = hr_original[:new_h, :new_w]
    
    print(f"  - Shape: {hr_original.shape}")
    print(f"  - Dtype: {hr_original.dtype}")
    print(f"  - Range: [{hr_original.min():.3f}, {hr_original.max():.3f}]")
    
    # Store all intermediate steps
    steps = []
    step_names = []
    
    # Step 0: Original HR
    steps.append(hr_original.copy())
    step_names.append("Step 0: Original HR")
    
    # 4. Apply Warping (M_k)
    print(f"\n[STEP 1] Applying Warping (geometric shift [0.0, 0.0])...")
    hr_warped = warp_op.apply(hr_original, seed=seed, downsampling_factor=downsampling_factor)
    print(f"  - Output shape: {hr_warped.shape}")
    print(f"  - Output range: [{hr_warped.min():.3f}, {hr_warped.max():.3f}]")
    print(f"  - Difference from original: mean={np.abs(hr_original - hr_warped).mean():.6f}")
    
    steps.append(hr_warped.copy())
    step_names.append("Step 1: After Warping")
    
    # 5. Apply Blur (B_k)
    print(f"\n[STEP 2] Applying PSF Blur (anisotropic Gaussian)...")
    hr_blurred = blur_op.apply(hr_warped)
    print(f"  - Output shape: {hr_blurred.shape}")
    print(f"  - Output range: [{hr_blurred.min():.3f}, {hr_blurred.max():.3f}]")
    print(f"  - Difference from warped: mean={np.abs(hr_warped - hr_blurred).mean():.6f}")
    
    steps.append(hr_blurred.copy())
    step_names.append("Step 2: After Blur (PSF)")
    
    # 6. Apply Downsampling (D)
    print(f"\n[STEP 3] Applying Downsampling ({downsampling_factor}x with average pooling)...")
    lr_downsampled = downsample_op.apply(hr_blurred)
    print(f"  - Output shape: {lr_downsampled.shape}")
    print(f"  - Output range: [{lr_downsampled.min():.3f}, {lr_downsampled.max():.3f}]")
    print(f"  - Downsampling ratio: {hr_blurred.shape[0]/lr_downsampled.shape[0]:.1f}x")
    
    steps.append(lr_downsampled.copy())
    step_names.append(f"Step 3: After Downsampling ({downsampling_factor}x)")
    
    # 7. Apply Noise (n_k)
    print(f"\n[STEP 4] Applying Noise (Poisson + Gaussian + ADC quantization)...")
    lr_final = noise_op.apply_noise_and_quantization(lr_downsampled, seed=seed)
    print(f"  - Output shape: {lr_final.shape}")
    print(f"  - Output range: [{lr_final.min():.3f}, {lr_final.max():.3f}]")
    print(f"  - Difference from downsampled: mean={np.abs(lr_downsampled - lr_final).mean():.6f}")
    
    steps.append(lr_final.copy())
    step_names.append("Step 4: After Noise (Final LR)")
    
    # 8. Save intermediate results
    print(f"\n[SAVING] Saving intermediate results to: {output_dir}")
    np.save(output_dir / "step0_hr_original.npy", steps[0])
    np.save(output_dir / "step1_hr_warped.npy", steps[1])
    np.save(output_dir / "step2_hr_blurred.npy", steps[2])
    np.save(output_dir / "step3_lr_downsampled.npy", steps[3])
    np.save(output_dir / "step4_lr_final.npy", steps[4])
    print(f"  ✓ Saved 5 numpy arrays")
    
    # 9. Create comprehensive visualization
    print(f"\n[VISUALIZATION] Creating step-by-step visualization...")
    
    # Main figure: All steps in sequence
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)
    
    # Row 1: All 5 steps
    print("  - Creating main pipeline visualization...")
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        visualize_step(steps[i], step_names[i], ax)
    
    # Row 2: Difference maps between consecutive steps
    print("  - Creating difference maps...")
    diff_titles = [
        "Diff: Original → Warped",
        "Diff: Warped → Blurred", 
        "Diff: Blurred → Downsampled*",
        "Diff: Downsampled → Noisy"
    ]
    
    # For steps 0-2 (HR images)
    for i in range(2):  # Only compare steps 0-1 and 1-2 (both HR)
        ax = fig.add_subplot(gs[1, i])
        visualize_difference(steps[i], steps[i+1], diff_titles[i], ax)
    
    # For step 2->3, we need to downsample step 2 for fair comparison
    ax = fig.add_subplot(gs[1, 2])
    hr_blurred_for_diff = steps[2]
    lr_blurred_downsampled = downsample_op.apply(hr_blurred_for_diff)
    visualize_difference(lr_blurred_downsampled, steps[3], diff_titles[2] + "\n(both at LR)", ax)
    
    # For step 3->4 (LR images)
    ax = fig.add_subplot(gs[1, 3])
    visualize_difference(steps[3], steps[4], diff_titles[3], ax)
    
    # Row 3: Zoomed comparison of HR original vs LR final
    print("  - Creating zoomed comparison...")
    
    # Original HR - center crop
    ax = fig.add_subplot(gs[2, 0:2])
    h, w = steps[0].shape
    crop_size = 128
    start_h, start_w = h//2 - crop_size//2, w//2 - crop_size//2
    hr_crop = steps[0][start_h:start_h+crop_size, start_w:start_w+crop_size]
    visualize_step(hr_crop, "HR Original (center crop 128x128)", ax)
    
    # LR final - center crop
    ax = fig.add_subplot(gs[2, 2:4])
    h_lr, w_lr = steps[4].shape
    crop_size_lr = crop_size // downsampling_factor
    start_h_lr, start_w_lr = h_lr//2 - crop_size_lr//2, w_lr//2 - crop_size_lr//2
    lr_crop = steps[4][start_h_lr:start_h_lr+crop_size_lr, start_w_lr:start_w_lr+crop_size_lr]
    visualize_step(lr_crop, f"LR Final (center crop {crop_size_lr}x{crop_size_lr})", ax)
    
    # Statistics summary
    ax = fig.add_subplot(gs[2, 4])
    ax.axis('off')
    summary_text = "PIPELINE STATISTICS\n" + "="*30 + "\n\n"
    summary_text += f"Original HR:\n"
    summary_text += f"  Shape: {steps[0].shape}\n"
    summary_text += f"  Range: [{steps[0].min():.3f}, {steps[0].max():.3f}]\n\n"
    summary_text += f"Final LR:\n"
    summary_text += f"  Shape: {steps[4].shape}\n"
    summary_text += f"  Range: [{steps[4].min():.3f}, {steps[4].max():.3f}]\n\n"
    summary_text += f"Degradation Effects:\n"
    summary_text += f"  Warping: {np.abs(steps[0]-steps[1]).mean():.6f}\n"
    summary_text += f"  Blur: {np.abs(steps[1]-steps[2]).mean():.6f}\n"
    summary_text += f"  Downsampling: {downsampling_factor}x reduction\n"
    summary_text += f"  Noise: {np.abs(steps[3]-steps[4]).mean():.6f}\n"
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'Step-by-Step Degradation Pipeline Analysis (LR Frame 0)\nInput: {Path(image_path).name}', 
                 fontsize=14, fontweight='bold')
    
    # Save main visualization
    output_path = output_dir / "pipeline_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved main visualization: {output_path}")
    plt.show()
    
    # 10. Create individual step visualizations
    print(f"\n  - Creating individual step visualizations...")
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    for i in range(5):
        visualize_step(steps[i], step_names[i], axes[i])
    
    plt.suptitle('Individual Pipeline Steps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "individual_steps.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved individual steps: {output_path}")
    plt.show()
    
    # 11. Create detailed difference analysis
    print(f"\n  - Creating detailed difference analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # HR differences
    visualize_difference(steps[0], steps[1], "Original → Warped", axes[0, 0])
    visualize_difference(steps[1], steps[2], "Warped → Blurred", axes[0, 1])
    
    # LR differences
    lr_blurred_for_comparison = downsample_op.apply(steps[2])
    visualize_difference(lr_blurred_for_comparison, steps[3], "Blurred → Downsampled\n(comparison at LR resolution)", axes[1, 0])
    visualize_difference(steps[3], steps[4], "Downsampled → Noisy", axes[1, 1])
    
    plt.suptitle('Difference Maps Between Pipeline Steps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "difference_maps.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved difference maps: {output_path}")
    plt.show()
    
    # 12. Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput files saved to: {output_dir.absolute()}")
    print(f"  - step0_hr_original.npy")
    print(f"  - step1_hr_warped.npy")
    print(f"  - step2_hr_blurred.npy")
    print(f"  - step3_lr_downsampled.npy")
    print(f"  - step4_lr_final.npy")
    print(f"  - pipeline_analysis.png (comprehensive visualization)")
    print(f"  - individual_steps.png (all steps side by side)")
    print(f"  - difference_maps.png (difference analysis)")
    
    print(f"\nPipeline Summary:")
    print(f"  Original HR: {steps[0].shape} → Final LR: {steps[4].shape}")
    print(f"  Downsampling: {downsampling_factor}x reduction")
    print(f"  Total degradation effect: {np.abs(steps[0].mean() - steps[4].mean()):.6f}")
    
    return steps


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze the degradation pipeline step-by-step with visualizations"
    )
    parser.add_argument(
        '--image', 
        type=str, 
        required=True,
        help='Path to input HR image (TIFF format)'
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
        default='analysis_output',
        help='Output directory for visualizations (default: analysis_output)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"ERROR: Image file not found: {args.image}")
        sys.exit(1)
    
    # Run analysis
    try:
        analyze_degradation_pipeline(
            image_path=args.image,
            config_path=args.config,
            output_dir=args.output,
            seed=args.seed
        )
    except Exception as e:
        print(f"\nERROR: Analysis failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
