#!/usr/bin/env python3
"""
Compute global percentile statistics across an entire dataset in RAW 0-65535 range.

This script:
1. Loads images in RAW uint16 format (0-65535 range) WITHOUT normalization
2. Filters out completely black images (all pixels == 0)
3. Filters out individual black pixels (pixel value == 0) from valid images
4. Computes histogram over 0-65535 range using only non-zero pixels
5. Saves histogram data (NPY) and visualization (PNG) for later combining
6. Computes 2nd and 98th percentiles from filtered histogram

Usage:
    python compute_global_stats.py --input-dir AOI_3_Paris_Train_SN2/PAN --output configs/stats_paris_sn2.yaml
"""

import numpy as np
from pathlib import Path
import argparse
import yaml
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src to path to import the exact same loader
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from utils.data_io import GeoTIFFLoader


def compute_global_percentiles(
    input_dir: Path,
    percentiles: list = [2, 98],
    bins: int = 65536,
    pattern: str = "*.tif",
    save_histogram: bool = True,
    histogram_output: Path = None,
    histogram_data_output: Path = None
):
    """
    Compute global percentiles across all images in RAW 0-65535 range.
    
    Filters out:
    1. Completely black images (all pixels == 0)
    2. Individual black pixels (pixel value == 0) from valid images
    
    Only non-zero pixels are used for histogram and percentile computation.
    Saves histogram data as NPY file for later combining across multiple folders.
    
    Args:
        input_dir: Directory containing images
        percentiles: List of percentiles to compute (e.g., [2, 98])
        bins: Number of histogram bins (default: 65536 for full 16-bit precision)
        pattern: Glob pattern for image files
        save_histogram: Whether to save histogram visualization
        histogram_output: Path to save histogram image (default: same as yaml with .png)
        histogram_data_output: Path to save histogram data (default: same as yaml with _hist.npz)
        
    Returns:
        Tuple of (statistics_dict, histogram, bin_edges)
    """
    print("="*70)
    print("Computing Global Percentile Statistics (RAW 0-65535 Range)")
    print("="*70)
    print(f"Input directory: {input_dir.absolute()}")
    print(f"File pattern: {pattern}")
    print(f"Histogram bins: {bins} (0-65535 range)")
    print(f"Percentiles: {percentiles}")
    print(f"Black filtering: ENABLED (images + pixels with value=0)")
    print()
    
    # Find all image files
    image_files = sorted(input_dir.glob(pattern))
    
    if not image_files:
        raise ValueError(f"No files matching '{pattern}' found in {input_dir}")
    
    print(f"Found {len(image_files)} image files")
    print()
    
    # Initialize GeoTIFFLoader WITHOUT normalization to get raw uint16 values
    loader = GeoTIFFLoader(normalize=False, target_dtype='uint16')
    
    # Initialize histogram
    # Range [0, 65535] for full 16-bit range
    hist = np.zeros(bins, dtype=np.int64)
    bin_edges = np.linspace(0, 65535, bins + 1)
    
    print("Building cumulative histogram across all images...")
    print("(Filtering out black patches AND black pixels)")
    total_pixels = 0
    total_black_patches = 0
    total_patches_processed = 0
    total_black_pixels_filtered = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image in RAW uint16 format (0-65535)
            image = loader.load_image(img_path)
            
            # Check if completely black (all pixels == 0)
            if np.all(image == 0):
                total_black_patches += 1
                total_patches_processed += 1
                continue  # Skip completely black patches
            
            total_patches_processed += 1
            
            # Filter out black pixels (value == 0) from this image
            # Only include non-zero pixels in histogram
            non_zero_pixels = image[image > 0]
            
            if non_zero_pixels.size == 0:
                # Shouldn't happen after the all-zero check, but be safe
                continue
            
            # Track how many black pixels we're filtering
            num_black_pixels = image.size - non_zero_pixels.size
            total_black_pixels_filtered += num_black_pixels
            
            # Build histogram ONLY from non-zero pixels
            h, _ = np.histogram(non_zero_pixels, bins=bin_edges)
            hist += h
            total_pixels += non_zero_pixels.size  # Count only non-zero pixels
            
        except Exception as e:
            print(f"\nWarning: Failed to process {img_path.name}: {e}")
            continue
    
    print(f"\nTotal patches processed: {total_patches_processed:,}")
    print(f"Black patches filtered (entire image): {total_black_patches:,}")
    print(f"Valid patches used: {total_patches_processed - total_black_patches:,}")
    print(f"Black pixels filtered (value=0): {total_black_pixels_filtered:,}")
    print(f"Non-zero pixels in histogram: {total_pixels:,}")
    print()
    
    if total_pixels == 0:
        raise ValueError("No valid pixels found after filtering black patches!")
    
    # Compute percentiles from cumulative histogram
    print("Computing percentiles from histogram...")
    cumulative = np.cumsum(hist)
    total_count = cumulative[-1]
    
    results = {}
    for p in percentiles:
        # Find bin index for this percentile
        target_count = (p / 100.0) * total_count
        bin_idx = np.searchsorted(cumulative, target_count)
        
        # Get value at bin edge (clamp to valid range)
        bin_idx = min(bin_idx, bins - 1)
        percentile_value = bin_edges[bin_idx]
        
        results[f'p{p}'] = float(percentile_value)
        print(f"  {p}th percentile: {percentile_value:.1f} (raw 16-bit value)")
    
    # Compute suggested normalization factor (98th percentile)
    suggested_norm = results.get('p98', 65535)
    results['suggested_normalization_factor'] = float(suggested_norm)
    print(f"\nSuggested normalization factor: {suggested_norm:.1f}")
    print(f"  (divide images by this value to map to [0, 1])")
    
    # Add metadata
    results['metadata'] = {
        'num_images_total': len(image_files),
        'num_images_valid': total_patches_processed - total_black_patches,
        'num_images_black': total_black_patches,
        'total_pixels': int(total_pixels),
        'total_black_pixels_filtered': int(total_black_pixels_filtered),
        'bins': bins,
        'range': [0, 65535],
        'normalization': 'none (raw uint16)',
        'loader': 'GeoTIFFLoader',
        'input_dir': str(input_dir.absolute()),
        'black_filtering': 'both images and pixels (value=0)'
    }
    
    # Save histogram data as NPZ for later combining
    if histogram_data_output:
        print()
        print("Saving histogram data for later combining...")
        np.savez_compressed(
            histogram_data_output,
            histogram=hist,
            bin_edges=bin_edges,
            metadata=results['metadata']
        )
        print(f"  Histogram data saved to: {histogram_data_output}")
    
    # Generate histogram visualization if requested
    if save_histogram:
        print()
        print("Generating histogram visualization...")
        plot_histogram(hist, bin_edges, results, percentiles, histogram_output)
    
    return results, hist, bin_edges


def plot_histogram(hist, bin_edges, stats, percentiles, output_path=None):
    """
    Generate and save histogram visualization with percentile markers.
    Plots on 0-65535 range (raw 16-bit values).
    
    Args:
        hist: Histogram counts
        bin_edges: Bin edge values (0-65535 range)
        stats: Statistics dictionary with percentile values
        percentiles: List of percentiles that were computed
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot 1: Linear scale histogram
    ax1.bar(bin_centers, hist, width=np.diff(bin_edges), 
            edgecolor='none', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Pixel Value (Raw 16-bit, 0-65535)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pixel Count', fontsize=12, fontweight='bold')
    ax1.set_title('Global Histogram - Linear Scale (0-65535 Range)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 65535)
    
    # Set custom x-axis ticks for important bit-depth boundaries
    custom_xticks = [0, 512, 1023, 2047, 4096, 8192, 16384, 32768, 65535]
    ax1.set_xticks(custom_xticks)
    ax1.set_xticklabels([f'{x:,}' for x in custom_xticks], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add percentile lines
    colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']
    for i, p in enumerate(percentiles):
        p_val = stats[f'p{p}']
        color = colors[i % len(colors)]
        ax1.axvline(p_val, color=color, linestyle='--', linewidth=2.5, 
                   label=f'{p}th percentile: {p_val:.1f}', alpha=0.8)
    ax1.legend(fontsize=11, loc='upper right')
    
    # Plot 2: Log scale histogram (better for seeing distribution)
    # Filter out zero counts for log scale
    nonzero_mask = hist > 0
    ax2.bar(bin_centers[nonzero_mask], hist[nonzero_mask], 
            width=np.diff(bin_edges)[nonzero_mask], 
            edgecolor='none', alpha=0.7, color='steelblue')
    ax2.set_xlabel('Pixel Value (Raw 16-bit, 0-65535)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pixel Count (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Global Histogram - Log Scale (0-65535 Range)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 65535)
    ax2.set_yscale('log')
    
    # Set custom x-axis ticks for important bit-depth boundaries
    ax2.set_xticks(custom_xticks)
    ax2.set_xticklabels([f'{x:,}' for x in custom_xticks], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add percentile lines
    for i, p in enumerate(percentiles):
        p_val = stats[f'p{p}']
        color = colors[i % len(colors)]
        ax2.axvline(p_val, color=color, linestyle='--', linewidth=2.5, 
                   label=f'{p}th percentile: {p_val:.1f}', alpha=0.8)
    ax2.legend(fontsize=11, loc='upper right')
    
    # Add metadata text
    metadata = stats['metadata']
    info_lines = [
        f"Valid Images: {metadata['num_images_valid']:,} | Black Images Filtered: {metadata['num_images_black']:,}",
        f"Total Pixels: {metadata['total_pixels']:,} | Bins: {metadata['bins']:,}",
        f"Suggested Normalization Factor: {stats.get('suggested_normalization_factor', 'N/A'):.1f}"
    ]
    info_text = '\n'.join(info_lines)
    
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Histogram plot saved to: {output_path}")
    
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compute global percentile statistics in RAW 0-65535 range (filters black patches)"
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='Directory containing image files (e.g., AOI_3_Paris_Train_SN2/PAN)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='global_stats.yaml',
        help='Output YAML file to save statistics (default: global_stats.yaml)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.tif',
        help='Glob pattern for image files (default: *.tif)'
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=65536,
        help='Number of histogram bins (default: 65536 for full 16-bit range)'
    )
    parser.add_argument(
        '--percentiles',
        type=int,
        nargs='+',
        default=[2, 98],
        help='Percentiles to compute (default: 2 98)'
    )
    parser.add_argument(
        '--no-histogram',
        action='store_true',
        help='Skip histogram visualization generation'
    )
    parser.add_argument(
        '--histogram-output',
        type=str,
        default=None,
        help='Path to save histogram plot (default: same as output with .png extension)'
    )
    parser.add_argument(
        '--histogram-data',
        type=str,
        default=None,
        help='Path to save histogram data NPZ (default: same as output with _hist.npz extension)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    
    # Determine histogram output paths
    if args.histogram_output:
        histogram_path = Path(args.histogram_output)
    else:
        histogram_path = output_path.with_suffix('.png')
    
    if args.histogram_data:
        histogram_data_path = Path(args.histogram_data)
    else:
        histogram_data_path = output_path.with_name(output_path.stem + '_hist.npz')
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Compute statistics
    try:
        stats, hist, bin_edges = compute_global_percentiles(
            input_dir=input_dir,
            percentiles=args.percentiles,
            bins=args.bins,
            pattern=args.pattern,
            save_histogram=not args.no_histogram,
            histogram_output=histogram_path,
            histogram_data_output=histogram_data_path
        )
        
        # Save to YAML
        print()
        print("="*70)
        print("Saving statistics...")
        print("="*70)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
        
        print(f"Statistics saved to: {output_path.absolute()}")
        print()
        print("Summary:")
        for key, value in stats.items():
            if key != 'metadata':
                print(f"  {key}: {value:.6f}")
        
        print()
        print("="*70)
        print("Done!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
