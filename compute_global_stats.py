#!/usr/bin/env python3
"""
Compute global percentile statistics across an entire dataset.

This script computes global 2nd and 98th percentiles across all images in a directory
using the exact same loading method as the degradation pipeline (GeoTIFFLoader).
Uses histogram-based method for memory efficiency on large datasets.

Usage:
    python compute_global_stats.py --input-dir AOI_3_Paris_Train_SN2/PAN --output global_stats_spacenet2.yaml
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
    histogram_output: Path = None
):
    """
    Compute global percentiles across all images in a directory using histogram method.
    
    Uses the exact same GeoTIFFLoader as the degradation pipeline to ensure consistency.
    
    Args:
        input_dir: Directory containing images
        percentiles: List of percentiles to compute (e.g., [2, 98])
        bins: Number of histogram bins (default: 65536 for high precision)
        pattern: Glob pattern for image files
        save_histogram: Whether to save histogram visualization
        histogram_output: Path to save histogram image (default: same as yaml with .png)
        
    Returns:
        Tuple of (statistics_dict, histogram, bin_edges)
    """
    print("="*70)
    print("Computing Global Percentile Statistics")
    print("="*70)
    print(f"Input directory: {input_dir.absolute()}")
    print(f"File pattern: {pattern}")
    print(f"Histogram bins: {bins}")
    print(f"Percentiles: {percentiles}")
    print()
    
    # Find all image files
    image_files = sorted(input_dir.glob(pattern))
    
    if not image_files:
        raise ValueError(f"No files matching '{pattern}' found in {input_dir}")
    
    print(f"Found {len(image_files)} image files")
    print()
    
    # Initialize GeoTIFFLoader with the exact same settings as pipeline
    loader = GeoTIFFLoader(normalize=True, target_dtype='float32')
    
    # Initialize histogram
    # Range [0, 1] because loader normalizes by dividing by 2047
    hist = np.zeros(bins, dtype=np.int64)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    
    print("Building cumulative histogram across all images...")
    total_pixels = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image using exact same method as pipeline
            image = loader.load_image(img_path)
            
            # Build histogram for this image
            h, _ = np.histogram(image.ravel(), bins=bin_edges)
            hist += h
            total_pixels += image.size
            
        except Exception as e:
            print(f"\nWarning: Failed to process {img_path.name}: {e}")
            continue
    
    print(f"\nTotal pixels processed: {total_pixels:,}")
    print()
    
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
        print(f"  {p}th percentile: {percentile_value:.6f}")
    
    # Add metadata
    results['metadata'] = {
        'num_images': len(image_files),
        'total_pixels': int(total_pixels),
        'bins': bins,
        'normalization': 'divide_by_2047',
        'loader': 'GeoTIFFLoader',
        'input_dir': str(input_dir.absolute())
    }
    
    # Generate histogram visualization if requested
    if save_histogram:
        print()
        print("Generating histogram visualization...")
        plot_histogram(hist, bin_edges, results, percentiles, histogram_output)
    
    return results, hist, bin_edges


def plot_histogram(hist, bin_edges, stats, percentiles, output_path=None):
    """
    Generate and save histogram visualization with percentile markers.
    
    Args:
        hist: Histogram counts
        bin_edges: Bin edge values
        stats: Statistics dictionary with percentile values
        percentiles: List of percentiles that were computed
        output_path: Path to save the plot (if None, derived from stats file)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot 1: Linear scale histogram
    ax1.bar(bin_centers, hist, width=np.diff(bin_edges), 
            edgecolor='none', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Normalized Pixel Value [0, 1]', fontsize=12)
    ax1.set_ylabel('Pixel Count', fontsize=12)
    ax1.set_title('Global Histogram - Linear Scale', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add percentile lines
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for i, p in enumerate(percentiles):
        p_val = stats[f'p{p}']
        color = colors[i % len(colors)]
        ax1.axvline(p_val, color=color, linestyle='--', linewidth=2, 
                   label=f'{p}th percentile: {p_val:.4f}')
    ax1.legend(fontsize=10)
    
    # Plot 2: Log scale histogram (better for seeing distribution)
    ax2.bar(bin_centers, hist, width=np.diff(bin_edges), 
            edgecolor='none', alpha=0.7, color='steelblue')
    ax2.set_xlabel('Normalized Pixel Value [0, 1]', fontsize=12)
    ax2.set_ylabel('Pixel Count (log scale)', fontsize=12)
    ax2.set_title('Global Histogram - Log Scale', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add percentile lines
    for i, p in enumerate(percentiles):
        p_val = stats[f'p{p}']
        color = colors[i % len(colors)]
        ax2.axvline(p_val, color=color, linestyle='--', linewidth=2, 
                   label=f'{p}th percentile: {p_val:.4f}')
    ax2.legend(fontsize=10)
    
    # Add metadata text
    metadata = stats['metadata']
    info_text = f"Images: {metadata['num_images']:,} | Pixels: {metadata['total_pixels']:,} | Bins: {metadata['bins']:,}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Histogram saved to: {output_path}")
    
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compute global percentile statistics across dataset using exact pipeline loading method"
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
        help='Number of histogram bins for precision (default: 65536)'
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
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    
    # Determine histogram output path
    if args.histogram_output:
        histogram_path = Path(args.histogram_output)
    else:
        histogram_path = output_path.with_suffix('.png')
    
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
            histogram_output=histogram_path
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
