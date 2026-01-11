#!/usr/bin/env python3
"""
Combine histograms from multiple folders and plot them on the same axis.

This script:
1. Loads histogram NPZ files from multiple folders (paris_sn2, vegas_sn2, etc.)
2. Plots all histograms on the same 0-65535 axis for comparison
3. Computes combined percentiles across all datasets
4. Helps decide the final normalization factor for the entire dataset

Usage:
    python combine_histograms.py --histograms stats_paris_sn2_hist.npz stats_vegas_sn2_hist.npz --output combined_stats.yaml
"""

import numpy as np
from pathlib import Path
import argparse
import yaml
import sys
import matplotlib.pyplot as plt
from typing import List, Tuple


def load_histogram_data(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load histogram data from NPZ file.
    
    Args:
        npz_path: Path to NPZ file saved by compute_global_stats.py
        
    Returns:
        Tuple of (histogram, bin_edges, metadata)
    """
    data = np.load(npz_path, allow_pickle=True)
    histogram = data['histogram']
    bin_edges = data['bin_edges']
    metadata = data['metadata'].item() if 'metadata' in data else {}
    return histogram, bin_edges, metadata


def combine_histograms(histogram_files: List[Path]) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Combine multiple histograms into one.
    
    Args:
        histogram_files: List of NPZ file paths
        
    Returns:
        Tuple of (combined_histogram, bin_edges, metadata_list)
    """
    print("="*70)
    print("Combining Histograms from Multiple Folders")
    print("="*70)
    print(f"Number of histogram files: {len(histogram_files)}")
    print()
    
    combined_hist = None
    bin_edges = None
    metadata_list = []
    
    for i, hist_file in enumerate(histogram_files):
        print(f"Loading {i+1}/{len(histogram_files)}: {hist_file.name}")
        hist, edges, meta = load_histogram_data(hist_file)
        
        if combined_hist is None:
            combined_hist = hist.copy()
            bin_edges = edges
            print(f"  Initialized with {len(hist)} bins, range [{edges[0]:.0f}, {edges[-1]:.0f}]")
        else:
            # Verify bins match
            if len(hist) != len(combined_hist):
                raise ValueError(f"Histogram bin count mismatch: {len(hist)} vs {len(combined_hist)}")
            if not np.allclose(edges, bin_edges):
                raise ValueError(f"Histogram bin edges mismatch")
            
            combined_hist += hist
            print(f"  Added histogram with {meta.get('total_pixels', 'unknown')} pixels")
        
        metadata_list.append(meta)
    
    print()
    print(f"Combined histogram total counts: {combined_hist.sum():,}")
    return combined_hist, bin_edges, metadata_list


def compute_combined_percentiles(hist: np.ndarray, bin_edges: np.ndarray, 
                                 percentiles: List[int]) -> dict:
    """
    Compute percentiles from combined histogram.
    
    Args:
        hist: Combined histogram
        bin_edges: Bin edges
        percentiles: List of percentiles to compute
        
    Returns:
        Dictionary with percentile values
    """
    print("Computing percentiles from combined histogram...")
    cumulative = np.cumsum(hist)
    total_count = cumulative[-1]
    
    results = {}
    for p in percentiles:
        target_count = (p / 100.0) * total_count
        bin_idx = np.searchsorted(cumulative, target_count)
        bin_idx = min(bin_idx, len(bin_edges) - 2)
        percentile_value = bin_edges[bin_idx]
        
        results[f'p{p}'] = float(percentile_value)
        print(f"  {p}th percentile: {percentile_value:.1f} (raw 16-bit value)")
    
    # Compute suggested normalization factor (98th percentile)
    suggested_norm = results.get('p98', 65535)
    results['suggested_normalization_factor'] = float(suggested_norm)
    print(f"\nSuggested normalization factor for ENTIRE dataset: {suggested_norm:.1f}")
    print(f"  (divide all images by this value to map to [0, 1])")
    
    return results


def plot_combined_histograms(individual_hists: List[Tuple[np.ndarray, str]], 
                            combined_hist: np.ndarray,
                            bin_edges: np.ndarray,
                            combined_stats: dict,
                            percentiles: List[int],
                            output_path: Path):
    """
    Plot individual and combined histograms on same axis.
    
    Args:
        individual_hists: List of (histogram, label) tuples
        combined_hist: Combined histogram
        bin_edges: Bin edges (0-65535 range)
        combined_stats: Statistics from combined histogram
        percentiles: List of percentiles computed
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot 1: Individual histograms overlaid (linear scale)
    ax1 = axes[0]
    for i, (hist, label) in enumerate(individual_hists):
        color = colors[i % len(colors)]
        ax1.plot(bin_centers, hist, alpha=0.6, linewidth=1.5, label=label, color=color)
    
    ax1.set_xlabel('Pixel Value (Raw 16-bit, 0-65535)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pixel Count', fontsize=12, fontweight='bold')
    ax1.set_title('Individual Histograms - Linear Scale (0-65535 Range)', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 65535)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Combined histogram (linear scale)
    ax2 = axes[1]
    ax2.bar(bin_centers, combined_hist, width=np.diff(bin_edges), 
            edgecolor='none', alpha=0.7, color='steelblue')
    ax2.set_xlabel('Pixel Value (Raw 16-bit, 0-65535)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pixel Count', fontsize=12, fontweight='bold')
    ax2.set_title('Combined Histogram - Linear Scale (0-65535 Range)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 65535)
    ax2.grid(True, alpha=0.3)
    
    # Add percentile lines
    percentile_colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']
    for i, p in enumerate(percentiles):
        p_val = combined_stats[f'p{p}']
        color = percentile_colors[i % len(percentile_colors)]
        ax2.axvline(p_val, color=color, linestyle='--', linewidth=2.5, 
                   label=f'{p}th percentile: {p_val:.1f}', alpha=0.8)
    ax2.legend(fontsize=10, loc='upper right')
    
    # Plot 3: Combined histogram (log scale)
    ax3 = axes[2]
    nonzero_mask = combined_hist > 0
    ax3.bar(bin_centers[nonzero_mask], combined_hist[nonzero_mask], 
            width=np.diff(bin_edges)[nonzero_mask],
            edgecolor='none', alpha=0.7, color='steelblue')
    ax3.set_xlabel('Pixel Value (Raw 16-bit, 0-65535)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Pixel Count (log scale)', fontsize=12, fontweight='bold')
    ax3.set_title('Combined Histogram - Log Scale (0-65535 Range)', 
                 fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 65535)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Add percentile lines
    for i, p in enumerate(percentiles):
        p_val = combined_stats[f'p{p}']
        color = percentile_colors[i % len(percentile_colors)]
        ax3.axvline(p_val, color=color, linestyle='--', linewidth=2.5, 
                   label=f'{p}th percentile: {p_val:.1f}', alpha=0.8)
    ax3.legend(fontsize=10, loc='upper right')
    
    # Add metadata text
    total_pixels = combined_hist.sum()
    info_lines = [
        f"Datasets Combined: {len(individual_hists)} | Total Pixels: {total_pixels:,}",
        f"Suggested Normalization Factor: {combined_stats.get('suggested_normalization_factor', 'N/A'):.1f}"
    ]
    info_text = '\n'.join(info_lines)
    
    fig.text(0.5, 0.01, info_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nCombined histogram plot saved to: {output_path}")
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Combine histograms from multiple folders and determine global normalization factor"
    )
    parser.add_argument(
        '--histograms', '-H',
        type=str,
        nargs='+',
        required=True,
        help='List of histogram NPZ files from compute_global_stats.py'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='combined_stats.yaml',
        help='Output YAML file for combined statistics (default: combined_stats.yaml)'
    )
    parser.add_argument(
        '--plot-output',
        type=str,
        default=None,
        help='Path to save combined histogram plot (default: same as output with .png)'
    )
    parser.add_argument(
        '--percentiles',
        type=int,
        nargs='+',
        default=[2, 98],
        help='Percentiles to compute (default: 2 98)'
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    histogram_files = [Path(h) for h in args.histograms]
    
    # Verify all files exist
    for hf in histogram_files:
        if not hf.exists():
            print(f"Error: Histogram file not found: {hf}")
            sys.exit(1)
    
    # Determine plot output path
    if args.plot_output:
        plot_path = Path(args.plot_output)
    else:
        plot_path = output_path.with_suffix('.png')
    
    try:
        # Combine histograms
        combined_hist, bin_edges, metadata_list = combine_histograms(histogram_files)
        
        # Compute combined percentiles
        print()
        combined_stats = compute_combined_percentiles(combined_hist, bin_edges, args.percentiles)
        
        # Add metadata
        combined_stats['metadata'] = {
            'num_datasets': len(histogram_files),
            'dataset_files': [str(hf) for hf in histogram_files],
            'total_pixels': int(combined_hist.sum()),
            'bins': len(combined_hist),
            'range': [0, 65535],
            'individual_datasets': metadata_list
        }
        
        # Save combined statistics
        print()
        print("="*70)
        print("Saving combined statistics...")
        print("="*70)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(combined_stats, f, default_flow_style=False, sort_keys=False)
        
        print(f"Combined statistics saved to: {output_path.absolute()}")
        
        # Plot combined histograms
        print()
        print("Generating combined histogram plots...")
        
        # Prepare individual histograms for plotting
        individual_hists = []
        for i, hf in enumerate(histogram_files):
            hist, _, meta = load_histogram_data(hf)
            label = hf.stem.replace('_hist', '')  # Remove _hist suffix for cleaner label
            individual_hists.append((hist, label))
        
        plot_combined_histograms(
            individual_hists=individual_hists,
            combined_hist=combined_hist,
            bin_edges=bin_edges,
            combined_stats=combined_stats,
            percentiles=args.percentiles,
            output_path=plot_path
        )
        
        print()
        print("="*70)
        print("Done!")
        print("="*70)
        print()
        print("Summary:")
        for key, value in combined_stats.items():
            if key != 'metadata':
                print(f"  {key}: {value:.1f}")
        print()
        print(f"RECOMMENDED: Use normalization factor = {combined_stats['suggested_normalization_factor']:.0f}")
        print(f"  Update your config to divide images by this value instead of 2047")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
