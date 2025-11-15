"""
Visualization utilities for degradation pipeline results.

Provides functions to visualize the effects of the degradation pipeline
and compare HR, LR1, and LR2 images.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def visualize_degradation_results(hr_image: np.ndarray,
                                 lr1_image: np.ndarray, 
                                 lr2_image: np.ndarray,
                                 title: str = "Degradation Pipeline Results",
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Visualize HR image and generated LR1, LR2 images side by side.
    
    Args:
        hr_image: High-resolution image (H, W)
        lr1_image: Low-resolution image 1 (H/factor, W/factor)
        lr2_image: Low-resolution image 2 (H/factor, W/factor)
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("Matplotlib not available, skipping visualization")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Display HR image
    axes[0].imshow(hr_image, cmap='gray', vmin=0, vmax=1 if hr_image.max() <= 1 else 255)
    axes[0].set_title(f'HR Image\n{hr_image.shape}')
    axes[0].axis('off')
    
    # Display LR1 image
    axes[1].imshow(lr1_image, cmap='gray', vmin=0, vmax=1 if lr1_image.max() <= 1 else 255)
    axes[1].set_title(f'LR1 (P1 Sensor)\n{lr1_image.shape}')
    axes[1].axis('off')
    
    # Display LR2 image
    axes[2].imshow(lr2_image, cmap='gray', vmin=0, vmax=1 if lr2_image.max() <= 1 else 255)
    axes[2].set_title(f'LR2 (P2 Sensor)\n{lr2_image.shape}')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Visualization saved to {save_path}")
    
    plt.show()


def plot_degradation_comparison(original: np.ndarray,
                               degraded_images: Dict[str, np.ndarray],
                               region: Optional[Tuple[int, int, int, int]] = None,
                               save_path: Optional[str] = None) -> None:
    """
    Compare original HR image with degraded versions in detail.
    
    Args:
        original: Original HR image
        degraded_images: Dict of degraded images {name: image}
        region: Optional region to zoom into (y1, x1, y2, x2)
        save_path: Optional path to save the figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("Matplotlib not available, skipping comparison plot")
        return
    
    n_images = len(degraded_images) + 1
    fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    # Plot full images in top row
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original HR')
    axes[0, 0].axis('off')
    
    for i, (name, img) in enumerate(degraded_images.items(), 1):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(name)
        axes[0, i].axis('off')
    
    # Plot zoomed regions in bottom row
    if region is not None:
        y1, x1, y2, x2 = region
        
        # Add rectangle to show zoom region
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
        axes[0, 0].add_patch(rect)
        
        # Show zoomed region
        axes[1, 0].imshow(original[y1:y2, x1:x2], cmap='gray')
        axes[1, 0].set_title('Original (Zoomed)')
        axes[1, 0].axis('off')
        
        for i, (name, img) in enumerate(degraded_images.items(), 1):
            # Calculate corresponding region in LR image
            factor = original.shape[0] // img.shape[0]
            lr_y1, lr_x1 = y1 // factor, x1 // factor
            lr_y2, lr_x2 = y2 // factor, x2 // factor
            
            axes[1, i].imshow(img[lr_y1:lr_y2, lr_x1:lr_x2], cmap='gray')
            axes[1, i].set_title(f'{name} (Zoomed)')
            axes[1, i].axis('off')
    else:
        # If no region specified, hide bottom row
        for i in range(n_images):
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Comparison plot saved to {save_path}")
    
    plt.show()


def plot_noise_analysis(clean_image: np.ndarray,
                       noisy_image: np.ndarray,
                       title: str = "Noise Analysis",
                       save_path: Optional[str] = None) -> None:
    """
    Analyze and visualize noise characteristics.
    
    Args:
        clean_image: Clean image before noise
        noisy_image: Image after noise addition
        title: Plot title
        save_path: Optional path to save the figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("Matplotlib not available, skipping noise analysis")
        return
    
    # Calculate noise
    noise = noisy_image.astype(np.float32) - clean_image.astype(np.float32)
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # Original images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(clean_image, cmap='gray')
    ax1.set_title('Clean Image')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(noisy_image, cmap='gray')
    ax2.set_title('Noisy Image')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(noise, cmap='RdBu_r', vmin=-3*noise.std(), vmax=3*noise.std())
    ax3.set_title('Noise')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, shrink=0.8)
    
    # Noise statistics
    ax4 = fig.add_subplot(gs[1, :])
    ax4.hist(noise.flatten(), bins=100, alpha=0.7, density=True)
    ax4.axvline(noise.mean(), color='red', linestyle='--', label=f'Mean: {noise.mean():.3f}')
    ax4.axvline(noise.std(), color='orange', linestyle='--', label=f'Std: {noise.std():.3f}')
    ax4.axvline(-noise.std(), color='orange', linestyle='--')
    ax4.set_title('Noise Distribution')
    ax4.set_xlabel('Noise Value')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # SNR analysis
    ax5 = fig.add_subplot(gs[2, 0])
    signal_power = np.mean(clean_image.astype(np.float32) ** 2)
    noise_power = np.mean(noise ** 2)
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    ax5.text(0.5, 0.5, f'SNR: {snr_db:.2f} dB\nSignal Power: {signal_power:.2f}\nNoise Power: {noise_power:.2f}',
             transform=ax5.transAxes, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax5.set_title('SNR Analysis')
    ax5.axis('off')
    
    # Noise statistics by region
    ax6 = fig.add_subplot(gs[2, 1:])
    if clean_image.size > 10000:  # Only for larger images
        # Divide image into regions and analyze noise
        h, w = noise.shape
        region_size = min(h, w) // 4
        regions_snr = []
        
        for i in range(0, h-region_size, region_size):
            for j in range(0, w-region_size, region_size):
                region_clean = clean_image[i:i+region_size, j:j+region_size]
                region_noise = noise[i:i+region_size, j:j+region_size]
                
                region_signal_power = np.mean(region_clean.astype(np.float32) ** 2)
                region_noise_power = np.mean(region_noise ** 2)
                
                if region_noise_power > 0:
                    region_snr = 10 * np.log10(region_signal_power / region_noise_power)
                    regions_snr.append(region_snr)
        
        if regions_snr:
            ax6.hist(regions_snr, bins=20, alpha=0.7)
            ax6.axvline(np.mean(regions_snr), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(regions_snr):.2f} dB')
            ax6.set_title('SNR Distribution Across Regions')
            ax6.set_xlabel('SNR (dB)')
            ax6.set_ylabel('Count')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Insufficient regions for analysis', 
                    transform=ax6.transAxes, ha='center', va='center')
            ax6.axis('off')
    else:
        ax6.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Noise analysis saved to {save_path}")
    
    plt.show()


def plot_blur_analysis(original: np.ndarray,
                      blurred: np.ndarray,
                      title: str = "Blur Analysis",
                      save_path: Optional[str] = None) -> None:
    """
    Analyze and visualize blur effects.
    
    Args:
        original: Original image before blurring
        blurred: Image after blurring
        title: Plot title
        save_path: Optional path to save the figure
    """
    if not MATPLOTLIB_AVAILABLE:
        logging.warning("Matplotlib not available, skipping blur analysis")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Images
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title('Blurred')
    axes[0, 1].axis('off')
    
    # Difference
    diff = original.astype(np.float32) - blurred.astype(np.float32)
    im = axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-diff.std()*3, vmax=diff.std()*3)
    axes[0, 2].set_title('Difference')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], shrink=0.8)
    
    # Edge analysis using gradients
    def compute_edge_strength(img):
        gy, gx = np.gradient(img.astype(np.float32))
        return np.sqrt(gx**2 + gy**2)
    
    edge_orig = compute_edge_strength(original)
    edge_blur = compute_edge_strength(blurred)
    
    axes[1, 0].imshow(edge_orig, cmap='hot')
    axes[1, 0].set_title('Original Edges')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edge_blur, cmap='hot')
    axes[1, 1].set_title('Blurred Edges')
    axes[1, 1].axis('off')
    
    # Edge strength comparison
    axes[1, 2].hist(edge_orig.flatten(), bins=50, alpha=0.5, label='Original', density=True)
    axes[1, 2].hist(edge_blur.flatten(), bins=50, alpha=0.5, label='Blurred', density=True)
    axes[1, 2].set_title('Edge Strength Distribution')
    axes[1, 2].set_xlabel('Edge Strength')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Blur analysis saved to {save_path}")
    
    plt.show()