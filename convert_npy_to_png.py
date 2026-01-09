#!/usr/bin/env python3
"""
Convert saved .npy arrays to .png images.

This script reads the .npy files from test_output and converts them to PNG images
without any matplotlib plotting that might introduce aliasing.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse


def convert_npy_to_png(npy_path, output_path, bit_depth=16, auto_normalize=True):
    """
    Convert a .npy file to PNG image.
    
    Args:
        npy_path: Path to .npy file
        output_path: Path to save PNG file
        bit_depth: 8 or 16 bit output (default: 16 to preserve 11-bit quantization)
        auto_normalize: If True, normalize each image to use full dynamic range (like matplotlib)
    """
    # Load the numpy array
    image = np.load(npy_path)
    
    print(f"Loading: {npy_path.name}")
    print(f"  Shape: {image.shape}")
    print(f"  Range: [{image.min():.6f}, {image.max():.6f}]")
    print(f"  Dtype: {image.dtype}")
    
    # Handle different array shapes
    if len(image.shape) == 3:
        # Multi-channel image
        if image.shape[2] == 1:
            # Single channel stored as (H, W, 1) - squeeze to (H, W)
            image = image.squeeze(axis=2)
            print(f"  Squeezed to shape: {image.shape}")
        elif image.shape[0] < min(image.shape[1], image.shape[2]):
            # Channels first (C, H, W) - take first channel
            image = image[0]
            print(f"  Extracted first channel, shape: {image.shape}")
        else:
            # Channels last (H, W, C) - take first channel
            image = image[:, :, 0]
            print(f"  Extracted first channel, shape: {image.shape}")
    
    # Ensure 2D grayscale image
    if len(image.shape) != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")
    
    # Auto-normalize to full range if requested (like matplotlib does)
    if auto_normalize:
        # Use percentile-based normalization like matplotlib (2% and 98% percentiles)
        # This clips extreme values and provides better contrast
        min_val = np.percentile(image, 2)
        max_val = np.percentile(image, 98)
        if max_val > min_val:
            image = np.clip((image - min_val) / (max_val - min_val), 0, 1)
            print(f"  Normalized using percentiles: [{min_val:.6f}, {max_val:.6f}] -> [0, 1]")
        else:
            print(f"  Warning: min_val >= max_val, using raw values")
    
    # Convert to appropriate bit depth
    if bit_depth == 16:
        # Convert [0,1] to [0, 65535] for full 16-bit range
        # This ensures proper display in image viewers
        image_scaled = np.clip(image * 65535, 0, 65535).astype(np.uint16)
        print(f"  Saving as 16-bit PNG (full range for proper display)")
        # PIL fromarray with uint16 automatically creates mode 'I' (32-bit in memory)
        # but will be saved correctly as 16-bit PNG
        img = Image.fromarray(image_scaled)
    else:
        # Convert [0,1] to [0, 255] for 8-bit
        image_scaled = np.clip(image * 255, 0, 255).astype(np.uint8)
        print(f"  Saving as 8-bit PNG")
        img = Image.fromarray(image_scaled)
    
    img.save(output_path)
    print(f"  Saved: {output_path.name}\n")


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description="Convert .npy arrays to PNG images"
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default='test_output',
        help='Directory containing .npy files (default: test_output)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='test_output/png',
        help='Directory to save PNG files (default: test_output/png)'
    )
    parser.add_argument(
        '--bit-depth',
        type=int,
        choices=[8, 16],
        default=16,
        help='Output bit depth: 8 or 16 (default: 16 to preserve 11-bit quantization)'
    )
    parser.add_argument(
        '--no-auto-normalize',
        action='store_true',
        help='Disable auto-normalization (by default, images are normalized to full range like matplotlib)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Converting .npy files to PNG")
    print("="*70)
    print(f"Input directory: {input_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Bit depth: {args.bit_depth}-bit\n")
    
    # Find all .npy files
    npy_files = list(input_dir.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return
    
    print(f"Found {len(npy_files)} .npy files\n")
    
    # Convert each file
    for npy_file in sorted(npy_files):
        output_file = output_dir / npy_file.name.replace('.npy', '.png')
        try:
            convert_npy_to_png(npy_file, output_file, bit_depth=args.bit_depth, 
                             auto_normalize=not args.no_auto_normalize)
        except Exception as e:
            print(f"ERROR converting {npy_file.name}: {e}\n")
    
    print("="*70)
    print("Conversion complete!")
    print("="*70)
    print(f"\nPNG files saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
