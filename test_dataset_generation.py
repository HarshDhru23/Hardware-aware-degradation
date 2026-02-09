#!/usr/bin/env python3
"""
Test dataset generation script - generates limited dataset with PNG visualization.

This script:
1. Processes only the first N images (default: 10)
2. Generates all 8 augmentations and 4 LR frames per augmentation
3. Saves both NPZ (with metadata) and PNG (for visual inspection)
4. Creates organized directory structure for easy verification

Usage:
    python test_dataset_generation.py \
        --input_dir data/input \
        --output_dir data/test_dataset \
        --config configs/default_config.yaml \
        --global_stats combined_stats.yaml \
        --num_images 10
"""

import argparse
import sys
import logging
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
import time
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import DegradationDataset


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def save_as_png(image_array: np.ndarray, output_path: Path, normalize: bool = True) -> None:
    """
    Save numpy array as PNG image.
    
    Args:
        image_array: Image array with shape [1, H, W] or [H, W]
        output_path: Path to save PNG (with .png extension)
        normalize: Whether to normalize to [0, 255] range
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for PNG conversion. Install with: pip install Pillow")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove channel dimension if present
    if image_array.ndim == 3 and image_array.shape[0] == 1:
        image_array = image_array[0]
    
    # Normalize to [0, 255] if needed
    if normalize:
        # Assume input is in [0, 1] range
        image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)
    else:
        # Assume input is already in appropriate range
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    
    # Save as PNG
    img = Image.fromarray(image_array, mode='L')  # 'L' for grayscale
    img.save(output_path)


def get_base_filename(filepath: Path) -> str:
    """Extract base filename without extension."""
    return filepath.stem


def save_hr_sample(sample: Dict, output_path: Path, save_png: bool = True) -> None:
    """
    Save HR augmented image with metadata as NPZ and optionally PNG.
    
    Args:
        sample: Sample dictionary from DegradationDataset
        output_path: Path to save (without extension)
        save_png: Whether to also save as PNG
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    hr_array = sample['hr'].cpu().numpy()  # [1, H, W]
    
    # Save NPZ
    save_dict = {
        'hr': hr_array,
        'original_filename': sample['metadata']['filename'],
        'augmentation_id': sample['metadata']['aug_idx'] + 1,
        'rotation': sample['metadata']['rotation'],
        'flip': int(sample['metadata']['flip']),
        'hr_shape': list(sample['hr'].shape),
        'downsampling_factor': sample['metadata']['downsampling_factor'],
    }
    np.savez_compressed(str(output_path) + '.npz', **save_dict)
    
    # Save PNG
    if save_png:
        save_as_png(hr_array, output_path.with_suffix('.png'), normalize=True)


def save_lr_sample(sample: Dict, lr_idx: int, output_path: Path, save_png: bool = True) -> None:
    """
    Save single LR frame with full metadata as NPZ and optionally PNG.
    
    Args:
        sample: Sample dictionary from DegradationDataset
        lr_idx: Index of the LR frame (0-based)
        output_path: Path to save (without extension)
        save_png: Whether to also save as PNG
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lr_frame = sample['lr'][lr_idx]
    lr_array = lr_frame.cpu().numpy()  # [1, H, W]
    shift_value = sample['shift_values'][lr_idx]
    flow_vector = sample['flow_vectors'][lr_idx]
    
    # Save NPZ
    save_dict = {
        'lr': lr_array,
        'hr': sample['hr'].cpu().numpy(),
        'shift_x': float(shift_value[0]),
        'shift_y': float(shift_value[1]),
        'flow_vector': flow_vector.cpu().numpy(),
        'original_filename': sample['metadata']['filename'],
        'augmentation_id': sample['metadata']['aug_idx'] + 1,
        'lr_frame_id': lr_idx + 1,
        'rotation': sample['metadata']['rotation'],
        'flip': int(sample['metadata']['flip']),
        'lr_shape': list(lr_frame.shape),
        'hr_shape': list(sample['hr'].shape),
        'downsampling_factor': sample['metadata']['downsampling_factor'],
    }
    
    # Add PSF parameters
    save_dict['psf_sigma_x'] = float(sample['psf_params']['sigma_x'][lr_idx])
    save_dict['psf_sigma_y'] = float(sample['psf_params']['sigma_y'][lr_idx])
    save_dict['psf_theta'] = float(sample['psf_params']['theta'][lr_idx])
    
    # Add PSF kernel if available
    psf_kernel = sample['psf_kernels'][lr_idx]
    if psf_kernel is not None:
        save_dict['psf_kernel'] = psf_kernel.cpu().numpy()
    
    np.savez_compressed(str(output_path) + '.npz', **save_dict)
    
    # Save PNG
    if save_png:
        save_as_png(lr_array, output_path.with_suffix('.png'), normalize=True)


def generate_test_dataset(input_dir: str,
                          output_dir: str,
                          config_path: str,
                          global_stats_path: str = None,
                          file_pattern: str = '*.tif',
                          num_images: int = 10,
                          copy_originals: bool = True,
                          logger: logging.Logger = None) -> Dict[str, Any]:
    """
    Generate test dataset with limited number of images.
    
    Args:
        input_dir: Directory containing HR images
        output_dir: Output directory for test dataset
        config_path: Path to degradation config
        global_stats_path: Path to global statistics (optional)
        file_pattern: Pattern for input files
        num_images: Number of input images to process (default: 10)
        copy_originals: Whether to copy original .tif files
        logger: Logger instance
        
    Returns:
        Statistics dictionary
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("GENERATING TEST DATASET WITH PNG VISUALIZATION")
    logger.info("="*80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Global stats: {global_stats_path}")
    logger.info(f"Number of images to process: {num_images}")
    logger.info(f"Copy originals: {copy_originals}")
    
    # Create dataset with augmentation enabled
    logger.info("\nInitializing DegradationDataset...")
    dataset = DegradationDataset(
        hr_image_dir=input_dir,
        config_path=config_path,
        global_stats_path=global_stats_path,
        augment=True,
        cache_size=100,
        file_pattern=file_pattern,
        seed=42
    )
    
    # Limit to first N images
    num_available_images = len(dataset.hr_files)
    num_images_to_process = min(num_images, num_available_images)
    
    num_augmentations = dataset.num_augmentations
    num_lr_frames = dataset.num_lr_frames
    total_samples = num_images_to_process * num_augmentations
    
    logger.info(f"\nDataset initialized:")
    logger.info(f"  Available HR images: {num_available_images}")
    logger.info(f"  Images to process: {num_images_to_process}")
    logger.info(f"  Augmentations per image: {num_augmentations}")
    logger.info(f"  Total samples to generate: {total_samples}")
    logger.info(f"  LR frames per sample: {num_lr_frames}")
    logger.info(f"  Downsampling factor: {dataset.downsampling_factor}")
    
    # Verify expected values
    if num_augmentations != 8:
        logger.warning(f"Expected 8 augmentations but got {num_augmentations}")
    if num_lr_frames != 4:
        logger.warning(f"Expected 4 LR frames but got {num_lr_frames}")
    
    # Copy original HR images (limited set)
    if copy_originals:
        logger.info("\n" + "="*80)
        logger.info("COPYING ORIGINAL HR IMAGES")
        logger.info("="*80)
        
        for i, hr_file in enumerate(dataset.hr_files[:num_images_to_process]):
            output_file = output_dir / hr_file.name
            if not output_file.exists():
                shutil.copy2(hr_file, output_file)
                logger.debug(f"Copied: {hr_file.name}")
    
    # Save dataset configuration
    dataset_info = {
        'mode': 'test',
        'total_samples': total_samples,
        'num_hr_images': num_images_to_process,
        'num_augmentations': num_augmentations,
        'num_lr_frames': num_lr_frames,
        'downsampling_factor': dataset.downsampling_factor,
        'hr_patch_size': dataset.hr_patch_size,
        'lr_patch_size': dataset.lr_patch_size,
        'augmentation_enabled': True,
        'structure': 'flat',
        'formats': ['npz', 'png'],
        'naming_format': {
            'original': '{basename}.tif',
            'hr_augmented': '{basename}_aug{N}.npz/.png',
            'lr_frames': '{basename}_aug{N}_lr{M}.npz/.png'
        },
        'config_path': str(config_path),
        'global_stats_path': str(global_stats_path) if global_stats_path else None,
        'file_pattern': file_pattern,
        'seed': 42,
        'hr_files': [str(dataset.hr_files[i]) for i in range(num_images_to_process)],
        'generation_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_dir / 'dataset_info.yaml', 'w') as f:
        yaml.dump(dataset_info, f, default_flow_style=False)
    
    logger.info("\n" + "="*80)
    logger.info("PROCESSING SAMPLES")
    logger.info("="*80)
    logger.info(f"Generating {num_augmentations} HR files per image (NPZ + PNG)")
    logger.info(f"Generating {num_lr_frames} LR files per augmentation (NPZ + PNG)")
    logger.info(f"Total files to generate per image:")
    logger.info(f"  - HR: {num_augmentations} × 2 formats = {num_augmentations * 2} files")
    logger.info(f"  - LR: {num_augmentations * num_lr_frames} × 2 formats = {num_augmentations * num_lr_frames * 2} files")
    logger.info(f"  - Total per image: {(num_augmentations + num_augmentations * num_lr_frames) * 2} files")
    
    # Statistics
    stats = {
        'total_samples': total_samples,
        'processed_samples': 0,
        'hr_files_created': 0,
        'lr_files_created': 0,
        'hr_pngs_created': 0,
        'lr_pngs_created': 0,
        'failed': 0,
        'total_time': 0.0,
        'errors': []
    }
    
    start_time = time.time()
    
    # Process limited samples
    with tqdm(total=total_samples, desc="Generating test dataset", unit="sample") as pbar:
        for idx in range(total_samples):
            try:
                # Get sample from dataset
                sample = dataset[idx]
                
                # Extract info for naming
                file_idx = idx // num_augmentations
                aug_idx = idx % num_augmentations
                hr_file = dataset.hr_files[file_idx]
                base_name = get_base_filename(hr_file)
                
                # Save HR: {basename}_aug{N}
                hr_filename = f"{base_name}_aug{aug_idx + 1}"
                hr_path = output_dir / hr_filename
                save_hr_sample(sample, hr_path, save_png=True)
                stats['hr_files_created'] += 1
                stats['hr_pngs_created'] += 1
                
                # Save each LR frame: {basename}_aug{N}_lr{M}
                for lr_idx in range(num_lr_frames):
                    lr_filename = f"{base_name}_aug{aug_idx + 1}_lr{lr_idx + 1}"
                    lr_path = output_dir / lr_filename
                    save_lr_sample(sample, lr_idx, lr_path, save_png=True)
                    stats['lr_files_created'] += 1
                    stats['lr_pngs_created'] += 1
                
                stats['processed_samples'] += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'hr_npz': stats['hr_files_created'],
                    'lr_npz': stats['lr_files_created'],
                    'hr_png': stats['hr_pngs_created'],
                    'lr_png': stats['lr_pngs_created']
                })
                
            except Exception as e:
                stats['failed'] += 1
                stats['errors'].append({
                    'sample_idx': idx,
                    'error': str(e)
                })
                logger.error(f"Failed to process sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                pbar.update(1)
    
    stats['total_time'] = time.time() - start_time
    
    # Save statistics
    with open(output_dir / 'generation_stats.yaml', 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("GENERATION COMPLETED")
    logger.info("="*80)
    logger.info(f"Processed images: {num_images_to_process}/{num_available_images}")
    logger.info(f"HR files (NPZ): {stats['hr_files_created']}")
    logger.info(f"HR files (PNG): {stats['hr_pngs_created']}")
    logger.info(f"LR files (NPZ): {stats['lr_files_created']}")
    logger.info(f"LR files (PNG): {stats['lr_pngs_created']}")
    logger.info(f"Total files: {stats['hr_files_created'] + stats['lr_files_created'] + stats['hr_pngs_created'] + stats['lr_pngs_created']}")
    logger.info(f"Failed samples: {stats['failed']}")
    logger.info(f"Total time: {stats['total_time']:.2f} seconds")
    logger.info(f"Average time per sample: {stats['total_time']/max(stats['processed_samples'], 1):.3f} seconds")
    
    # Show structure example
    if num_images_to_process > 0:
        example_base = get_base_filename(dataset.hr_files[0])
        logger.info("\n" + "="*80)
        logger.info("OUTPUT STRUCTURE")
        logger.info("="*80)
        logger.info(f"Original HR:   {example_base}.tif")
        logger.info(f"")
        logger.info(f"Augmented HR:  {example_base}_aug1.npz + {example_base}_aug1.png")
        logger.info(f"               {example_base}_aug2.npz + {example_base}_aug2.png")
        logger.info(f"               ... (up to aug8)")
        logger.info(f"")
        logger.info(f"LR frames:     {example_base}_aug1_lr1.npz + {example_base}_aug1_lr1.png")
        logger.info(f"               {example_base}_aug1_lr2.npz + {example_base}_aug1_lr2.png")
        logger.info(f"               {example_base}_aug1_lr3.npz + {example_base}_aug1_lr3.png")
        logger.info(f"               {example_base}_aug1_lr4.npz + {example_base}_aug1_lr4.png")
        logger.info(f"               ... (same for aug2-aug8)")
        logger.info(f"")
        logger.info(f"✓ You can now open the PNG files to visually verify the degradation!")
    
    logger.info(f"\nOutput directory: {output_dir}")
    
    if stats['errors']:
        logger.warning(f"\nErrors encountered in {len(stats['errors'])} samples")
        logger.warning("See generation_stats.yaml for details")
    
    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate test dataset with PNG visualization"
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing HR GeoTIFF images'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for test dataset'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to degradation configuration file'
    )
    
    parser.add_argument(
        '--global_stats',
        type=str,
        default=None,
        help='Path to global statistics YAML (optional but recommended)'
    )
    
    parser.add_argument(
        '--file_pattern',
        type=str,
        default='*.tif',
        help='File pattern for input images (default: *.tif)'
    )
    
    parser.add_argument(
        '--num_images',
        type=int,
        default=10,
        help='Number of input images to process (default: 10)'
    )
    
    parser.add_argument(
        '--no_copy_originals',
        action='store_true',
        help='Do not copy original .tif files to output directory'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Validate paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file does not exist: {config_path}")
        sys.exit(1)
    
    if args.global_stats:
        global_stats_path = Path(args.global_stats)
        if not global_stats_path.exists():
            logger.warning(f"Global stats file does not exist: {global_stats_path}")
            logger.warning("Proceeding without global statistics")
            args.global_stats = None
    
    # Check for Pillow
    try:
        import PIL
    except ImportError:
        logger.error("Pillow is required for PNG generation. Install with: pip install Pillow")
        sys.exit(1)
    
    # Generate test dataset
    try:
        stats = generate_test_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            global_stats_path=args.global_stats,
            file_pattern=args.file_pattern,
            num_images=args.num_images,
            copy_originals=not args.no_copy_originals,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Test dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("\n" + "="*80)
    logger.info("TEST DATASET GENERATION SUCCESSFUL")
    logger.info("="*80)
    logger.info(f"\n✓ Dataset saved to: {Path(args.output_dir).absolute()}")
    logger.info(f"✓ Open the .png files to visually verify the degradation quality")
    logger.info(f"✓ Check different augmentations and LR frames")
    logger.info(f"\nQuick verification tips:")
    logger.info(f"  1. Compare HR augmented images to check rotation/flip")
    logger.info(f"  2. Compare LR frames to verify blur and downsampling")
    logger.info(f"  3. Check if noise is visible in LR frames")
    logger.info(f"  4. Verify that LR frames have slight variations (sub-pixel shifts)")


if __name__ == "__main__":
    main()
