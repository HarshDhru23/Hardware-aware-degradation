#!/usr/bin/env python3
"""
Generate and save complete training-ready dataset to disk.

This script generates the full dataset following the exact same rules as DegradationDataset:
- 8-way augmentation (4 rotations × 2 flips)
- Global percentile normalization
- Multi-frame LR generation with proper shifts
- PSF kernels and parameters
- All metadata preserved

The generated dataset can be directly loaded for training without on-the-fly generation.

Usage:
    python generate_training_dataset.py \
        --input_dir data/input \
        --output_dir data/training_dataset \
        --config configs/default_config.yaml \
        --global_stats configs/global_stats.yaml
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
import json

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


def save_sample(sample: Dict, output_path: Path, format: str = 'npz') -> None:
    """
    Save a single sample to disk.
    
    Args:
        sample: Sample dictionary from DegradationDataset
        output_path: Path to save the sample (without extension)
        format: Save format ('npz' or 'pt')
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'npz':
        # Convert torch tensors to numpy for npz format
        save_dict = {
            'hr': sample['hr'].cpu().numpy(),
            'lr_frames': np.array([lr.cpu().numpy() for lr in sample['lr']]),
            'flow_vectors': sample['flow_vectors'].cpu().numpy(),
            'shift_values': np.array(sample['shift_values']),
        }
        
        # Add PSF kernels if they exist (some might be None)
        psf_kernels_list = []
        for kernel in sample['psf_kernels']:
            if kernel is not None:
                psf_kernels_list.append(kernel.cpu().numpy())
            else:
                psf_kernels_list.append(None)
        
        # Save PSF parameters
        save_dict['psf_sigma_x'] = np.array(sample['psf_params']['sigma_x'])
        save_dict['psf_sigma_y'] = np.array(sample['psf_params']['sigma_y'])
        save_dict['psf_theta'] = np.array(sample['psf_params']['theta'])
        
        # Save non-None PSF kernels with indices
        for i, kernel in enumerate(psf_kernels_list):
            if kernel is not None:
                save_dict[f'psf_kernel_{i}'] = kernel
        
        # Save metadata as JSON string
        save_dict['metadata'] = json.dumps(sample['metadata'])
        
        np.savez_compressed(str(output_path) + '.npz', **save_dict)
        
    elif format == 'pt':
        # Save as PyTorch file (keeps tensors as tensors)
        torch.save(sample, str(output_path) + '.pt')
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def generate_dataset(input_dir: str,
                     output_dir: str,
                     config_path: str,
                     global_stats_path: str = None,
                     file_pattern: str = '*.tif',
                     format: str = 'npz',
                     num_workers: int = 0,
                     logger: logging.Logger = None) -> Dict[str, Any]:
    """
    Generate and save the complete training dataset.
    
    Args:
        input_dir: Directory containing HR images
        output_dir: Output directory for generated dataset
        config_path: Path to degradation config
        global_stats_path: Path to global statistics (optional)
        file_pattern: Pattern for input files
        format: Save format ('npz' or 'pt')
        num_workers: Number of workers (currently not used, for future parallel processing)
        logger: Logger instance
        
    Returns:
        Statistics dictionary
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("GENERATING TRAINING-READY DATASET")
    logger.info("="*80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Global stats: {global_stats_path}")
    logger.info(f"Format: {format}")
    
    # Create dataset with augmentation enabled
    logger.info("\nInitializing DegradationDataset...")
    dataset = DegradationDataset(
        hr_image_dir=input_dir,
        config_path=config_path,
        global_stats_path=global_stats_path,
        augment=True,  # Enable 8-way augmentation
        cache_size=100,
        file_pattern=file_pattern,
        seed=42  # Fixed seed for reproducibility
    )
    
    total_samples = len(dataset)
    logger.info(f"\nDataset initialized:")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  HR images: {len(dataset.hr_files)}")
    logger.info(f"  Augmentations per image: {dataset.num_augmentations}")
    logger.info(f"  LR frames per sample: {dataset.num_lr_frames}")
    logger.info(f"  Downsampling factor: {dataset.downsampling_factor}")
    
    # Create directory structure
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    # Save dataset configuration and metadata
    dataset_info = {
        'total_samples': total_samples,
        'num_hr_images': len(dataset.hr_files),
        'num_augmentations': dataset.num_augmentations,
        'num_lr_frames': dataset.num_lr_frames,
        'downsampling_factor': dataset.downsampling_factor,
        'hr_patch_size': dataset.hr_patch_size,
        'lr_patch_size': dataset.lr_patch_size,
        'augmentation_enabled': True,
        'format': format,
        'config_path': str(config_path),
        'global_stats_path': str(global_stats_path) if global_stats_path else None,
        'file_pattern': file_pattern,
        'seed': 42,
        'hr_files': [str(f) for f in dataset.hr_files],
        'generation_date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save dataset info
    with open(output_dir / 'dataset_info.yaml', 'w') as f:
        yaml.dump(dataset_info, f, default_flow_style=False)
    
    logger.info("\n" + "="*80)
    logger.info("PROCESSING SAMPLES")
    logger.info("="*80)
    
    # Statistics
    stats = {
        'total_samples': total_samples,
        'processed': 0,
        'failed': 0,
        'total_time': 0.0,
        'errors': []
    }
    
    start_time = time.time()
    
    # Process each sample
    with tqdm(total=total_samples, desc="Generating dataset", unit="sample") as pbar:
        for idx in range(total_samples):
            try:
                # Get sample from dataset
                sample = dataset[idx]
                
                # Create output path: sample_000000.npz
                sample_path = samples_dir / f"sample_{idx:06d}"
                
                # Save sample
                save_sample(sample, sample_path, format=format)
                
                stats['processed'] += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'processed': stats['processed'],
                    'failed': stats['failed']
                })
                
            except Exception as e:
                stats['failed'] += 1
                stats['errors'].append({
                    'sample_idx': idx,
                    'error': str(e)
                })
                logger.error(f"Failed to process sample {idx}: {e}")
                pbar.update(1)
    
    stats['total_time'] = time.time() - start_time
    
    # Save processing statistics
    stats_file = output_dir / 'generation_stats.yaml'
    with open(stats_file, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("GENERATION COMPLETED")
    logger.info("="*80)
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Successfully processed: {stats['processed']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Total time: {stats['total_time']:.2f} seconds")
    logger.info(f"Average time per sample: {stats['total_time']/max(stats['processed'], 1):.3f} seconds")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dataset info saved to: {output_dir / 'dataset_info.yaml'}")
    
    if stats['errors']:
        logger.warning(f"\nErrors encountered in {len(stats['errors'])} samples")
        logger.warning("See generation_stats.yaml for details")
    
    return stats


def verify_dataset(output_dir: str, num_samples: int = 5, logger: logging.Logger = None) -> bool:
    """
    Verify generated dataset by loading random samples.
    
    Args:
        output_dir: Dataset directory
        num_samples: Number of samples to verify
        logger: Logger instance
        
    Returns:
        True if verification passed
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("VERIFYING DATASET")
    logger.info("="*80)
    
    output_dir = Path(output_dir)
    samples_dir = output_dir / "samples"
    
    # Load dataset info
    dataset_info_path = output_dir / 'dataset_info.yaml'
    if not dataset_info_path.exists():
        logger.error("Dataset info file not found")
        return False
    
    with open(dataset_info_path, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    logger.info(f"Dataset format: {dataset_info['format']}")
    logger.info(f"Total samples: {dataset_info['total_samples']}")
    
    # Verify random samples
    import random
    total_samples = dataset_info['total_samples']
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    format_ext = '.npz' if dataset_info['format'] == 'npz' else '.pt'
    
    for idx in sample_indices:
        sample_path = samples_dir / f"sample_{idx:06d}{format_ext}"
        
        if not sample_path.exists():
            logger.error(f"Sample {idx} not found: {sample_path}")
            return False
        
        try:
            if dataset_info['format'] == 'npz':
                data = np.load(sample_path, allow_pickle=True)
                hr_shape = data['hr'].shape
                num_lr = len(data['lr_frames'])
                metadata = json.loads(str(data['metadata']))
                
                logger.info(f"Sample {idx}:")
                logger.info(f"  HR shape: {hr_shape}")
                logger.info(f"  LR frames: {num_lr}")
                logger.info(f"  Filename: {metadata['filename']}")
                logger.info(f"  Augmentation: rot={metadata['rotation']}°, flip={metadata['flip']}")
                
            elif dataset_info['format'] == 'pt':
                data = torch.load(sample_path)
                hr_shape = data['hr'].shape
                num_lr = len(data['lr'])
                metadata = data['metadata']
                
                logger.info(f"Sample {idx}:")
                logger.info(f"  HR shape: {hr_shape}")
                logger.info(f"  LR frames: {num_lr}")
                logger.info(f"  Filename: {metadata['filename']}")
                logger.info(f"  Augmentation: rot={metadata['rotation']}°, flip={metadata['flip']}")
        
        except Exception as e:
            logger.error(f"Failed to load sample {idx}: {e}")
            return False
    
    logger.info("\nVerification PASSED ✓")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate training-ready dataset from DegradationDataset"
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
        help='Output directory for generated dataset'
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
        '--format',
        type=str,
        choices=['npz', 'pt'],
        default='npz',
        help='Save format: npz (compressed) or pt (PyTorch)'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify dataset after generation'
    )
    
    parser.add_argument(
        '--num_verify_samples',
        type=int,
        default=5,
        help='Number of samples to verify'
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
    
    # Generate dataset
    try:
        stats = generate_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            global_stats_path=args.global_stats,
            file_pattern=args.file_pattern,
            format=args.format,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Verify if requested
    if args.verify:
        if not verify_dataset(args.output_dir, args.num_verify_samples, logger):
            logger.error("Dataset verification failed")
            sys.exit(1)
    
    logger.info("\n" + "="*80)
    logger.info("DATASET GENERATION SUCCESSFUL")
    logger.info("="*80)
    logger.info(f"\nYou can now use this dataset for training by loading from:")
    logger.info(f"  {Path(args.output_dir).absolute()}")
    logger.info(f"\nUse the PreGeneratedDataset class to load this dataset.")


if __name__ == "__main__":
    main()
