#!/usr/bin/env python3
"""
Main processing script for the Hardware-aware Degradation Pipeline.

This script processes a batch of HR GeoTIFF images from the SpaceNet dataset
and generates corresponding LR1 and LR2 patches for training the MFSR model.

Usage:
    python process_images.py --input_dir data/input --output_dir data/output --config configs/default_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import time
from tqdm import tqdm

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from degradation import DegradationPipeline
from config import ConfigManager
from utils.data_io import GeoTIFFLoader, PatchExtractor, save_image_patches
from utils.validation import validate_image, validate_file_path
from utils.visualization import visualize_degradation_results


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def process_single_image(image_path: Path,
                        pipeline: DegradationPipeline,
                        loader: GeoTIFFLoader,
                        extractor: PatchExtractor,
                        output_dir: Path,
                        config: ConfigManager,
                        logger: logging.Logger) -> Dict[str, Any]:
    """
    Process a single HR image through the degradation pipeline.
    
    Args:
        image_path: Path to HR GeoTIFF image
        pipeline: Degradation pipeline instance
        loader: GeoTIFF loader instance
        extractor: Patch extractor instance
        output_dir: Output directory for patches
        config: Configuration manager
        logger: Logger instance
        
    Returns:
        Processing statistics dictionary
    """
    stats = {
        'filename': image_path.name,
        'success': False,
        'patches_extracted': 0,
        'processing_time': 0.0,
        'error': None
    }
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing {image_path.name}")
        
        # Load HR image
        hr_image = loader.load_image(image_path)
        validate_image(hr_image, name=f"HR image {image_path.name}")
        
        # Check image dimensions compatibility
        if not pipeline.validate_image_dimensions(hr_image):
            logger.warning(f"Adjusting image dimensions for {image_path.name}")
            # Crop image to make it compatible
            h, w = hr_image.shape[:2]
            factor = pipeline.downsampling_factor
            new_h = (h // factor) * factor
            new_w = (w // factor) * factor
            hr_image = hr_image[:new_h, :new_w]
        
        # Generate LR images using degradation pipeline
        lr1_image, lr2_image = pipeline.process_image(hr_image, seed=42)
        
        # Extract patches
        patches = extractor.extract_patches(hr_image, lr1_image, lr2_image)
        
        if not patches:
            logger.warning(f"No valid patches extracted from {image_path.name}")
            stats['error'] = "No valid patches"
            return stats
        
        # Save patches
        base_filename = image_path.stem
        patch_output_dir = output_dir / base_filename
        patch_output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = save_image_patches(
            patches, 
            patch_output_dir,
            base_filename=base_filename,
            format=config.get('output_format', 'npy')
        )
        
        # Save visualization if enabled
        if config.get('save_visualization', False):
            vis_output_dir = output_dir / "visualizations"
            vis_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Select a representative patch for visualization
            if patches:
                sample_patch = patches[len(patches) // 2]  # Middle patch
                visualize_degradation_results(
                    sample_patch['hr'],
                    sample_patch['lr1'], 
                    sample_patch['lr2'],
                    title=f"Degradation Results - {image_path.name}",
                    save_path=vis_output_dir / f"{base_filename}_degradation.png"
                )
        
        # Update statistics
        stats['success'] = True
        stats['patches_extracted'] = len(patches)
        stats['processing_time'] = time.time() - start_time
        
        logger.info(f"Successfully processed {image_path.name}: {len(patches)} patches extracted")
        
    except Exception as e:
        stats['error'] = str(e)
        stats['processing_time'] = time.time() - start_time
        logger.error(f"Failed to process {image_path.name}: {e}")
    
    return stats


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Hardware-aware Degradation Pipeline for ISRO MFSR Project"
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
        help='Output directory for LR patches'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--pattern', 
        type=str, 
        default='*.tif',
        help='File pattern for input images'
    )
    
    parser.add_argument(
        '--max_images', 
        type=int, 
        default=None,
        help='Maximum number of images to process'
    )
    
    parser.add_argument(
        '--log_file', 
        type=str, 
        default=None,
        help='Optional log file path'
    )
    
    parser.add_argument(
        '--dry_run', 
        action='store_true',
        help='Dry run mode - just list files to be processed'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Load configuration
    try:
        config = ConfigManager(config_path)
    except Exception as e:
        print(f"Error: Failed to load configuration: {e}")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging(
        log_level=config.get('log_level', 'INFO'),
        log_file=args.log_file
    )
    
    logger.info("Starting Hardware-aware Degradation Pipeline")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {config_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find input files
    loader = GeoTIFFLoader(
        normalize=config.get('normalize', True),
        target_dtype=config.get('target_dtype', 'float32')
    )
    
    input_files = loader.find_geotiff_files(input_dir, pattern=args.pattern)
    
    if not input_files:
        logger.error(f"No GeoTIFF files found in {input_dir} with pattern {args.pattern}")
        sys.exit(1)
    
    # Limit number of files if specified
    if args.max_images:
        input_files = input_files[:args.max_images]
    
    logger.info(f"Found {len(input_files)} images to process")
    
    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE - Files to be processed:")
        for i, file_path in enumerate(input_files, 1):
            logger.info(f"  {i:3d}: {file_path}")
        logger.info(f"Total: {len(input_files)} files")
        return
    
    # Initialize pipeline components
    try:
        pipeline = DegradationPipeline(config.get_all())
        
        extractor = PatchExtractor(
            hr_patch_size=config.get('hr_patch_size', 256),
            lr_patch_size=config.get('lr_patch_size', 64),
            stride=config.get('patch_stride', None),
            min_valid_pixels=config.get('min_valid_pixels', 0.95)
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Process images
    total_stats = {
        'total_files': len(input_files),
        'successful_files': 0,
        'failed_files': 0,
        'total_patches': 0,
        'total_processing_time': 0.0,
        'errors': []
    }
    
    start_time = time.time()
    
    # Process each image
    for image_path in tqdm(input_files, desc="Processing images"):
        file_stats = process_single_image(
            image_path, pipeline, loader, extractor, 
            output_dir, config, logger
        )
        
        # Update total statistics
        total_stats['total_processing_time'] += file_stats['processing_time']
        
        if file_stats['success']:
            total_stats['successful_files'] += 1
            total_stats['total_patches'] += file_stats['patches_extracted']
        else:
            total_stats['failed_files'] += 1
            total_stats['errors'].append({
                'filename': file_stats['filename'],
                'error': file_stats['error']
            })
    
    # Print final statistics
    total_time = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {total_stats['total_files']}")
    logger.info(f"Successful: {total_stats['successful_files']}")
    logger.info(f"Failed: {total_stats['failed_files']}")
    logger.info(f"Total patches extracted: {total_stats['total_patches']}")
    logger.info(f"Total processing time: {total_stats['total_processing_time']:.2f} seconds")
    logger.info(f"Wall clock time: {total_time:.2f} seconds")
    logger.info(f"Average patches per image: {total_stats['total_patches'] / max(total_stats['successful_files'], 1):.1f}")
    
    if total_stats['errors']:
        logger.warning(f"Errors encountered in {len(total_stats['errors'])} files:")
        for error in total_stats['errors']:
            logger.warning(f"  {error['filename']}: {error['error']}")
    
    # Save processing summary
    summary_file = output_dir / "processing_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Hardware-aware Degradation Pipeline - Processing Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Configuration: {config_path}\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Processing time: {total_time:.2f} seconds\n")
        f.write(f"Total files: {total_stats['total_files']}\n")
        f.write(f"Successful: {total_stats['successful_files']}\n")
        f.write(f"Failed: {total_stats['failed_files']}\n")
        f.write(f"Total patches: {total_stats['total_patches']}\n")
        
        if total_stats['errors']:
            f.write("\nErrors:\n")
            for error in total_stats['errors']:
                f.write(f"  {error['filename']}: {error['error']}\n")
    
    logger.info(f"Processing summary saved to {summary_file}")
    
    if total_stats['failed_files'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()