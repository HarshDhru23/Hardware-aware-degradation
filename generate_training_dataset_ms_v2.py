#!/usr/bin/env python3
"""
Generate and save complete MS training-ready dataset to disk (Version 2).

This version uses a flat naming structure matching the ISRO format:
- {basename}.tif                    -> Original HR image
- {basename}_aug{N}.npz             -> Augmented HR (N=1-8)
- {basename}_aug{N}_lr{M}.npz       -> LR frames (M=1-4)

All files are stored in a flat directory structure with no subdirectories.

The generated dataset follows the exact same rules as MSDataset:
- 8-way augmentation (4 rotations x 2 flips)
- Global percentile normalization
- Multi-frame LR generation with proper shifts
- PSF kernels and parameters
- All metadata preserved

Usage:
    python generate_training_dataset_ms_v2.py \
        --input_dir data/input \
        --output_dir data/training_dataset_ms \
        --config configs/default_config.yaml \
        --global_stats configs/global_stats_ms.yaml
"""

import argparse
import sys
import logging
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
import time
import shutil

# Add repo root to import path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import MSDataset


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def save_hr_sample(sample: Dict, output_path: Path) -> None:
    """
    Save HR augmented MS image with metadata.

    Args:
        sample: Sample dictionary from MSDataset
        output_path: Path to save the HR sample (without extension)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "hr": sample["hr"].cpu().numpy(),  # [C, H, W]
        "original_filename": sample["metadata"]["filename"],
        "augmentation_id": sample["metadata"]["aug_idx"] + 1,  # 1-based
        "rotation": sample["metadata"]["rotation"],
        "flip": int(sample["metadata"]["flip"]),
        "num_bands": sample["metadata"]["num_bands"],
        "hr_shape": list(sample["hr"].shape),
        "downsampling_factor": sample["metadata"]["downsampling_factor"],
    }

    np.savez_compressed(str(output_path) + ".npz", **save_dict)


def save_lr_sample(sample: Dict, lr_idx: int, output_path: Path) -> None:
    """
    Save single LR MS frame with full metadata.

    Args:
        sample: Sample dictionary from MSDataset
        lr_idx: Index of the LR frame (0-based)
        output_path: Path to save the LR sample (without extension)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lr_frame = sample["lr"][lr_idx]
    shift_value = sample["shift_values"][lr_idx]

    save_dict = {
        "lr": lr_frame.cpu().numpy(),  # [C, H, W]
        "hr": sample["hr"].cpu().numpy(),  # Include corresponding HR
        # Actual shift values used (stochastic samples in stochastic mode, not means)
        "actual_shift_x": float(shift_value[0]),
        "actual_shift_y": float(shift_value[1]),
        # Legacy fields for compatibility
        "shift_x": float(shift_value[0]),
        "shift_y": float(shift_value[1]),
        "original_filename": sample["metadata"]["filename"],
        "augmentation_id": sample["metadata"]["aug_idx"] + 1,  # 1-based
        "lr_frame_id": lr_idx + 1,  # 1-based
        "rotation": sample["metadata"]["rotation"],
        "flip": int(sample["metadata"]["flip"]),
        "num_bands": sample["metadata"]["num_bands"],
        "lr_shape": list(lr_frame.shape),
        "hr_shape": list(sample["hr"].shape),
        "downsampling_factor": sample["metadata"]["downsampling_factor"],
    }

    save_dict["psf_sigma_x"] = float(sample["psf_params"]["sigma_x"][lr_idx])
    save_dict["psf_sigma_y"] = float(sample["psf_params"]["sigma_y"][lr_idx])
    save_dict["psf_theta"] = float(sample["psf_params"]["theta"][lr_idx])

    psf_kernel = sample["psf_kernels"][lr_idx]
    if psf_kernel is not None:
        save_dict["psf_kernel"] = psf_kernel.cpu().numpy()

    np.savez_compressed(str(output_path) + ".npz", **save_dict)


def get_base_filename(filepath: Path) -> str:
    """Extract base filename without extension."""
    return filepath.stem


def generate_dataset(
    input_dir: str,
    output_dir: str,
    config_path: str,
    global_stats_path: str = None,
    file_pattern: str = "*.tif",
    copy_originals: bool = True,
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """
    Generate and save the complete MS training dataset in flat structure.

    Args:
        input_dir: Directory containing HR MS images
        output_dir: Output directory for generated dataset
        config_path: Path to degradation config
        global_stats_path: Path to global statistics (optional)
        file_pattern: Pattern for input files
        copy_originals: Whether to copy original .tif files
        logger: Logger instance

    Returns:
        Statistics dictionary
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("GENERATING MS TRAINING-READY DATASET (FLAT STRUCTURE)")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Global stats: {global_stats_path}")
    logger.info(f"Copy originals: {copy_originals}")

    logger.info("\nInitializing MSDataset...")
    dataset = MSDataset(
        hr_image_dir=input_dir,
        config_path=config_path,
        global_stats_path=global_stats_path,
        augment=True,
        cache_size=100,
        file_pattern=file_pattern,
        seed=42,
    )

    total_samples = len(dataset)
    num_hr_images = len(dataset.hr_files)
    num_augmentations = dataset.num_augmentations
    num_lr_frames = dataset.num_lr_frames

    logger.info("\nDataset initialized:")
    logger.info(f"  HR MS images: {num_hr_images}")
    logger.info(f"  Bands: {dataset.num_bands}")
    logger.info(f"  Augmentations per image: {num_augmentations}")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  LR frames per sample: {num_lr_frames}")
    logger.info(f"  Downsampling factor: {dataset.downsampling_factor}")

    if num_augmentations != 8:
        logger.warning(f"Expected 8 augmentations but got {num_augmentations}")

    if copy_originals:
        logger.info("\n" + "=" * 80)
        logger.info("COPYING ORIGINAL HR MS IMAGES")
        logger.info("=" * 80)

        for hr_file in tqdm(dataset.hr_files, desc="Copying originals", unit="file"):
            output_file = output_dir / hr_file.name
            if not output_file.exists():
                shutil.copy2(hr_file, output_file)
                logger.debug(f"Copied: {hr_file.name}")

    dataset_info = {
        "modality": "ms",
        "total_samples": total_samples,
        "num_hr_images": num_hr_images,
        "num_bands": dataset.num_bands,
        "num_augmentations": num_augmentations,
        "num_lr_frames": num_lr_frames,
        "downsampling_factor": dataset.downsampling_factor,
        "hr_patch_size": dataset.hr_patch_size,
        "lr_patch_size": dataset.lr_patch_size,
        "augmentation_enabled": True,
        "structure": "flat",
        "naming_format": {
            "original": "{basename}.tif",
            "hr_augmented": "{basename}_aug{N}.npz (N=1-8)",
            "lr_frames": "{basename}_aug{N}_lr{M}.npz (M=1-num_lr_frames)",
        },
        "config_path": str(config_path),
        "global_stats_path": str(global_stats_path) if global_stats_path else None,
        "file_pattern": file_pattern,
        "seed": 42,
        "hr_files": [str(f) for f in dataset.hr_files],
        "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_dir / "dataset_info.yaml", "w") as f:
        yaml.dump(dataset_info, f, default_flow_style=False)

    logger.info("\n" + "=" * 80)
    logger.info("PROCESSING SAMPLES")
    logger.info("=" * 80)
    logger.info(f"Generating {num_augmentations} HR files per image")
    logger.info(f"Generating {num_lr_frames} LR files per augmentation")
    logger.info(
        f"Total files to generate: {num_augmentations * num_hr_images} HR + "
        f"{num_augmentations * num_hr_images * num_lr_frames} LR"
    )

    stats = {
        "total_samples": total_samples,
        "processed_samples": 0,
        "hr_files_created": 0,
        "lr_files_created": 0,
        "failed": 0,
        "total_time": 0.0,
        "errors": [],
    }

    start_time = time.time()

    with tqdm(total=total_samples, desc="Generating dataset", unit="sample") as pbar:
        for idx in range(total_samples):
            try:
                sample = dataset[idx]

                file_idx = idx // num_augmentations
                aug_idx = idx % num_augmentations
                hr_file = dataset.hr_files[file_idx]
                base_name = get_base_filename(hr_file)

                hr_filename = f"{base_name}_aug{aug_idx + 1}"
                hr_path = output_dir / hr_filename
                save_hr_sample(sample, hr_path)
                stats["hr_files_created"] += 1

                for lr_idx in range(num_lr_frames):
                    lr_filename = f"{base_name}_aug{aug_idx + 1}_lr{lr_idx + 1}"
                    lr_path = output_dir / lr_filename
                    save_lr_sample(sample, lr_idx, lr_path)
                    stats["lr_files_created"] += 1

                stats["processed_samples"] += 1

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "hr": stats["hr_files_created"],
                        "lr": stats["lr_files_created"],
                        "failed": stats["failed"],
                    }
                )

            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append({"sample_idx": idx, "error": str(e)})
                logger.error(f"Failed to process sample {idx}: {e}")
                pbar.update(1)

    stats["total_time"] = time.time() - start_time

    with open(output_dir / "generation_stats.yaml", "w") as f:
        yaml.dump(stats, f, default_flow_style=False)

    logger.info("\n" + "=" * 80)
    logger.info("GENERATION COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Original HR images: {num_hr_images}")
    logger.info(f"HR augmented files: {stats['hr_files_created']}")
    logger.info(f"LR frame files: {stats['lr_files_created']}")
    logger.info(
        f"Total files created: {stats['hr_files_created'] + stats['lr_files_created']}"
    )
    logger.info(f"Failed samples: {stats['failed']}")
    logger.info(f"Total time: {stats['total_time']:.2f} seconds")
    logger.info(
        f"Average time per sample: "
        f"{stats['total_time']/max(stats['processed_samples'], 1):.3f} seconds"
    )
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"Dataset info saved to: {output_dir / 'dataset_info.yaml'}")

    if num_hr_images > 0:
        example_base = get_base_filename(dataset.hr_files[0])
        logger.info("\n" + "=" * 80)
        logger.info("NAMING STRUCTURE EXAMPLES")
        logger.info("=" * 80)
        logger.info(f"Original HR:   {example_base}.tif")
        logger.info(
            f"Augmented HR:  {example_base}_aug1.npz ... {example_base}_aug8.npz"
        )
        logger.info(
            f"LR frames:     {example_base}_aug1_lr1.npz ... "
            f"{example_base}_aug1_lr{num_lr_frames}.npz"
        )

    if stats["errors"]:
        logger.warning(f"\nErrors encountered in {len(stats['errors'])} samples")
        logger.warning("See generation_stats.yaml for details")

    return stats


def verify_dataset(output_dir: str, num_samples: int = 3, logger: logging.Logger = None) -> bool:
    """
    Verify generated MS dataset by loading random samples.

    Args:
        output_dir: Dataset directory
        num_samples: Number of images to verify (checks all their augmentations)
        logger: Logger instance

    Returns:
        True if verification passed
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 80)
    logger.info("VERIFYING DATASET")
    logger.info("=" * 80)

    output_dir = Path(output_dir)
    dataset_info_path = output_dir / "dataset_info.yaml"
    if not dataset_info_path.exists():
        logger.error("Dataset info file not found")
        return False

    with open(dataset_info_path, "r") as f:
        dataset_info = yaml.safe_load(f)

    logger.info(f"Structure: {dataset_info['structure']}")
    logger.info(f"Total samples: {dataset_info['total_samples']}")
    logger.info(f"HR images: {dataset_info['num_hr_images']}")
    logger.info(f"Bands: {dataset_info['num_bands']}")
    logger.info(f"Augmentations: {dataset_info['num_augmentations']}")
    logger.info(f"LR frames: {dataset_info['num_lr_frames']}")

    hr_files = [Path(f).name for f in dataset_info["hr_files"]]

    import random

    sample_files = random.sample(hr_files, min(num_samples, len(hr_files)))

    num_augmentations = dataset_info["num_augmentations"]
    num_lr_frames = dataset_info["num_lr_frames"]

    for hr_filename in sample_files:
        base_name = Path(hr_filename).stem
        logger.info(f"\nVerifying: {base_name}")

        original_path = output_dir / hr_filename
        if original_path.exists():
            logger.debug(f"  Original file found: {hr_filename}")
        else:
            logger.debug(f"  Original file not found (may be expected): {hr_filename}")

        for aug_idx in range(1, num_augmentations + 1):
            hr_aug_path = output_dir / f"{base_name}_aug{aug_idx}.npz"
            if not hr_aug_path.exists():
                logger.error(f"  HR augmentation not found: {hr_aug_path.name}")
                return False

            try:
                hr_data = np.load(hr_aug_path)
                hr_shape = hr_data["hr"].shape
                aug_id = hr_data["augmentation_id"]

                logger.debug(
                    f"  {hr_aug_path.name}: HR shape={hr_shape}, aug_id={aug_id}"
                )

                for lr_idx in range(1, num_lr_frames + 1):
                    lr_path = output_dir / f"{base_name}_aug{aug_idx}_lr{lr_idx}.npz"
                    if not lr_path.exists():
                        logger.error(f"    LR frame not found: {lr_path.name}")
                        return False

                    lr_data = np.load(lr_path)
                    lr_shape = lr_data["lr"].shape
                    lr_frame_id = lr_data["lr_frame_id"]
                    shift_x = lr_data["shift_x"]
                    shift_y = lr_data["shift_y"]

                    logger.debug(
                        f"    {lr_path.name}: LR shape={lr_shape}, frame_id={lr_frame_id}, "
                        f"shift=({shift_x:.4f}, {shift_y:.4f})"
                    )

            except Exception as e:
                logger.error(f"  Failed to load {hr_aug_path.name}: {e}")
                return False

        logger.info(
            f"  All {num_augmentations} augmentations with {num_lr_frames} LR frames verified"
        )

    logger.info("\nVerification PASSED")
    return True


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate MS training-ready dataset in flat structure (Version 2)"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing HR MS GeoTIFF images",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated dataset",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to degradation configuration file",
    )

    parser.add_argument(
        "--global_stats",
        type=str,
        default=None,
        help="Path to global statistics YAML (optional but recommended)",
    )

    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.tif",
        help="File pattern for input images (default: *.tif)",
    )

    parser.add_argument(
        "--no_copy_originals",
        action="store_true",
        help="Do not copy original .tif files to output directory",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset after generation",
    )

    parser.add_argument(
        "--num_verify_samples",
        type=int,
        default=3,
        help="Number of images to verify (checks all their augmentations)",
    )

    args = parser.parse_args()

    logger = setup_logging(args.log_level)

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

    try:
        generate_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            global_stats_path=args.global_stats,
            file_pattern=args.file_pattern,
            copy_originals=not args.no_copy_originals,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if args.verify:
        if not verify_dataset(args.output_dir, args.num_verify_samples, logger):
            logger.error("Dataset verification failed")
            sys.exit(1)

    logger.info("\n" + "=" * 80)
    logger.info("MS DATASET GENERATION SUCCESSFUL")
    logger.info("=" * 80)
    logger.info(f"\nDataset structure:")
    logger.info(f"  All files in: {Path(args.output_dir).absolute()}")
    logger.info(f"  Naming format:")
    logger.info(f"    - Original:    {{basename}}.tif")
    logger.info(f"    - HR augmented: {{basename}}_aug{{N}}.npz (N=1-8)")
    logger.info(f"    - LR frames:    {{basename}}_aug{{N}}_lr{{M}}.npz")
    logger.info(f"\nYou can load individual files or use glob patterns:")
    logger.info(f"  - All HR aug:  {{basename}}_aug*.npz")
    logger.info(f"  - All LR:      {{basename}}_*_lr*.npz")
    logger.info(f"  - Specific aug: {{basename}}_aug1_lr*.npz")


if __name__ == "__main__":
    main()
