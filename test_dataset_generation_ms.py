#!/usr/bin/env python3
"""
Test MS dataset generation script - generates a limited MS dataset with PNG previews.

This script:
1. Processes only the first N images (default: 10)
2. Generates all 8 augmentations and LR frames per augmentation
3. Saves NPZ files with metadata plus RGB preview PNG files
4. Creates a flat directory structure for quick server-side verification

Usage:
    python test_dataset_generation_ms.py \
        --input_dir data/input \
        --output_dir data/test_dataset_ms \
        --config configs/default_config.yaml \
        --global_stats combined_stats.yaml \
        --num_images 10
"""

import argparse
import sys
import logging
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Tuple
import time
import shutil

# Add src to path
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


def parse_rgb_bands(bands: str) -> Tuple[int, int, int]:
    """Parse comma-separated RGB band indices, e.g. '0,1,2'."""
    parts = [p.strip() for p in bands.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--rgb_bands must have exactly 3 comma-separated indices, e.g. 0,1,2")
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as e:
        raise ValueError("--rgb_bands must contain integers, e.g. 0,1,2") from e


def save_rgb_png(image_chw: np.ndarray, output_path: Path, rgb_bands: Tuple[int, int, int]) -> None:
    """
    Save a CHW MS image tensor as RGB PNG preview.

    Args:
        image_chw: Image array [C, H, W] in [0, 1] range
        output_path: Path to save PNG
        rgb_bands: 3-tuple of channel indices for (R, G, B)
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow is required for PNG conversion. Install with: pip install Pillow") from e

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image_chw.ndim != 3:
        raise ValueError(f"Expected [C, H, W] array for MS preview, got shape {image_chw.shape}")

    c, h, w = image_chw.shape
    r_idx, g_idx, b_idx = rgb_bands
    for idx in (r_idx, g_idx, b_idx):
        if idx < 0 or idx >= c:
            raise ValueError(f"RGB band index {idx} out of range for image with {c} bands")

    rgb = np.stack([image_chw[r_idx], image_chw[g_idx], image_chw[b_idx]], axis=-1)  # [H, W, 3]
    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    Image.fromarray(rgb_uint8, mode="RGB").save(output_path)


def get_base_filename(filepath: Path) -> str:
    """Extract base filename without extension."""
    return filepath.stem


def save_hr_sample(sample: Dict, output_path: Path, save_png: bool, rgb_bands: Tuple[int, int, int]) -> None:
    """Save one HR MS sample as NPZ and optional RGB preview PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hr_array = sample["hr"].cpu().numpy()  # [C, H, W]

    save_dict = {
        "hr": hr_array,
        "original_filename": sample["metadata"]["filename"],
        "augmentation_id": sample["metadata"]["aug_idx"] + 1,
        "rotation": sample["metadata"]["rotation"],
        "flip": int(sample["metadata"]["flip"]),
        "num_bands": sample["metadata"]["num_bands"],
        "hr_shape": list(sample["hr"].shape),
        "downsampling_factor": sample["metadata"]["downsampling_factor"],
    }

    np.savez_compressed(str(output_path) + ".npz", **save_dict)

    if save_png:
        save_rgb_png(hr_array, output_path.with_suffix(".png"), rgb_bands=rgb_bands)


def save_lr_sample(
    sample: Dict,
    lr_idx: int,
    output_path: Path,
    save_png: bool,
    rgb_bands: Tuple[int, int, int],
) -> None:
    """Save one LR MS frame as NPZ and optional RGB preview PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lr_frame = sample["lr"][lr_idx]
    lr_array = lr_frame.cpu().numpy()  # [C, H, W]
    shift_value = sample["shift_values"][lr_idx]

    save_dict = {
        "lr": lr_array,
        "hr": sample["hr"].cpu().numpy(),
        "actual_shift_x": float(shift_value[0]),
        "actual_shift_y": float(shift_value[1]),
        "shift_x": float(shift_value[0]),
        "shift_y": float(shift_value[1]),
        "original_filename": sample["metadata"]["filename"],
        "augmentation_id": sample["metadata"]["aug_idx"] + 1,
        "lr_frame_id": lr_idx + 1,
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

    if save_png:
        save_rgb_png(lr_array, output_path.with_suffix(".png"), rgb_bands=rgb_bands)


def generate_test_dataset(
    input_dir: str,
    output_dir: str,
    config_path: str,
    global_stats_path: str = None,
    file_pattern: str = "*.tif",
    num_images: int = 10,
    copy_originals: bool = True,
    save_png: bool = True,
    rgb_bands: Tuple[int, int, int] = (0, 1, 2),
    logger: logging.Logger = None,
) -> Dict[str, Any]:
    """Generate MS test dataset with optional PNG previews."""
    if logger is None:
        logger = logging.getLogger(__name__)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("GENERATING MS TEST DATASET")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Global stats: {global_stats_path}")
    logger.info(f"Number of images to process: {num_images}")
    logger.info(f"Copy originals: {copy_originals}")
    logger.info(f"Save PNG previews: {save_png}")
    if save_png:
        logger.info(f"RGB preview bands: {rgb_bands}")

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

    num_available_images = len(dataset.hr_files)
    num_images_to_process = min(num_images, num_available_images)

    num_augmentations = dataset.num_augmentations
    num_lr_frames = dataset.num_lr_frames
    total_samples = num_images_to_process * num_augmentations

    logger.info("\nDataset initialized:")
    logger.info(f"  Available HR MS images: {num_available_images}")
    logger.info(f"  Images to process: {num_images_to_process}")
    logger.info(f"  Bands: {dataset.num_bands}")
    logger.info(f"  Augmentations per image: {num_augmentations}")
    logger.info(f"  Total samples to generate: {total_samples}")
    logger.info(f"  LR frames per sample: {num_lr_frames}")
    logger.info(f"  Downsampling factor: {dataset.downsampling_factor}")

    if copy_originals:
        logger.info("\n" + "=" * 80)
        logger.info("COPYING ORIGINAL HR MS IMAGES")
        logger.info("=" * 80)

        for hr_file in dataset.hr_files[:num_images_to_process]:
            output_file = output_dir / hr_file.name
            if not output_file.exists():
                shutil.copy2(hr_file, output_file)
                logger.debug(f"Copied: {hr_file.name}")

    dataset_info = {
        "mode": "test_ms",
        "modality": "ms",
        "total_samples": total_samples,
        "num_hr_images": num_images_to_process,
        "num_bands": dataset.num_bands,
        "num_augmentations": num_augmentations,
        "num_lr_frames": num_lr_frames,
        "downsampling_factor": dataset.downsampling_factor,
        "hr_patch_size": dataset.hr_patch_size,
        "lr_patch_size": dataset.lr_patch_size,
        "augmentation_enabled": True,
        "structure": "flat",
        "formats": ["npz", "png"] if save_png else ["npz"],
        "preview_rgb_bands": list(rgb_bands) if save_png else None,
        "naming_format": {
            "original": "{basename}.tif",
            "hr_augmented": "{basename}_aug{N}.npz/.png",
            "lr_frames": "{basename}_aug{N}_lr{M}.npz/.png",
        },
        "config_path": str(config_path),
        "global_stats_path": str(global_stats_path) if global_stats_path else None,
        "file_pattern": file_pattern,
        "seed": 42,
        "hr_files": [str(dataset.hr_files[i]) for i in range(num_images_to_process)],
        "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_dir / "dataset_info.yaml", "w") as f:
        yaml.dump(dataset_info, f, default_flow_style=False)

    stats = {
        "total_samples": total_samples,
        "processed_samples": 0,
        "hr_files_created": 0,
        "lr_files_created": 0,
        "hr_pngs_created": 0,
        "lr_pngs_created": 0,
        "failed": 0,
        "total_time": 0.0,
        "errors": [],
    }

    start_time = time.time()

    with tqdm(total=total_samples, desc="Generating MS test dataset", unit="sample") as pbar:
        for idx in range(total_samples):
            try:
                sample = dataset[idx]

                file_idx = idx // num_augmentations
                aug_idx = idx % num_augmentations
                hr_file = dataset.hr_files[file_idx]
                base_name = get_base_filename(hr_file)

                hr_filename = f"{base_name}_aug{aug_idx + 1}"
                hr_path = output_dir / hr_filename
                save_hr_sample(sample, hr_path, save_png=save_png, rgb_bands=rgb_bands)
                stats["hr_files_created"] += 1
                if save_png:
                    stats["hr_pngs_created"] += 1

                for lr_idx in range(num_lr_frames):
                    lr_filename = f"{base_name}_aug{aug_idx + 1}_lr{lr_idx + 1}"
                    lr_path = output_dir / lr_filename
                    save_lr_sample(sample, lr_idx, lr_path, save_png=save_png, rgb_bands=rgb_bands)
                    stats["lr_files_created"] += 1
                    if save_png:
                        stats["lr_pngs_created"] += 1

                stats["processed_samples"] += 1

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "hr_npz": stats["hr_files_created"],
                        "lr_npz": stats["lr_files_created"],
                        "failed": stats["failed"],
                    }
                )
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append({"sample_idx": idx, "error": str(e)})
                logger.error(f"Failed to process sample {idx}: {e}")
                import traceback

                traceback.print_exc()
                pbar.update(1)

    stats["total_time"] = time.time() - start_time

    with open(output_dir / "generation_stats.yaml", "w") as f:
        yaml.dump(stats, f, default_flow_style=False)

    logger.info("\n" + "=" * 80)
    logger.info("GENERATION COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Processed images: {num_images_to_process}/{num_available_images}")
    logger.info(f"HR files (NPZ): {stats['hr_files_created']}")
    logger.info(f"LR files (NPZ): {stats['lr_files_created']}")
    if save_png:
        logger.info(f"HR files (PNG): {stats['hr_pngs_created']}")
        logger.info(f"LR files (PNG): {stats['lr_pngs_created']}")
    logger.info(f"Failed samples: {stats['failed']}")
    logger.info(f"Total time: {stats['total_time']:.2f} seconds")
    logger.info(f"Average time per sample: {stats['total_time']/max(stats['processed_samples'], 1):.3f} seconds")
    logger.info(f"Output directory: {output_dir}")

    if stats["errors"]:
        logger.warning(f"Errors encountered in {len(stats['errors'])} samples")
        logger.warning("See generation_stats.yaml for details")

    return stats


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate MS test dataset with optional PNG previews")

    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing HR MS GeoTIFF images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for test dataset")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to degradation configuration file")
    parser.add_argument("--global_stats", type=str, default=None, help="Path to global statistics YAML (optional but recommended)")
    parser.add_argument("--file_pattern", type=str, default="*.tif", help="File pattern for input images (default: *.tif)")
    parser.add_argument("--num_images", type=int, default=10, help="Number of input images to process (default: 10)")
    parser.add_argument("--no_copy_originals", action="store_true", help="Do not copy original .tif files to output directory")
    parser.add_argument("--no_png", action="store_true", help="Do not save RGB preview PNG files")
    parser.add_argument("--rgb_bands", type=str, default="0,1,2", help="Comma-separated RGB band indices for previews (default: 0,1,2)")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
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
        rgb_bands = parse_rgb_bands(args.rgb_bands)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if not args.no_png:
        try:
            import PIL  # noqa: F401
        except ImportError:
            logger.error("Pillow is required for PNG generation. Install with: pip install Pillow")
            sys.exit(1)

    try:
        generate_test_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            global_stats_path=args.global_stats,
            file_pattern=args.file_pattern,
            num_images=args.num_images,
            copy_originals=not args.no_copy_originals,
            save_png=not args.no_png,
            rgb_bands=rgb_bands,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"MS test dataset generation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("MS TEST DATASET GENERATION FINISHED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
