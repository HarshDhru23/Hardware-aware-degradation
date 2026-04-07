"""Tests for multispectral dataset behavior and label consistency."""

from pathlib import Path
import numpy as np
import pytest
from typing import cast

try:
    import rasterio
    from rasterio.transform import from_origin
except Exception:
    rasterio = None
    from_origin = None

try:
    import torch
except Exception:
    torch = None

pytestmark = pytest.mark.skipif(
    rasterio is None or torch is None,
    reason="requires rasterio and torch",
)


@pytest.fixture
def ms_temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with one synthetic 8-band TIFF."""
    assert rasterio is not None
    assert from_origin is not None

    tif_path = tmp_path / "sample_ms.tif"
    h = w = 64
    ms_array = np.random.randint(0, 2048, size=(8, h, w), dtype=np.uint16)

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=8,
        dtype="uint16",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        for band_idx in range(8):
            dst.write(ms_array[band_idx], band_idx + 1)

    return tmp_path


def test_geotiffloader_reads_multiband_as_chw(ms_temp_dir: Path):
    """GeoTIFFLoader should preserve all bands for multiband TIFFs."""
    from src.utils.data_io import GeoTIFFLoader

    loader = GeoTIFFLoader(normalize=False, target_dtype="uint16")
    image = loader.load_image(ms_temp_dir / "sample_ms.tif")

    assert image.shape == (8, 64, 64)


def test_msdataset_sample_labels_match_pan_style(ms_temp_dir: Path):
    """MSDataset sample output should follow PAN-style label keys."""
    from src.dataset import MSDataset

    dataset = MSDataset(
        hr_image_dir=ms_temp_dir,
        config_path="configs/default_config.yaml",
        augment=False,
        seed=7,
    )

    sample = dataset[0]

    expected_keys = {
        "hr",
        "lr",
        "psf_kernels",
        "psf_params",
        "shift_values",
        "metadata",
    }
    assert expected_keys.issubset(sample.keys())

    assert sample["hr"].ndim == 3
    assert sample["hr"].shape[0] == 8
    assert isinstance(sample["lr"], list)
    assert len(sample["lr"]) == dataset.num_lr_frames
    assert sample["lr"][0].shape[0] == 8

    assert "flow_vectors" not in sample
    assert len(sample["shift_values"]) == dataset.num_lr_frames


def test_msdataset_frame_count_consistent_with_pipeline(ms_temp_dir: Path):
    """Dataset frame count should stay aligned with pipeline frame count."""
    from src.dataset import MSDataset

    dataset = MSDataset(
        hr_image_dir=ms_temp_dir,
        config_path="configs/default_config.yaml",
        augment=False,
        seed=11,
    )

    assert dataset.num_lr_frames == dataset.pipeline.num_lr_frames
    assert dataset.downsampling_factor == dataset.pipeline.downsampling_factor


def test_collate_fn_ms_preserves_pan_style_keys(ms_temp_dir: Path):
    """MS collate output should keep PAN-style dataset keys without flow vectors."""
    from src.dataset import MSDataset, collate_fn_ms

    dataset = MSDataset(
        hr_image_dir=ms_temp_dir,
        config_path="configs/default_config.yaml",
        augment=False,
        seed=13,
    )
    sample = dataset[0]

    batch = collate_fn_ms([sample, sample])
    hr_batch = cast(torch.Tensor, batch["hr"])

    assert "flow_vectors" not in batch
    assert hr_batch.shape[0] == 2
    assert isinstance(batch["lr"], list)
