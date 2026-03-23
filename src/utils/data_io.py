"""
Data I/O utilities for GeoTIFF files and patch extraction.

Handles loading WorldView-3 PAN GeoTIFF files from SpaceNet dataset
and extracting matching patches for HR/LR1/LR2 triplets.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import os
import glob
from pathlib import Path
import logging

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class GeoTIFFLoader:
    """
    Loader for WorldView-3 PAN GeoTIFF files from SpaceNet dataset.
    
    Handles both rasterio (preferred) and PIL fallback for GeoTIFF loading.
    """
    
    def __init__(self, normalize: bool = True, target_dtype: str = 'float32'):
        """
        Initialize GeoTIFF loader.
        
        Args:
            normalize: Whether to normalize pixel values to [0, 1]
            target_dtype: Target data type ('float32', 'float64', 'uint8', 'uint16')
        """
        self.normalize = normalize
        self.target_dtype = target_dtype
        self.logger = logging.getLogger(__name__)
        
        if not RASTERIO_AVAILABLE and not PIL_AVAILABLE:
            raise ImportError("Either rasterio or PIL must be installed for GeoTIFF support")
    
    def load_image(self, filepath: Union[str, Path]) -> np.ndarray:
        """
        Load a single GeoTIFF image.
        
        Args:
            filepath: Path to GeoTIFF file
            
        Returns:
            Image array (H, W) for grayscale
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self.logger.debug(f"Loading GeoTIFF: {filepath}")
        
        try:
            if RASTERIO_AVAILABLE:
                image = self._load_with_rasterio(filepath)
            elif PIL_AVAILABLE:
                image = self._load_with_pil(filepath)
            else:
                raise ImportError("No suitable image loading library available")
            
            # Convert to target dtype
            image = self._convert_dtype(image)
            
            self.logger.debug(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to load {filepath}: {e}")
            raise ValueError(f"Could not load GeoTIFF file: {filepath}") from e
    
    def _load_with_rasterio(self, filepath: Path) -> np.ndarray:
        """Load GeoTIFF using rasterio (preferred method)."""
        with rasterio.open(filepath) as src:
            # Keep PAN as 2D while preserving multiband images as [C, H, W].
            if src.count == 1:
                image = src.read(1)
            else:
                image = src.read()
            original_dtype = src.dtypes[0]
            
            self.logger.debug(f"Original image dtype: {original_dtype}, range: [{image.min()}, {image.max()}]")
            
            # Handle different bit depths
            if src.dtypes[0] in ['uint16', 'int16']:
                # 16-bit images - normalize by 11-bit max (2047) to match sensor ADC
                if self.normalize:
                    image = image.astype(np.float32) / 2047.0
                    self.logger.debug(f"Normalized 16-bit image by 2047 (11-bit ADC) to [0, 1] range")
            elif src.dtypes[0] in ['uint8']:
                # 8-bit images
                if self.normalize:
                    image = image.astype(np.float32) / 255.0
                    self.logger.debug(f"Normalized 8-bit image to [0, 1] range")
            else:
                # Float images
                image = image.astype(np.float32)
                self.logger.debug(f"Float image, keeping as-is")
                
        return image
    
    def _load_with_pil(self, filepath: Path) -> np.ndarray:
        """Load GeoTIFF using PIL (fallback method)."""
        with Image.open(filepath) as img:
            # Convert to numpy array
            image = np.array(img)
            original_dtype = image.dtype
            
            self.logger.debug(f"Original image dtype: {original_dtype}, range: [{image.min()}, {image.max()}]")
            
            # Handle different bit depths
            if image.dtype == np.uint16:
                if self.normalize:
                    image = image.astype(np.float32) / 2047.0
                    self.logger.debug(f"Normalized 16-bit image by 2047 (11-bit ADC) to [0, 1] range")
            elif image.dtype == np.uint8:
                if self.normalize:
                    image = image.astype(np.float32) / 255.0
                    self.logger.debug(f"Normalized 8-bit image to [0, 1] range")
            else:
                image = image.astype(np.float32)
                self.logger.debug(f"Float image, keeping as-is")
                
        return image
    
    def _convert_dtype(self, image: np.ndarray) -> np.ndarray:
        """Convert image to target dtype."""
        if self.target_dtype == 'float32':
            return image.astype(np.float32)
        elif self.target_dtype == 'float64':
            return image.astype(np.float64)
        elif self.target_dtype == 'uint8':
            if self.normalize:
                return (image * 255).astype(np.uint8)
            else:
                return np.clip(image, 0, 255).astype(np.uint8)
        elif self.target_dtype == 'uint16':
            if self.normalize:
                # Convert back to 11-bit range (0-2047) stored in uint16
                return (image * 2047).astype(np.uint16)
            else:
                return np.clip(image, 0, 2047).astype(np.uint16)
        else:
            raise ValueError(f"Unsupported target dtype: {self.target_dtype}")
    
    def load_batch(self, filepaths: List[Union[str, Path]]) -> List[np.ndarray]:
        """
        Load multiple GeoTIFF files.
        
        Args:
            filepaths: List of file paths
            
        Returns:
            List of loaded images
        """
        images = []
        for filepath in filepaths:
            try:
                image = self.load_image(filepath)
                images.append(image)
            except Exception as e:
                self.logger.warning(f"Skipping {filepath}: {e}")
                continue
        
        self.logger.info(f"Successfully loaded {len(images)}/{len(filepaths)} images")
        return images
    
    def find_geotiff_files(self, directory: Union[str, Path], 
                          pattern: str = "*.tif") -> List[Path]:
        """
        Find all GeoTIFF files in a directory.
        
        Args:
            directory: Directory to search
            pattern: File pattern (default: "*.tif")
            
        Returns:
            List of found file paths
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Search for files
        files = list(directory.glob(pattern))
        files.extend(list(directory.glob(pattern.replace('.tif', '.tiff'))))
        
        self.logger.info(f"Found {len(files)} GeoTIFF files in {directory}")
        return sorted(files)


class PatchExtractor:
    """
    Extract corresponding patches from HR and LR images.
    
    Ensures spatial correspondence between HR patches and LR1/LR2 patches.
    """
    
    def __init__(self, 
                 hr_patch_size: int = 256,
                 lr_patch_size: int = 64,
                 stride: Optional[int] = None,
                 min_valid_pixels: float = 0.95):
        """
        Initialize patch extractor.
        
        Args:
            hr_patch_size: Size of HR patches (default: 256x256)
            lr_patch_size: Size of LR patches (default: 64x64) 
            stride: Stride for patch extraction (default: hr_patch_size)
            min_valid_pixels: Minimum fraction of valid (non-zero) pixels
        """
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = lr_patch_size
        self.stride = stride or hr_patch_size
        self.min_valid_pixels = min_valid_pixels
        
        # Verify patch size compatibility
        factor = hr_patch_size // lr_patch_size
        if hr_patch_size % lr_patch_size != 0:
            raise ValueError(f"HR patch size {hr_patch_size} must be divisible by LR patch size {lr_patch_size}")
        
        self.downsampling_factor = factor
        self.logger = logging.getLogger(__name__)
    
    def extract_patches(self, 
                       hr_image: np.ndarray,
                       lr1_image: np.ndarray,
                       lr2_image: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Extract corresponding patches from HR, LR1, and LR2 images.
        
        Args:
            hr_image: High-resolution image (H, W)
            lr1_image: Low-resolution image 1 (H/factor, W/factor)
            lr2_image: Low-resolution image 2 (H/factor, W/factor)
            
        Returns:
            List of patch dictionaries with keys: 'hr', 'lr1', 'lr2', 'coords'
        """
        # Validate input dimensions
        self._validate_dimensions(hr_image, lr1_image, lr2_image)
        
        patches = []
        hr_h, hr_w = hr_image.shape[:2]
        
        # Calculate patch positions
        y_positions = range(0, hr_h - self.hr_patch_size + 1, self.stride)
        x_positions = range(0, hr_w - self.hr_patch_size + 1, self.stride)
        
        for y in y_positions:
            for x in x_positions:
                # Extract HR patch
                hr_patch = hr_image[y:y+self.hr_patch_size, x:x+self.hr_patch_size]
                
                # Calculate corresponding LR coordinates
                lr_y = y // self.downsampling_factor
                lr_x = x // self.downsampling_factor
                
                # Extract LR patches
                lr1_patch = lr1_image[lr_y:lr_y+self.lr_patch_size, lr_x:lr_x+self.lr_patch_size]
                lr2_patch = lr2_image[lr_y:lr_y+self.lr_patch_size, lr_x:lr_x+self.lr_patch_size]
                
                # Validate patch quality
                if self._is_valid_patch(hr_patch) and self._is_valid_patch(lr1_patch) and self._is_valid_patch(lr2_patch):
                    patch_data = {
                        'hr': hr_patch.copy(),
                        'lr1': lr1_patch.copy(),
                        'lr2': lr2_patch.copy(),
                        'coords': {'hr': (y, x), 'lr': (lr_y, lr_x)}
                    }
                    patches.append(patch_data)
        
        self.logger.info(f"Extracted {len(patches)} valid patches")
        return patches
    
    def _validate_dimensions(self, hr_image: np.ndarray, lr1_image: np.ndarray, lr2_image: np.ndarray):
        """Validate that image dimensions are compatible."""
        hr_h, hr_w = hr_image.shape[:2]
        lr1_h, lr1_w = lr1_image.shape[:2]
        lr2_h, lr2_w = lr2_image.shape[:2]
        
        expected_lr_h = hr_h // self.downsampling_factor
        expected_lr_w = hr_w // self.downsampling_factor
        
        if (lr1_h != expected_lr_h or lr1_w != expected_lr_w or
            lr2_h != expected_lr_h or lr2_w != expected_lr_w):
            raise ValueError(
                f"Image dimension mismatch. HR: {hr_image.shape}, "
                f"LR1: {lr1_image.shape}, LR2: {lr2_image.shape}. "
                f"Expected LR shape: ({expected_lr_h}, {expected_lr_w})"
            )
    
    def _is_valid_patch(self, patch: np.ndarray) -> bool:
        """Check if patch has sufficient valid pixels."""
        if patch.size == 0:
            return False
        
        # Count non-zero pixels (assuming zero indicates invalid/missing data)
        valid_pixels = np.count_nonzero(patch)
        valid_fraction = valid_pixels / patch.size
        
        return valid_fraction >= self.min_valid_pixels


def save_image_patches(patches: List[Dict[str, np.ndarray]], 
                      output_dir: Union[str, Path],
                      base_filename: str = "patch",
                      format: str = "npy") -> List[Path]:
    """
    Save extracted patches to disk.
    
    Args:
        patches: List of patch dictionaries from PatchExtractor
        output_dir: Output directory
        base_filename: Base filename for patches
        format: Save format ('npy', 'png', 'tiff')
        
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for i, patch_data in enumerate(patches):
        if format == "npy":
            # Save as numpy files
            for patch_type in ['hr', 'lr1', 'lr2']:
                filename = f"{base_filename}_{i:06d}_{patch_type}.npy"
                filepath = output_dir / filename
                np.save(filepath, patch_data[patch_type])
                saved_files.append(filepath)
        
        elif format in ["png", "tiff"]:
            # Save as image files
            for patch_type in ['hr', 'lr1', 'lr2']:
                filename = f"{base_filename}_{i:06d}_{patch_type}.{format}"
                filepath = output_dir / filename
                
                # Convert to appropriate format
                patch = patch_data[patch_type]
                if patch.dtype in [np.float32, np.float64]:
                    # Assume normalized [0, 1] range
                    patch = (patch * 255).astype(np.uint8)
                elif patch.dtype == np.uint16:
                    # Convert 16-bit to 8-bit
                    patch = (patch / 256).astype(np.uint8)
                
                if PIL_AVAILABLE:
                    Image.fromarray(patch).save(filepath)
                    saved_files.append(filepath)
                else:
                    logging.warning(f"PIL not available, skipping {format} save")
    
    logging.info(f"Saved {len(saved_files)} patch files to {output_dir}")
    return saved_files