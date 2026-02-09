"""
PyTorch Dataset for Hardware-aware Degradation Pipeline.

Generates degraded LR images on-the-fly from HR images with:
- Multi-frame LR generation (2 or 4 frames based on downsampling factor)
- Anisotropic Gaussian PSF blur
- Sub-pixel shifts
- Poisson-Gaussian noise
- Global percentile normalization
- 8-way data augmentation (4 rotations × 2 flips)
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
import logging
from functools import lru_cache

from .degradation.pipeline import DegradationPipeline
from .utils.data_io import GeoTIFFLoader
from .config import ConfigManager


class DegradationDataset(Dataset):
    """
    PyTorch Dataset for on-the-fly hardware-aware image degradation.
    
    Loads HR images and generates corresponding LR images dynamically during training.
    Supports multi-frame LR generation with different sub-pixel shifts and PSF variations.
    """
    
    def __init__(
        self,
        hr_image_dir: Union[str, Path],
        config_path: Union[str, Path],
        global_stats_path: Optional[Union[str, Path]] = None,
        augment: bool = True,
        cache_size: int = 100,
        file_pattern: str = "*.tif",
        seed: Optional[int] = None
    ):
        """
        Initialize DegradationDataset.
        
        Args:
            hr_image_dir: Directory containing HR GeoTIFF images
            config_path: Path to degradation configuration YAML
            global_stats_path: Path to global percentile statistics YAML (optional)
            augment: Enable 8-way augmentation (4 rotations × 2 flips)
            cache_size: Number of HR images to cache in memory (LRU cache)
            file_pattern: Glob pattern for image files (default: *.tif)
            seed: Random seed for reproducibility (optional)
        """
        super().__init__()
        
        self.hr_image_dir = Path(hr_image_dir)
        self.config_path = Path(config_path)
        self.global_stats_path = Path(global_stats_path) if global_stats_path else None
        self.augment = augment
        self.cache_size = cache_size
        self.seed = seed
        
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = ConfigManager(str(config_path))
        self.config_dict = self.config.get_all()
        
        # Load global statistics if provided
        self.global_stats = None
        if self.global_stats_path and self.global_stats_path.exists():
            with open(self.global_stats_path, 'r') as f:
                self.global_stats = yaml.safe_load(f)
            self.logger.info(f"Loaded global stats from {self.global_stats_path}")
            self.logger.info(f"  p2: {self.global_stats.get('p2', 'N/A')}")
            self.logger.info(f"  p98: {self.global_stats.get('p98', 'N/A')}")
        
        # Find all HR image files
        self.hr_files = sorted(list(self.hr_image_dir.glob(file_pattern)))
        if not self.hr_files:
            raise ValueError(f"No files matching '{file_pattern}' found in {hr_image_dir}")
        
        self.logger.info(f"Found {len(self.hr_files)} HR images in {hr_image_dir}")
        
        # Initialize data loader (without normalization - we'll apply global norm later)
        self.loader = GeoTIFFLoader(normalize=False, target_dtype='uint16')
        
        # Initialize degradation pipeline
        self.pipeline = DegradationPipeline(self.config_dict)
        
        # Get parameters from config
        self.num_lr_frames = self.config.get('num_lr_frames', 4)
        self.downsampling_factor = self.config.get('downsampling_factor', 4)
        self.hr_patch_size = self.config.get('hr_patch_size', 256)
        self.lr_patch_size = self.hr_patch_size // self.downsampling_factor
        
        # Augmentation: 4 rotations (0°, 90°, 180°, 270°) × 2 flips (no flip, horizontal flip)
        self.augmentations = []
        if self.augment:
            for rot in [0, 1, 2, 3]:  # k for np.rot90
                for flip in [False, True]:
                    self.augmentations.append({'rot': rot, 'flip': flip})
        else:
            self.augmentations = [{'rot': 0, 'flip': False}]
        
        self.num_augmentations = len(self.augmentations)
        
        self.logger.info(f"Dataset initialized:")
        self.logger.info(f"  HR images: {len(self.hr_files)}")
        self.logger.info(f"  Augmentations: {self.num_augmentations}")
        self.logger.info(f"  Total samples: {len(self)}")
        self.logger.info(f"  LR frames per sample: {self.num_lr_frames}")
        self.logger.info(f"  Downsampling factor: {self.downsampling_factor}")
        self.logger.info(f"  HR patch size: {self.hr_patch_size}")
        self.logger.info(f"  LR patch size: {self.lr_patch_size}")
    
    def __len__(self) -> int:
        """Return total number of samples (images × augmentations)."""
        return len(self.hr_files) * self.num_augmentations
    
    def _load_hr_image_cached(self, file_idx: int) -> torch.Tensor:
        """
        Load HR image with LRU caching.
        
        Args:
            file_idx: Index of the HR file
            
        Returns:
            HR image as torch.Tensor (raw uint16, converted to float32)
        """
        filepath = self.hr_files[file_idx]
        image = self.loader.load_image(filepath)  # Returns numpy array
        # Convert to torch tensor immediately
        image_tensor = torch.from_numpy(image).float()
        return image_tensor
    
    def _apply_global_normalization(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply global percentile normalization to image.
        
        Args:
            image: Input image tensor (raw uint16 or already normalized)
            
        Returns:
            Normalized image in [0, 1] range
        """
        if self.global_stats is None:
            # Fallback to simple division by max value
            return image / 65535.0
        
        p2 = self.global_stats.get('p2', 0)
        p98 = self.global_stats.get('p98', 65535)
        
        # Apply percentile normalization
        image_normalized = (image - p2) / (p98 - p2)
        image_normalized = torch.clamp(image_normalized, 0, 1)
        
        return image_normalized
    
    def _apply_augmentation(self, image: torch.Tensor, aug_params: Dict) -> torch.Tensor:
        """
        Apply rotation and flip augmentation.
        
        Args:
            image: Input image tensor
            aug_params: Dict with 'rot' (0-3 for 90° rotations) and 'flip' (bool)
            
        Returns:
            Augmented image tensor
        """
        # Apply rotation (k*90 degrees counterclockwise)
        if aug_params['rot'] > 0:
            # torch.rot90 rotates in dims (0, 1) which are H, W for 2D tensor
            image = torch.rot90(image, k=aug_params['rot'], dims=(0, 1))
        
        # Apply horizontal flip
        if aug_params['flip']:
            image = torch.flip(image, dims=[1])  # Flip along width dimension
        
        return image.contiguous()  # Ensure contiguous memory
    
    def _extract_psf_info(self) -> List[Dict]:
        """
        Extract PSF parameters and kernels from blur operators.
        
        Returns:
            List of dicts, one per LR frame, containing:
                - 'sigma_x': float
                - 'sigma_y': float
                - 'theta': float (radians)
                - 'kernel': torch.Tensor (2D PSF kernel)
        """
        psf_info = []
        
        for i in range(self.num_lr_frames):
            blur_op = self.pipeline.blur_operators[i]
            
            # Get PSF parameters from operator
            kernel = None
            if hasattr(blur_op, 'psf_kernel') and blur_op.psf_kernel is not None:
                # Convert numpy kernel to torch tensor
                kernel = torch.from_numpy(blur_op.psf_kernel).float()
            
            psf_params = {
                'sigma_x': blur_op.sigma_x,
                'sigma_y': blur_op.sigma_y,
                'theta': blur_op.theta,
                'kernel': kernel
            }
            
            psf_info.append(psf_params)
        
        return psf_info
    
    def _generate_flow_vectors(self, lr_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Generate ground-truth optical flow vectors from known sub-pixel shifts.
        
        Flow vectors represent the displacement needed to warp each LR frame to the reference frame.
        Since shift_values are [(dx0, dy0), (dx1, dy1), ...] where frame i is shifted by (dxi, dyi)
        relative to HR space, the flow to warp frame i to frame 0 is: -(dxi - dx0, dyi - dy0)
        
        Args:
            lr_shape: Shape of LR images (H, W)
            
        Returns:
            Flow tensor of shape [num_frames, 2, H, W] where channels are (flow_x, flow_y)
        """
        H, W = lr_shape
        num_frames = len(self.pipeline.shift_values)
        
        # Create flow tensor
        flow_vectors = torch.zeros(num_frames, 2, H, W)
        
        # Get reference shift (first frame)
        ref_shift = self.pipeline.shift_values[0]  # [dx, dy]
        
        # For each frame, compute flow relative to reference frame
        for i, shift in enumerate(self.pipeline.shift_values):
            # Flow is negative of the relative shift (to warp TO reference)
            # Also scale by downsampling factor since shifts are in HR space
            flow_x = -(shift[0] - ref_shift[0]) / self.downsampling_factor
            flow_y = -(shift[1] - ref_shift[1]) / self.downsampling_factor
            
            # Fill entire flow field with constant flow (since shifts are global)
            flow_vectors[i, 0, :, :] = flow_x  # x-component
            flow_vectors[i, 1, :, :] = flow_y  # y-component
        
        return flow_vectors
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index (includes augmentation)
            
        Returns:
            Dictionary containing:
                'hr': HR image tensor [1, H, W]
                'lr': List of LR image tensors [1, H/scale, W/scale]
                'psf_kernels': List of PSF kernel tensors [Kh, Kw]
                'psf_params': Dict with 'sigma_x', 'sigma_y', 'theta' lists
                'shift_values': List of shift values [(sx, sy), ...]
                'metadata': Dict with filename, augmentation, etc.
        """
        # Map global index to (file_idx, aug_idx)
        file_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations
        aug_params = self.augmentations[aug_idx]
        
        # Load HR image (returns torch.Tensor)
        hr_image = self._load_hr_image_cached(file_idx)
        
        # Apply augmentation to HR image
        hr_image_aug = self._apply_augmentation(hr_image, aug_params)
        
        # Apply global normalization
        hr_image_normalized = self._apply_global_normalization(hr_image_aug)
        
        # Convert to numpy for degradation pipeline (pipeline expects numpy)
        hr_image_np = hr_image_normalized.cpu().numpy()
        
        # Generate LR frames using degradation pipeline
        # Use deterministic seed based on idx for reproducibility
        frame_seed = (self.seed if self.seed is not None else 0) + idx
        lr_frames_np = self.pipeline.generate_lr_frames(hr_image_np, seed=frame_seed)
        
        # Convert LR frames from numpy to torch
        lr_frames = [torch.from_numpy(lr).float() for lr in lr_frames_np]
        
        # Extract PSF information (already returns torch tensors)
        psf_info = self._extract_psf_info()
        
        # Convert to PyTorch tensors with channel dimension
        hr_tensor = hr_image_normalized.unsqueeze(0)  # [1, H, W]
        lr_tensors = [lr.unsqueeze(0) for lr in lr_frames]  # List of [1, H', W']
        
        # PSF kernels as tensors (already torch.Tensor from _extract_psf_info)
        psf_kernel_tensors = []
        psf_sigma_x = []
        psf_sigma_y = []
        psf_theta = []
        
        for psf in psf_info:
            psf_kernel_tensors.append(psf['kernel'])  # Already torch.Tensor or None
            psf_sigma_x.append(psf['sigma_x'])
            psf_sigma_y.append(psf['sigma_y'])
            psf_theta.append(psf['theta'])
        
        # Shift values from pipeline
        shift_values = self.pipeline.shift_values  # List of [dx, dy]
        
        # Generate flow vectors from known shifts
        lr_shape = lr_frames[0].shape  # (H, W) without channel dim
        flow_vectors = self._generate_flow_vectors(lr_shape)
        
        # Metadata
        metadata = {
            'filename': self.hr_files[file_idx].name,
            'file_idx': file_idx,
            'aug_idx': aug_idx,
            'rotation': aug_params['rot'] * 90,  # degrees
            'flip': aug_params['flip'],
            'num_lr_frames': self.num_lr_frames,
            'downsampling_factor': self.downsampling_factor,
            'hr_shape': hr_tensor.shape,
            'lr_shape': lr_tensors[0].shape
        }
        
        return {
            'hr': hr_tensor,
            'lr': lr_tensors,  # List of tensors
            'flow_vectors': flow_vectors,  # [num_frames, 2, H, W]
            'psf_kernels': psf_kernel_tensors,  # List of tensors
            'psf_params': {
                'sigma_x': psf_sigma_x,
                'sigma_y': psf_sigma_y,
                'theta': psf_theta
            },
            'shift_values': shift_values,  # List of [dx, dy]
            'metadata': metadata
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Custom collate function for DegradationDataset.
    
    Handles batching of variable-length lists (LR frames, PSF kernels).
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched dictionary with:
            'hr': [B, 1, H, W]
            'lr': List of [B, 1, H', W'] (one per frame)
            'psf_kernels': List of [B, Kh, Kw] (one per frame)
            'psf_params': Dict with batched lists
            'shift_values': Tensor [B, num_frames, 2]
            'metadata': List of metadata dicts
    """
    # Stack HR images
    hr_batch = torch.stack([sample['hr'] for sample in batch], dim=0)  # [B, 1, H, W]
    
    # Stack LR frames (each frame becomes [B, 1, H', W'])
    num_lr_frames = len(batch[0]['lr'])
    lr_batch = []
    for frame_idx in range(num_lr_frames):
        lr_frame_batch = torch.stack([sample['lr'][frame_idx] for sample in batch], dim=0)
        lr_batch.append(lr_frame_batch)
    
    # Stack PSF kernels
    psf_kernels_batch = []
    for frame_idx in range(num_lr_frames):
        kernels = [sample['psf_kernels'][frame_idx] for sample in batch]
        if kernels[0] is not None:
            psf_kernel_batch = torch.stack(kernels, dim=0)
            psf_kernels_batch.append(psf_kernel_batch)
        else:
            psf_kernels_batch.append(None)
    
    # Batch PSF parameters
    psf_params_batch = {
        'sigma_x': [sample['psf_params']['sigma_x'] for sample in batch],
        'sigma_y': [sample['psf_params']['sigma_y'] for sample in batch],
        'theta': [sample['psf_params']['theta'] for sample in batch]
    }
    
    # Stack shift values
    shift_values_batch = torch.tensor(
        [sample['shift_values'] for sample in batch], 
        dtype=torch.float32
    )  # [B, num_frames, 2]
    
    # Collect metadata
    metadata_batch = [sample['metadata'] for sample in batch]
    
    return {
        'hr': hr_batch,
        'lr': lr_batch,
        'psf_kernels': psf_kernels_batch,
        'psf_params': psf_params_batch,
        'shift_values': shift_values_batch,
        'metadata': metadata_batch
    }


def create_dataloader(
    hr_image_dir: Union[str, Path],
    config_path: Union[str, Path],
    global_stats_path: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = True,
    cache_size: int = 100,
    seed: Optional[int] = None
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the DegradationDataset.
    
    Args:
        hr_image_dir: Directory containing HR images
        config_path: Path to degradation config
        global_stats_path: Path to global percentile stats
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Shuffle data
        augment: Enable augmentation
        cache_size: LRU cache size for HR images
        seed: Random seed
        
    Returns:
        PyTorch DataLoader
    """
    dataset = DegradationDataset(
        hr_image_dir=hr_image_dir,
        config_path=config_path,
        global_stats_path=global_stats_path,
        augment=augment,
        cache_size=cache_size,
        seed=seed
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return dataloader


class MSDataset(Dataset):
    """
    PyTorch Dataset for Multi-Spectral (MS) hardware-aware image degradation.
    
    Handles both:
    - WorldView-3: 8-band MS pansharpened to 0.3m
    - WorldView-2: 4-band MS pansharpened to 0.5m
    
    Applies the same degradation pipeline as PAN, but processes each band independently.
    """
    
    def __init__(
        self,
        hr_image_dir: Union[str, Path],
        config_path: Union[str, Path],
        global_stats_path: Optional[Union[str, Path]] = None,
        augment: bool = True,
        cache_size: int = 100,
        file_pattern: str = "*.tif",
        seed: Optional[int] = None
    ):
        """
        Initialize MSDataset.
        
        Args:
            hr_image_dir: Directory containing HR MS GeoTIFF images
            config_path: Path to degradation configuration YAML
            global_stats_path: Path to global percentile statistics YAML (optional)
            augment: Enable 8-way augmentation (4 rotations × 2 flips)
            cache_size: Number of HR images to cache in memory
            file_pattern: Glob pattern for image files (default: *.tif)
            seed: Random seed for reproducibility (optional)
        """
        super().__init__()
        
        self.hr_image_dir = Path(hr_image_dir)
        self.config_path = Path(config_path)
        self.global_stats_path = Path(global_stats_path) if global_stats_path else None
        self.augment = augment
        self.cache_size = cache_size
        self.seed = seed
        
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = ConfigManager(str(config_path))
        self.config_dict = self.config.get_all()
        
        # Load global statistics if provided
        self.global_stats = None
        if self.global_stats_path and self.global_stats_path.exists():
            with open(self.global_stats_path, 'r') as f:
                self.global_stats = yaml.safe_load(f)
            self.logger.info(f"Loaded global stats from {self.global_stats_path}")
        
        # Find all HR image files
        self.hr_files = sorted(list(self.hr_image_dir.glob(file_pattern)))
        if not self.hr_files:
            raise ValueError(f"No files matching '{file_pattern}' found in {hr_image_dir}")
        
        self.logger.info(f"Found {len(self.hr_files)} HR MS images in {hr_image_dir}")
        
        # Initialize data loader (without normalization - we'll apply global norm later)
        self.loader = GeoTIFFLoader(normalize=False, target_dtype='uint16')
        
        # Detect number of bands from first image
        first_image = self.loader.load_image(self.hr_files[0])
        if len(first_image.shape) == 2:
            raise ValueError("MS images should have multiple bands, got grayscale image")
        self.num_bands = first_image.shape[0] if first_image.shape[0] < first_image.shape[-1] else first_image.shape[-1]
        
        self.logger.info(f"Detected {self.num_bands}-band MS data")
        if self.num_bands == 8:
            self.logger.info("  WorldView-3: 8-band MS pansharpened to 0.3m")
        elif self.num_bands == 4:
            self.logger.info("  WorldView-2: 4-band MS pansharpened to 0.5m")
        else:
            self.logger.warning(f"  Unexpected band count: {self.num_bands}")
        
        # Initialize degradation pipeline (will be applied per-band)
        self.pipeline = DegradationPipeline(self.config_dict)
        
        # Get parameters from config
        self.num_lr_frames = self.config.get('num_lr_frames', 2)
        self.downsampling_factor = self.config.get('downsampling_factor', 4)
        self.hr_patch_size = self.config.get('hr_patch_size', 256)
        self.lr_patch_size = self.hr_patch_size // self.downsampling_factor
        
        # Augmentation: 4 rotations × 2 flips
        self.augmentations = []
        if self.augment:
            for rot in [0, 1, 2, 3]:
                for flip in [False, True]:
                    self.augmentations.append({'rot': rot, 'flip': flip})
        else:
            self.augmentations = [{'rot': 0, 'flip': False}]
        
        self.num_augmentations = len(self.augmentations)
        
        self.logger.info(f"MS Dataset initialized:")
        self.logger.info(f"  HR images: {len(self.hr_files)}")
        self.logger.info(f"  Bands: {self.num_bands}")
        self.logger.info(f"  Augmentations: {self.num_augmentations}")
        self.logger.info(f"  Total samples: {len(self)}")
        self.logger.info(f"  LR frames per sample: {self.num_lr_frames}")
        self.logger.info(f"  Downsampling factor: {self.downsampling_factor}")
    
    def __len__(self) -> int:
        """Return total number of samples (images × augmentations)."""
        return len(self.hr_files) * self.num_augmentations
    
    def _load_hr_image_cached(self, file_idx: int) -> torch.Tensor:
        """
        Load HR MS image with LRU caching.
        
        Args:
            file_idx: Index of the HR file
            
        Returns:
            HR image as torch.Tensor [C, H, W] (raw uint16, converted to float32)
        """
        filepath = self.hr_files[file_idx]
        image = self.loader.load_image(filepath)  # Returns numpy array
        
        # Ensure channels-first format [C, H, W]
        if len(image.shape) == 3:
            if image.shape[0] > image.shape[-1]:
                # Already channels-first [C, H, W]
                image_tensor = torch.from_numpy(image).float()
            else:
                # Channels-last [H, W, C] -> convert to [C, H, W]
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            raise ValueError(f"Expected 3D image, got shape {image.shape}")
        
        return image_tensor
    
    def _apply_global_normalization(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply global percentile normalization to MS image.
        
        Args:
            image: Input image tensor [C, H, W] (raw uint16)
            
        Returns:
            Normalized image in [0, 1] range [C, H, W]
        """
        if self.global_stats is None:
            # Fallback to simple division by max value
            return image / 65535.0
        
        p2 = self.global_stats.get('p2', 0)
        p98 = self.global_stats.get('p98', 65535)
        
        # Apply percentile normalization (same for all bands)
        image_normalized = (image - p2) / (p98 - p2)
        image_normalized = torch.clamp(image_normalized, 0, 1)
        
        return image_normalized
    
    def _apply_augmentation(self, image: torch.Tensor, aug_params: Dict) -> torch.Tensor:
        """
        Apply rotation and flip augmentation to multi-band image.
        
        Args:
            image: Input image tensor [C, H, W]
            aug_params: Dict with 'rot' (0-3 for 90° rotations) and 'flip' (bool)
            
        Returns:
            Augmented image tensor [C, H, W]
        """
        # Apply rotation (k*90 degrees on spatial dims)
        if aug_params['rot'] > 0:
            # Rotate spatial dimensions (H, W) = dims (1, 2) for [C, H, W]
            image = torch.rot90(image, k=aug_params['rot'], dims=(1, 2))
        
        # Apply horizontal flip on width dimension
        if aug_params['flip']:
            image = torch.flip(image, dims=[2])  # Flip along width
        
        return image.contiguous()
    
    def _process_band(self, band: torch.Tensor, seed: int) -> List[torch.Tensor]:
        """
        Process a single band through degradation pipeline.
        
        Args:
            band: Single-band image [H, W]
            seed: Random seed for this band processing
            
        Returns:
            List of degraded LR frames for this band
        """
        # Convert to numpy for pipeline
        band_np = band.cpu().numpy()
        
        # Generate LR frames
        lr_frames_np = self.pipeline.generate_lr_frames(band_np, seed=seed)
        
        # Convert back to torch
        lr_frames = [torch.from_numpy(lr).float() for lr in lr_frames_np]
        
        return lr_frames
    
    def _extract_psf_info(self) -> List[Dict]:
        """
        Extract PSF parameters and kernels from blur operators.
        
        Returns:
            List of dicts, one per LR frame
        """
        psf_info = []
        
        for i in range(self.num_lr_frames):
            blur_op = self.pipeline.blur_operators[i]
            
            kernel = None
            if hasattr(blur_op, 'psf_kernel') and blur_op.psf_kernel is not None:
                kernel = torch.from_numpy(blur_op.psf_kernel).float()
            
            psf_params = {
                'sigma_x': blur_op.sigma_x,
                'sigma_y': blur_op.sigma_y,
                'theta': blur_op.theta,
                'kernel': kernel
            }
            
            psf_info.append(psf_params)
        
        return psf_info
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single MS sample.
        
        Args:
            idx: Sample index (includes augmentation)
            
        Returns:
            Dictionary containing:
                'hr': HR image tensor [C, H, W]
                'lr': List of LR image tensors [C, H/scale, W/scale]
                'psf_kernels': List of PSF kernel tensors [Kh, Kw]
                'psf_params': Dict with 'sigma_x', 'sigma_y', 'theta' lists
                'shift_values': List of shift values [(sx, sy), ...]
                'metadata': Dict with filename, bands, augmentation, etc.
        """
        # Map global index to (file_idx, aug_idx)
        file_idx = idx // self.num_augmentations
        aug_idx = idx % self.num_augmentations
        aug_params = self.augmentations[aug_idx]
        
        # Load HR MS image [C, H, W]
        hr_image = self._load_hr_image_cached(file_idx)
        
        # Apply augmentation
        hr_image_aug = self._apply_augmentation(hr_image, aug_params)
        
        # Apply global normalization
        hr_image_normalized = self._apply_global_normalization(hr_image_aug)
        
        # Process each band through degradation pipeline
        # Use band-specific seed for deterministic but varied degradation per band
        base_seed = (self.seed if self.seed is not None else 0) + idx
        
        lr_frames_all_bands = []  # Will be [num_frames][num_bands, H', W']
        
        for frame_idx in range(self.num_lr_frames):
            lr_frame_bands = []
            
            for band_idx in range(self.num_bands):
                # Get single band
                band = hr_image_normalized[band_idx]  # [H, W]
                
                # Process this band (use combined seed for reproducibility)
                band_seed = base_seed + frame_idx * 1000 + band_idx
                lr_frames_band = self._process_band(band, seed=band_seed)
                
                # Take the corresponding frame
                lr_frame_bands.append(lr_frames_band[frame_idx])
            
            # Stack all bands for this frame [C, H', W']
            lr_frame_stacked = torch.stack(lr_frame_bands, dim=0)
            lr_frames_all_bands.append(lr_frame_stacked)
        
        # Extract PSF information (same for all bands)
        psf_info = self._extract_psf_info()
        
        # Prepare PSF data
        psf_kernel_tensors = [psf['kernel'] for psf in psf_info]
        psf_sigma_x = [psf['sigma_x'] for psf in psf_info]
        psf_sigma_y = [psf['sigma_y'] for psf in psf_info]
        psf_theta = [psf['theta'] for psf in psf_info]
        
        # Shift values from pipeline
        shift_values = self.pipeline.shift_values
        
        # Metadata
        metadata = {
            'filename': self.hr_files[file_idx].name,
            'file_idx': file_idx,
            'aug_idx': aug_idx,
            'rotation': aug_params['rot'] * 90,
            'flip': aug_params['flip'],
            'num_bands': self.num_bands,
            'num_lr_frames': self.num_lr_frames,
            'downsampling_factor': self.downsampling_factor,
            'hr_shape': hr_image_normalized.shape,
            'lr_shape': lr_frames_all_bands[0].shape
        }
        
        return {
            'hr': hr_image_normalized,  # [C, H, W]
            'lr': lr_frames_all_bands,  # List of [C, H', W']
            'psf_kernels': psf_kernel_tensors,
            'psf_params': {
                'sigma_x': psf_sigma_x,
                'sigma_y': psf_sigma_y,
                'theta': psf_theta
            },
            'shift_values': shift_values,
            'metadata': metadata
        }


def collate_fn_ms(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Custom collate function for MSDataset.
    
    Handles batching of multi-band MS images.
    
    Args:
        batch: List of samples from MSDataset.__getitem__
        
    Returns:
        Batched dictionary with:
            'hr': [B, C, H, W]
            'lr': List of [B, C, H', W'] (one per frame)
            'psf_kernels': List of [B, Kh, Kw] (one per frame)
            'psf_params': Dict with batched lists
            'shift_values': Tensor [B, num_frames, 2]
            'metadata': List of metadata dicts
    """
    # Stack HR images
    hr_batch = torch.stack([sample['hr'] for sample in batch], dim=0)  # [B, C, H, W]
    
    # Stack LR frames
    num_lr_frames = len(batch[0]['lr'])
    lr_batch = []
    for frame_idx in range(num_lr_frames):
        lr_frame_batch = torch.stack([sample['lr'][frame_idx] for sample in batch], dim=0)
        lr_batch.append(lr_frame_batch)  # [B, C, H', W']
    
    # Stack PSF kernels
    psf_kernels_batch = []
    for frame_idx in range(num_lr_frames):
        kernels = [sample['psf_kernels'][frame_idx] for sample in batch]
        if kernels[0] is not None:
            psf_kernel_batch = torch.stack(kernels, dim=0)
            psf_kernels_batch.append(psf_kernel_batch)
        else:
            psf_kernels_batch.append(None)
    
    # Batch PSF parameters
    psf_params_batch = {
        'sigma_x': [sample['psf_params']['sigma_x'] for sample in batch],
        'sigma_y': [sample['psf_params']['sigma_y'] for sample in batch],
        'theta': [sample['psf_params']['theta'] for sample in batch]
    }
    
    # Stack shift values
    shift_values_batch = torch.tensor(
        [sample['shift_values'] for sample in batch], 
        dtype=torch.float32
    )
    
    # Collect metadata
    metadata_batch = [sample['metadata'] for sample in batch]
    
    return {
        'hr': hr_batch,
        'lr': lr_batch,
        'psf_kernels': psf_kernels_batch,
        'psf_params': psf_params_batch,
        'shift_values': shift_values_batch,
        'metadata': metadata_batch
    }


def create_ms_dataloader(
    hr_image_dir: Union[str, Path],
    config_path: Union[str, Path],
    global_stats_path: Optional[Union[str, Path]] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = True,
    cache_size: int = 100,
    seed: Optional[int] = None
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the MSDataset.
    
    Args:
        hr_image_dir: Directory containing HR MS images
        config_path: Path to degradation config
        global_stats_path: Path to global percentile stats
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Shuffle data
        augment: Enable augmentation
        cache_size: LRU cache size for HR images
        seed: Random seed
        
    Returns:
        PyTorch DataLoader for MS data
    """
    dataset = MSDataset(
        hr_image_dir=hr_image_dir,
        config_path=config_path,
        global_stats_path=global_stats_path,
        augment=augment,
        cache_size=cache_size,
        seed=seed
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_ms,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return dataloader


if __name__ == '__main__':
    """Example usage and testing."""
    import sys
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("="*70)
    print("Testing DegradationDataset (PAN)")
    print("="*70)
    
    # Test PAN dataset
    pan_dataset = DegradationDataset(
        hr_image_dir='test_images/PAN',  # Replace with your directory
        config_path='configs/default_config.yaml',
        global_stats_path='configs/combined_global_stats.yaml',  # Optional
        augment=True,
        cache_size=10
    )
    
    print(f"\nPAN Dataset size: {len(pan_dataset)}")
    
    # Get a sample
    pan_sample = pan_dataset[0]
    print(f"\nPAN Sample 0:")
    print(f"  HR shape: {pan_sample['hr'].shape}")
    print(f"  Number of LR frames: {len(pan_sample['lr'])}")
    print(f"  LR[0] shape: {pan_sample['lr'][0].shape}")
    print(f"  PSF params (frame 0): sigma_x={pan_sample['psf_params']['sigma_x'][0]:.3f}, "
          f"sigma_y={pan_sample['psf_params']['sigma_y'][0]:.3f}, "
          f"theta={pan_sample['psf_params']['theta'][0]:.3f}")
    
    print("\n" + "="*70)
    print("Testing MSDataset (Multi-Spectral)")
    print("="*70)
    
    # Test MS dataset
    ms_dataset = MSDataset(
        hr_image_dir='test_images/PS-MS',  # Replace with your directory
        config_path='configs/default_config.yaml',
        global_stats_path='configs/combined_global_stats_ms.yaml',  # Optional
        augment=True,
        cache_size=10
    )
    
    print(f"\nMS Dataset size: {len(ms_dataset)}")
    
    # Get a sample
    ms_sample = ms_dataset[0]
    print(f"\nMS Sample 0:")
    print(f"  HR shape: {ms_sample['hr'].shape}")
    print(f"  Number of bands: {ms_sample['metadata']['num_bands']}")
    print(f"  Number of LR frames: {len(ms_sample['lr'])}")
    print(f"  LR[0] shape: {ms_sample['lr'][0].shape}")
    
    # Test MS DataLoader
    print("\nTesting MS DataLoader...")
    ms_dataloader = create_ms_dataloader(
        hr_image_dir='test_images/PS-MS',
        config_path='configs/default_config.yaml',
        batch_size=2,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        shuffle=True
    )
    
    for batch_idx, batch in enumerate(ms_dataloader):
        print(f"\nMS Batch {batch_idx}:")
        print(f"  HR batch shape: {batch['hr'].shape}")  # [B, C, H, W]
        print(f"  LR batch shapes: {[lr.shape for lr in batch['lr']]}")  # List of [B, C, H', W']
        print(f"  Shift values shape: {batch['shift_values'].shape}")
        if batch_idx >= 1:  # Test 2 batches
            break
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
    print(f"theta={sample['psf_params']['theta'][0]:.3f}")
    print(f"  Shift values: {sample['shift_values']}")
    print(f"  Metadata: {sample['metadata']}")
     
    # Test DataLoader
    print("\nTesting DataLoader...")
    dataloader = create_dataloader(
        hr_image_dir='test_images',
        config_path='configs/default_config.yaml',
        batch_size=4,
        num_workers=2,
        shuffle=True
    )
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  HR batch shape: {batch['hr'].shape}")
        print(f"  LR batch shapes: {[lr.shape for lr in batch['lr']]}")
        print(f"  Shift values shape: {batch['shift_values'].shape}")
        if batch_idx >= 2:  # Test first 3 batches
            break
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)
