"""
Individual operators for the degradation pipeline.

Each operator implements a specific part of the observation model:
y_k = D * B_k * M_k * x + n_k
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import warnings


class WarpingOperator:
    """
    Warping operator M_k that applies geometric transformations.
    
    For P1 sensor: Identity (no shift)
    For P2 sensor: (0.5, 0.5) LR pixel shift = (2, 2) HR pixel shift for 4x factor
    """
    
    def __init__(self, shift_x: float = 0.0, shift_y: float = 0.0):
        """
        Initialize warping operator.
        
        Args:
            shift_x: Horizontal shift in HR pixels (range: 0-4 for 4x factor)
            shift_y: Vertical shift in HR pixels (range: 0-4 for 4x factor)
        """
        self.shift_x = np.clip(shift_x, 0.0, 4.0)
        self.shift_y = np.clip(shift_y, 0.0, 4.0)
        
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply warping transformation to image.
        
        Args:
            image: Input HR image (H, W) or (H, W, C)
        
        Returns:
            Warped image with same shape as input
        """
        if self.shift_x == 0.0 and self.shift_y == 0.0:
            return image.copy()
        
        # Create transformation matrix
        transformation_matrix = np.array([
            [1, 0, self.shift_x],
            [0, 1, self.shift_y]
        ], dtype=np.float32)
        
        # Apply affine transformation
        if len(image.shape) == 2:
            h, w = image.shape
            warped = cv2.warpAffine(
                image, 
                transformation_matrix, 
                (w, h), 
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
        else:
            h, w, c = image.shape
            warped = cv2.warpAffine(
                image, 
                transformation_matrix, 
                (w, h), 
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
        
        return warped


class BlurOperator:
    """
    Blur operator B_k that applies optical and motion blur.
    
    Optical blur: 2D Gaussian kernel
    Motion blur: 1D vertical kernel (TDI velocity mismatch simulation)
    """
    
    def __init__(self, 
                 optical_sigma: float = 1.0,
                 optical_kernel_size: int = 5,
                 motion_kernel_size: int = 3):
        """
        Initialize blur operator.
        
        Args:
            optical_sigma: Gaussian blur standard deviation (range: 0.5-3.0)
            optical_kernel_size: Gaussian kernel size (range: 3-15, odd numbers)
            motion_kernel_size: Motion blur kernel size (range: 1-9, odd numbers)
        """
        self.optical_sigma = np.clip(optical_sigma, 0.5, 3.0)
        self.optical_kernel_size = max(3, min(15, optical_kernel_size))
        if self.optical_kernel_size % 2 == 0:
            self.optical_kernel_size += 1
            
        self.motion_kernel_size = max(1, min(9, motion_kernel_size))
        if self.motion_kernel_size % 2 == 0:
            self.motion_kernel_size += 1
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply optical and motion blur to image.
        
        Args:
            image: Input image (H, W) or (H, W, C)
        
        Returns:
            Blurred image with same shape as input
        """
        # Apply optical blur (2D Gaussian)
        blurred = gaussian_filter(image, sigma=self.optical_sigma)
        
        # Apply motion blur (1D vertical kernel)
        if self.motion_kernel_size > 1:
            # Create 1D motion blur kernel (vertical direction)
            motion_kernel = np.ones((self.motion_kernel_size, 1)) / self.motion_kernel_size
            
            if len(image.shape) == 2:
                blurred = cv2.filter2D(blurred, -1, motion_kernel)
            else:
                # Apply to each channel separately
                for c in range(image.shape[2]):
                    blurred[:, :, c] = cv2.filter2D(blurred[:, :, c], -1, motion_kernel)
        
        return blurred


class DownsamplingOperator:
    """
    Downsampling operator D that combines sensor PSF and downsampling.
    
    Uses average pooling with kernel_size=factor and stride=factor to simulate:
    1. Sensor PSF (box filter averaging)
    2. Downsampling (undersampling)
    """
    
    def __init__(self, downsampling_factor: int = 4):
        """
        Initialize downsampling operator.
        
        Args:
            downsampling_factor: Downsampling factor (range: 2-8)
        """
        self.factor = max(2, min(8, downsampling_factor))
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply downsampling with sensor PSF simulation.
        
        Args:
            image: Input HR image (H, W) or (H, W, C)
        
        Returns:
            Downsampled LR image (H/factor, W/factor) or (H/factor, W/factor, C)
        """
        if len(image.shape) == 2:
            h, w = image.shape
            # Ensure dimensions are divisible by factor
            h_new = (h // self.factor) * self.factor
            w_new = (w // self.factor) * self.factor
            image_cropped = image[:h_new, :w_new]
            
            # Apply average pooling
            pooled = image_cropped.reshape(
                h_new // self.factor, self.factor,
                w_new // self.factor, self.factor
            ).mean(axis=(1, 3))
            
        else:
            h, w, c = image.shape
            # Ensure dimensions are divisible by factor
            h_new = (h // self.factor) * self.factor
            w_new = (w // self.factor) * self.factor
            image_cropped = image[:h_new, :w_new, :]
            
            # Apply average pooling to each channel
            pooled = image_cropped.reshape(
                h_new // self.factor, self.factor,
                w_new // self.factor, self.factor, c
            ).mean(axis=(1, 3))
        
        return pooled


class NoiseOperator:
    """
    Noise operator n_k that adds Gaussian and Poisson noise.
    
    Gaussian noise: Additive with configurable mean and variance
    Poisson noise: Signal-dependent noise
    """
    
    def __init__(self, 
                 gaussian_mean: float = 0.0,
                 gaussian_std: float = 5.0,
                 poisson_lambda: float = 1.0,
                 enable_gaussian: bool = True,
                 enable_poisson: bool = True):
        """
        Initialize noise operator.
        
        Args:
            gaussian_mean: Gaussian noise mean (range: -10.0 to 10.0)
            gaussian_std: Gaussian noise standard deviation (range: 1.0-20.0)
            poisson_lambda: Poisson noise scaling factor (range: 0.1-5.0)
            enable_gaussian: Whether to add Gaussian noise
            enable_poisson: Whether to add Poisson noise
        """
        self.gaussian_mean = np.clip(gaussian_mean, -10.0, 10.0)
        self.gaussian_std = np.clip(gaussian_std, 1.0, 20.0)
        self.poisson_lambda = np.clip(poisson_lambda, 0.1, 5.0)
        self.enable_gaussian = enable_gaussian
        self.enable_poisson = enable_poisson
    
    def apply(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Add noise to image.
        
        Args:
            image: Input LR image (H, W) or (H, W, C)
            seed: Random seed for reproducible noise
        
        Returns:
            Noisy image with same shape and dtype as input
        """
        if seed is not None:
            np.random.seed(seed)
        
        noisy_image = image.copy().astype(np.float32)
        
        # Add Gaussian noise
        if self.enable_gaussian:
            gaussian_noise = np.random.normal(
                self.gaussian_mean, 
                self.gaussian_std, 
                image.shape
            ).astype(np.float32)
            noisy_image += gaussian_noise
        
        # Add Poisson noise (signal-dependent)
        if self.enable_poisson:
            # Ensure positive values for Poisson noise
            positive_image = np.maximum(noisy_image, 0.0)
            # Scale by lambda and apply Poisson noise
            scaled_image = positive_image * self.poisson_lambda
            poisson_noise = np.random.poisson(scaled_image).astype(np.float32)
            # Convert back to original scale
            poisson_noise = poisson_noise / self.poisson_lambda
            # Add the noise component
            noise_component = poisson_noise - positive_image
            noisy_image += noise_component
        
        # Clip to valid range and convert back to original dtype
        if image.dtype == np.uint8:
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            noisy_image = np.clip(noisy_image, 0, 65535).astype(np.uint16)
        else:
            noisy_image = noisy_image.astype(image.dtype)
        
        return noisy_image