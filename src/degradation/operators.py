"""
Individual operators for the degradation pipeline.

Each operator implements a specific part of the observation model:
y_k = D * B_k * M_k * x + n_k
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import warnings


class WarpingOperator:
    """
    Warping operator M_k that applies geometric transformations.
    
    For P1 sensor: Identity (no shift)
    For P2 sensor: Stochastic shift drawn from Gaussian distribution
                   - Mean: 0.5 LR pixels (half-pixel offset)
                   - Std: 0.1 LR pixels (sensor jitter/misalignment)
    """
    
    def __init__(self, shift_x: float = 0.0, shift_y: float = 0.0, 
                 stochastic: bool = False, shift_mean: float = 0.5, shift_std: float = 0.1):
        """
        Initialize warping operator.
        
        Args:
            shift_x: Horizontal shift in HR pixels (deterministic mode)
            shift_y: Vertical shift in HR pixels (deterministic mode)
            stochastic: If True, sample shifts from Gaussian distribution
            shift_mean: Mean shift value for stochastic mode (in LR pixels)
            shift_std: Std deviation for stochastic mode (in LR pixels)
        """
        self.shift_x = np.clip(shift_x, 0.0, 4.0)
        self.shift_y = np.clip(shift_y, 0.0, 4.0)
        self.stochastic = stochastic
        self.shift_mean = shift_mean
        self.shift_std = shift_std
        
    def apply(self, image: np.ndarray, seed: Optional[int] = None, downsampling_factor: int = 4) -> np.ndarray:
        """
        Apply warping transformation to image.
        
        Args:
            image: Input HR image (H, W) or (H, W, C)
            seed: Random seed for stochastic shift (only used if stochastic=True)
            downsampling_factor: Downsampling factor to convert LR shift to HR shift
        
        Returns:
            Warped image with same shape as input
        """
        # Determine actual shift values
        if self.stochastic:
            # Sample from Gaussian distribution
            if seed is not None:
                np.random.seed(seed)
            # Sample shift in LR pixels, then convert to HR pixels
            shift_x_lr = np.random.normal(self.shift_mean, self.shift_std)
            shift_y_lr = np.random.normal(self.shift_mean, self.shift_std)
            shift_x_hr = shift_x_lr * downsampling_factor
            shift_y_hr = shift_y_lr * downsampling_factor
            # Clip to reasonable range
            shift_x_hr = np.clip(shift_x_hr, -4.0, 4.0)
            shift_y_hr = np.clip(shift_y_hr, -4.0, 4.0)
        else:
            # Use deterministic shifts
            shift_x_hr = self.shift_x
            shift_y_hr = self.shift_y
        
        if shift_x_hr == 0.0 and shift_y_hr == 0.0:
            return image.copy()
        
        # Create transformation matrix with computed shifts
        transformation_matrix = np.array([
            [1, 0, shift_x_hr],
            [0, 1, shift_y_hr]
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
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize blur operator from config.
        
        Args:
            config: Configuration dictionary containing blur parameters
        """
        # Load values from config - these MUST come from config file
        # No fallback defaults to ensure config changes are always reflected
        self.optical_sigma = config.get('optical_sigma')
        self.optical_kernel_size = config.get('optical_kernel_size')
        self.motion_kernel_size = config.get('motion_kernel_size')
        
        if self.optical_sigma is None or self.optical_kernel_size is None or self.motion_kernel_size is None:
            raise ValueError("BlurOperator requires 'optical_sigma', 'optical_kernel_size', and 'motion_kernel_size' in config")
        
        # Validate and clip to safe ranges
        self.optical_sigma = np.clip(self.optical_sigma, 0.5, 3.0)
        self.optical_kernel_size = max(3, min(15, self.optical_kernel_size))
        if self.optical_kernel_size % 2 == 0:
            self.optical_kernel_size += 1
            
        self.motion_kernel_size = max(1, min(9, self.motion_kernel_size))
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
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize downsampling operator from config.
        
        Args:
            config: Configuration dictionary containing downsampling parameters
        """
        # Load value from config - MUST come from config file
        downsampling_factor = config.get('downsampling_factor')
        if downsampling_factor is None:
            raise ValueError("DownsamplingOperator requires 'downsampling_factor' in config")
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
    Noise operator n_k that adds realistic Poisson-Gaussian noise.
    
    Models the complete noise in digital imaging sensors:
    - Poisson noise: Signal-dependent photon shot noise (sqrt(N) statistics)
    - Gaussian noise: Signal-independent read noise from electronics
    
    Combined model: y = Poisson(x * gain) / gain + Gaussian(0, sigma_read)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize noise operator from config.
        
        Args:
            config: Configuration dictionary containing noise parameters
        """
        # Load values from config - these MUST come from config file
        # No fallback defaults to ensure config changes are always reflected
        self.gaussian_mean = config.get('gaussian_mean')
        self.gaussian_std = config.get('gaussian_std')
        self.poisson_lambda = config.get('poisson_lambda')
        self.enable_gaussian = config.get('enable_gaussian')
        self.enable_poisson = config.get('enable_poisson')
        
        # Photon gain factor for Poisson noise (higher = more photons = less relative noise)
        # For satellite sensors: 100-1000 is typical for well-lit scenes
        self.photon_gain = config.get('photon_gain')
        
        # Validate all required parameters are present
        if any(x is None for x in [self.gaussian_std, self.enable_gaussian, self.enable_poisson, self.photon_gain]):
            raise ValueError("NoiseOperator requires 'gaussian_std', 'enable_gaussian', 'enable_poisson', and 'photon_gain' in config")
        
        # Validate and clip to safe ranges
        self.gaussian_mean = np.clip(self.gaussian_mean, -10.0, 10.0)
        self.gaussian_std = np.clip(self.gaussian_std, 0.0, 20.0)
        self.poisson_lambda = np.clip(self.poisson_lambda, 0.1, 5.0)
        self.photon_gain = np.clip(self.photon_gain, 10.0, 10000.0)
    
    def apply(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Add realistic Poisson-Gaussian noise to image.
        
        Implements the physical noise model for digital sensors:
        1. Photon shot noise (Poisson): sqrt(N) noise where N = photon count
        2. Read noise (Gaussian): Electronic noise independent of signal
        
        For normalized [0,1] images:
        - Scales to photon counts: x → x * photon_gain
        - Adds Poisson noise: Poisson(x * gain)
        - Scales back: / photon_gain
        - Adds Gaussian read noise
        
        Args:
            image: Input LR image (H, W) or (H, W, C), normalized to [0, 1]
            seed: Random seed for reproducible noise
        
        Returns:
            Noisy image with same shape and dtype, clipped to [0, 1]
        """
        if seed is not None:
            np.random.seed(seed)
        
        noisy_image = image.copy().astype(np.float64)  # Use float64 for precision
        
        # Step 1: Add Poisson noise (signal-dependent photon shot noise)
        if self.enable_poisson:
            # Scale normalized image to photon counts
            # For [0,1] image: 0.5 intensity → photon_gain/2 photons
            photon_counts = noisy_image * self.photon_gain
            
            # Ensure non-negative (Poisson requires λ >= 0)
            photon_counts = np.maximum(photon_counts, 0.0)
            
            # Generate Poisson-distributed photon counts
            # Variance = Mean for Poisson distribution
            noisy_photons = np.random.poisson(photon_counts).astype(np.float64)
            
            # Scale back to [0,1] range
            noisy_image = noisy_photons / self.photon_gain
        
        # Step 2: Add Gaussian noise (signal-independent read noise)
        if self.enable_gaussian:
            # Electronic read noise from sensor circuits
            read_noise = np.random.normal(
                self.gaussian_mean,
                self.gaussian_std,
                image.shape
            ).astype(np.float64)
            noisy_image += read_noise
        
        # Clip to valid range [0, 1] for normalized images
        noisy_image = np.clip(noisy_image, 0.0, 1.0)
        
        # Convert back to original dtype
        return noisy_image.astype(image.dtype)