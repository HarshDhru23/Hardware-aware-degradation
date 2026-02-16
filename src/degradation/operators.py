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
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import bicubic interpolation with antialiasing
from utils import bicubic_core


class WarpingOperator:
    """
    Warping operator M_k that applies geometric transformations.
    
    For P1 sensor: Identity (no shift)
    For P2 sensor: Stochastic shift drawn from Gaussian distribution
                   - Mean: 0.5 LR pixels (half-pixel offset)
                   - Std: 0.1 LR pixels (sensor jitter/misalignment)
    """
    
    def __init__(self, shift_x: float = 0.0, shift_y: float = 0.0, 
                 stochastic: bool = False, shift_mean_x: float = 0.0, shift_mean_y: float = 0.0,
                 shift_variance: float = 0.08):
        """
        Initialize warping operator.
        
        Args:
            shift_x: Horizontal shift in HR pixels (deterministic mode)
            shift_y: Vertical shift in HR pixels (deterministic mode)
            stochastic: If True, sample shifts from Gaussian distribution around mean
            shift_mean_x: Mean horizontal shift for stochastic mode (in LR pixels)
            shift_mean_y: Mean vertical shift for stochastic mode (in LR pixels)
            shift_variance: Variance for stochastic shifts (std = sqrt(variance))
                          For 2x: 0.01-0.15, For 4x: ~0.03 (12% of nominal shift)
        """
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.stochastic = stochastic
        self.shift_mean_x = shift_mean_x
        self.shift_mean_y = shift_mean_y
        self.shift_std = np.sqrt(shift_variance)  # Convert variance to std deviation
        
    def apply(self, image: np.ndarray, seed: Optional[int] = None, downsampling_factor: int = 4) -> np.ndarray:
        """
        Apply warping transformation to image.
        
        Args:
            image: Input HR image (H, W) or (H, W, C)
            seed: Random seed for stochastic shift (only used if stochastic=True)
            downsampling_factor: Downsampling factor to convert LR shift to HR shift
        
        Returns:
            Warped image with same shape as input
            
        Note:
            The actual shift values used are stored in self.last_shift_x_lr and self.last_shift_y_lr
        """
        # Determine actual shift values
        if self.stochastic:
            # Sample from Gaussian distribution around nominal shift
            if seed is not None:
                np.random.seed(seed)
            # Sample shift in LR pixels around the mean, then convert to HR pixels
            shift_x_lr = np.random.normal(self.shift_mean_x, self.shift_std)
            shift_y_lr = np.random.normal(self.shift_mean_y, self.shift_std)
            shift_x_hr = shift_x_lr * downsampling_factor
            shift_y_hr = shift_y_lr * downsampling_factor
            # Clip to reasonable range
            shift_x_hr = np.clip(shift_x_hr, -4.0 * downsampling_factor, 4.0 * downsampling_factor)
            shift_y_hr = np.clip(shift_y_hr, -4.0 * downsampling_factor, 4.0 * downsampling_factor)
            # Recompute LR shifts from clipped HR shifts for accurate ground truth
            shift_x_lr = shift_x_hr / downsampling_factor
            shift_y_lr = shift_y_hr / downsampling_factor
        else:
            # Use deterministic shifts
            shift_x_hr = self.shift_x
            shift_y_hr = self.shift_y
            shift_x_lr = shift_x_hr / downsampling_factor
            shift_y_lr = shift_y_hr / downsampling_factor
        
        # Store actual shift values for access after apply() call
        self.last_shift_x_lr = shift_x_lr
        self.last_shift_y_lr = shift_y_lr
        
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
    Anisotropic Gaussian PSF blur operator B_k.
    
    Models the satellite sensor Point Spread Function (PSF) as a 2D anisotropic Gaussian
    with different sigma values in x and y directions, and optional rotation.
    
    PSF(x, y) = exp(-((x*cos(θ) + y*sin(θ))²/(2σ_x²) + (-x*sin(θ) + y*cos(θ))²/(2σ_y²)))
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize anisotropic blur operator from config.
        
        Args:
            config: Configuration dictionary containing PSF blur parameters
        """
        # Load values from config
        self.sigma_x = config.get('psf_sigma_x')
        self.sigma_y = config.get('psf_sigma_y')
        self.theta = config.get('psf_theta', 0.0)  # Orientation angle in degrees
        self.kernel_size = config.get('psf_kernel_size')
        
        if self.sigma_x is None or self.sigma_y is None or self.kernel_size is None:
            raise ValueError("BlurOperator requires 'psf_sigma_x', 'psf_sigma_y', and 'psf_kernel_size' in config")
        
        # Validate and clip to safe ranges
        self.sigma_x = np.clip(self.sigma_x, 0.5, 2.0)
        self.sigma_y = np.clip(self.sigma_y, 0.5, 2.0)
        self.theta = np.clip(self.theta, 0.0, 180.0)
        self.kernel_size = max(3, min(15, self.kernel_size))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply anisotropic Gaussian PSF blur to image.
        
        Args:
            image: Input image (H, W) or (H, W, C)
        
        Returns:
            Blurred image with same shape as input
        """
        # For isotropic case (sigma_x == sigma_y and theta == 0), use simple Gaussian
        if abs(self.sigma_x - self.sigma_y) < 0.01 and abs(self.theta) < 0.01:
            blurred = gaussian_filter(image, sigma=self.sigma_x)
        else:
            # Create anisotropic Gaussian kernel
            kernel = self._create_anisotropic_gaussian_kernel()
            
            # Apply convolution
            if len(image.shape) == 2:
                blurred = cv2.filter2D(image, -1, kernel)
            else:
                # Apply to each channel separately
                blurred = np.zeros_like(image)
                for c in range(image.shape[2]):
                    blurred[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)
        
        return blurred
    
    def _create_anisotropic_gaussian_kernel(self) -> np.ndarray:
        """
        Create 2D anisotropic Gaussian kernel with rotation.
        
        Returns:
            Normalized 2D Gaussian kernel of size (kernel_size, kernel_size)
        """
        # Create coordinate grid
        ax = np.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        
        # Convert theta to radians
        theta_rad = np.deg2rad(self.theta)
        
        # Rotation matrix components
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        
        # Rotate coordinates
        x_rot = xx * cos_theta + yy * sin_theta
        y_rot = -xx * sin_theta + yy * cos_theta
        
        # Anisotropic Gaussian
        kernel = np.exp(-(x_rot**2 / (2 * self.sigma_x**2) + y_rot**2 / (2 * self.sigma_y**2)))
        
        # Normalize
        kernel = kernel / kernel.sum()
        
        return kernel


class DownsamplingOperator:
    """
    Downsampling operator D with bicubic anti-aliasing.
    
    Uses bicubic interpolation with anti-aliasing to properly handle high-frequency content
    and prevent aliasing artifacts during downsampling.
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
        Apply bicubic downsampling with anti-aliasing.
        
        Args:
            image: Input HR image (H, W) or (H, W, C)
        
        Returns:
            Downsampled LR image (H/factor, W/factor) or (H/factor, W/factor, C)
        """
        # Calculate output shape
        if len(image.shape) == 2:
            h, w = image.shape
            output_shape = (h // self.factor, w // self.factor)
        else:
            h, w, c = image.shape
            output_shape = (h // self.factor, w // self.factor)
        
        # Use bicubic interpolation with anti-aliasing
        # scale_factor = 1 / downsampling_factor (e.g., 0.25 for 4x downsampling)
        scale_factor = 1.0 / self.factor
        
        # Apply bicubic downsampling with anti-aliasing
        downsampled = bicubic_core.imresize(
            image, 
            scale_factor=scale_factor,
            output_shape=output_shape,
            kernel='cubic',
            antialiasing=True
        )
        
        return downsampled


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
        
        # Quantization parameters (ADC simulation)
        self.enable_quantization = config.get('enable_quantization', True)
        self.quantization_bits = config.get('quantization_bits', 11)  # WorldView-3 default
        
        # Validate all required parameters are present
        if any(x is None for x in [self.gaussian_std, self.enable_gaussian, self.enable_poisson, self.photon_gain]):
            raise ValueError("NoiseOperator requires 'gaussian_std', 'enable_gaussian', 'enable_poisson', and 'photon_gain' in config")
        
        # Validate and clip to safe ranges
        self.gaussian_mean = np.clip(self.gaussian_mean, -10.0, 10.0)
        self.gaussian_std = np.clip(self.gaussian_std, 0.0001, 0.1)  # Reduced range for normalized images
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
    
    def apply_poisson_only(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply ONLY Poisson noise (photon shot noise) to HR image.
        This should be called BEFORE downsampling to simulate physical sensor behavior.
        
        Args:
            image: Input HR image (H, W) or (H, W, C), normalized to [0, 1]
            seed: Random seed for reproducible noise
        
        Returns:
            Image with Poisson noise, clipped to [0, 1]
        """
        if not self.enable_poisson:
            return image.copy()
        
        if seed is not None:
            np.random.seed(seed)
        
        noisy_image = image.copy().astype(np.float64)
        
        # Scale to photon counts
        photon_counts = noisy_image * self.photon_gain
        photon_counts = np.maximum(photon_counts, 0.0)
        
        # Apply Poisson noise
        noisy_photons = np.random.poisson(photon_counts).astype(np.float64)
        
        # Scale back to [0,1]
        noisy_image = noisy_photons / self.photon_gain
        noisy_image = np.clip(noisy_image, 0.0, 1.0)
        
        return noisy_image.astype(image.dtype)
    
    def apply_gaussian_only(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply ONLY Gaussian noise (read noise) to image.
        
        Args:
            image: Input image (H, W) or (H, W, C), normalized to [0, 1]
            seed: Random seed for reproducible noise
        
        Returns:
            Image with Gaussian read noise, clipped to [0, 1]
        """
        if not self.enable_gaussian:
            return image.copy()
        
        if seed is not None:
            np.random.seed(seed)
        
        noisy_image = image.copy().astype(np.float64)
        
        # Add Gaussian read noise
        read_noise = np.random.normal(
            self.gaussian_mean,
            self.gaussian_std,
            image.shape
        ).astype(np.float64)
        noisy_image += read_noise
        
        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0.0, 1.0)
        
        return noisy_image.astype(image.dtype)
    
    def apply_noise_and_quantization(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply complete noise model: Poisson + Gaussian + ADC quantization.
        
        This should be applied to LR images AFTER downsampling.
        
        Physical ordering for sensor:
        1. Downsampling (spatial integration)
        2. Poisson noise (photon shot noise on integrated signal)
        3. Gaussian noise (electronic read noise)
        4. ADC quantization
        
        Photon gain formula: photon_gain = normalizing_number * sensor_factor
        - normalizing_number = 2047 (for 11-bit ADC)
        - sensor_factor = 10-20 (random per sensor, fixed at ~14.65 for 30k gain)
        - Typical range: 20k - 40k
        - Using: 30000 for our simulations
        
        Args:
            image: Input LR image (H, W) or (H, W, C), normalized to [0, 1]
            seed: Random seed for reproducible noise
        
        Returns:
            Image with Poisson + Gaussian noise and quantization, clipped to [0, 1]
        """
        if seed is not None:
            np.random.seed(seed)
        
        noisy_image = image.copy().astype(np.float64)
        
        # Step 1: Apply Poisson noise (photon shot noise on LR after downsampling)
        if self.enable_poisson:
            # Scale to photon counts
            photon_counts = noisy_image * self.photon_gain
            photon_counts = np.maximum(photon_counts, 0.0)
            
            # Apply Poisson noise
            noisy_photons = np.random.poisson(photon_counts).astype(np.float64)
            
            # Scale back to [0,1]
            noisy_image = noisy_photons / self.photon_gain
            noisy_image = np.clip(noisy_image, 0.0, 1.0)
        
        # Step 2: Add Gaussian read noise
        if self.enable_gaussian:
            read_noise = np.random.normal(
                self.gaussian_mean,
                self.gaussian_std,
                image.shape
            ).astype(np.float64)
            noisy_image += read_noise
            
            # Clip to valid range before quantization
            noisy_image = np.clip(noisy_image, 0.0, 1.0)
        
        # Step 3: Apply ADC quantization (WorldView-3 is 11-bit)
        if self.enable_quantization:
            max_value = (2 ** self.quantization_bits) - 1  # e.g., 2047 for 11-bit
            
            # Quantize: [0,1] → [0, max_value] → round → [0,1]
            quantized = np.round(noisy_image * max_value)
            noisy_image = quantized / max_value
        
        return noisy_image.astype(image.dtype)