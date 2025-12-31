"""
Main degradation pipeline implementing the observation model:
y_k = D * B_k * M_k * x + n_k

This pipeline generates two LR images (LR1, LR2) from one HR image,
simulating the P1 and P2 satellite sensors with (0.5, 0.5) pixel offset.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
import logging
from .operators import WarpingOperator, BlurOperator, DownsamplingOperator, NoiseOperator


class DegradationPipeline:
    """
    Hardware-aware degradation pipeline for satellite sensor simulation.
    
    Implements the complete observation model to generate synthetic training data
    for the ISRO Multi-Frame Super-Resolution project.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the degradation pipeline with configuration parameters.
        
        Args:
            config: Configuration dictionary containing all pipeline parameters
        """
        self.config = config
        self.downsampling_factor = config.get('downsampling_factor', 4)
        
        # Initialize operators for LR1 (P1 sensor - reference frame)
        self.warp_lr1 = WarpingOperator(
            shift_x=0.0,  # No shift for reference frame
            shift_y=0.0
        )
        
        # Pass full config to BlurOperator - it will read its own parameters
        self.blur_lr1 = BlurOperator(config)
        
        # Initialize operators for LR2 (P2 sensor - shifted frame)
        # Use stochastic shift with Gaussian distribution
        shift_mean = config.get('shift_mean', 0.5)  # Mean shift in LR pixels
        shift_std = config.get('shift_std', 0.1)    # Std dev for shift jitter
        
        self.warp_lr2 = WarpingOperator(
            shift_x=0.0,  # Not used in stochastic mode
            shift_y=0.0,
            stochastic=True,
            shift_mean=shift_mean,
            shift_std=shift_std
        )
        
        # Pass full config to BlurOperator - it will read its own parameters
        self.blur_lr2 = BlurOperator(config)
        
        # Pass full config to DownsamplingOperator
        self.downsample = DownsamplingOperator(config)
        
        # Pass full config to NoiseOperators - they will read their own parameters
        self.noise_lr1 = NoiseOperator(config)
        self.noise_lr2 = NoiseOperator(config)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def generate_lr1(self, hr_image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate LR1 (reference frame) from HR image.
        
        CORRECT Physical Pipeline: 
        x -> M_1 (warp) -> B_1 (blur) -> n_poisson (photon shot) -> D (downsample) -> n_gaussian (read noise) -> y_1
        
        Args:
            hr_image: High-resolution input image (H, W) or (H, W, C)
            seed: Random seed for noise reproducibility
            
        Returns:
            LR1 image (H/factor, W/factor) or (H/factor, W/factor, C)
        """
        self.logger.debug("Generating LR1 (reference frame)")
        
        # Step 1: Warping (M_1) - Identity operation
        warped = self.warp_lr1.apply(hr_image, seed=seed, downsampling_factor=self.downsampling_factor)
        
        # Step 2: Blurring (B_1) - Optical + Motion blur
        blurred = self.blur_lr1.apply(warped)
        
        # Step 3: Poisson noise (photon shot noise) - BEFORE downsampling (on HR)
        poisson_noisy = self.noise_lr1.apply_poisson_only(blurred, seed=seed)
        
        # Step 4: Downsampling with sensor PSF (D) - Spatial integration smooths noise
        downsampled = self.downsample.apply(poisson_noisy)
        
        # Step 5: Gaussian noise (electronic read noise) - AFTER downsampling (on LR)
        lr1 = self.noise_lr1.apply_gaussian_only(downsampled, seed=seed)
        
        return lr1
    
    def generate_lr2(self, hr_image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate LR2 (shifted frame) from HR image.
        
        CORRECT Physical Pipeline:
        x -> M_2 (warp+shift) -> B_2 (blur) -> n_poisson (photon shot) -> D (downsample) -> n_gaussian (read noise) -> y_2
        
        Args:
            hr_image: High-resolution input image (H, W) or (H, W, C)
            seed: Random seed for noise reproducibility
            
        Returns:
            LR2 image (H/factor, W/factor) or (H/factor, W/factor, C)
        """
        self.logger.debug("Generating LR2 (shifted frame with stochastic shift)")
        
        # Step 1: Warping (M_2) - Stochastic sub-pixel shift from Gaussian distribution
        warped = self.warp_lr2.apply(hr_image, seed=seed, downsampling_factor=self.downsampling_factor)
        
        # Step 2: Blurring (B_2) - Optical + Motion blur
        blurred = self.blur_lr2.apply(warped)
        
        # Step 3: Poisson noise (photon shot noise) - BEFORE downsampling (on HR)
        # Use different seed for independent noise realization
        poisson_seed = seed + 1 if seed is not None else None
        poisson_noisy = self.noise_lr2.apply_poisson_only(blurred, seed=poisson_seed)
        
        # Step 4: Downsampling with sensor PSF (D) - Spatial integration smooths noise
        downsampled = self.downsample.apply(poisson_noisy)
        
        # Step 5: Gaussian noise (electronic read noise) - AFTER downsampling (on LR)
        lr2 = self.noise_lr2.apply_gaussian_only(downsampled, seed=poisson_seed)
        
        return lr2
    
    def process_image(self, hr_image: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single HR image to generate both LR1 and LR2.
        
        Args:
            hr_image: High-resolution input image (H, W) or (H, W, C)
            seed: Random seed for reproducible results
            
        Returns:
            Tuple of (LR1, LR2) images
        """
        self.logger.info(f"Processing HR image of shape {hr_image.shape}")
        
        # Validate input
        if len(hr_image.shape) not in [2, 3]:
            raise ValueError(f"Input image must be 2D or 3D, got shape {hr_image.shape}")
        
        # Generate both LR images
        lr1 = self.generate_lr1(hr_image, seed=seed)
        lr2 = self.generate_lr2(hr_image, seed=seed)
        
        self.logger.info(f"Generated LR1: {lr1.shape}, LR2: {lr2.shape}")
        
        return lr1, lr2
    
    def get_config(self) -> Dict[str, Any]:
        """Get current pipeline configuration."""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update pipeline configuration and reinitialize operators.
        
        Args:
            new_config: New configuration parameters
        """
        self.config.update(new_config)
        
        # Reinitialize operators with new config
        self.__init__(self.config)
        
        self.logger.info("Pipeline configuration updated")
    
    def validate_image_dimensions(self, hr_image: np.ndarray) -> bool:
        """
        Validate that HR image dimensions are compatible with downsampling factor.
        
        Args:
            hr_image: HR image to validate
            
        Returns:
            True if dimensions are valid, False otherwise
        """
        if len(hr_image.shape) == 2:
            h, w = hr_image.shape
        else:
            h, w, _ = hr_image.shape
        
        return (h % self.downsampling_factor == 0 and 
                w % self.downsampling_factor == 0)
    
    def get_output_dimensions(self, hr_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Calculate output LR dimensions given HR shape.
        
        Args:
            hr_shape: Shape of HR image
            
        Returns:
            Shape of output LR images
        """
        if len(hr_shape) == 2:
            h, w = hr_shape
            return (h // self.downsampling_factor, w // self.downsampling_factor)
        else:
            h, w, c = hr_shape
            return (h // self.downsampling_factor, w // self.downsampling_factor, c)