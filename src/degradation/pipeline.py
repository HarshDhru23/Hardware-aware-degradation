"""
Main degradation pipeline implementing the observation model:
y_k = D * B_k * M_k * x + n_k

This pipeline supports both deterministic and stochastic sub-pixel shifts:

Deterministic mode (shift_mode='deterministic'):
- Mode 2: [(0.0, 0.0), (0.5, 0.5)]
- Mode 4: [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]

Stochastic mode (shift_mode='stochastic'):
- Shifts sampled from Gaussian distribution around nominal values
- Variance: 0.01-0.15 for 2x, ~0.03 for 4x (to simulate sensor jitter)
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any, List
import logging
from .operators import WarpingOperator, BlurOperator, DownsamplingOperator, NoiseOperator


class DegradationPipeline:
    """
    Hardware-aware degradation pipeline for satellite sensor simulation.
    
    Implements the complete observation model to generate synthetic training data
    for the ISRO Multi-Frame Super-Resolution project with 4 LR frames.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the degradation pipeline with configuration parameters.
        
        Args:
            config: Configuration dictionary containing all pipeline parameters
        """
        self.config = config
        self.downsampling_factor = config.get('downsampling_factor', 4)
        self.downsampling_mode = config.get('downsampling_mode', 4)
        self.shift_mode = config.get('shift_mode', 'deterministic')
        
        # Set shift values and num_lr_frames based on downsampling_mode
        if self.downsampling_mode == 2:
            self.shift_values = [[0.0, 0.0], [0.5, 0.5]]
            self.num_lr_frames = 2
            self.shift_variance = config.get('shift_variance_2x', 0.08)
        elif self.downsampling_mode == 4:
            self.shift_values = [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75]]
            self.num_lr_frames = 4
            self.shift_variance = config.get('shift_variance_4x', 0.03)
        else:
            raise ValueError(f"downsampling_mode must be 2 or 4, got {self.downsampling_mode}")
        
        # Allow config override of shift_values if explicitly provided
        if 'shift_values' in config:
            self.shift_values = config['shift_values']
        if 'num_lr_frames' in config:
            self.num_lr_frames = config['num_lr_frames']
        
        # Initialize operators for each LR frame
        self.warp_operators = []
        self.blur_operators = []
        self.noise_operators = []
        
        for i, (shift_lr_x, shift_lr_y) in enumerate(self.shift_values[:self.num_lr_frames]):
            # Convert LR shift to HR shift
            shift_hr_x = shift_lr_x * self.downsampling_factor
            shift_hr_y = shift_lr_y * self.downsampling_factor
            
            # Warping operator with deterministic or stochastic shift
            is_stochastic = (self.shift_mode == 'stochastic')
            warp_op = WarpingOperator(
                shift_x=shift_hr_x,
                shift_y=shift_hr_y,
                stochastic=is_stochastic,
                shift_mean_x=shift_lr_x,  # Use nominal shift as mean
                shift_mean_y=shift_lr_y,
                shift_variance=self.shift_variance
            )
            self.warp_operators.append(warp_op)
            
            # Blur operator (anisotropic Gaussian PSF) - shared parameters
            blur_op = BlurOperator(config)
            self.blur_operators.append(blur_op)
            
            # Noise operator - shared parameters
            noise_op = NoiseOperator(config)
            self.noise_operators.append(noise_op)
        
        # Downsampling operator (shared across all frames)
        self.downsample = DownsamplingOperator(config)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def generate_lr_frame(self, hr_image: np.ndarray, frame_idx: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a single LR frame from HR image with specified shift.
        
        Physical Pipeline:
        x -> M_k (warp) -> B_k (anisotropic PSF blur) -> D (downsample) -> n_k (Poisson+Gaussian+ADC) -> y_k
        
        Args:
            hr_image: High-resolution input image (H, W) or (H, W, C)
            frame_idx: Frame index (0, 1, 2, or 3)
            seed: Random seed for noise reproducibility
            
        Returns:
            LR frame image (H/factor, W/factor) or (H/factor, W/factor, C)
        """
        shift_lr = self.shift_values[frame_idx]
        self.logger.debug(f"Generating LR frame {frame_idx} with shift {shift_lr}")
        
        # Step 1: Warping (M_k) - Deterministic sub-pixel shift
        warped = self.warp_operators[frame_idx].apply(
            hr_image, 
            seed=seed, 
            downsampling_factor=self.downsampling_factor
        )
        
        # Step 2: Anisotropic Gaussian PSF blur (B_k)
        blurred = self.blur_operators[frame_idx].apply(warped)
        
        # Step 3: Downsampling with sensor PSF (D) - Spatial integration
        downsampled = self.downsample.apply(blurred)
        
        # Step 4: Noise + ADC - AFTER downsampling (on LR)
        # Applies: Poisson (photon shot) + Gaussian (read noise) + Quantization (ADC)
        frame_seed = seed + frame_idx if seed is not None else None
        lr_frame = self.noise_operators[frame_idx].apply_noise_and_quantization(
            downsampled, 
            seed=frame_seed
        )
        
        return lr_frame
    
    def generate_lr_frames(self, hr_image: np.ndarray, seed: Optional[int] = None) -> List[np.ndarray]:
        """
        Generate LR frames from HR image based on downsampling_mode.
        
        Args:
            hr_image: High-resolution input image (H, W) or (H, W, C)
            seed: Random seed for reproducibility
            
        Returns:
            List of LR frames with sub-pixel shifts:
            - Mode 2: 2 frames [(0,0), (0.5,0.5)]
            - Mode 4: 4 frames [(0,0), (0.25,0.25), (0.5,0.5), (0.75,0.75)]
        """
        self.logger.info(f"Generating {self.num_lr_frames} LR frames from HR image of shape {hr_image.shape}")
        
        lr_frames = []
        for i in range(self.num_lr_frames):
            lr_frame = self.generate_lr_frame(hr_image, frame_idx=i, seed=seed)
            lr_frames.append(lr_frame)
        
        return lr_frames
    
    # Backward compatibility methods
    def generate_lr1(self, hr_image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Generate LR1 (frame 0 with shift 0,0). Kept for backward compatibility."""
        return self.generate_lr_frame(hr_image, frame_idx=0, seed=seed)
    
    def generate_lr2(self, hr_image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Generate LR2 (frame 2 with shift 0.5,0.5). Kept for backward compatibility."""
        return self.generate_lr_frame(hr_image, frame_idx=2, seed=seed)
    
    def process_image(self, hr_image: np.ndarray, seed: Optional[int] = None) -> List[np.ndarray]:
        """
        Process a single HR image to generate LR frames based on downsampling_mode.
        
        Args:
            hr_image: High-resolution input image (H, W) or (H, W, C)
            seed: Random seed for reproducible results
            
        Returns:
            List of LR frames with shifts:
            - Mode 2: 2 frames [(0,0), (0.5,0.5)]
            - Mode 4: 4 frames [(0,0), (0.25,0.25), (0.5,0.5), (0.75,0.75)]
        """
        self.logger.info(f"Processing HR image of shape {hr_image.shape}")
        
        # Validate input
        if len(hr_image.shape) not in [2, 3]:
            raise ValueError(f"Input image must be 2D or 3D, got shape {hr_image.shape}")
        
        # Generate all 4 LR frames
        lr_frames = self.generate_lr_frames(hr_image, seed=seed)
        
        self.logger.info(f"Generated {len(lr_frames)} LR frames, each with shape: {lr_frames[0].shape}")
        
        return lr_frames
    
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