"""
Validation utilities for images and configuration parameters.

Provides functions to validate input data and configuration parameters
to ensure they meet the requirements of the degradation pipeline.
"""

import numpy as np
from typing import Dict, Any, List, Union, Tuple
import logging


def validate_image(image: np.ndarray, 
                  name: str = "image",
                  min_size: Tuple[int, int] = (64, 64),
                  max_size: Tuple[int, int] = (8192, 8192),
                  allowed_dtypes: List[type] = None) -> bool:
    """
    Validate image array properties.
    
    Args:
        image: Image array to validate
        name: Name for error messages
        min_size: Minimum (height, width) 
        max_size: Maximum (height, width)
        allowed_dtypes: List of allowed data types
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    logger = logging.getLogger(__name__)
    
    if allowed_dtypes is None:
        allowed_dtypes = [np.uint8, np.uint16, np.float32, np.float64]
    
    # Check if it's a numpy array
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be a numpy array, got {type(image)}")
    
    # Check dimensions
    if len(image.shape) not in [2, 3]:
        raise ValueError(f"{name} must be 2D or 3D, got shape {image.shape}")
    
    # Check size limits
    h, w = image.shape[:2]
    min_h, min_w = min_size
    max_h, max_w = max_size
    
    if h < min_h or w < min_w:
        raise ValueError(f"{name} size {(h, w)} is smaller than minimum {min_size}")
    
    if h > max_h or w > max_w:
        raise ValueError(f"{name} size {(h, w)} is larger than maximum {max_size}")
    
    # Check data type
    if image.dtype.type not in allowed_dtypes:
        raise ValueError(f"{name} dtype {image.dtype} not in allowed types {allowed_dtypes}")
    
    # Check for invalid values
    if np.any(np.isnan(image)):
        raise ValueError(f"{name} contains NaN values")
    
    if np.any(np.isinf(image)):
        raise ValueError(f"{name} contains infinite values")
    
    # Check value ranges based on dtype
    if image.dtype == np.uint8:
        if np.any(image < 0) or np.any(image > 255):
            raise ValueError(f"{name} with uint8 dtype has values outside [0, 255]")
    elif image.dtype == np.uint16:
        if np.any(image < 0) or np.any(image > 65535):
            raise ValueError(f"{name} with uint16 dtype has values outside [0, 65535]")
    elif image.dtype in [np.float32, np.float64]:
        # For float images, check if they're in a reasonable range
        if np.any(image < -10) or np.any(image > 10):
            logger.warning(f"{name} with float dtype has values outside [-10, 10]")
    
    logger.debug(f"{name} validation passed: shape={image.shape}, dtype={image.dtype}")
    return True


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate pipeline configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    logger = logging.getLogger(__name__)
    
    # Define parameter ranges
    param_ranges = {
        'downsampling_factor': (2, 8),
        'optical_sigma': (0.5, 3.0),
        'optical_kernel_size': (3, 15),
        'motion_kernel_size': (1, 9),
        'gaussian_mean': (-10.0, 10.0),
        'gaussian_std': (0.0001, 0.1),  # For normalized [0,1] images: 0.0001-0.001 (very low), 0.001-0.01 (typical)
        'poisson_lambda': (0.1, 5.0),
        'hr_patch_size': (64, 1024),
        'lr_patch_size': (16, 256),
    }
    
    # Define required parameters
    required_params = ['downsampling_factor']
    
    # Check required parameters
    for param in required_params:
        if param not in config:
            raise ValueError(f"Required parameter '{param}' missing from config")
    
    # Validate parameter ranges
    for param, value in config.items():
        if param in param_ranges:
            min_val, max_val = param_ranges[param]
            if not isinstance(value, (int, float)):
                raise ValueError(f"Parameter '{param}' must be numeric, got {type(value)}")
            if value < min_val or value > max_val:
                raise ValueError(f"Parameter '{param}' = {value} outside valid range [{min_val}, {max_val}]")
    
    # Special validations
    
    # Kernel sizes must be odd
    for kernel_param in ['optical_kernel_size', 'motion_kernel_size']:
        if kernel_param in config:
            if config[kernel_param] % 2 == 0:
                raise ValueError(f"Parameter '{kernel_param}' must be odd, got {config[kernel_param]}")
    
    # Downsampling factor validation
    factor = config['downsampling_factor']
    if not isinstance(factor, int) or factor < 2:
        raise ValueError(f"downsampling_factor must be integer >= 2, got {factor}")
    
    # Downsampling mode validation
    if 'downsampling_mode' in config:
        mode = config['downsampling_mode']
        if mode not in [2, 4]:
            raise ValueError(f"downsampling_mode must be 2 or 4, got {mode}")
    
    # Patch size compatibility
    if 'hr_patch_size' in config and 'lr_patch_size' in config:
        hr_size = config['hr_patch_size']
        lr_size = config['lr_patch_size']
        if hr_size % lr_size != 0:
            raise ValueError(f"hr_patch_size ({hr_size}) must be divisible by lr_patch_size ({lr_size})")
        
        derived_factor = hr_size // lr_size
        if derived_factor != factor:
            raise ValueError(
                f"Patch size ratio ({derived_factor}) must match downsampling_factor ({factor})"
            )
    
    # Boolean parameter validation
    bool_params = ['enable_gaussian', 'enable_poisson', 'normalize']
    for param in bool_params:
        if param in config and not isinstance(config[param], bool):
            raise ValueError(f"Parameter '{param}' must be boolean, got {type(config[param])}")
    
    # String parameter validation
    if 'target_dtype' in config:
        valid_dtypes = ['float32', 'float64', 'uint8', 'uint16']
        if config['target_dtype'] not in valid_dtypes:
            raise ValueError(f"target_dtype must be one of {valid_dtypes}, got {config['target_dtype']}")
    
    logger.info("Configuration validation passed")
    return True


def validate_patch_compatibility(hr_patch_size: int, 
                                lr_patch_size: int, 
                                downsampling_factor: int) -> bool:
    """
    Validate that patch sizes are compatible with downsampling factor.
    
    Args:
        hr_patch_size: HR patch size
        lr_patch_size: LR patch size  
        downsampling_factor: Downsampling factor
        
    Returns:
        True if compatible, raises ValueError if not
    """
    if hr_patch_size // downsampling_factor != lr_patch_size:
        raise ValueError(
            f"Patch sizes incompatible: HR={hr_patch_size}, LR={lr_patch_size}, "
            f"factor={downsampling_factor}. Expected LR size: {hr_patch_size // downsampling_factor}"
        )
    
    return True


def validate_image_dimensions(image: np.ndarray, 
                            downsampling_factor: int,
                            name: str = "image") -> bool:
    """
    Validate that image dimensions are compatible with downsampling factor.
    
    Args:
        image: Image array
        downsampling_factor: Downsampling factor
        name: Name for error messages
        
    Returns:
        True if compatible, raises ValueError if not
    """
    h, w = image.shape[:2]
    
    if h % downsampling_factor != 0:
        raise ValueError(
            f"{name} height {h} not divisible by downsampling factor {downsampling_factor}"
        )
    
    if w % downsampling_factor != 0:
        raise ValueError(
            f"{name} width {w} not divisible by downsampling factor {downsampling_factor}"
        )
    
    return True


def validate_file_path(filepath: Union[str, Any], 
                      must_exist: bool = True,
                      extensions: List[str] = None) -> bool:
    """
    Validate file path.
    
    Args:
        filepath: File path to validate
        must_exist: Whether file must exist
        extensions: List of allowed extensions (e.g., ['.tif', '.tiff'])
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    from pathlib import Path
    
    try:
        path = Path(filepath)
    except Exception as e:
        raise ValueError(f"Invalid file path: {filepath}") from e
    
    if must_exist and not path.exists():
        raise ValueError(f"File does not exist: {filepath}")
    
    if extensions:
        if path.suffix.lower() not in [ext.lower() for ext in extensions]:
            raise ValueError(f"File extension {path.suffix} not in allowed extensions {extensions}")
    
    return True