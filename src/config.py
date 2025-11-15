"""
Configuration management utilities.

Handles loading, validating, and managing YAML configuration files
for the degradation pipeline.
"""

import yaml
from typing import Dict, Any, Union, Optional
from pathlib import Path
import logging

from ..utils.validation import validate_config


class ConfigManager:
    """
    Configuration manager for degradation pipeline.
    
    Handles loading YAML configs, validation, and parameter access.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.logger = logging.getLogger(__name__)
        self.config = {}
        
        if config_path is not None:
            self.load_config(config_path)
        else:
            self.load_default_config()
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # Validate configuration
            validate_config(self.config)
            
            self.logger.info(f"Configuration loaded from {config_path}")
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_path}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {e}") from e
    
    def load_default_config(self) -> None:
        """Load default configuration."""
        # Get the default config file path
        current_dir = Path(__file__).parent.parent.parent
        default_config_path = current_dir / "configs" / "default_config.yaml"
        
        if default_config_path.exists():
            self.load_config(default_config_path)
        else:
            # Fallback to hardcoded defaults
            self.config = self._get_hardcoded_defaults()
            validate_config(self.config)
            self.logger.warning("Using hardcoded default configuration")
    
    def _get_hardcoded_defaults(self) -> Dict[str, Any]:
        """Get hardcoded default configuration as fallback."""
        return {
            'downsampling_factor': 4,
            'optical_sigma': 1.0,
            'optical_kernel_size': 5,
            'motion_kernel_size': 3,
            'enable_gaussian': True,
            'enable_poisson': True,
            'gaussian_mean': 0.0,
            'gaussian_std': 5.0,
            'poisson_lambda': 1.0,
            'normalize': True,
            'target_dtype': 'float32',
            'hr_patch_size': 256,
            'lr_patch_size': 64,
            'patch_stride': 256,
            'min_valid_pixels': 0.95,
            'input_format': 'geotiff',
            'output_format': 'npy',
            'save_visualization': True,
            'log_level': 'INFO',
            'verbose': True
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        
        # Re-validate config after modification
        try:
            validate_config(self.config)
        except ValueError as e:
            # Revert the change if validation fails
            if key in self.config:
                del self.config[key]
            raise ValueError(f"Invalid configuration value for '{key}': {e}") from e
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of updates
        """
        old_config = self.config.copy()
        
        try:
            self.config.update(updates)
            validate_config(self.config)
        except ValueError as e:
            # Revert changes if validation fails
            self.config = old_config
            raise ValueError(f"Invalid configuration update: {e}") from e
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save config to {output_path}: {e}") from e
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration parameters."""
        return self.config.copy()
    
    def merge_with_file(self, config_path: Union[str, Path]) -> None:
        """
        Merge current config with another config file.
        
        Args:
            config_path: Path to config file to merge
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                new_config = yaml.safe_load(f)
            
            # Merge configurations (new_config takes precedence)
            merged_config = self.config.copy()
            merged_config.update(new_config)
            
            # Validate merged config
            validate_config(merged_config)
            
            self.config = merged_config
            self.logger.info(f"Configuration merged with {config_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to merge config from {config_path}: {e}") from e
    
    def create_config_template(self, output_path: Union[str, Path]) -> None:
        """
        Create a configuration template with all available parameters.
        
        Args:
            output_path: Path to save template
        """
        template = {
            '# Core pipeline parameters': None,
            'downsampling_factor': 4,
            
            '# Optical blur parameters': None,
            'optical_sigma': 1.0,
            'optical_kernel_size': 5,
            
            '# Motion blur parameters': None,
            'motion_kernel_size': 3,
            
            '# Noise parameters': None,
            'enable_gaussian': True,
            'enable_poisson': True,
            'gaussian_mean': 0.0,
            'gaussian_std': 5.0,
            'poisson_lambda': 1.0,
            
            '# Data parameters': None,
            'normalize': True,
            'target_dtype': 'float32',
            
            '# Patch extraction parameters': None,
            'hr_patch_size': 256,
            'lr_patch_size': 64,
            'patch_stride': 256,
            'min_valid_pixels': 0.95,
            
            '# I/O parameters': None,
            'input_format': 'geotiff',
            'output_format': 'npy',
            'save_visualization': True,
            
            '# Logging parameters': None,
            'log_level': 'INFO',
            'verbose': True
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for key, value in template.items():
                if key.startswith('#'):
                    f.write(f"\n{key}\n")
                elif value is not None:
                    f.write(f"{key}: {value}\n")
        
        self.logger.info(f"Configuration template saved to {output_path}")
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"ConfigManager({len(self.config)} parameters)"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = ["Configuration Parameters:"]
        for key, value in sorted(self.config.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)