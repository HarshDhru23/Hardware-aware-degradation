"""
Hardware-aware Degradation Pipeline for ISRO Multi-Frame Super-Resolution (MFSR)

This package implements the observation model: y_k = D * B_k * M_k * x + n_k
where:
- x: High-resolution ground truth image
- M_k: Warping operator (0.5 pixel shift for P1/P2 sensors)
- B_k: Blur operator (optical + motion blur)
- D: Downsampling operator (includes sensor PSF)
- n_k: Additive noise (Gaussian + Poisson)
- y_k: k-th observed low-resolution image
"""

from .pipeline import DegradationPipeline
from .operators import WarpingOperator, BlurOperator, DownsamplingOperator, NoiseOperator

__version__ = "1.0.0"
__all__ = ["DegradationPipeline", "WarpingOperator", "BlurOperator", "DownsamplingOperator", "NoiseOperator"]