"""
Methods module for cmu-10799-diffusion.

This module contains implementations of generative modeling methods:
- DDPM (Denoising Diffusion Probabilistic Models)
"""

from .base import BaseMethod
from .ddpm import DDPM
from .flow_matching import FlowMatching

__all__ = [
    'BaseMethod',
    'DDPM',
    'FlowMatching',
]
