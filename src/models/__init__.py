"""
Models module for cmu-10799-diffusion.

This module contains the neural network architectures used for
diffusion models and flow matching.
"""

from .unet import UNet, create_model_from_config
from .unet_meanflow import UNetMeanFlow, create_meanflow_model_from_config
from .blocks import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)

__all__ = [
    # Main model
    'UNet',
    'create_model_from_config',
    'UNetMeanFlow',
    'create_meanflow_model_from_config',
    # Building blocks
    'SinusoidalPositionalEmbedding',
    'TimestepEmbedding', 
    'ResBlock',
    'AttentionBlock',
    'Downsample',
    'Upsample',
    'GroupNorm32',
]
