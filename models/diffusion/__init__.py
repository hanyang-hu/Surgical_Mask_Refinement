"""Diffusion model module."""

from .unet import LatentDiffusionUNet, RGBConditionedLatentDiffusionUNet
from .scheduler import LatentDiffusionScheduler
from .conditioner import (
    RGBTokenProjector,
    CrossAttentionBlock,
    RGBConditioner,
)
from .latent_interface import FrozenVAELatentInterface
from .time_embedding import TimestepEmbedding, get_timestep_embedding
from .losses import DiffusionLoss, diffusion_epsilon_loss

__all__ = [
    # U-Net architectures
    "LatentDiffusionUNet",
    "RGBConditionedLatentDiffusionUNet",
    # Scheduler
    "LatentDiffusionScheduler",
    # RGB conditioning
    "RGBTokenProjector",
    "CrossAttentionBlock",
    "RGBConditioner",
    # VAE interface
    "FrozenVAELatentInterface",
    # Time embedding
    "TimestepEmbedding",
    "get_timestep_embedding",
    # Loss functions
    "DiffusionLoss",
    "diffusion_epsilon_loss",
]
