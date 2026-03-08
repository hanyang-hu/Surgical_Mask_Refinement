"""VAE encoder for compressing masks into latent space.

Encodes binary/grayscale segmentation masks into a continuous latent representation.
Implements a convolutional encoder with progressive downsampling to create a spatial
latent representation suitable for latent diffusion models.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """Residual block with conv + norm + activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "batch",
        activation: str = "silu"
    ):
        super().__init__()
        
        # Norm layer
        if norm == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm == "instance":
            norm_layer = nn.InstanceNorm2d
        elif norm == "group":
            norm_layer = lambda channels: nn.GroupNorm(32, channels)
        else:
            raise ValueError(f"Unknown norm: {norm}")
        
        # Activation
        if activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "silu":
            act = nn.SiLU(inplace=True)
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = norm_layer(out_channels)
        self.act1 = act
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = norm_layer(out_channels)
        self.act2 = act
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = out + identity
        out = self.act2(out)
        
        return out


class DownBlock(nn.Module):
    """Downsampling block with residual blocks."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 1,
        norm: str = "batch",
        activation: str = "silu"
    ):
        super().__init__()
        
        # Downsample first
        self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        
        # Residual blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(out_channels, out_channels, norm, activation))
        self.res_blocks = nn.Sequential(*res_blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample and apply residual blocks."""
        x = self.downsample(x)
        x = self.res_blocks(x)
        return x


class MaskEncoder(nn.Module):
    """Convolutional encoder for mask VAE.
    
    Compresses spatial mask [B, 1, H, W] into spatial latent code [B, latent_channels, h, w].
    
    Architecture:
    - Input: [B, 1, 512, 512]
    - Progressive downsampling: 512 -> 256 -> 128 -> 64 -> 32
    - Channel progression: 1 -> base -> base*2 -> base*4 -> base*8
    - Output spatial latent: [B, latent_channels, 32, 32]
    - Two heads: mu and logvar for reparameterization
    
    Args:
        in_channels: Number of input channels (default: 1 for binary mask)
        base_channels: Base channel width (default: 32)
        channel_multipliers: Channel multipliers for each stage (default: [1, 2, 4, 8])
        latent_channels: Number of latent channels (default: 8)
        num_res_blocks: Number of residual blocks per stage (default: 1)
        norm: Normalization type ('batch', 'instance', 'group')
        activation: Activation function ('silu', 'relu', 'gelu')
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_multipliers: list = [1, 2, 4, 8],
        latent_channels: int = 8,
        num_res_blocks: int = 1,
        norm: str = "batch",
        activation: str = "silu"
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling blocks
        down_blocks = []
        in_ch = base_channels
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            down_blocks.append(
                DownBlock(in_ch, out_ch, num_res_blocks, norm, activation)
            )
            in_ch = out_ch
        self.down_blocks = nn.Sequential(*down_blocks)
        
        # Middle residual block
        self.mid_block = ResidualBlock(in_ch, in_ch, norm, activation)
        
        # Latent projection heads
        self.conv_mu = nn.Conv2d(in_ch, latent_channels, 3, padding=1)
        self.conv_logvar = nn.Conv2d(in_ch, latent_channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode mask to latent distribution parameters.
        
        Args:
            x: Input mask [B, 1, H, W] where H=W=512
            
        Returns:
            Tuple of (mu, logvar) each with shape [B, latent_channels, H_latent, W_latent]
            For default config with 4 downsampling stages: [B, latent_channels, 32, 32]
        """
        # Initial conv
        h = self.conv_in(x)  # [B, base_channels, 512, 512]
        
        # Progressive downsampling
        h = self.down_blocks(h)  # [B, base*8, 32, 32] for 4 stages
        
        # Middle block
        h = self.mid_block(h)
        
        # Project to latent parameters
        mu = self.conv_mu(h)       # [B, latent_channels, 32, 32]
        logvar = self.conv_logvar(h)  # [B, latent_channels, 32, 32]
        
        return mu, logvar
    
    def get_latent_shape(self, input_size: int = 512) -> Tuple[int, int]:
        """Calculate latent spatial dimensions given input size.
        
        Args:
            input_size: Input spatial dimension (assumes square)
            
        Returns:
            (latent_height, latent_width)
        """
        # Each downsampling stage divides by 2
        num_stages = len(self.channel_multipliers)
        latent_size = input_size // (2 ** num_stages)
        return (latent_size, latent_size)
