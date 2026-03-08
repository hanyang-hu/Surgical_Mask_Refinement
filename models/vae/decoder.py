"""VAE decoder for reconstructing masks from latent codes.

Decodes latent representation back to mask space through progressive upsampling.
Mirrors the encoder architecture to reconstruct full-resolution masks.
"""

import torch
import torch.nn as nn
from typing import Optional


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


class UpBlock(nn.Module):
    """Upsampling block with residual blocks."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 1,
        norm: str = "batch",
        activation: str = "silu"
    ):
        super().__init__()
        
        # Upsample first (using transpose conv)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        
        # Residual blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(out_channels, out_channels, norm, activation))
        self.res_blocks = nn.Sequential(*res_blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample and apply residual blocks."""
        x = self.upsample(x)
        x = self.res_blocks(x)
        return x


class MaskDecoder(nn.Module):
    """Convolutional decoder for mask VAE.
    
    Reconstructs full-resolution mask from spatial latent code.
    Mirrors the encoder architecture with progressive upsampling.
    
    Architecture:
    - Input: [B, latent_channels, 32, 32]
    - Progressive upsampling: 32 -> 64 -> 128 -> 256 -> 512
    - Channel progression: latent_channels -> base*8 -> base*4 -> base*2 -> base
    - Output: [B, 1, 512, 512] (logits, no sigmoid)
    
    Args:
        latent_channels: Number of latent channels (default: 8)
        base_channels: Base channel width (default: 32)
        channel_multipliers: Channel multipliers for each stage (default: [1, 2, 4, 8])
        out_channels: Number of output channels (default: 1 for binary mask)
        num_res_blocks: Number of residual blocks per stage (default: 1)
        norm: Normalization type ('batch', 'instance', 'group')
        activation: Activation function ('silu', 'relu', 'gelu')
    """
    
    def __init__(
        self,
        latent_channels: int = 8,
        base_channels: int = 32,
        channel_multipliers: list = [1, 2, 4, 8],
        out_channels: int = 1,
        num_res_blocks: int = 1,
        norm: str = "batch",
        activation: str = "silu"
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        
        # Initial convolution from latent channels to highest feature dimension
        highest_channels = base_channels * channel_multipliers[-1]
        self.conv_in = nn.Conv2d(latent_channels, highest_channels, 3, padding=1)
        
        # Middle residual block
        self.mid_block = ResidualBlock(highest_channels, highest_channels, norm, activation)
        
        # Upsampling blocks (mirror encoder in reverse)
        # Encoder: base -> base*1 -> base*2 -> base*4 -> base*8 (4 down blocks)
        # Decoder: base*8 -> base*4 -> base*2 -> base*1 -> base (4 up blocks)
        up_blocks = []
        reversed_mults = list(reversed(channel_multipliers))
        in_ch = highest_channels
        for i, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult
            # Always add upsampling block to mirror encoder
            up_blocks.append(
                UpBlock(in_ch, out_ch, num_res_blocks, norm, activation)
            )
            in_ch = out_ch
        self.up_blocks = nn.Sequential(*up_blocks)
        
        # Final convolution to output channels (logits, no activation)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to mask logits.
        
        Args:
            z: Latent code [B, latent_channels, H_latent, W_latent]
               For default config: [B, latent_channels, 32, 32]
            
        Returns:
            Reconstructed mask LOGITS [B, out_channels, H, W]
            For default config: [B, 1, 512, 512]
            
            Note: Returns logits (not probabilities). Use with BCEWithLogitsLoss.
        """
        # Initial conv
        h = self.conv_in(z)  # [B, base*8, 32, 32]
        
        # Middle block
        h = self.mid_block(h)
        
        # Progressive upsampling
        h = self.up_blocks(h)  # [B, base, 512, 512]
        
        # Output logits (no sigmoid)
        logits = self.conv_out(h)  # [B, 1, 512, 512]
        
        return logits
