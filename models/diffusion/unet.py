"""UNet architecture for latent diffusion.

Conditional UNet that predicts noise in VAE latent space,
conditioned on coarse mask latent (NO CLIP in this baseline).
"""

import torch
import torch.nn as nn
from typing import Optional

from .time_embedding import TimestepEmbedding


class ResBlock(nn.Module):
    """Residual block with time embedding injection.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        time_embed_dim: Time embedding dimension
        norm: Normalization type ('group', 'batch', 'none')
        activation: Activation function ('silu', 'relu')
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        norm: str = 'group',
        activation: str = 'silu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Normalization
        if norm == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        elif norm == 'batch':
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        # Activation
        if activation == 'silu':
            self.act = nn.SiLU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_embed_dim, out_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, C_in, H, W]
            time_emb: Time embedding [B, time_embed_dim]
            
        Returns:
            Output tensor [B, C_out, H, W]
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_proj = self.time_mlp(self.act(time_emb))[:, :, None, None]
        h = h + time_proj
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip(x)


class DownBlock(nn.Module):
    """Downsampling block with residual blocks.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        time_embed_dim: Time embedding dimension
        num_res_blocks: Number of residual blocks
        downsample: Whether to downsample at end
        norm: Normalization type
        activation: Activation function
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_res_blocks: int = 1,
        downsample: bool = True,
        norm: str = 'group',
        activation: str = 'silu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_embed_dim,
                norm=norm,
                activation=activation,
                dropout=dropout,
            )
            for i in range(num_res_blocks)
        ])
        
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> tuple:
        """Forward pass.
        
        Args:
            x: Input tensor [B, C_in, H, W]
            time_emb: Time embedding [B, time_embed_dim]
            
        Returns:
            Tuple of (output, skip_connection)
        """
        h = x
        for res_block in self.res_blocks:
            h = res_block(h, time_emb)
        
        skip = h
        
        if self.downsample is not None:
            h = self.downsample(h)
        
        return h, skip


class UpBlock(nn.Module):
    """Upsampling block with residual blocks and skip connections.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        skip_channels: Skip connection channels
        time_embed_dim: Time embedding dimension
        num_res_blocks: Number of residual blocks
        upsample: Whether to upsample at start
        norm: Normalization type
        activation: Activation function
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_embed_dim: int,
        num_res_blocks: int = 1,
        upsample: bool = True,
        norm: str = 'group',
        activation: str = 'silu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if upsample:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample = None
        
        self.res_blocks = nn.ModuleList([
            ResBlock(
                in_channels + skip_channels if i == 0 else out_channels,
                out_channels,
                time_embed_dim,
                norm=norm,
                activation=activation,
                dropout=dropout,
            )
            for i in range(num_res_blocks)
        ])
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, C_in, H, W]
            skip: Skip connection [B, C_skip, H, W]
            time_emb: Time embedding [B, time_embed_dim]
            
        Returns:
            Output tensor [B, C_out, H, W]
        """
        if self.upsample is not None:
            x = self.upsample(x)
        
        # Concatenate skip connection
        h = torch.cat([x, skip], dim=1)
        
        for res_block in self.res_blocks:
            h = res_block(h, time_emb)
        
        return h


class LatentDiffusionUNet(nn.Module):
    """UNet for latent diffusion denoising (NO CLIP conditioning in this baseline).
    
    Predicts noise in VAE latent space, conditioned on coarse mask latent.
    
    Conditioning strategy:
    - Concatenate noisy refined latent z_t with coarse latent z_coarse
    - Input becomes [z_t || z_coarse] with shape [B, 16, 32, 32]
    - Output is predicted noise epsilon [B, 8, 32, 32]
    
    Args:
        in_channels: Input channels (16 = 8 for z_t + 8 for z_coarse)
        out_channels: Output channels (8 for epsilon)
        base_channels: Base channel multiplier
        channel_multipliers: Channel multipliers for each level
        num_res_blocks: Number of residual blocks per level
        time_embed_dim: Time embedding dimension
        norm: Normalization type
        activation: Activation function
        dropout: Dropout probability
        
    Example:
        >>> model = LatentDiffusionUNet(
        ...     in_channels=16,
        ...     out_channels=8,
        ...     base_channels=64,
        ...     channel_multipliers=[1, 2, 4],
        ...     time_embed_dim=256
        ... )
        >>> z_t = torch.randn(2, 8, 32, 32)
        >>> z_coarse = torch.randn(2, 8, 32, 32)
        >>> t = torch.tensor([100, 500])
        >>> eps_pred = model(z_t, t, z_coarse)  # Shape: [2, 8, 32, 32]
    """
    
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 8,
        base_channels: int = 64,
        channel_multipliers: list = [1, 2, 4],
        num_res_blocks: int = 1,
        time_embed_dim: int = 256,
        norm: str = 'group',
        activation: str = 'silu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_embed = TimestepEmbedding(
            embedding_dim=base_channels,
            hidden_dim=time_embed_dim,
            output_dim=time_embed_dim,
            activation=activation,
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling path
        channels = [base_channels]
        self.down_blocks = nn.ModuleList()
        
        for i, mult in enumerate(channel_multipliers):
            in_ch = channels[-1]
            out_ch = base_channels * mult
            downsample = (i < len(channel_multipliers) - 1)  # Don't downsample last block
            
            self.down_blocks.append(
                DownBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    time_embed_dim=time_embed_dim,
                    num_res_blocks=num_res_blocks,
                    downsample=downsample,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                )
            )
            channels.append(out_ch)
        
        # Bottleneck
        bottleneck_ch = base_channels * channel_multipliers[-1]
        self.mid_block = ResBlock(
            in_channels=bottleneck_ch,
            out_channels=bottleneck_ch,
            time_embed_dim=time_embed_dim,
            norm=norm,
            activation=activation,
            dropout=dropout,
        )
        
        # Upsampling path (reverse of down path)
        self.up_blocks = nn.ModuleList()
        
        # Reverse channel multipliers for symmetry
        reversed_mults = list(reversed(channel_multipliers))
        
        for i in range(len(channel_multipliers)):
            # Current level multiplier
            curr_mult = reversed_mults[i]
            # Next level multiplier (or 1 if we're at the last up block)
            next_mult = reversed_mults[i + 1] if i < len(channel_multipliers) - 1 else 1
            
            in_ch = base_channels * curr_mult
            out_ch = base_channels * next_mult
            skip_ch = in_ch  # Skip connection has same channels as current level
            upsample = (i > 0)  # Upsample AFTER first block (first block matches bottleneck)
            
            self.up_blocks.append(
                UpBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    skip_channels=skip_ch,
                    time_embed_dim=time_embed_dim,
                    num_res_blocks=num_res_blocks,
                    upsample=upsample,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                )
            )
        
        # Output projection
        if norm == 'group':
            self.output_norm = nn.GroupNorm(num_groups=8, num_channels=base_channels)
        elif norm == 'batch':
            self.output_norm = nn.BatchNorm2d(base_channels)
        else:
            self.output_norm = nn.Identity()
        
        if activation == 'silu':
            self.output_act = nn.SiLU()
        elif activation == 'relu':
            self.output_act = nn.ReLU()
        else:
            self.output_act = nn.Identity()
        
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        z_coarse: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of UNet.
        
        Args:
            z_t: Noisy refined latent [B, 8, H, W]
            timesteps: Diffusion timesteps [B]
            z_coarse: Coarse mask latent [B, 8, H, W]
            
        Returns:
            Predicted noise epsilon [B, 8, H, W]
        """
        # Concatenate z_t and z_coarse as conditioning
        x = torch.cat([z_t, z_coarse], dim=1)  # [B, 16, H, W]
        
        # Time embedding
        time_emb = self.time_embed(timesteps)  # [B, time_embed_dim]
        
        # Input projection
        h = self.input_conv(x)  # [B, base_channels, H, W]
        
        # Downsampling path
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, time_emb)
            skips.append(skip)
        
        # Bottleneck
        h = self.mid_block(h, time_emb)
        
        # Upsampling path
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, time_emb)
        
        # Output projection
        h = self.output_norm(h)
        h = self.output_act(h)
        h = self.output_conv(h)
        
        return h


class RGBConditionedLatentDiffusionUNet(nn.Module):
    """RGB-conditioned UNet for latent diffusion with cross-attention.
    
    Extends the baseline latent diffusion model with RGB CLIP token conditioning
    via cross-attention in the bottleneck layer.
    
    Conditioning strategy:
    - Concatenate noisy refined latent z_t with coarse latent z_coarse
    - Input becomes [z_t || z_coarse] with shape [B, 16, 32, 32]
    - RGB tokens are projected and used for cross-attention at the bottleneck
    - Output is predicted noise epsilon [B, 8, 32, 32]
    
    Args:
        in_channels: Input channels (16 = 8 for z_t + 8 for z_coarse)
        out_channels: Output channels (8 for epsilon)
        base_channels: Base channel multiplier
        channel_multipliers: Channel multipliers for each level
        num_res_blocks: Number of residual blocks per level
        time_embed_dim: Time embedding dimension
        norm: Normalization type
        activation: Activation function
        dropout: Dropout probability
        rgb_token_dim: RGB CLIP token dimension (default: 768)
        rgb_projected_dim: Projected RGB token dimension (default: 256)
        rgb_num_heads: Number of cross-attention heads (default: 4)
        rgb_dropout: RGB conditioning dropout (default: 0.0)
        
    Example:
        >>> model = RGBConditionedLatentDiffusionUNet(
        ...     in_channels=16,
        ...     out_channels=8,
        ...     base_channels=64,
        ...     channel_multipliers=[1, 2, 4],
        ...     time_embed_dim=256,
        ...     rgb_token_dim=768,
        ...     rgb_projected_dim=256,
        ...     rgb_num_heads=4
        ... )
        >>> z_t = torch.randn(2, 8, 32, 32)
        >>> z_coarse = torch.randn(2, 8, 32, 32)
        >>> t = torch.tensor([100, 500])
        >>> rgb_tokens = torch.randn(2, 196, 768)
        >>> eps_pred = model(z_t, t, z_coarse, rgb_tokens)  # Shape: [2, 8, 32, 32]
    """
    
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 8,
        base_channels: int = 64,
        channel_multipliers: list = [1, 2, 4],
        num_res_blocks: int = 1,
        time_embed_dim: int = 256,
        norm: str = 'group',
        activation: str = 'silu',
        dropout: float = 0.0,
        rgb_token_dim: int = 768,
        rgb_projected_dim: int = 256,
        rgb_num_heads: int = 4,
        rgb_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.time_embed_dim = time_embed_dim
        self.rgb_token_dim = rgb_token_dim
        self.rgb_projected_dim = rgb_projected_dim
        
        # Import conditioner here to avoid circular imports
        from .conditioner import RGBConditioner
        
        # Time embedding
        self.time_embed = TimestepEmbedding(
            embedding_dim=base_channels,
            hidden_dim=time_embed_dim,
            output_dim=time_embed_dim,
            activation=activation,
        )
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling path
        channels = [base_channels]
        self.down_blocks = nn.ModuleList()
        
        for i, mult in enumerate(channel_multipliers):
            in_ch = channels[-1]
            out_ch = base_channels * mult
            downsample = (i < len(channel_multipliers) - 1)  # Don't downsample last block
            
            self.down_blocks.append(
                DownBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    time_embed_dim=time_embed_dim,
                    num_res_blocks=num_res_blocks,
                    downsample=downsample,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                )
            )
            channels.append(out_ch)
        
        # Bottleneck with RGB conditioning
        bottleneck_ch = base_channels * channel_multipliers[-1]
        
        # First residual block
        self.mid_block1 = ResBlock(
            in_channels=bottleneck_ch,
            out_channels=bottleneck_ch,
            time_embed_dim=time_embed_dim,
            norm=norm,
            activation=activation,
            dropout=dropout,
        )
        
        # RGB cross-attention conditioning
        self.rgb_conditioner = RGBConditioner(
            token_dim=rgb_token_dim,
            projected_dim=rgb_projected_dim,
            latent_dim=bottleneck_ch,
            num_heads=rgb_num_heads,
            dropout=rgb_dropout,
        )
        
        # Second residual block after conditioning
        self.mid_block2 = ResBlock(
            in_channels=bottleneck_ch,
            out_channels=bottleneck_ch,
            time_embed_dim=time_embed_dim,
            norm=norm,
            activation=activation,
            dropout=dropout,
        )
        
        # Upsampling path (reverse of down path)
        self.up_blocks = nn.ModuleList()
        
        # Reverse channel multipliers for symmetry
        reversed_mults = list(reversed(channel_multipliers))
        
        for i in range(len(channel_multipliers)):
            # Current level multiplier
            curr_mult = reversed_mults[i]
            # Next level multiplier (or 1 if we're at the last up block)
            next_mult = reversed_mults[i + 1] if i < len(channel_multipliers) - 1 else 1
            
            in_ch = base_channels * curr_mult
            out_ch = base_channels * next_mult
            skip_ch = in_ch  # Skip connection has same channels as current level
            upsample = (i > 0)  # Upsample AFTER first block (first block matches bottleneck)
            
            self.up_blocks.append(
                UpBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    skip_channels=skip_ch,
                    time_embed_dim=time_embed_dim,
                    num_res_blocks=num_res_blocks,
                    upsample=upsample,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                )
            )
        
        # Output projection
        if norm == 'group':
            self.output_norm = nn.GroupNorm(num_groups=8, num_channels=base_channels)
        elif norm == 'batch':
            self.output_norm = nn.BatchNorm2d(base_channels)
        else:
            self.output_norm = nn.Identity()
        
        if activation == 'silu':
            self.output_act = nn.SiLU()
        elif activation == 'relu':
            self.output_act = nn.ReLU()
        else:
            self.output_act = nn.Identity()
        
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        z_coarse: torch.Tensor,
        rgb_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of RGB-conditioned UNet.
        
        Args:
            z_t: Noisy refined latent [B, 8, H, W]
            timesteps: Diffusion timesteps [B]
            z_coarse: Coarse mask latent [B, 8, H, W]
            rgb_tokens: Precomputed RGB CLIP tokens [B, N_tokens, 768]
            
        Returns:
            Predicted noise epsilon [B, 8, H, W]
        """
        # Validate RGB tokens
        if rgb_tokens is None:
            raise ValueError("rgb_tokens cannot be None for RGB-conditioned model")
        
        if rgb_tokens.dim() != 3:
            raise ValueError(
                f"Expected rgb_tokens to be 3D [B, N_tokens, token_dim], "
                f"but got shape {rgb_tokens.shape}"
            )
        
        if rgb_tokens.shape[2] != self.rgb_token_dim:
            raise ValueError(
                f"Expected rgb_tokens token_dim={self.rgb_token_dim}, "
                f"but got {rgb_tokens.shape[2]}"
            )
        
        # Concatenate z_t and z_coarse as conditioning
        x = torch.cat([z_t, z_coarse], dim=1)  # [B, 16, H, W]
        
        # Time embedding
        time_emb = self.time_embed(timesteps)  # [B, time_embed_dim]
        
        # Input projection
        h = self.input_conv(x)  # [B, base_channels, H, W]
        
        # Downsampling path
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, time_emb)
            skips.append(skip)
        
        # Bottleneck with RGB conditioning
        h = self.mid_block1(h, time_emb)
        h = self.rgb_conditioner(h, rgb_tokens)  # Cross-attention with RGB tokens
        h = self.mid_block2(h, time_emb)
        
        # Upsampling path
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, time_emb)
        
        # Output projection
        h = self.output_norm(h)
        h = self.output_act(h)
        h = self.output_conv(h)
        
        return h
