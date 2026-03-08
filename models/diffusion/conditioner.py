"""RGB Token Conditioning Module for Latent Diffusion.

This module provides RGB token projection for cross-attention conditioning
in the latent diffusion U-Net.
"""

import torch
import torch.nn as nn
from typing import Optional


class RGBTokenProjector(nn.Module):
    """Projects precomputed RGB CLIP tokens for cross-attention conditioning.
    
    Takes frozen CLIP vision tokens [B, N_tokens, token_dim] and projects them
    to a lower dimension suitable for cross-attention in the diffusion U-Net.
    
    Args:
        token_dim: Input CLIP token dimension (default: 768 for CLIP ViT-B/16@224px)
        projected_dim: Output projection dimension for cross-attention (default: 256)
        dropout: Dropout probability (default: 0.0)
        use_layer_norm: Whether to apply LayerNorm before projection (default: True)
    
    Input:
        rgb_tokens: [B, N_tokens, token_dim], precomputed CLIP visual tokens
    
    Output:
        projected_tokens: [B, N_tokens, projected_dim]
    
    Example:
        >>> projector = RGBTokenProjector(token_dim=768, projected_dim=256)
        >>> rgb_tokens = torch.randn(4, 196, 768)  # Batch of 4, 196 tokens, 768-dim
        >>> projected = projector(rgb_tokens)  # [4, 196, 256]
    """
    
    def __init__(
        self,
        token_dim: int = 768,
        projected_dim: int = 256,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        self.token_dim = token_dim
        self.projected_dim = projected_dim
        
        # Optional normalization
        self.norm = nn.LayerNorm(token_dim) if use_layer_norm else nn.Identity()
        
        # Linear projection
        self.projection = nn.Linear(token_dim, projected_dim)
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize projection weights
        # Use small initialization to start conditioning gently
        nn.init.xavier_uniform_(self.projection.weight, gain=0.02)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, rgb_tokens: torch.Tensor) -> torch.Tensor:
        """Project RGB tokens.
        
        Args:
            rgb_tokens: [B, N_tokens, token_dim] precomputed CLIP tokens
        
        Returns:
            projected_tokens: [B, N_tokens, projected_dim]
        """
        # Validate input shape
        if rgb_tokens.dim() != 3:
            raise ValueError(
                f"Expected rgb_tokens to be 3D [B, N_tokens, token_dim], "
                f"but got shape {rgb_tokens.shape}"
            )
        
        if rgb_tokens.shape[2] != self.token_dim:
            raise ValueError(
                f"Expected token_dim={self.token_dim}, but got {rgb_tokens.shape[2]}"
            )
        
        # Optional: Warn if token count is unexpected
        # For CLIP ViT-B/16@224px, we expect 196 tokens (14x14 patches, CLS removed)
        # For CLIP ViT-L/14@336px, we expect 576 tokens (24x24 patches, CLS removed)
        num_tokens = rgb_tokens.shape[1]
        if num_tokens not in [196, 576]:
            import warnings
            warnings.warn(
                f"Unexpected number of tokens: {num_tokens}. "
                f"Expected 196 (ViT-B/16@224px) or 576 (ViT-L/14@336px). "
                f"This may indicate a mismatch in CLIP preprocessing."
            )
        
        # Normalize
        x = self.norm(rgb_tokens)
        
        # Project
        x = self.projection(x)
        
        # Dropout
        x = self.dropout(x)
        
        return x  # [B, N_tokens, projected_dim]


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for conditioning on RGB tokens.
    
    Applies cross-attention where:
    - Queries come from latent feature maps (spatial features)
    - Keys and Values come from projected RGB tokens
    
    This allows latent features to attend to RGB visual information.
    
    Args:
        latent_dim: Channel dimension of latent features (query dimension)
        condition_dim: Dimension of conditioning tokens (key/value dimension)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.0)
    
    Input:
        latent_features: [B, latent_dim, H, W] spatial latent features
        condition_tokens: [B, N_tokens, condition_dim] projected RGB tokens
    
    Output:
        conditioned_features: [B, latent_dim, H, W] attended features
    
    Example:
        >>> cross_attn = CrossAttentionBlock(latent_dim=256, condition_dim=256, num_heads=4)
        >>> latent = torch.randn(4, 256, 8, 8)  # Batch of 4, 256 channels, 8x8 spatial
        >>> tokens = torch.randn(4, 196, 256)   # Batch of 4, 196 tokens, 256-dim
        >>> output = cross_attn(latent, tokens)  # [4, 256, 8, 8]
    """
    
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_heads = num_heads
        
        # Ensure dimensions are divisible by num_heads
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        # Query projection from latent features
        self.to_q = nn.Linear(latent_dim, latent_dim)
        
        # Key and Value projections from condition tokens
        self.to_k = nn.Linear(condition_dim, latent_dim)
        self.to_v = nn.Linear(condition_dim, latent_dim)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )
        
        # Normalization
        self.norm_latent = nn.LayerNorm(latent_dim)
        self.norm_condition = nn.LayerNorm(condition_dim)
        
        self.scale = (latent_dim // num_heads) ** -0.5
    
    def forward(
        self,
        latent_features: torch.Tensor,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention.
        
        Args:
            latent_features: [B, latent_dim, H, W] spatial features
            condition_tokens: [B, N_tokens, condition_dim] RGB tokens
        
        Returns:
            conditioned_features: [B, latent_dim, H, W] attended features
        """
        B, C, H, W = latent_features.shape
        
        # Reshape latent features to sequence format
        # [B, C, H, W] -> [B, H*W, C]
        x = latent_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Normalize
        x_norm = self.norm_latent(x)
        cond_norm = self.norm_condition(condition_tokens)
        
        # Compute Q, K, V
        q = self.to_q(x_norm)  # [B, H*W, latent_dim]
        k = self.to_k(cond_norm)  # [B, N_tokens, latent_dim]
        v = self.to_v(cond_norm)  # [B, N_tokens, latent_dim]
        
        # Reshape for multi-head attention
        # [B, N, latent_dim] -> [B, num_heads, N, head_dim]
        head_dim = self.latent_dim // self.num_heads
        
        q = q.reshape(B, H * W, self.num_heads, head_dim).transpose(1, 2)
        k = k.reshape(B, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, -1, self.num_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        # [B, num_heads, H*W, head_dim] @ [B, num_heads, head_dim, N_tokens]
        # -> [B, num_heads, H*W, N_tokens]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        # [B, num_heads, H*W, N_tokens] @ [B, num_heads, N_tokens, head_dim]
        # -> [B, num_heads, H*W, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        # [B, num_heads, H*W, head_dim] -> [B, H*W, latent_dim]
        attn_output = attn_output.transpose(1, 2).reshape(B, H * W, self.latent_dim)
        
        # Output projection
        out = self.to_out(attn_output)
        
        # Residual connection
        out = out + x
        
        # Reshape back to spatial format
        # [B, H*W, C] -> [B, C, H, W]
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return out


class RGBConditioner(nn.Module):
    """Complete RGB conditioning module with projection and cross-attention.
    
    This is a convenience module that combines:
    1. Token projection (768 -> projected_dim)
    2. Cross-attention block for conditioning
    
    Args:
        token_dim: Input CLIP token dimension (default: 768)
        projected_dim: Projected token dimension (default: 256)
        latent_dim: Latent feature dimension for cross-attention (default: 256)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.0)
    
    Input:
        latent_features: [B, latent_dim, H, W]
        rgb_tokens: [B, N_tokens, token_dim]
    
    Output:
        conditioned_features: [B, latent_dim, H, W]
    """
    
    def __init__(
        self,
        token_dim: int = 768,
        projected_dim: int = 256,
        latent_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.projector = RGBTokenProjector(
            token_dim=token_dim,
            projected_dim=projected_dim,
            dropout=dropout,
        )
        
        self.cross_attention = CrossAttentionBlock(
            latent_dim=latent_dim,
            condition_dim=projected_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
    
    def forward(
        self,
        latent_features: torch.Tensor,
        rgb_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RGB conditioning.
        
        Args:
            latent_features: [B, latent_dim, H, W]
            rgb_tokens: [B, N_tokens, token_dim]
        
        Returns:
            conditioned_features: [B, latent_dim, H, W]
        """
        # Project tokens
        projected_tokens = self.projector(rgb_tokens)
        
        # Apply cross-attention
        conditioned = self.cross_attention(latent_features, projected_tokens)
        
        return conditioned
        pass
