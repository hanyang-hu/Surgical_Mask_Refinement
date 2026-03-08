"""Adapter layers for transforming CLIP tokens.

Optional learnable adapters to project or transform CLIP tokens
before feeding into diffusion model conditioning.
"""

from typing import Optional
import torch
import torch.nn as nn


class TokenProjection(nn.Module):
    """Simple linear projection for token sequences.
    
    Projects CLIP token features to a different dimension.
    Useful for adapting CLIP features to diffusion model conditioning.
    
    Args:
        input_dim: Input feature dimension (e.g., 768 for CLIP ViT-B/16)
        output_dim: Output feature dimension
        bias: Whether to use bias (default: True)
        
    Example:
        >>> projection = TokenProjection(768, 512)
        >>> tokens = torch.randn(4, 196, 768)
        >>> projected = projection(tokens)
        >>> print(projected.shape)  # [4, 196, 512]
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
    ):
        """Initialize token projection."""
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.projection = nn.Linear(input_dim, output_dim, bias=bias)
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project tokens.
        
        Args:
            tokens: Token sequence [B, N, input_dim]
            
        Returns:
            Projected tokens [B, N, output_dim]
        """
        return self.projection(tokens)
    
    def __repr__(self) -> str:
        return f"TokenProjection({self.input_dim} -> {self.output_dim})"


class SpatialFeatureAdapter(nn.Module):
    """Adapter for spatial feature maps using 1x1 convolution.
    
    Projects spatial feature maps to a different channel dimension.
    Uses 1x1 convolution which is equivalent to per-pixel linear projection.
    
    Args:
        input_channels: Input channel dimension (e.g., 768 for CLIP ViT-B/16)
        output_channels: Output channel dimension
        bias: Whether to use bias (default: True)
        
    Example:
        >>> adapter = SpatialFeatureAdapter(768, 512)
        >>> features = torch.randn(4, 768, 14, 14)
        >>> adapted = adapter(features)
        >>> print(adapted.shape)  # [4, 512, 14, 14]
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        bias: bool = True,
    ):
        """Initialize spatial adapter."""
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # 1x1 convolution for channel-wise projection
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Adapt spatial features.
        
        Args:
            features: Spatial feature map [B, input_channels, H, W]
            
        Returns:
            Adapted features [B, output_channels, H, W]
        """
        return self.conv(features)
    
    def __repr__(self) -> str:
        return f"SpatialFeatureAdapter({self.input_channels} -> {self.output_channels})"


class MLPAdapter(nn.Module):
    """MLP adapter with LayerNorm and GELU activation.
    
    Two-layer MLP for more expressive token adaptation.
    Useful when simple linear projection is not sufficient.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        hidden_dim: Hidden layer dimension (default: 4x input_dim)
        dropout: Dropout probability (default: 0.0)
        
    Example:
        >>> adapter = MLPAdapter(768, 512, hidden_dim=2048)
        >>> tokens = torch.randn(4, 196, 768)
        >>> adapted = adapter(tokens)
        >>> print(adapted.shape)  # [4, 196, 512]
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Initialize MLP adapter."""
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dim is None:
            hidden_dim = input_dim * 4
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply MLP adapter.
        
        Args:
            tokens: Token sequence [B, N, input_dim]
            
        Returns:
            Adapted tokens [B, N, output_dim]
        """
        return self.net(tokens)
    
    def __repr__(self) -> str:
        return (
            f"MLPAdapter({self.input_dim} -> {self.hidden_dim} -> {self.output_dim})"
        )


def build_adapter(
    adapter_type: str,
    input_dim: int,
    output_dim: int,
    **kwargs
) -> nn.Module:
    """Factory function to build adapter modules.
    
    Args:
        adapter_type: Type of adapter ('linear', 'conv1x1', 'mlp')
        input_dim: Input dimension
        output_dim: Output dimension
        **kwargs: Additional arguments for specific adapter types
        
    Returns:
        Adapter module
        
    Example:
        >>> adapter = build_adapter('linear', 768, 512)
        >>> adapter = build_adapter('mlp', 768, 512, hidden_dim=2048)
    """
    if adapter_type == "linear":
        return TokenProjection(input_dim, output_dim, **kwargs)
    elif adapter_type == "conv1x1":
        return SpatialFeatureAdapter(input_dim, output_dim, **kwargs)
    elif adapter_type == "mlp":
        return MLPAdapter(input_dim, output_dim, **kwargs)
    else:
        raise ValueError(
            f"Unknown adapter_type: {adapter_type}. "
            f"Supported types: 'linear', 'conv1x1', 'mlp'"
        )
