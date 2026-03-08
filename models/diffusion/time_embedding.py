"""Time embedding for diffusion timesteps.

Sinusoidal position embeddings for encoding diffusion timesteps.
"""

import torch
import torch.nn as nn
import math


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: Integer timesteps [B]
        embedding_dim: Dimension of embedding
        max_period: Maximum period for sinusoidal waves
        
    Returns:
        Timestep embeddings [B, embedding_dim]
    """
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
    ).to(timesteps.device)
    
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    if embedding_dim % 2:
        # Pad if odd dimension
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    
    return embedding


class TimestepEmbedding(nn.Module):
    """Timestep embedding module with MLP projection.
    
    Projects sinusoidal timestep embeddings to desired dimension.
    
    Args:
        embedding_dim: Dimension of sinusoidal embedding
        hidden_dim: Dimension of hidden layer (typically 4 * embedding_dim)
        output_dim: Output dimension (time_embed_dim for U-Net)
        activation: Activation function ('silu', 'relu', 'gelu')
        
    Example:
        >>> time_embed = TimestepEmbedding(128, 512, 256)
        >>> t = torch.tensor([0, 50, 100, 999])
        >>> emb = time_embed(t)  # Shape: [4, 256]
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = 'silu',
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Activation function
        if activation == 'silu':
            act_fn = nn.SiLU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # MLP: embedding_dim -> hidden_dim -> output_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Create time embeddings.
        
        Args:
            timesteps: Integer timesteps [B]
            
        Returns:
            Time embeddings [B, output_dim]
        """
        # Get sinusoidal embedding
        emb = get_timestep_embedding(timesteps, self.embedding_dim)
        
        # Project through MLP
        emb = self.mlp(emb)
        
        return emb
