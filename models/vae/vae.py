"""Complete VAE model combining encoder and decoder.

Provides end-to-end mask VAE with reparameterization trick for learning
a continuous latent space of surgical instrument masks.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .encoder import MaskEncoder
from .decoder import MaskDecoder


class MaskVAE(nn.Module):
    """Variational Autoencoder for segmentation masks.
    
    Learns to compress and reconstruct binary segmentation masks,
    creating a smooth spatial latent space suitable for diffusion modeling.
    
    Architecture:
    - Encoder: mask [B, 1, 512, 512] -> (mu, logvar) [B, latent_channels, 32, 32]
    - Reparameterization: (mu, logvar) -> z [B, latent_channels, 32, 32]
    - Decoder: z [B, latent_channels, 32, 32] -> logits [B, 1, 512, 512]
    
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
        
        # Encoder and decoder
        self.encoder = MaskEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm=norm,
            activation=activation
        )
        
        self.decoder = MaskDecoder(
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            norm=norm,
            activation=activation
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * sigma.
        
        Args:
            mu: Mean of latent distribution [B, latent_channels, H, W]
            logvar: Log variance of latent distribution [B, latent_channels, H, W]
            
        Returns:
            Sampled latent code z [B, latent_channels, H, W]
        """
        if self.training:
            # Sample epsilon from standard normal
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # Deterministic during inference
            z = mu
        
        return z
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode mask to latent distribution parameters.
        
        Args:
            x: Input mask [B, 1, H, W]
            
        Returns:
            Tuple of (mu, logvar)
        """
        mu, logvar = self.encoder(x)
        return mu, logvar
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to mask logits.
        
        Args:
            z: Latent code [B, latent_channels, H_latent, W_latent]
            
        Returns:
            Reconstructed mask logits [B, 1, H, W]
        """
        logits = self.decoder(z)
        return logits
        
    def forward(
        self, 
        x: torch.Tensor,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass with reconstruction and latent sampling.
        
        Args:
            x: Input mask [B, 1, H, W]
            return_dict: If True, return dict; if False, return tuple
            
        Returns:
            Dictionary containing:
                - 'recon_logits': Reconstructed mask logits [B, 1, H, W]
                - 'mu': Latent mean [B, latent_channels, H_latent, W_latent]
                - 'logvar': Latent log variance [B, latent_channels, H_latent, W_latent]
                - 'z': Sampled latent code [B, latent_channels, H_latent, W_latent]
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_logits = self.decode(z)
        
        if return_dict:
            return {
                'recon_logits': recon_logits,
                'mu': mu,
                'logvar': logvar,
                'z': z
            }
        else:
            return recon_logits, mu, logvar, z
    
    def sample(
        self,
        num_samples: int,
        latent_size: int = 32,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Sample random masks from prior N(0, I).
        
        Args:
            num_samples: Number of samples to generate
            latent_size: Spatial size of latent (default: 32)
            device: Device to generate samples on
            
        Returns:
            Sampled mask logits [num_samples, 1, H, W]
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from standard normal prior
        z = torch.randn(
            num_samples, self.latent_channels, latent_size, latent_size,
            device=device
        )
        
        # Decode
        with torch.no_grad():
            logits = self.decode(z)
        
        return logits
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct masks deterministically (using mean, no sampling).
        
        Args:
            x: Input mask [B, 1, H, W]
            
        Returns:
            Reconstructed mask probabilities [B, 1, H, W]
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
            logits = self.decode(mu)
            probs = torch.sigmoid(logits)
        
        return probs
    
    def get_latent_shape(self, input_size: int = 512) -> Tuple[int, int, int]:
        """Get the shape of the latent representation.
        
        Args:
            input_size: Input spatial dimension
            
        Returns:
            (latent_channels, latent_height, latent_width)
        """
        h, w = self.encoder.get_latent_shape(input_size)
        return (self.latent_channels, h, w)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        total_params = encoder_params + decoder_params
        
        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'total': total_params
        }
