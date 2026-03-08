"""Frozen VAE Latent Interface for Latent Diffusion.

Provides a clean interface for encoding masks to latent space and decoding
latents back to masks using a pre-trained, frozen VAE.

This interface is designed for use in latent diffusion training where:
- The VAE is fully trained and frozen (no fine-tuning)
- We encode coarse/refined masks to latent representations
- We decode latents back to mask space for evaluation
- We use deterministic encoding (mu) rather than sampling
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Union, Optional
import yaml

from models.vae import MaskVAE
from utils.checkpoint import load_checkpoint


class FrozenVAELatentInterface(nn.Module):
    """Frozen VAE interface for latent diffusion.
    
    Loads a pre-trained VAE checkpoint and provides clean encode/decode
    methods for latent diffusion training. All VAE parameters are frozen.
    
    Args:
        model_config_path: Path to VAE model config (configs/model/vae.yaml)
        checkpoint_path: Path to trained VAE checkpoint (outputs/vae/checkpoints/best.pt)
        device: Device to load model on ('cuda' or 'cpu')
        use_mu_only: If True, use deterministic encoding (z=mu). If False, sample from posterior.
                     Default: True (recommended for first diffusion baseline)
        
    Example:
        >>> interface = FrozenVAELatentInterface(
        ...     model_config_path='configs/model/vae.yaml',
        ...     checkpoint_path='outputs/vae/checkpoints/best.pt',
        ...     device='cuda',
        ...     use_mu_only=True
        ... )
        >>> 
        >>> # Encode a mask to latent
        >>> mask = torch.randn(4, 1, 512, 512).cuda()
        >>> z = interface.encode_mask(mask)  # Returns z = mu, shape [4, 8, 32, 32]
        >>> 
        >>> # Decode latent back to mask
        >>> recon_logits = interface.decode_latent(z)  # Shape [4, 1, 512, 512]
        >>> 
        >>> # Check frozen status
        >>> assert interface.is_frozen()
        >>> assert interface.count_trainable_parameters() == 0
    """
    
    def __init__(
        self,
        model_config_path: Union[str, Path],
        checkpoint_path: Union[str, Path],
        device: str = 'cuda',
        use_mu_only: bool = True,
    ):
        super().__init__()
        
        self.model_config_path = Path(model_config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.use_mu_only = use_mu_only
        
        # Validate paths
        if not self.model_config_path.exists():
            raise FileNotFoundError(
                f"Model config not found: {self.model_config_path}"
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )
        
        # Load VAE
        self.vae = self._load_vae()
        
        # Freeze VAE
        self._freeze_vae()
        
        # Store latent shape info
        self._latent_channels = self.vae.latent_channels
        self._latent_spatial_size = None  # Will be set on first encode
        
        print(f"Frozen VAE Latent Interface initialized:")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Use mu only: {self.use_mu_only}")
        print(f"  Trainable parameters: {self.count_trainable_parameters()}")
        print(f"  Frozen: {self.is_frozen()}")
    
    def _load_vae(self) -> MaskVAE:
        """Load VAE model from config and checkpoint.
        
        Returns:
            Loaded and initialized VAE model
        """
        # Load config
        with open(self.model_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create model
        vae = MaskVAE(
            in_channels=config['model']['in_channels'],
            base_channels=config['model']['base_channels'],
            channel_multipliers=config['model']['channel_multipliers'],
            latent_channels=config['model']['latent_channels'],
            num_res_blocks=config['model'].get('num_res_blocks', 1),
            norm=config['model'].get('norm', 'batch'),
            activation=config['model'].get('activation', 'silu'),
        )
        
        # Load checkpoint
        checkpoint = load_checkpoint(
            checkpoint_path=str(self.checkpoint_path),
            model=vae,
            device=self.device,
            strict=True
        )
        
        # Move to device and set to eval mode
        vae = vae.to(self.device)
        vae.eval()
        
        return vae
    
    def _freeze_vae(self):
        """Freeze all VAE parameters (encoder and decoder).
        
        Sets requires_grad=False for all parameters to prevent gradients
        from being computed or parameters from being updated.
        """
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Ensure eval mode
        self.vae.eval()
    
    def is_frozen(self) -> bool:
        """Check if VAE is fully frozen.
        
        Returns:
            True if all parameters have requires_grad=False
        """
        return all(not param.requires_grad for param in self.vae.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters.
        
        Returns:
            Number of parameters with requires_grad=True (should be 0)
        """
        return sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
    
    def count_total_parameters(self) -> int:
        """Count total number of parameters in VAE.
        
        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.vae.parameters())
    
    @property
    def latent_shape(self) -> tuple:
        """Get expected latent shape for a single sample.
        
        Returns:
            (channels, height, width) tuple for latent space
        """
        if self._latent_spatial_size is None:
            # Infer from a dummy input
            with torch.no_grad():
                dummy = torch.zeros(1, 1, 512, 512, device=self.device)
                mu, logvar = self.vae.encode(dummy)
                self._latent_spatial_size = mu.shape[-2:]
        
        return (self._latent_channels, *self._latent_spatial_size)
    
    @torch.no_grad()
    def encode_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Encode a mask to latent representation.
        
        Args:
            mask: Input binary mask [B, 1, H, W]
            
        Returns:
            Latent representation [B, C, h, w]
            - If use_mu_only=True: returns mu (deterministic)
            - If use_mu_only=False: returns sampled z from posterior
        """
        if mask.dim() != 4:
            raise ValueError(
                f"Expected 4D mask tensor [B, 1, H, W], got shape {mask.shape}"
            )
        
        # Encode - returns (mu, logvar) tuple
        mu, logvar = self.vae.encode(mask)
        
        # Return mu or sampled z
        if self.use_mu_only:
            return mu
        else:
            # Sample from posterior
            return self.vae.reparameterize(mu, logvar)
    
    @torch.no_grad()
    def encode_coarse_mask(self, coarse_mask: torch.Tensor) -> torch.Tensor:
        """Encode coarse mask to latent representation.
        
        This is a convenience method with explicit naming for clarity.
        
        Args:
            coarse_mask: Input coarse binary mask [B, 1, H, W]
            
        Returns:
            Latent representation [B, C, h, w]
        """
        return self.encode_mask(coarse_mask)
    
    @torch.no_grad()
    def encode_refined_mask(self, refined_mask: torch.Tensor) -> torch.Tensor:
        """Encode refined mask to latent representation.
        
        This is a convenience method with explicit naming for clarity.
        
        Args:
            refined_mask: Input refined binary mask [B, 1, H, W]
            
        Returns:
            Latent representation [B, C, h, w]
        """
        return self.encode_mask(refined_mask)
    
    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to mask logits.
        
        Args:
            z: Latent representation [B, C, h, w]
            
        Returns:
            Reconstruction logits [B, 1, H, W]
            
        Note:
            Returns logits, not probabilities. Apply sigmoid for probabilities.
        """
        if z.dim() != 4:
            raise ValueError(
                f"Expected 4D latent tensor [B, C, h, w], got shape {z.shape}"
            )
        
        # Decode
        recon_logits = self.vae.decode(z)
        
        return recon_logits
    
    @torch.no_grad()
    def decode_to_probs(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to mask probabilities.
        
        Convenience method that applies sigmoid to decode output.
        
        Args:
            z: Latent representation [B, C, h, w]
            
        Returns:
            Reconstruction probabilities [B, 1, H, W] in range [0, 1]
        """
        logits = self.decode_latent(z)
        return torch.sigmoid(logits)
    
    @torch.no_grad()
    def reconstruct_mask(self, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode and decode a mask (full reconstruction).
        
        Args:
            mask: Input binary mask [B, 1, H, W]
            
        Returns:
            Dictionary containing:
            - 'z': Latent representation [B, C, h, w]
            - 'recon_logits': Reconstruction logits [B, 1, H, W]
            - 'recon_probs': Reconstruction probabilities [B, 1, H, W]
        """
        z = self.encode_mask(mask)
        recon_logits = self.decode_latent(z)
        recon_probs = torch.sigmoid(recon_logits)
        
        return {
            'z': z,
            'recon_logits': recon_logits,
            'recon_probs': recon_probs,
        }
    
    @staticmethod
    def threshold_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Threshold logits to binary mask.
        
        Args:
            logits: Input logits [B, 1, H, W]
            threshold: Threshold value (default: 0.5)
            
        Returns:
            Binary mask [B, 1, H, W]
        """
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()
    
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode mask to latent.
        
        This is the default forward behavior when using the interface
        as a nn.Module.
        
        Args:
            mask: Input binary mask [B, 1, H, W]
            
        Returns:
            Latent representation [B, C, h, w]
        """
        return self.encode_mask(mask)
    
    def __repr__(self) -> str:
        return (
            f"FrozenVAELatentInterface(\n"
            f"  checkpoint={self.checkpoint_path},\n"
            f"  device={self.device},\n"
            f"  use_mu_only={self.use_mu_only},\n"
            f"  latent_shape={self.latent_shape},\n"
            f"  total_params={self.count_total_parameters():,},\n"
            f"  trainable_params={self.count_trainable_parameters()},\n"
            f"  frozen={self.is_frozen()}\n"
            f")"
        )
