"""Loss functions for VAE training.

Implements reconstruction losses (BCE, Dice) and KL divergence for mask VAE training.
Supports beta-VAE weighting for controlling the strength of the KL regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0
) -> torch.Tensor:
    """Dice loss computed from logits.
    
    Args:
        logits: Predicted logits [B, C, H, W]
        target: Target mask [B, C, H, W] with values in {0, 1}
        smooth: Smoothing constant to avoid division by zero
        
    Returns:
        Dice loss (scalar)
    """
    # Apply sigmoid to logits
    probs = torch.sigmoid(logits)
    
    # Flatten spatial dimensions
    probs_flat = probs.view(probs.size(0), -1)  # [B, C*H*W]
    target_flat = target.view(target.size(0), -1)  # [B, C*H*W]
    
    # Compute intersection and union
    intersection = (probs_flat * target_flat).sum(dim=1)  # [B]
    union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)  # [B]
    
    # Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)  # [B]
    
    # Dice loss (1 - dice coefficient)
    loss = 1.0 - dice.mean()
    
    return loss


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between N(mu, sigma) and N(0, I).
    
    KL(N(mu, sigma) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    Args:
        mu: Mean of latent distribution [B, latent_channels, H, W]
        logvar: Log variance of latent distribution [B, latent_channels, H, W]
        
    Returns:
        KL divergence (scalar, averaged over batch)
    """
    # KL divergence formula
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])  # [B]
    
    # Average over batch
    kl = kl.mean()
    
    return kl


def vae_loss(
    recon_logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    recon_loss_type: str = "bce_dice",
    beta: float = 1e-4,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0
) -> Dict[str, torch.Tensor]:
    """Compute total VAE loss with reconstruction and KL divergence.
    
    Args:
        recon_logits: Reconstructed mask logits [B, 1, H, W]
        target: Target mask [B, 1, H, W] with values in {0, 1}
        mu: Latent mean [B, latent_channels, H_latent, W_latent]
        logvar: Latent log variance [B, latent_channels, H_latent, W_latent]
        recon_loss_type: Type of reconstruction loss ('bce', 'dice', 'bce_dice')
        beta: KL weighting coefficient (beta-VAE)
        bce_weight: Weight for BCE loss when using 'bce_dice'
        dice_weight: Weight for Dice loss when using 'bce_dice'
        
    Returns:
        Dictionary containing:
            - 'loss': Total weighted loss
            - 'recon_loss': Reconstruction loss
            - 'kl_loss': KL divergence
            - 'bce_loss': BCE loss (if computed)
            - 'dice_loss': Dice loss (if computed)
    """
    # Compute KL divergence
    kl_loss = kl_divergence(mu, logvar)
    
    # Compute reconstruction loss
    losses = {}
    
    if recon_loss_type == "bce":
        # Binary cross-entropy from logits
        bce_loss = F.binary_cross_entropy_with_logits(recon_logits, target, reduction='mean')
        recon_loss = bce_loss
        losses['bce_loss'] = bce_loss
        
    elif recon_loss_type == "dice":
        # Dice loss
        dice_loss_val = dice_loss_from_logits(recon_logits, target)
        recon_loss = dice_loss_val
        losses['dice_loss'] = dice_loss_val
        
    elif recon_loss_type == "bce_dice":
        # Combined BCE + Dice
        bce_loss = F.binary_cross_entropy_with_logits(recon_logits, target, reduction='mean')
        dice_loss_val = dice_loss_from_logits(recon_logits, target)
        recon_loss = bce_weight * bce_loss + dice_weight * dice_loss_val
        losses['bce_loss'] = bce_loss
        losses['dice_loss'] = dice_loss_val
        
    else:
        raise ValueError(
            f"Unknown reconstruction loss type: {recon_loss_type}. "
            f"Choose from: 'bce', 'dice', 'bce_dice'"
        )
    
    # Total loss: reconstruction + beta * KL
    total_loss = recon_loss + beta * kl_loss
    
    # Return all loss components
    losses.update({
        'loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    })
    
    return losses


class VAELoss(nn.Module):
    """VAE loss module for easy integration with training loops.
    
    Combines reconstruction loss and KL divergence with configurable weighting.
    
    Args:
        recon_loss_type: Type of reconstruction loss ('bce', 'dice', 'bce_dice')
        beta: KL weighting coefficient (default: 1e-4 for beta-VAE)
        bce_weight: Weight for BCE loss when using 'bce_dice' (default: 1.0)
        dice_weight: Weight for Dice loss when using 'bce_dice' (default: 1.0)
    """
    
    def __init__(
        self,
        recon_loss_type: str = "bce_dice",
        beta: float = 1e-4,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0
    ):
        super().__init__()
        self.recon_loss_type = recon_loss_type
        self.beta = beta
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(
        self,
        recon_logits: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss.
        
        Args:
            recon_logits: Reconstructed mask logits [B, 1, H, W]
            target: Target mask [B, 1, H, W]
            mu: Latent mean
            logvar: Latent log variance
            
        Returns:
            Dictionary with loss components
        """
        return vae_loss(
            recon_logits=recon_logits,
            target=target,
            mu=mu,
            logvar=logvar,
            recon_loss_type=self.recon_loss_type,
            beta=self.beta,
            bce_weight=self.bce_weight,
            dice_weight=self.dice_weight
        )
