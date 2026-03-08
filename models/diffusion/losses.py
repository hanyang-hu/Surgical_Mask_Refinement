"""Diffusion model losses.

Loss functions for training diffusion models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def diffusion_epsilon_loss(
    predicted_eps: torch.Tensor,
    target_eps: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Simple epsilon-prediction MSE loss for diffusion training.
    
    Args:
        predicted_eps: Model prediction [B, C, H, W]
        target_eps: Actual noise added [B, C, H, W]
        reduction: Reduction type ('mean', 'sum', 'none')
        
    Returns:
        Loss scalar or tensor
    """
    loss = F.mse_loss(predicted_eps, target_eps, reduction=reduction)
    return loss


class DiffusionLoss(nn.Module):
    """Loss for diffusion model training.
    
    Computes epsilon-prediction loss between predicted and actual noise.
    Supports MSE and L1 loss functions.
    
    Args:
        loss_type: Loss function type ('mse' or 'l1')
        reduction: Reduction type ('mean', 'sum', 'none')
        
    Example:
        >>> loss_fn = DiffusionLoss(loss_type='mse')
        >>> pred_eps = torch.randn(4, 8, 32, 32)
        >>> target_eps = torch.randn(4, 8, 32, 32)
        >>> loss = loss_fn(pred_eps, target_eps)
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        reduction: str = 'mean',
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type not in ['mse', 'l1']:
            raise ValueError(f"Unknown loss type: {loss_type}. Must be 'mse' or 'l1'")
    
    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor:
        """Compute diffusion loss.
        
        Args:
            predicted_noise: Model prediction [B, C, H, W]
            target_noise: Actual noise added [B, C, H, W]
            return_dict: If True, return dict with metrics; if False, return scalar
            
        Returns:
            Loss scalar (or dict if return_dict=True)
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(predicted_noise, target_noise, reduction=self.reduction)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, target_noise, reduction=self.reduction)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        if return_dict:
            return {
                'loss': loss,
                'loss_type': self.loss_type,
            }
        
        return loss
