"""Diffusion noise scheduler for latent diffusion.

Handles noise scheduling and forward diffusion (q_sample) for training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class LatentDiffusionScheduler:
    """Noise scheduler for latent diffusion process.
    
    Implements DDPM forward process with configurable beta schedule.
    Used during training to add noise to latents.
    
    Args:
        num_train_timesteps: Number of diffusion steps (default: 1000)
        beta_schedule: Beta schedule type ('linear' or 'cosine')
        beta_start: Starting beta value (default: 1e-4)
        beta_end: Ending beta value (default: 2e-2)
        device: Device to store tensors on
        
    Example:
        >>> scheduler = LatentDiffusionScheduler(num_train_timesteps=1000)
        >>> z0 = torch.randn(4, 8, 32, 32)
        >>> t = scheduler.sample_timesteps(4, device='cuda')
        >>> noise = torch.randn_like(z0)
        >>> zt = scheduler.q_sample(z0, t, noise)
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: Optional[str] = None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device if device else 'cpu'
        
        # Compute beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Compute alpha schedule
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]), self.alphas_cumprod[:-1]
        ])
        
        # Precompute values for q_sample
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Precompute values for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Move to device
        self.to(self.device)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def to(self, device: str):
        """Move all tensors to device."""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
    
    def sample_timesteps(self, batch_size: int, device: Optional[str] = None) -> torch.Tensor:
        """Sample random timesteps for training.
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on
            
        Returns:
            Random timesteps [B] in range [0, num_train_timesteps)
        """
        if device is None:
            device = self.device
        
        return torch.randint(
            0, self.num_train_timesteps, (batch_size,), 
            device=device, dtype=torch.long
        )
    
    def q_sample(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion: add noise to x_start at timestep t.
        
        Implements q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        
        Args:
            x_start: Clean latent [B, C, H, W]
            t: Timesteps [B]
            noise: Optional pre-sampled noise [B, C, H, W]
            
        Returns:
            Noisy latent x_t [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Get sqrt_alpha_bar and sqrt_one_minus_alpha_bar for timesteps t
        sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        # Compute noisy sample
        x_t = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
        
        return x_t
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract values from a at indices t and reshape for broadcasting.
        
        Args:
            a: 1D tensor to extract from
            t: Timestep indices [B]
            x_shape: Shape of tensor to broadcast to [B, C, H, W]
            
        Returns:
            Extracted values reshaped to [B, 1, 1, 1]
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Get alpha_bar values at timesteps t.
        
        Args:
            t: Timesteps [B]
            
        Returns:
            Alpha_bar values [B]
        """
        return self.alphas_cumprod[t]
    
    def predict_x0_from_eps(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        eps: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise eps.
        
        Uses the formula:
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        
        Args:
            x_t: Noisy latent [B, C, H, W]
            t: Timesteps [B]
            eps: Predicted noise [B, C, H, W]
            
        Returns:
            Predicted clean latent x_0 [B, C, H, W]
        """
        sqrt_alpha_bar_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t
        
        return x0_pred
    
    def __repr__(self) -> str:
        return (
            f"LatentDiffusionScheduler(\n"
            f"  num_train_timesteps={self.num_train_timesteps},\n"
            f"  beta_schedule='{self.beta_schedule}',\n"
            f"  beta_start={self.beta_start},\n"
            f"  beta_end={self.beta_end},\n"
            f"  device='{self.device}'\n"
            f")"
        )
