"""Diffusion model trainer implementation.

Handles training loop for latent diffusion baseline (no CLIP yet).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm import tqdm
import logging
import numpy as np
import wandb
import torchvision.utils as vutils

from models.diffusion import (
    FrozenVAELatentInterface,
    LatentDiffusionUNet,
    LatentDiffusionScheduler,
    DiffusionLoss
)
from utils.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class LatentDiffusionTrainer:
    """Trainer for latent diffusion baseline model (without CLIP).
    
    Manages diffusion training including:
    - Frozen VAE latent encoding/decoding
    - Noise scheduling and diffusion process  
    - Epsilon-prediction training
    - Validation with reconstruction visualization
    - Checkpointing and WandB logging
    - Overfit mode for debugging
    
    Args:
        model: LatentDiffusionUNet model
        vae_interface: FrozenVAELatentInterface for encoding/decoding
        scheduler: LatentDiffusionScheduler for noise scheduling
        train_loader: Training dataloader
        val_loader: Validation dataloader
        optimizer: Optimizer
        criterion: Diffusion loss function
        config: Training configuration dict
        device: Device to train on
        lr_scheduler: Optional learning rate scheduler
        use_wandb: Whether to use WandB logging
    """
    
    def __init__(
        self,
        model: LatentDiffusionUNet,
        vae_interface: FrozenVAELatentInterface,
        scheduler: LatentDiffusionScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: DiffusionLoss,
        config: Dict[str, Any],
        device: str = "cuda",
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_wandb: bool = True,
    ):
        self.model = model.to(device)
        self.vae_interface = vae_interface
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.use_wandb = use_wandb
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Extract config values
        self.epochs = config['train']['epochs']
        self.grad_clip_norm = config['train'].get('grad_clip_norm', None)
        self.log_every_n_steps = config['train'].get('log_every_n_steps', 50)
        
        # Checkpoint config
        self.output_dir = Path(config['checkpoint']['output_dir'])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.vis_dir = self.output_dir / "visualizations"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best = config['checkpoint'].get('save_best', True)
        self.save_every_n_epochs = config['checkpoint'].get('save_every_n_epochs', 50)
        self.monitor_metric = config['checkpoint'].get('monitor', 'val/loss')
        
        # Evaluation config
        self.eval_every_n_epochs = config['eval'].get('eval_every_n_epochs', 50)
        self.num_vis_samples = config['eval'].get('num_visualizations', 8)
        self.save_visualizations = config['eval'].get('save_visualizations', True)
        
        # Verify VAE is frozen
        self._verify_vae_frozen()
        
        logger.info(f"LatentDiffusionTrainer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Training samples: {len(train_loader.dataset)}")
        logger.info(f"  Validation samples: {len(val_loader.dataset)}")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Model parameters: {self.count_parameters(self.model):,}")
        logger.info(f"  VAE frozen: {self.vae_interface.is_frozen()}")
        logger.info(f"  Output dir: {self.output_dir}")
    
    def _verify_vae_frozen(self):
        """Verify that VAE has no trainable parameters."""
        if not self.vae_interface.is_frozen():
            raise RuntimeError("VAE is not frozen! All VAE parameters must have requires_grad=False")
        
        trainable_params = self.vae_interface.count_trainable_parameters()
        if trainable_params != 0:
            raise RuntimeError(f"VAE has {trainable_params} trainable parameters! Expected 0.")
        
        logger.info(f"✓ VAE is frozen (0 trainable parameters, {self.vae_interface.count_total_parameters():,} total)")
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of average training metrics
        """
        self.model.train()
        
        # Accumulators
        loss_sum = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Extract masks from batch
            coarse_mask = batch['coarse_mask'].to(self.device)  # [B, 1, 512, 512]
            refined_mask = batch['refined_mask'].to(self.device)  # [B, 1, 512, 512]
            
            # Validate shapes
            assert coarse_mask.shape[1:] == (1, 512, 512), f"Expected [B,1,512,512], got {coarse_mask.shape}"
            assert refined_mask.shape[1:] == (1, 512, 512), f"Expected [B,1,512,512], got {refined_mask.shape}"
            
            # Encode to latents with frozen VAE (no gradients)
            with torch.no_grad():
                z_coarse = self.vae_interface.encode_coarse_mask(coarse_mask)  # [B, 8, 32, 32]
                z_refined = self.vae_interface.encode_refined_mask(refined_mask)  # [B, 8, 32, 32]
            
            # Sample timesteps
            batch_size = z_refined.shape[0]
            timesteps = self.scheduler.sample_timesteps(batch_size, device=self.device)  # [B]
            
            # Sample noise
            noise = torch.randn_like(z_refined)  # [B, 8, 32, 32]
            
            # Add noise to latent (forward diffusion)
            z_t = self.scheduler.q_sample(z_refined, timesteps, noise)  # [B, 8, 32, 32]
            
            # Predict noise with U-Net
            # Input: [z_t || z_coarse] = [B, 16, 32, 32]
            self.optimizer.zero_grad()
            eps_pred = self.model(z_t, timesteps, z_coarse)  # [B, 8, 32, 32]
            
            # Compute epsilon-prediction loss
            loss = self.criterion(eps_pred, noise)
            
            # Check for NaN/Inf
            if not torch.isfinite(loss):
                logger.error(f"Non-finite loss detected at step {self.global_step}")
                logger.error(f"  Loss: {loss.item()}")
                logger.error(f"  z_t stats: min={z_t.min():.4f}, max={z_t.max():.4f}, mean={z_t.mean():.4f}")
                logger.error(f"  eps_pred stats: min={eps_pred.min():.4f}, max={eps_pred.max():.4f}, mean={eps_pred.mean():.4f}")
                raise ValueError("Non-finite loss detected")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            
            self.optimizer.step()
            
            # Accumulate metrics
            loss_sum += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Log to wandb
            if self.use_wandb and (self.global_step % self.log_every_n_steps == 0):
                wandb_metrics = {
                    'train/loss': loss.item(),
                    'train/step': self.global_step,
                    'train/epoch': self.current_epoch,
                    'train/z_coarse_std': z_coarse.std().item(),
                    'train/z_refined_std': z_refined.std().item(),
                    'train/eps_pred_std': eps_pred.std().item(),
                }
                
                if self.lr_scheduler is not None:
                    wandb_metrics['train/lr'] = self.lr_scheduler.get_last_lr()[0]
                
                wandb.log(wandb_metrics, step=self.global_step)
        
        # Compute average
        avg_loss = loss_sum / num_batches
        
        return {'train/loss': avg_loss}
    
    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary of average validation metrics
        """
        self.model.eval()
        
        # Accumulators
        loss_sum = 0.0
        num_batches = 0
        
        # Store first batch for visualization
        vis_batch = None
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch_idx, batch in enumerate(pbar):
            # Extract masks
            coarse_mask = batch['coarse_mask'].to(self.device)
            refined_mask = batch['refined_mask'].to(self.device)
            
            # Encode to latents
            z_coarse = self.vae_interface.encode_coarse_mask(coarse_mask)
            z_refined = self.vae_interface.encode_refined_mask(refined_mask)
            
            # Sample timesteps
            batch_size = z_refined.shape[0]
            timesteps = self.scheduler.sample_timesteps(batch_size, device=self.device)
            
            # Sample noise
            noise = torch.randn_like(z_refined)
            
            # Add noise
            z_t = self.scheduler.q_sample(z_refined, timesteps, noise)
            
            # Predict noise
            eps_pred = self.model(z_t, timesteps, z_coarse)
            
            # Compute loss
            loss = self.criterion(eps_pred, noise)
            
            loss_sum += loss.item()
            num_batches += 1
            
            # Store first batch for visualization
            if batch_idx == 0:
                vis_batch = {
                    'coarse_mask': coarse_mask[:self.num_vis_samples],
                    'refined_mask': refined_mask[:self.num_vis_samples],
                    'z_coarse': z_coarse[:self.num_vis_samples],
                    'z_refined': z_refined[:self.num_vis_samples],
                    'z_t': z_t[:self.num_vis_samples],
                    'eps_pred': eps_pred[:self.num_vis_samples],
                    'noise': noise[:self.num_vis_samples],
                    'timesteps': timesteps[:self.num_vis_samples],
                }
            
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        # Compute average
        avg_loss = loss_sum / num_batches
        
        # Generate visualizations
        if self.save_visualizations and vis_batch is not None:
            self.save_eval_visualization(vis_batch, epoch=self.current_epoch)
        
        return {'val/loss': avg_loss}
    
    def save_eval_visualization(self, vis_batch: Dict, epoch: int):
        """Save evaluation visualization grid.
        
        Visualizes:
        1. Coarse GT mask
        2. Refined GT mask
        3. Decoded coarse latent
        4. Decoded predicted x0 (denoised latent)
        5. Error map (optional)
        
        Args:
            vis_batch: Dictionary with coarse_mask, refined_mask, z_coarse, z_t, eps_pred, noise, timesteps
            epoch: Current epoch number
        """
        # Predict x0 from epsilon
        x0_pred = self.scheduler.predict_x0_from_eps(
            vis_batch['z_t'],
            vis_batch['timesteps'],
            vis_batch['eps_pred']
        )  # [N, 8, 32, 32]
        
        # Decode latents through frozen VAE
        with torch.no_grad():
            decoded_coarse = self.vae_interface.decode_to_probs(vis_batch['z_coarse'])  # [N, 1, 512, 512]
            decoded_pred = self.vae_interface.decode_to_probs(x0_pred)  # [N, 1, 512, 512]
        
        # Get GT masks
        coarse_gt = vis_batch['coarse_mask']  # [N, 1, 512, 512]
        refined_gt = vis_batch['refined_mask']  # [N, 1, 512, 512]
        
        # Create comparison grid: [coarse_gt, refined_gt, decoded_coarse, decoded_pred]
        comparison = torch.cat([coarse_gt, refined_gt, decoded_coarse, decoded_pred], dim=3)  # [N, 1, 512, 512*4]
        
        # Make grid
        grid = vutils.make_grid(comparison, nrow=1, padding=2, normalize=False, pad_value=1.0)
        
        # Save to disk
        save_path = self.vis_dir / f"epoch_{epoch:04d}.png"
        vutils.save_image(grid, save_path)
        
        # Log to wandb
        if self.use_wandb:
            grid_np = grid.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
            grid_np = (grid_np * 255).astype(np.uint8)
            if grid_np.shape[2] == 1:
                grid_np = grid_np.squeeze(2)
            
            wandb.log({
                'val/visualization': wandb.Image(
                    grid_np,
                    caption=f"Epoch {epoch}: Coarse GT | Refined GT | Decoded Coarse | Decoded Pred"
                )
            }, step=self.global_step)
        
        logger.info(f"Saved visualization to {save_path}")
    
    def save_checkpoint_wrapper(self, name: str, **extra_data):
        """Save checkpoint with given name.
        
        Args:
            name: Checkpoint filename (e.g., 'best.pt', 'latest.pt')
            **extra_data: Additional data to save in checkpoint
        """
        save_path = self.checkpoint_dir / name
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            global_step=self.global_step,
            save_path=str(save_path),
            scheduler=self.lr_scheduler,
            best_val_loss=self.best_val_loss,
            config=self.config,
            **extra_data
        )
    
    def load_checkpoint_wrapper(self, checkpoint_path: str):
        """Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            device=self.device
        )
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
    
    def fit(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Log epoch metrics
            logger.info(f"Epoch {epoch+1}/{self.epochs} - Train: " + 
                       ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
            
            if self.use_wandb:
                wandb.log({**train_metrics, 'epoch': epoch}, step=self.global_step)
            
            # Validation
            if (epoch + 1) % self.eval_every_n_epochs == 0:
                val_metrics = self.validate_epoch()
                
                logger.info(f"Epoch {epoch+1}/{self.epochs} - Val: " + 
                           ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
                
                if self.use_wandb:
                    wandb.log(val_metrics, step=self.global_step)
                
                # Check for best model
                val_loss = val_metrics.get(self.monitor_metric, val_metrics.get('val/loss', float('inf')))
                
                if self.save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint_wrapper('best.pt', is_best=True)
                    logger.info(f"Saved best model with {self.monitor_metric}={val_loss:.4f}")
            
            # Save checkpoints
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint_wrapper('latest.pt')
                self.save_checkpoint_wrapper(f'epoch_{epoch+1:04d}.pt')
            
            # Learning rate scheduler step
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
