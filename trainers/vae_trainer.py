"""VAE trainer implementation.

Handles training loop for mask VAE with reconstruction and KL losses.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm import tqdm
import logging
import numpy as np
import wandb
from PIL import Image
import torchvision.utils as vutils

from models.vae import MaskVAE, VAELoss
from utils.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class VAETrainer:
    """Trainer for mask-only VAE.
    
    Manages VAE training including:
    - Training and validation loops
    - Loss computation (BCE + Dice + KL)
    - Checkpointing (best + latest)
    - Visualization of reconstructions
    - WandB logging in online mode
    - Overfit mode for debugging
    
    Args:
        model: MaskVAE model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        optimizer: Optimizer
        criterion: VAE loss function
        config: Training configuration dict
        device: Device to train on
        scheduler: Optional learning rate scheduler
        use_wandb: Whether to use WandB logging
    """
    
    def __init__(
        self,
        model: MaskVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: VAELoss,
        config: Dict[str, Any],
        device: str = "cuda",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_wandb: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.scheduler = scheduler
        self.use_wandb = use_wandb
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Extract config values
        self.epochs = config['train']['epochs']
        self.grad_clip_norm = config['train'].get('grad_clip_norm', None)
        self.log_every_n_steps = config['train'].get('log_every_n_steps', 50)
        self.val_every_n_epochs = config['train'].get('val_every_n_epochs', 1)
        self.save_every_n_epochs = config['train'].get('save_every_n_epochs', 1)
        
        # Checkpoint config
        self.output_dir = Path(config['checkpoint']['output_dir'])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.vis_dir = self.output_dir / "reconstructions"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best = config['checkpoint'].get('save_best', True)
        self.monitor_metric = config['checkpoint'].get('monitor', 'val/loss')
        
        # Visualization config
        self.num_vis_samples = config['visualization'].get('num_samples', 4)
        self.save_reconstructions = config['visualization'].get('save_reconstructions', True)
        
        logger.info(f"VAETrainer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Training samples: {len(train_loader.dataset)}")
        logger.info(f"  Validation samples: {len(val_loader.dataset)}")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Output dir: {self.output_dir}")
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of average training metrics
        """
        self.model.train()
        
        # Accumulators for metrics
        metrics_sum = {
            'loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'bce_loss': 0.0,
            'dice_loss': 0.0,
        }
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Extract mask from batch
            if isinstance(batch, dict):
                if 'mask' in batch:
                    x = batch['mask'].to(self.device)
                elif 'refined_mask' in batch:
                    x = batch['refined_mask'].to(self.device)
                else:
                    raise KeyError(f"Batch keys: {batch.keys()}. Expected 'mask' or 'refined_mask'")
            else:
                # Assume it's a tuple (id, source, mask) or similar
                x = batch[-1].to(self.device) if isinstance(batch, (tuple, list)) else batch.to(self.device)
            
            # Validate input shape
            if x.shape[1:] != (1, 512, 512):
                raise ValueError(f"Expected mask shape [B, 1, 512, 512], got {x.shape}")
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x)
            
            # Compute loss
            losses = self.criterion(
                outputs['recon_logits'],
                x,
                outputs['mu'],
                outputs['logvar']
            )
            loss = losses['loss']
            
            # Check for NaN/Inf
            if not torch.isfinite(loss):
                logger.error(f"Non-finite loss detected at step {self.global_step}")
                logger.error(f"  Loss: {loss.item()}")
                logger.error(f"  Input stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
                raise ValueError("Non-finite loss detected")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            
            self.optimizer.step()
            
            # Accumulate metrics
            for key in metrics_sum.keys():
                if key in losses:
                    metrics_sum[key] += losses[key].item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{losses['recon_loss'].item():.4f}",
                'kl': f"{losses['kl_loss'].item():.4f}",
            })
            
            # Log to wandb
            if self.use_wandb and (self.global_step % self.log_every_n_steps == 0):
                wandb_metrics = {
                    'train/loss': loss.item(),
                    'train/recon_loss': losses['recon_loss'].item(),
                    'train/kl_loss': losses['kl_loss'].item(),
                    'train/step': self.global_step,
                    'train/epoch': self.current_epoch,
                }
                if 'bce_loss' in losses:
                    wandb_metrics['train/bce_loss'] = losses['bce_loss'].item()
                if 'dice_loss' in losses:
                    wandb_metrics['train/dice_loss'] = losses['dice_loss'].item()
                
                wandb.log(wandb_metrics, step=self.global_step)
        
        # Compute averages
        avg_metrics = {f'train/{k}': v / num_batches for k, v in metrics_sum.items()}
        
        return avg_metrics
    
    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary of average validation metrics
        """
        self.model.eval()
        
        # Accumulators for metrics
        metrics_sum = {
            'loss': 0.0,
            'recon_loss': 0.0,
            'kl_loss': 0.0,
            'bce_loss': 0.0,
            'dice_loss': 0.0,
        }
        num_batches = 0
        
        # Store first batch for visualization
        first_batch_input = None
        first_batch_output = None
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch_idx, batch in enumerate(pbar):
            # Extract mask from batch
            if isinstance(batch, dict):
                if 'mask' in batch:
                    x = batch['mask'].to(self.device)
                elif 'refined_mask' in batch:
                    x = batch['refined_mask'].to(self.device)
                else:
                    raise KeyError(f"Batch keys: {batch.keys()}. Expected 'mask' or 'refined_mask'")
            else:
                x = batch[-1].to(self.device) if isinstance(batch, (tuple, list)) else batch.to(self.device)
            
            # Forward pass
            outputs = self.model(x)
            
            # Compute loss
            losses = self.criterion(
                outputs['recon_logits'],
                x,
                outputs['mu'],
                outputs['logvar']
            )
            
            # Accumulate metrics
            for key in metrics_sum.keys():
                if key in losses:
                    metrics_sum[key] += losses[key].item()
            num_batches += 1
            
            # Store first batch for visualization
            if batch_idx == 0:
                first_batch_input = x[:self.num_vis_samples]
                first_batch_output = outputs['recon_logits'][:self.num_vis_samples]
            
            # Update progress bar
            pbar.set_postfix({'val_loss': f"{losses['loss'].item():.4f}"})
        
        # Compute averages
        avg_metrics = {f'val/{k}': v / num_batches for k, v in metrics_sum.items()}
        
        # Generate and save visualizations
        if self.save_reconstructions and first_batch_input is not None:
            self.save_reconstruction_grid(
                first_batch_input,
                first_batch_output,
                epoch=self.current_epoch
            )
        
        return avg_metrics
    
    def save_reconstruction_grid(
        self,
        inputs: torch.Tensor,
        recon_logits: torch.Tensor,
        epoch: int
    ):
        """Save reconstruction visualization grid.
        
        Args:
            inputs: Input masks [N, 1, 512, 512]
            recon_logits: Reconstructed logits [N, 1, 512, 512]
            epoch: Current epoch number
        """
        # Convert logits to probabilities
        recon_probs = torch.sigmoid(recon_logits)
        
        # Threshold at 0.5 for binary mask
        recon_binary = (recon_probs > 0.5).float()
        
        # Create grid: [input, prob_map, binary]
        # Stack along width dimension for side-by-side comparison
        comparison = torch.cat([inputs, recon_probs, recon_binary], dim=3)  # [N, 1, H, W*3]
        
        # Make grid
        grid = vutils.make_grid(comparison, nrow=1, padding=2, normalize=False, pad_value=1.0)
        
        # Save to disk
        save_path = self.vis_dir / f"epoch_{epoch:04d}.png"
        vutils.save_image(grid, save_path)
        
        # Log to wandb
        if self.use_wandb:
            # Create labeled image for wandb
            grid_np = grid.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
            grid_np = (grid_np * 255).astype(np.uint8)
            if grid_np.shape[2] == 1:
                grid_np = grid_np.squeeze(2)
            
            wandb.log({
                'val/reconstructions': wandb.Image(
                    grid_np,
                    caption=f"Epoch {epoch}: Input | Prob Map | Binary (threshold=0.5)"
                )
            }, step=self.global_step)
        
        logger.info(f"Saved reconstruction grid to {save_path}")
    
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
            scheduler=self.scheduler,
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
            scheduler=self.scheduler,
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
            if (epoch + 1) % self.val_every_n_epochs == 0:
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
                
                # Optionally save epoch checkpoint
                if self.config['checkpoint'].get('save_epoch_checkpoints', False):
                    self.save_checkpoint_wrapper(f'epoch_{epoch+1:04d}.pt')
            
            # Learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
                if self.use_wandb:
                    wandb.log({'train/lr': self.scheduler.get_last_lr()[0]}, step=self.global_step)
        
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        
    def visualize_reconstructions(self, num_samples: int = 8):
        """Visualize mask reconstructions.
        
        TODO: Sample from validation set
        TODO: Generate reconstructions
        TODO: Log to tensorboard/wandb
        """
        pass
