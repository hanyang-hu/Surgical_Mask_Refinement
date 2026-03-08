"""Checkpoint utilities for saving and loading model state.

Provides utilities for checkpoint management including best model tracking
and checkpoint cleanup.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    save_path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    **kwargs
):
    """Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        global_step: Global training step
        save_path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        **kwargs: Additional items to save (e.g., best_metric, config)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add any additional items
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda",
    strict: bool = True,
) -> Dict:
    """Load checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load to
        strict: Whether to strictly enforce matching keys
        
    Returns:
        Dictionary with checkpoint metadata (epoch, step, etc.)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint


def cleanup_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 3,
    exclude_patterns: Optional[list] = None
):
    """Remove old checkpoints, keeping only recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        exclude_patterns: Patterns to exclude from cleanup (e.g., ['best', 'latest'])
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return
    
    if exclude_patterns is None:
        exclude_patterns = ['best', 'latest']
    
    # Find all epoch checkpoints
    epoch_checkpoints = []
    for ckpt in checkpoint_dir.glob("epoch_*.pt"):
        # Skip excluded patterns
        if any(pattern in ckpt.name for pattern in exclude_patterns):
            continue
        epoch_checkpoints.append(ckpt)
    
    # Sort by modification time
    epoch_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Remove old checkpoints
    for ckpt in epoch_checkpoints[keep_last_n:]:
        logger.info(f"Removing old checkpoint: {ckpt.name}")
        ckpt.unlink()
