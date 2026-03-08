"""Logging utilities for training and evaluation.

Provides utilities for tensorboard, wandb, and console logging.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        
    Returns:
        Configured logger
        
    TODO: Configure logger with formatters
    TODO: Add console and file handlers
    """
    pass


class TensorboardLogger:
    """Tensorboard logging wrapper.
    
    TODO: Implement tensorboard logging
    TODO: Support scalars, images, histograms
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        # TODO: Initialize tensorboard writer
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        pass
        
    def log_image(self, tag: str, image, step: int):
        """Log image."""
        pass
        
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram."""
        pass


class WandbLogger:
    """Weights & Biases logging wrapper.
    
    TODO: Implement wandb logging
    """
    
    def __init__(self, project_name: str, config: dict):
        self.project_name = project_name
        self.config = config
        # TODO: Initialize wandb run
        
    def log(self, metrics: dict, step: int):
        """Log metrics."""
        pass
