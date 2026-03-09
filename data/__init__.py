"""Data module for dataset indexing, loading, and preprocessing.

This module provides utilities for:
- Building metadata index from dataset directory
- Creating train/val/test splits
- PyTorch dataset classes for paired RGB-mask loading
- Data augmentation and transforms
- Token-based dataset for precomputed features
"""

from .dataset import SurgicalMaskRefinementDataset, VAEDataset
from .splits import create_splits, load_split

try:
    from .token_dataset import TokenConditionedMaskDataset, create_token_dataset
except ImportError:  # Optional dependency (cv2) may be unavailable in inference-only envs
    TokenConditionedMaskDataset = None
    create_token_dataset = None

__all__ = [
    "SurgicalMaskRefinementDataset",
    "VAEDataset",
    "TokenConditionedMaskDataset",
    "create_splits",
    "load_split",
    "create_token_dataset",
]
