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
from .token_dataset import TokenConditionedMaskDataset, create_token_dataset

__all__ = [
    "SurgicalMaskRefinementDataset",
    "VAEDataset",
    "TokenConditionedMaskDataset",
    "create_splits",
    "load_split",
    "create_token_dataset",
]
