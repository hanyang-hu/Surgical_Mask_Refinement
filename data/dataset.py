"""PyTorch dataset classes for surgical mask data.

Provides dataset classes for loading paired RGB images and masks
with optional augmentation.
"""

from pathlib import Path
from typing import Dict, Optional, Callable, List, Union
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class SurgicalMaskRefinementDataset(Dataset):
    """Unified dataset for paired RGB images and segmentation masks.
    
    Loads RGB, coarse_mask, and refined_mask triplets from split files.
    Supports flexible loading modes for different use cases.
    
    Args:
        metadata_dir: Directory containing split JSON files
        split: Which split to load ('train', 'val', 'test')
        source: Which source to load ('all', 'real_world', 'synthetic')
        load_images: Whether to load actual image data (default: True)
        return_paths: Whether to include file paths in returned samples (default: False)
        apply_transforms: Whether to apply preprocessing transforms (default: False)
        transform: Optional paired transform callable (if None, no transforms applied)
        
    Example:
        >>> # Load training data with transforms
        >>> from data.transforms import build_transforms
        >>> transform = build_transforms(train=True, augment=True)
        >>> dataset = SurgicalMaskRefinementDataset(
        ...     metadata_dir='data/metadata',
        ...     split='train',
        ...     apply_transforms=True,
        ...     transform=transform
        ... )
        >>> sample = dataset[0]
        >>> print(sample['rgb'].shape)  # torch.Size([3, 512, 512])
        
        >>> # Load validation data, paths only (for precomputing)
        >>> dataset = SurgicalMaskRefinementDataset(
        ...     metadata_dir='data/metadata',
        ...     split='val',
        ...     load_images=False,
        ...     return_paths=True
        ... )
        >>> sample = dataset[0]
        >>> print(sample['rgb_path'])  # Path to RGB image
    """
    
    def __init__(
        self,
        metadata_dir: Union[str, Path],
        split: str = "train",
        source: str = "all",
        load_images: bool = True,
        return_paths: bool = False,
        apply_transforms: bool = False,
        transform: Optional[Callable] = None,
    ):
        """Initialize dataset."""
        self.metadata_dir = Path(metadata_dir)
        self.split = split
        self.source = source
        self.load_images = load_images
        self.return_paths = return_paths
        self.apply_transforms = apply_transforms
        self.transform = transform
        
        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'")
        
        # Load samples from split file
        split_file = self.metadata_dir / f"{split}.json"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Have you run the split generation script?"
            )
        
        with open(split_file, 'r') as f:
            all_samples = json.load(f)
        
        # Filter by source if requested
        if source == "all":
            self.samples = all_samples
        elif source in ["real_world", "synthetic"]:
            self.samples = [s for s in all_samples if s['source'] == source]
        else:
            raise ValueError(
                f"Invalid source '{source}'. Must be 'all', 'real_world', or 'synthetic'"
            )
        
        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found for split '{split}' and source '{source}'"
            )
        
        print(f"Loaded {len(self.samples)} samples from {split} split (source: {source})")
        
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def _load_image(self, path: str) -> Image.Image:
        """Load an image from disk.
        
        Args:
            path: Path to image file
            
        Returns:
            PIL Image
        """
        return Image.open(path)
        
    def __getitem__(self, idx: int) -> Dict[str, Union[str, Image.Image, torch.Tensor]]:
        """Load and return a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample data based on configuration:
                - 'id': Unique sample identifier (always included)
                - 'file_stem': Original filename stem (always included)
                - 'source': Source type (always included)
                - 'rgb': RGB PIL Image or tensor (if load_images=True)
                - 'coarse_mask': Coarse mask PIL Image or tensor (if load_images=True)
                - 'refined_mask': Refined mask PIL Image or tensor (if load_images=True)
                - 'rgb_path': Path to RGB image (if return_paths=True)
                - 'coarse_mask_path': Path to coarse mask (if return_paths=True)
                - 'refined_mask_path': Path to refined mask (if return_paths=True)
        """
        sample_meta = self.samples[idx]
        
        # Always include metadata
        sample = {
            'id': sample_meta['id'],
            'file_stem': sample_meta['file_stem'],
            'source': sample_meta['source'],
        }
        
        # Add paths if requested
        if self.return_paths:
            sample['rgb_path'] = sample_meta['rgb_path']
            sample['coarse_mask_path'] = sample_meta['coarse_mask_path']
            sample['refined_mask_path'] = sample_meta['refined_mask_path']
        
        # Load images if requested
        if self.load_images:
            # Load RGB image
            rgb = self._load_image(sample_meta['rgb_path'])
            
            # Load masks
            coarse_mask = self._load_image(sample_meta['coarse_mask_path'])
            refined_mask = self._load_image(sample_meta['refined_mask_path'])
            
            # Apply paired transforms if requested
            if self.apply_transforms and self.transform is not None:
                rgb, coarse_mask, refined_mask = self.transform(rgb, coarse_mask, refined_mask)
            
            sample['rgb'] = rgb
            sample['coarse_mask'] = coarse_mask
            sample['refined_mask'] = refined_mask
        
        return sample
    
    def get_source_counts(self) -> Dict[str, int]:
        """Get counts of samples by source.
        
        Returns:
            Dictionary mapping source name to count
        """
        from collections import defaultdict
        counts = defaultdict(int)
        for sample in self.samples:
            counts[sample['source']] += 1
        return dict(counts)


class VAEDataset(Dataset):
    """Dataset for VAE training on masks only.
    
    Loads only mask images for VAE pretraining.
    This is a convenience wrapper around SurgicalMaskRefinementDataset
    that focuses on masks.
    
    Args:
        metadata_dir: Directory containing split JSON files
        split: Which split to load ('train', 'val', 'test')
        source: Which source to load ('all', 'real_world', 'synthetic')
        mask_type: Which mask to load ('refined', 'coarse', 'both')
        apply_transforms: Whether to apply transforms (default: False)
        mask_transform: Optional transform for masks (should handle single mask)
    """
    
    def __init__(
        self,
        metadata_dir: Union[str, Path],
        split: str = "train",
        source: str = "all",
        mask_type: str = "refined",
        apply_transforms: bool = False,
        mask_transform: Optional[Callable] = None,
    ):
        """Initialize VAE dataset."""
        self.base_dataset = SurgicalMaskRefinementDataset(
            metadata_dir=metadata_dir,
            split=split,
            source=source,
            load_images=True,
            return_paths=False,
            apply_transforms=False,  # We handle transforms separately for masks
            transform=None,
        )
        
        if mask_type not in ['refined', 'coarse', 'both']:
            raise ValueError(
                f"Invalid mask_type '{mask_type}'. Must be 'refined', 'coarse', or 'both'"
            )
        
        self.mask_type = mask_type
        self.apply_transforms = apply_transforms
        self.mask_transform = mask_transform
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, Image.Image, torch.Tensor]]:
        """Load and return a mask sample.
        
        Returns:
            Dictionary containing:
                - 'id': Sample identifier
                - 'source': Source type
                - 'mask': Mask image/tensor (if mask_type is 'refined' or 'coarse')
                - 'refined_mask': Refined mask (if mask_type is 'both')
                - 'coarse_mask': Coarse mask (if mask_type is 'both')
        """
        sample = self.base_dataset[idx]
        
        result = {
            'id': sample['id'],
            'source': sample['source'],
        }
        
        if self.mask_type == 'refined':
            mask = sample['refined_mask']
            if self.apply_transforms and self.mask_transform is not None:
                mask = self.mask_transform(mask)
            result['mask'] = mask
        elif self.mask_type == 'coarse':
            mask = sample['coarse_mask']
            if self.apply_transforms and self.mask_transform is not None:
                mask = self.mask_transform(mask)
            result['mask'] = mask
        else:  # both
            refined = sample['refined_mask']
            coarse = sample['coarse_mask']
            if self.apply_transforms and self.mask_transform is not None:
                refined = self.mask_transform(refined)
                coarse = self.mask_transform(coarse)
            result['refined_mask'] = refined
            result['coarse_mask'] = coarse
        
        return result