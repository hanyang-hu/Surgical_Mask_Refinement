"""Data augmentation and preprocessing transforms.

Provides composition of transforms for RGB images and masks,
ensuring consistent augmentation across paired data.
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Tuple, Optional, Callable, Union
from PIL import Image
import random
import numpy as np


def binarize_mask(
    mask: Union[torch.Tensor, Image.Image, np.ndarray],
    threshold: float = 0.5
) -> torch.Tensor:
    """Binarize mask to exact {0.0, 1.0} values.
    
    Args:
        mask: Input mask (tensor, PIL Image, or numpy array)
        threshold: Threshold for binarization (default: 0.5 for [0,1] range, use 127 for [0,255])
        
    Returns:
        Binary float tensor with values {0.0, 1.0}
    """
    # Convert to tensor if needed
    if isinstance(mask, Image.Image):
        mask = TF.to_tensor(mask)
    elif isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # Normalize to [0,1] if in [0,255]
        if mask.max() > 1.0:
            mask = mask / 255.0
    
    # Ensure it's a tensor
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Expected tensor, PIL Image, or numpy array, got {type(mask)}")
    
    # Binarize
    binary_mask = (mask > threshold).float()
    
    return binary_mask


class PairedResize:
    """Resize RGB and masks with appropriate interpolation.
    
    Resizes RGB with bilinear interpolation and masks with nearest-neighbor
    interpolation to preserve label integrity.
    
    Args:
        size: Target size (height, width) or single int
        rgb_interpolation: Interpolation mode for RGB (default: bilinear)
        mask_interpolation: Interpolation mode for masks (default: nearest)
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        rgb_interpolation: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
        mask_interpolation: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
    ):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.rgb_interpolation = rgb_interpolation
        self.mask_interpolation = mask_interpolation
    
    def __call__(
        self, 
        rgb: Image.Image, 
        coarse_mask: Image.Image, 
        refined_mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """Resize RGB and masks.
        
        Args:
            rgb: RGB PIL Image
            coarse_mask: Coarse mask PIL Image
            refined_mask: Refined mask PIL Image
            
        Returns:
            Tuple of resized (rgb, coarse_mask, refined_mask)
        """
        rgb = TF.resize(rgb, self.size, self.rgb_interpolation)
        coarse_mask = TF.resize(coarse_mask, self.size, self.mask_interpolation)
        refined_mask = TF.resize(refined_mask, self.size, self.mask_interpolation)
        
        return rgb, coarse_mask, refined_mask


class PairedRandomHorizontalFlip:
    """Apply same random horizontal flip to RGB and masks.
    
    Args:
        p: Probability of flipping (default: 0.5)
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(
        self, 
        rgb: Image.Image, 
        coarse_mask: Image.Image, 
        refined_mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """Apply horizontal flip.
        
        Args:
            rgb: RGB PIL Image
            coarse_mask: Coarse mask PIL Image
            refined_mask: Refined mask PIL Image
            
        Returns:
            Tuple of (rgb, coarse_mask, refined_mask), possibly flipped
        """
        if random.random() < self.p:
            rgb = TF.hflip(rgb)
            coarse_mask = TF.hflip(coarse_mask)
            refined_mask = TF.hflip(refined_mask)
        
        return rgb, coarse_mask, refined_mask


class PairedRandomAffine:
    """Apply same random affine transformation to RGB and masks.
    
    Uses mild parameters suitable for medical/surgical image refinement.
    
    Args:
        degrees: Range of rotation degrees (default: 10)
        translate: Tuple of max absolute translations (default: (0.05, 0.05))
        scale: Range of scale factor (default: (0.95, 1.05))
        rgb_interpolation: Interpolation for RGB (default: bilinear)
        mask_interpolation: Interpolation for masks (default: nearest)
    """
    
    def __init__(
        self,
        degrees: float = 10,
        translate: Optional[Tuple[float, float]] = (0.05, 0.05),
        scale: Optional[Tuple[float, float]] = (0.95, 1.05),
        rgb_interpolation: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
        mask_interpolation: TF.InterpolationMode = TF.InterpolationMode.NEAREST,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.rgb_interpolation = rgb_interpolation
        self.mask_interpolation = mask_interpolation
    
    def __call__(
        self, 
        rgb: Image.Image, 
        coarse_mask: Image.Image, 
        refined_mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """Apply affine transformation.
        
        Args:
            rgb: RGB PIL Image
            coarse_mask: Coarse mask PIL Image
            refined_mask: Refined mask PIL Image
            
        Returns:
            Tuple of transformed (rgb, coarse_mask, refined_mask)
        """
        # Get random parameters
        angle = random.uniform(-self.degrees, self.degrees)
        
        if self.translate is not None:
            max_dx = self.translate[0] * rgb.width
            max_dy = self.translate[1] * rgb.height
            translate = (
                random.uniform(-max_dx, max_dx),
                random.uniform(-max_dy, max_dy)
            )
        else:
            translate = (0, 0)
        
        if self.scale is not None:
            scale = random.uniform(self.scale[0], self.scale[1])
        else:
            scale = 1.0
        
        shear = 0  # No shear for surgical images
        
        # Apply same transformation to all
        rgb = TF.affine(
            rgb, angle, translate, scale, shear, 
            self.rgb_interpolation
        )
        coarse_mask = TF.affine(
            coarse_mask, angle, translate, scale, shear, 
            self.mask_interpolation
        )
        refined_mask = TF.affine(
            refined_mask, angle, translate, scale, shear, 
            self.mask_interpolation
        )
        
        return rgb, coarse_mask, refined_mask


class RGBOnlyColorJitter:
    """Apply color jitter only to RGB image, not masks.
    
    Args:
        brightness: Brightness jitter factor (default: 0.1)
        contrast: Contrast jitter factor (default: 0.1)
        saturation: Saturation jitter factor (default: 0.1)
        hue: Hue jitter factor (default: 0.05)
    """
    
    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.05,
    ):
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
    
    def __call__(
        self, 
        rgb: Image.Image, 
        coarse_mask: Image.Image, 
        refined_mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """Apply color jitter to RGB only.
        
        Args:
            rgb: RGB PIL Image
            coarse_mask: Coarse mask PIL Image (unchanged)
            refined_mask: Refined mask PIL Image (unchanged)
            
        Returns:
            Tuple of (jittered_rgb, coarse_mask, refined_mask)
        """
        rgb = self.color_jitter(rgb)
        return rgb, coarse_mask, refined_mask


class ToTensorPair:
    """Convert RGB and masks to tensors with proper formatting.
    
    Args:
        binarize_masks: Whether to binarize masks to {0.0, 1.0} (default: True)
        normalize_rgb: Whether to normalize RGB with ImageNet stats (default: False)
    """
    
    def __init__(
        self,
        binarize_masks: bool = True,
        normalize_rgb: bool = False,
    ):
        self.binarize_masks = binarize_masks
        self.normalize_rgb = normalize_rgb
        
        if normalize_rgb:
            # ImageNet normalization
            self.rgb_normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.rgb_normalize = None
    
    def __call__(
        self, 
        rgb: Image.Image, 
        coarse_mask: Image.Image, 
        refined_mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to tensors.
        
        Args:
            rgb: RGB PIL Image
            coarse_mask: Coarse mask PIL Image
            refined_mask: Refined mask PIL Image
            
        Returns:
            Tuple of tensors:
                - rgb: [3, H, W] float tensor in [0,1] or normalized
                - coarse_mask: [1, H, W] float tensor
                - refined_mask: [1, H, W] float tensor
        """
        # Convert RGB to tensor [3, H, W] in [0, 1]
        rgb = TF.to_tensor(rgb)
        
        # Normalize RGB if requested
        if self.rgb_normalize is not None:
            rgb = self.rgb_normalize(rgb)
        
        # Convert masks to tensors [1, H, W]
        coarse_mask = TF.to_tensor(coarse_mask)
        refined_mask = TF.to_tensor(refined_mask)
        
        # Binarize masks if requested
        if self.binarize_masks:
            coarse_mask = binarize_mask(coarse_mask, threshold=0.5)
            refined_mask = binarize_mask(refined_mask, threshold=0.5)
        
        return rgb, coarse_mask, refined_mask


class PairedCompose:
    """Compose multiple paired transforms.
    
    Args:
        transforms: List of paired transform callables
    """
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(
        self, 
        rgb: Image.Image, 
        coarse_mask: Image.Image, 
        refined_mask: Image.Image
    ) -> Tuple[Union[Image.Image, torch.Tensor], 
               Union[Image.Image, torch.Tensor], 
               Union[Image.Image, torch.Tensor]]:
        """Apply all transforms sequentially.
        
        Args:
            rgb: RGB PIL Image
            coarse_mask: Coarse mask PIL Image
            refined_mask: Refined mask PIL Image
            
        Returns:
            Tuple of transformed (rgb, coarse_mask, refined_mask)
        """
        for t in self.transforms:
            rgb, coarse_mask, refined_mask = t(rgb, coarse_mask, refined_mask)
        return rgb, coarse_mask, refined_mask


def build_transforms(
    train: bool = True,
    image_size: Union[int, Tuple[int, int]] = 512,
    augment: bool = False,
    binarize: bool = True,
    normalize_rgb: bool = False,
) -> PairedCompose:
    """Build paired transform pipeline for RGB and masks.
    
    Args:
        train: Whether this is for training (vs. validation/test)
        image_size: Target image size (default: 512)
        augment: Whether to apply data augmentation (only used if train=True)
        binarize: Whether to binarize masks to {0.0, 1.0} (default: True)
        normalize_rgb: Whether to normalize RGB with ImageNet stats (default: False)
        
    Returns:
        PairedCompose transform that takes (rgb, coarse_mask, refined_mask)
        and returns processed versions
        
    Example:
        >>> transform = build_transforms(train=True, augment=True)
        >>> rgb_tensor, coarse_tensor, refined_tensor = transform(rgb_pil, coarse_pil, refined_pil)
    """
    transforms_list = []
    
    # Always resize to fixed size
    transforms_list.append(PairedResize(image_size))
    
    # Add augmentation for training if requested
    if train and augment:
        # Mild geometric augmentation
        transforms_list.append(PairedRandomHorizontalFlip(p=0.5))
        transforms_list.append(PairedRandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ))
        
        # RGB-only appearance augmentation
        transforms_list.append(RGBOnlyColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05
        ))
    
    # Convert to tensors (always last)
    transforms_list.append(ToTensorPair(
        binarize_masks=binarize,
        normalize_rgb=normalize_rgb
    ))
    
    return PairedCompose(transforms_list)
