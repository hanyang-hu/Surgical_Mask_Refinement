"""Image processing utilities.

Utilities for loading, preprocessing, and postprocessing images.
"""

import torch
import numpy as np
from PIL import Image
from typing import Union


def load_image(path: str) -> np.ndarray:
    """Load image from file.
    
    Args:
        path: Path to image file
        
    Returns:
        Image as numpy array
        
    TODO: Implement image loading with PIL
    TODO: Handle RGB and grayscale images
    """
    pass


def save_image(image: Union[np.ndarray, torch.Tensor], path: str):
    """Save image to file.
    
    Args:
        image: Image as numpy array or tensor
        path: Output path
        
    TODO: Implement image saving
    TODO: Handle tensor-to-numpy conversion
    TODO: Handle value range normalization
    """
    pass


def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """Resize image to target size.
    
    TODO: Implement resize with PIL or cv2
    """
    pass


def normalize_image(image: np.ndarray, mean: list, std: list) -> np.ndarray:
    """Normalize image with mean and std.
    
    TODO: Implement normalization
    """
    pass


def denormalize_image(image: np.ndarray, mean: list, std: list) -> np.ndarray:
    """Denormalize image.
    
    TODO: Implement denormalization
    """
    pass
