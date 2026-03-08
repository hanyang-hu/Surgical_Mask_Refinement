"""Evaluation metrics for segmentation.

Provides functions for computing segmentation metrics like IoU, Dice,
precision, recall, etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    threshold: float = 0.5
) -> torch.Tensor:
    """Compute Dice coefficient for binary masks.
    
    Args:
        pred: Predicted mask [B, 1, H, W] or [B, H, W] (values in [0, 1])
        target: Target mask [B, 1, H, W] or [B, H, W] (binary: 0 or 1)
        smooth: Smoothing constant for numerical stability
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dice score per sample [B] or scalar if batch size is 1
        
    Formula:
        Dice = 2 * |pred ∩ target| / (|pred| + |target|)
    """
    # Ensure same shape
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    
    # Binarize prediction
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # Flatten spatial dimensions [B, 1, H, W] -> [B, H*W]
    pred_flat = pred_binary.reshape(pred_binary.shape[0], -1)
    target_flat = target_binary.reshape(target_binary.shape[0], -1)
    
    # Compute intersection and cardinalities
    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)
    
    # Dice coefficient
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    return dice


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    threshold: float = 0.5
) -> torch.Tensor:
    """Compute Intersection over Union (IoU) for binary masks.
    
    Args:
        pred: Predicted mask [B, 1, H, W] or [B, H, W] (values in [0, 1])
        target: Target mask [B, 1, H, W] or [B, H, W] (binary: 0 or 1)
        smooth: Smoothing constant for numerical stability
        threshold: Threshold for binarizing predictions
        
    Returns:
        IoU score per sample [B] or scalar if batch size is 1
        
    Formula:
        IoU = |pred ∩ target| / |pred ∪ target|
    """
    # Ensure same shape
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    
    # Binarize prediction
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # Flatten spatial dimensions [B, 1, H, W] -> [B, H*W]
    pred_flat = pred_binary.reshape(pred_binary.shape[0], -1)
    target_flat = target_binary.reshape(target_binary.shape[0], -1)
    
    # Compute intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    
    # IoU coefficient
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def binary_cross_entropy(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = False
) -> torch.Tensor:
    """Compute binary cross-entropy loss per sample.
    
    Args:
        pred: Predicted mask [B, 1, H, W] (logits if from_logits=True, probs otherwise)
        target: Target mask [B, 1, H, W] (binary: 0 or 1)
        from_logits: Whether pred contains logits or probabilities
        
    Returns:
        BCE loss per sample [B]
    """
    if from_logits:
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    else:
        loss = F.binary_cross_entropy(pred, target, reduction='none')
    
    # Average over spatial dimensions [B, 1, H, W] -> [B]
    loss_per_sample = loss.reshape(loss.shape[0], -1).mean(dim=1)
    
    return loss_per_sample


# Legacy aliases for backward compatibility
def compute_iou(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> float:
    """Compute Intersection over Union (IoU).
    
    Legacy function. Use iou_score() for batch processing.
    
    Args:
        pred: Predicted mask (numpy array or torch tensor)
        target: Ground truth mask
        threshold: Threshold for binarization
        
    Returns:
        IoU score (scalar)
    """
    # Convert to torch if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    # Add batch dimension if needed
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(0)
    
    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(0)
    
    iou = iou_score(pred, target, threshold=threshold)
    return iou.item()


def compute_dice(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> float:
    """Compute Dice coefficient.
    
    Legacy function. Use dice_score() for batch processing.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        threshold: Threshold for binarization
        
    Returns:
        Dice score (scalar)
    """
    # Convert to torch if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    # Add batch dimension if needed
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(0)
    
    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(0)
    
    dice = dice_score(pred, target, threshold=threshold)
    return dice.item()
    pass


def compute_precision_recall(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> tuple:
    """Compute precision and recall.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        threshold: Threshold for binarization
        
    Returns:
        Tuple of (precision, recall)
        
    TODO: Implement precision = TP / (TP + FP)
    TODO: Implement recall = TP / (TP + FN)
    """
    pass


def compute_f1_score(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall.
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F1 score
        
    TODO: Implement F1 = 2 * (precision * recall) / (precision + recall)
    """
    pass


def compute_all_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> dict:
    """Compute all segmentation metrics.
    
    Args:
        pred: Predicted mask
        target: Ground truth mask
        threshold: Threshold for binarization
        
    Returns:
        Dictionary with all metrics
        
    TODO: Compute and return IoU, Dice, precision, recall, F1
    """
    pass
