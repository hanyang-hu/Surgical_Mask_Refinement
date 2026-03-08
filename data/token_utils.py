"""Utilities for loading precomputed CLIP tokens.

Minimal helper functions for loading saved tokens from disk.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import torch


def get_token_path(
    token_dir: Union[str, Path],
    split: str,
    source: str,
    file_stem: str
) -> Path:
    """Construct path to saved token file.
    
    Args:
        token_dir: Base token directory (e.g., 'outputs/clip_tokens')
        split: Split name ('train', 'val', 'test')
        source: Source name ('real_world', 'synthetic')
        file_stem: File stem (e.g., '1672')
        
    Returns:
        Path to token file
        
    Example:
        >>> path = get_token_path('outputs/clip_tokens', 'train', 'real_world', '1672')
        >>> print(path)
        outputs/clip_tokens/train/real_world/1672.pt
    """
    token_dir = Path(token_dir)
    return token_dir / split / source / f"{file_stem}.pt"


def load_token_file(token_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """Load a saved token file.
    
    Args:
        token_path: Path to token .pt file
        
    Returns:
        Dictionary containing token data and metadata
        
    Example:
        >>> data = load_token_file('outputs/clip_tokens/train/real_world/1672.pt')
        >>> tokens = data['tokens']  # [196, 768]
        >>> print(data['id'])  # 'real_world_1672'
    """
    return torch.load(token_path)


def load_tokens_for_sample(
    token_dir: Union[str, Path],
    split: str,
    source: str,
    file_stem: str,
    device: Optional[str] = None
) -> torch.Tensor:
    """Load tokens for a specific sample.
    
    Args:
        token_dir: Base token directory
        split: Split name
        source: Source name
        file_stem: File stem
        device: Device to load tensors on (default: None, keeps on CPU)
        
    Returns:
        Token tensor [N, C]
        
    Example:
        >>> tokens = load_tokens_for_sample(
        ...     'outputs/clip_tokens',
        ...     'train',
        ...     'real_world',
        ...     '1672',
        ...     device='cuda'
        ... )
        >>> print(tokens.shape)  # torch.Size([196, 768])
    """
    token_path = get_token_path(token_dir, split, source, file_stem)
    data = load_token_file(token_path)
    tokens = data['tokens']
    
    if device is not None:
        tokens = tokens.to(device)
    
    return tokens


def verify_token_files_exist(
    token_dir: Union[str, Path],
    metadata_samples: list
) -> Dict[str, int]:
    """Verify that token files exist for a list of samples.
    
    Args:
        token_dir: Base token directory
        metadata_samples: List of sample metadata dicts with 'split', 'source', 'file_stem'
        
    Returns:
        Dictionary with counts of existing and missing files
        
    Example:
        >>> from data.splits import load_split
        >>> samples = load_split('data/metadata', 'train')
        >>> stats = verify_token_files_exist('outputs/clip_tokens', samples)
        >>> print(f"Existing: {stats['existing']}, Missing: {stats['missing']}")
    """
    token_dir = Path(token_dir)
    
    stats = {
        'existing': 0,
        'missing': 0,
        'missing_files': []
    }
    
    for sample in metadata_samples:
        # Get split, source, file_stem from sample
        split = sample.get('split', 'unknown')
        source = sample.get('source', 'unknown')
        file_stem = sample.get('file_stem', sample.get('id', 'unknown'))
        
        token_path = get_token_path(token_dir, split, source, file_stem)
        
        if token_path.exists():
            stats['existing'] += 1
        else:
            stats['missing'] += 1
            stats['missing_files'].append(str(token_path))
    
    return stats
