"""Dataset splitting utilities.

Create train/val/test splits from the full dataset index.
Supports stratified splitting and reproducible random seeds.
"""

from typing import Dict, List, Tuple
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def stratified_split(
    samples: List[Dict],
    split_ratios: Dict[str, float],
    stratify_key: str,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create stratified train/val/test splits.
    
    Args:
        samples: List of sample dictionaries
        split_ratios: Dictionary with 'train', 'val', 'test' ratios
        stratify_key: Key to stratify by (e.g., 'source')
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    # Group samples by stratification key
    grouped_samples = defaultdict(list)
    for sample in samples:
        key_value = sample[stratify_key]
        grouped_samples[key_value].append(sample)
    
    # Set random seed
    random.seed(seed)
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    # Split each group proportionally
    for group_name, group_samples in grouped_samples.items():
        # Shuffle group samples
        shuffled = group_samples.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * split_ratios['train'])
        val_end = train_end + int(n * split_ratios['val'])
        
        train_samples.extend(shuffled[:train_end])
        val_samples.extend(shuffled[train_end:val_end])
        test_samples.extend(shuffled[val_end:])
    
    # Shuffle final splits (maintains reproducibility with seed)
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    return train_samples, val_samples, test_samples


def create_splits(
    metadata_file: str,
    output_dir: str,
    split_ratios: Dict[str, float] = None,
    seed: int = 42,
    stratify_by: str = "source"
) -> Tuple[List, List, List]:
    """Create train/val/test splits from dataset index.
    
    Args:
        metadata_file: Path to full dataset index JSON
        output_dir: Directory to save split files
        split_ratios: Dictionary with 'train', 'val', 'test' ratios
        seed: Random seed for reproducibility
        stratify_by: Stratification key (default: 'source')
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    if split_ratios is None:
        split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    
    # Validate split ratios
    total = sum(split_ratios.values())
    if not 0.99 <= total <= 1.01:  # Allow small floating point errors
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    # Load metadata
    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_path, 'r') as f:
        all_samples = json.load(f)
    
    print(f"Loaded {len(all_samples)} samples from {metadata_file}")
    print(f"Split ratios: train={split_ratios['train']:.2f}, "
          f"val={split_ratios['val']:.2f}, test={split_ratios['test']:.2f}")
    print(f"Random seed: {seed}")
    print(f"Stratifying by: {stratify_by}")
    print("=" * 60)
    
    # Create stratified splits
    train_samples, val_samples, test_samples = stratified_split(
        all_samples, split_ratios, stratify_by, seed
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    for split_name, split_samples in splits.items():
        output_file = output_path / f"{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(split_samples, f, indent=2)
        print(f"Saved {split_name}: {len(split_samples)} samples -> {output_file}")
    
    # Print statistics by source
    print("\n" + "=" * 60)
    print("SPLIT STATISTICS BY SOURCE")
    print("=" * 60)
    
    for split_name, split_samples in splits.items():
        source_counts = defaultdict(int)
        for sample in split_samples:
            source_counts[sample[stratify_by]] += 1
        
        print(f"\n{split_name.upper()} ({len(split_samples)} total):")
        for source, count in sorted(source_counts.items()):
            percentage = (count / len(split_samples)) * 100 if split_samples else 0
            print(f"  - {source}: {count} ({percentage:.1f}%)")
    
    return train_samples, val_samples, test_samples


def load_split(split_path: str) -> List[Dict]:
    """Load a specific split from file.
    
    Args:
        split_path: Path to split JSON file
        
    Returns:
        List of sample dictionaries
    """
    with open(split_path, 'r') as f:
        return json.load(f)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate train/val/test splits from dataset index"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Path to dataset index JSON (e.g., data/metadata/all_samples.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/metadata",
        help="Output directory for split files (default: data/metadata)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--stratify_by",
        type=str,
        default="source",
        help="Field to stratify by (default: source)"
    )
    
    args = parser.parse_args()
    
    split_ratios = {
        'train': args.train_ratio,
        'val': args.val_ratio,
        'test': args.test_ratio
    }
    
    create_splits(
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
        split_ratios=split_ratios,
        seed=args.seed,
        stratify_by=args.stratify_by
    )


if __name__ == "__main__":
    main()
