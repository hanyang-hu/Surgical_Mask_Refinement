"""Debug script for token-aware dataset.

This script helps inspect and verify the TokenConditionedMaskDataset:
- Loads the dataset with specified configuration
- Prints dataset statistics and source counts
- Inspects individual samples
- Validates mask and token formats
- Tests DataLoader batching
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.token_dataset import TokenConditionedMaskDataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Debug and inspect token-aware mask dataset"
    )
    
    # Dataset arguments
    parser.add_argument(
        '--metadata_dir',
        type=str,
        default='data/metadata',
        help='Path to metadata directory (default: data/metadata)'
    )
    parser.add_argument(
        '--token_dir',
        type=str,
        default='outputs/clip_tokens',
        help='Path to precomputed tokens (default: outputs/clip_tokens)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Split to load (default: train)'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='all',
        choices=['all', 'real_world', 'synthetic'],
        help='Source filter (default: all)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=512,
        help='Target image size for masks (default: 512)'
    )
    
    # Inspection arguments
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1,
        help='Number of samples to inspect (default: 1)'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Starting index for sample inspection (default: 0)'
    )
    parser.add_argument(
        '--load_spatial_map',
        action='store_true',
        help='Load spatial feature maps if available'
    )
    parser.add_argument(
        '--return_paths',
        action='store_true',
        help='Include file paths in samples'
    )
    
    # DataLoader testing
    parser.add_argument(
        '--test_dataloader',
        action='store_true',
        help='Test PyTorch DataLoader batching'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for DataLoader test (default: 4)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of DataLoader workers (default: 0)'
    )
    
    return parser.parse_args()


def print_separator(char='=', length=70):
    """Print a separator line."""
    print('\n' + char * length)


def print_dataset_info(dataset: TokenConditionedMaskDataset):
    """Print dataset information and statistics."""
    print_separator()
    print("DATASET INFORMATION")
    print_separator()
    
    print(f"Split: {dataset.split}")
    print(f"Source: {dataset.source}")
    print(f"Total samples: {len(dataset)}")
    print(f"Image size: {dataset.image_size}")
    print(f"Load spatial map: {dataset.load_spatial_map}")
    print(f"Strict tokens: {dataset.strict_tokens}")
    
    # Print source counts
    source_counts = dataset.get_source_counts()
    print(f"\nSource distribution:")
    for source, count in sorted(source_counts.items()):
        percentage = (count / len(dataset)) * 100
        print(f"  {source}: {count} ({percentage:.1f}%)")


def inspect_sample(sample: dict, idx: int):
    """Inspect and print details of a single sample."""
    print_separator('-')
    print(f"SAMPLE {idx}")
    print_separator('-')
    
    # Basic metadata
    print(f"ID: {sample['id']}")
    print(f"File stem: {sample['file_stem']}")
    print(f"Source: {sample['source']}")
    
    # Paths (if available)
    if 'coarse_mask_path' in sample:
        print(f"\nPaths:")
        print(f"  Coarse mask: {sample['coarse_mask_path']}")
        print(f"  Refined mask: {sample['refined_mask_path']}")
        print(f"  Token file: {sample['token_path']}")
    
    # Coarse mask
    print(f"\nCoarse Mask:")
    coarse = sample['coarse_mask']
    print(f"  Shape: {coarse.shape}")
    print(f"  Dtype: {coarse.dtype}")
    print(f"  Range: [{coarse.min():.4f}, {coarse.max():.4f}]")
    unique_vals = torch.unique(coarse)
    print(f"  Unique values: {len(unique_vals)} values")
    if len(unique_vals) <= 10:
        print(f"    {unique_vals.tolist()}")
    
    # Check if binary
    is_binary = torch.all((coarse == 0.0) | (coarse == 1.0))
    print(f"  Is binary {{0.0, 1.0}}: {is_binary}")
    if not is_binary:
        print(f"    WARNING: Mask should be binary!")
    
    # Refined mask
    print(f"\nRefined Mask:")
    refined = sample['refined_mask']
    print(f"  Shape: {refined.shape}")
    print(f"  Dtype: {refined.dtype}")
    print(f"  Range: [{refined.min():.4f}, {refined.max():.4f}]")
    unique_vals = torch.unique(refined)
    print(f"  Unique values: {len(unique_vals)} values")
    if len(unique_vals) <= 10:
        print(f"    {unique_vals.tolist()}")
    
    # Check if binary
    is_binary = torch.all((refined == 0.0) | (refined == 1.0))
    print(f"  Is binary {{0.0, 1.0}}: {is_binary}")
    if not is_binary:
        print(f"    WARNING: Mask should be binary!")
    
    # RGB tokens
    print(f"\nRGB Tokens:")
    tokens = sample['rgb_tokens']
    print(f"  Shape: {tokens.shape}")
    print(f"  Dtype: {tokens.dtype}")
    print(f"  Range: [{tokens.min():.4f}, {tokens.max():.4f}]")
    print(f"  Mean: {tokens.mean():.4f}")
    print(f"  Std: {tokens.std():.4f}")
    
    # Check expected shape
    expected_shape = (196, 768)
    if tokens.shape != torch.Size(expected_shape):
        print(f"    WARNING: Expected shape {expected_shape}, got {tokens.shape}")
    else:
        print(f"  ✓ Shape matches expected [196, 768]")
    
    # Spatial map (if available)
    if 'rgb_spatial_map' in sample:
        print(f"\nRGB Spatial Map:")
        spatial = sample['rgb_spatial_map']
        print(f"  Shape: {spatial.shape}")
        print(f"  Dtype: {spatial.dtype}")
        print(f"  Range: [{spatial.min():.4f}, {spatial.max():.4f}]")


def test_dataloader(dataset: TokenConditionedMaskDataset, batch_size: int, num_workers: int):
    """Test DataLoader batching."""
    print_separator()
    print("DATALOADER TEST")
    print_separator()
    
    print(f"Creating DataLoader:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Shuffle: False")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    print(f"\nDataLoader created successfully")
    print(f"  Total batches: {len(dataloader)}")
    
    # Load first batch
    print(f"\nLoading first batch...")
    try:
        batch = next(iter(dataloader))
        print(f"✓ Batch loaded successfully")
        
        print(f"\nBatch contents:")
        print(f"  ID: {len(batch['id'])} samples")
        print(f"  Coarse mask shape: {batch['coarse_mask'].shape}")
        print(f"  Refined mask shape: {batch['refined_mask'].shape}")
        print(f"  RGB tokens shape: {batch['rgb_tokens'].shape}")
        
        # Validate shapes
        B = len(batch['id'])
        print(f"\nShape validation (batch_size={B}):")
        
        expected_coarse = (B, 1, dataset.image_size[0], dataset.image_size[1])
        if batch['coarse_mask'].shape == expected_coarse:
            print(f"  ✓ Coarse mask: {batch['coarse_mask'].shape}")
        else:
            print(f"  ✗ Coarse mask: expected {expected_coarse}, got {batch['coarse_mask'].shape}")
        
        expected_refined = (B, 1, dataset.image_size[0], dataset.image_size[1])
        if batch['refined_mask'].shape == expected_refined:
            print(f"  ✓ Refined mask: {batch['refined_mask'].shape}")
        else:
            print(f"  ✗ Refined mask: expected {expected_refined}, got {batch['refined_mask'].shape}")
        
        expected_tokens = (B, 196, 768)
        if batch['rgb_tokens'].shape == expected_tokens:
            print(f"  ✓ RGB tokens: {batch['rgb_tokens'].shape}")
        else:
            print(f"  ✗ RGB tokens: expected {expected_tokens}, got {batch['rgb_tokens'].shape}")
        
        # Check spatial map if present
        if 'rgb_spatial_map' in batch:
            expected_spatial = (B, 768, 14, 14)
            if batch['rgb_spatial_map'].shape == expected_spatial:
                print(f"  ✓ RGB spatial map: {batch['rgb_spatial_map'].shape}")
            else:
                print(f"  ✗ RGB spatial map: expected {expected_spatial}, got {batch['rgb_spatial_map'].shape}")
        
        # Memory usage estimate
        mask_memory = batch['coarse_mask'].element_size() * batch['coarse_mask'].nelement()
        mask_memory += batch['refined_mask'].element_size() * batch['refined_mask'].nelement()
        token_memory = batch['rgb_tokens'].element_size() * batch['rgb_tokens'].nelement()
        total_memory = mask_memory + token_memory
        
        print(f"\nMemory usage (per batch):")
        print(f"  Masks: {mask_memory / 1024 / 1024:.2f} MB")
        print(f"  Tokens: {token_memory / 1024 / 1024:.2f} MB")
        print(f"  Total: {total_memory / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main debug script."""
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("TOKEN-AWARE MASK DATASET DEBUG")
    print("=" * 70)
    
    # Create dataset
    print("\nInitializing dataset...")
    try:
        dataset = TokenConditionedMaskDataset(
            metadata_dir=args.metadata_dir,
            token_dir=args.token_dir,
            split=args.split,
            source=args.source,
            image_size=args.image_size,
            load_spatial_map=args.load_spatial_map,
            return_paths=args.return_paths,
            strict_tokens=True
        )
        print("✓ Dataset initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print dataset info
    print_dataset_info(dataset)
    
    # Inspect samples
    if args.num_samples > 0:
        print_separator()
        print(f"INSPECTING {args.num_samples} SAMPLE(S)")
        print_separator()
        
        for i in range(args.num_samples):
            idx = args.start_idx + i
            
            if idx >= len(dataset):
                print(f"\nWarning: Index {idx} out of range (dataset size: {len(dataset)})")
                break
            
            try:
                sample = dataset[idx]
                inspect_sample(sample, idx)
            except Exception as e:
                print(f"\n✗ Failed to load sample {idx}: {e}")
                import traceback
                traceback.print_exc()
    
    # Test DataLoader
    if args.test_dataloader:
        try:
            test_dataloader(dataset, args.batch_size, args.num_workers)
        except Exception as e:
            print(f"\n✗ DataLoader test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print_separator()
    print("DEBUG COMPLETE")
    print_separator()
    print()


if __name__ == '__main__':
    main()
