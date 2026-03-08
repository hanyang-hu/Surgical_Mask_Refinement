"""Test DataLoader with preprocessing transforms."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import SurgicalMaskRefinementDataset
from data.transforms import build_transforms
import torch
from torch.utils.data import DataLoader


def test_dataloader():
    """Test DataLoader with transforms."""
    print("=" * 70)
    print("DATALOADER TEST WITH TRANSFORMS")
    print("=" * 70)
    
    # Build transforms
    transform = build_transforms(train=True, augment=True, image_size=512)
    
    # Create dataset
    dataset = SurgicalMaskRefinementDataset(
        metadata_dir='data/metadata',
        split='train',
        source='all',
        load_images=True,
        return_paths=False,
        apply_transforms=True,
        transform=transform,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
        drop_last=False,
    )
    
    print(f"Number of batches: {len(dataloader)}")
    print()
    
    # Load first batch
    print("Loading first batch...")
    batch = next(iter(dataloader))
    
    print(f"Batch keys: {list(batch.keys())}")
    print()
    
    # Print batch info
    print("Batch contents:")
    print(f"  RGB shape: {batch['rgb'].shape}")
    print(f"  RGB dtype: {batch['rgb'].dtype}")
    print(f"  RGB min/max: [{batch['rgb'].min():.3f}, {batch['rgb'].max():.3f}]")
    print()
    
    print(f"  Coarse mask shape: {batch['coarse_mask'].shape}")
    print(f"  Coarse mask dtype: {batch['coarse_mask'].dtype}")
    print(f"  Coarse mask min/max: [{batch['coarse_mask'].min():.3f}, {batch['coarse_mask'].max():.3f}]")
    print(f"  Coarse mask unique values: {torch.unique(batch['coarse_mask']).cpu().numpy()}")
    print()
    
    print(f"  Refined mask shape: {batch['refined_mask'].shape}")
    print(f"  Refined mask dtype: {batch['refined_mask'].dtype}")
    print(f"  Refined mask min/max: [{batch['refined_mask'].min():.3f}, {batch['refined_mask'].max():.3f}]")
    print(f"  Refined mask unique values: {torch.unique(batch['refined_mask']).cpu().numpy()}")
    print()
    
    # Check metadata
    print(f"  IDs (first 3): {batch['id'][:3]}")
    print(f"  Sources (first 3): {batch['source'][:3]}")
    print()
    
    # Verify shapes are correct
    assert batch['rgb'].shape == (8, 3, 512, 512), "RGB shape incorrect"
    assert batch['coarse_mask'].shape == (8, 1, 512, 512), "Coarse mask shape incorrect"
    assert batch['refined_mask'].shape == (8, 1, 512, 512), "Refined mask shape incorrect"
    
    # Verify mask values are binary
    coarse_unique = torch.unique(batch['coarse_mask']).cpu().numpy()
    refined_unique = torch.unique(batch['refined_mask']).cpu().numpy()
    assert set(coarse_unique.tolist()).issubset({0.0, 1.0}), "Coarse mask not binary"
    assert set(refined_unique.tolist()).issubset({0.0, 1.0}), "Refined mask not binary"
    
    print("=" * 70)
    print("✓ All checks passed!")
    print("=" * 70)
    
    # Test a few more batches
    print("\nTesting 5 more batches...")
    for i, batch in enumerate(dataloader):
        if i >= 4:
            break
        print(f"  Batch {i+2}: RGB {batch['rgb'].shape}, masks {batch['coarse_mask'].shape}")
    
    print("\n✓ DataLoader test completed successfully!")


if __name__ == "__main__":
    test_dataloader()
