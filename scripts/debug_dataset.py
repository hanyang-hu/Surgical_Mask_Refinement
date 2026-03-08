"""Debug script for visualizing dataset samples and verifying data loading.

Use this script to:
- Verify dataset index is built correctly
- Check that image/mask pairs load properly
- Visualize sample images and masks
- Test data augmentation pipeline
- Verify preprocessing transforms
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import SurgicalMaskRefinementDataset
from data.transforms import build_transforms
import torch
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Debug dataset loading")
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="data/metadata",
        help="Directory containing split JSON files (default: data/metadata)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Split to debug (default: train)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "real_world", "synthetic"],
        help="Source to load (default: all)"
    )
    parser.add_argument(
        "--load_images",
        action="store_true",
        help="Load actual images (otherwise just verify metadata)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples to inspect (default: 3)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize samples with matplotlib"
    )
    parser.add_argument(
        "--save_viz",
        type=str,
        default=None,
        help="Save visualization to file (e.g., debug_viz.png)"
    )
    parser.add_argument(
        "--apply_transforms",
        action="store_true",
        help="Apply preprocessing transforms (resize to 512x512, normalize)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation (only with --apply_transforms)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size for transforms (default: 512)"
    )
    
    return parser.parse_args()


def tensor_to_numpy(img):
    """Convert tensor or PIL Image to numpy array for visualization.
    
    Args:
        img: PIL Image or torch.Tensor
        
    Returns:
        numpy array suitable for matplotlib
    """
    if torch.is_tensor(img):
        # Convert tensor to numpy
        img_np = img.cpu().numpy()
        
        # Handle different tensor shapes
        if img_np.ndim == 3:  # [C, H, W]
            if img_np.shape[0] == 3:  # RGB
                # Transpose to [H, W, C] and denormalize if needed
                img_np = img_np.transpose(1, 2, 0)
                # If normalized to [-1, 1], convert back to [0, 1]
                if img_np.min() < 0:
                    img_np = (img_np + 1) / 2
                # Clip to valid range
                img_np = np.clip(img_np, 0, 1)
            elif img_np.shape[0] == 1:  # Grayscale mask
                img_np = img_np.squeeze(0)
        elif img_np.ndim == 2:  # Already [H, W]
            pass
        else:
            raise ValueError(f"Unexpected tensor shape: {img_np.shape}")
        
        return img_np
    else:
        # PIL Image - convert directly to numpy
        return np.array(img)


def get_image_info(img):
    """Get size/shape information from PIL Image or tensor.
    
    Args:
        img: PIL Image or torch.Tensor
        
    Returns:
        String describing image dimensions
    """
    if torch.is_tensor(img):
        return f"tensor {tuple(img.shape)}"
    else:
        return f"PIL {img.size}"


def visualize_sample(rgb, coarse_mask, refined_mask, sample_id, save_path=None):
    """Visualize a single RGB-mask triple.
    
    Args:
        rgb: RGB PIL Image or tensor
        coarse_mask: Coarse mask PIL Image or tensor
        refined_mask: Refined mask PIL Image or tensor
        sample_id: Sample identifier
        save_path: Optional path to save figure
    """
    # Convert to numpy
    rgb_np = tensor_to_numpy(rgb)
    coarse_np = tensor_to_numpy(coarse_mask)
    refined_np = tensor_to_numpy(refined_mask)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB image
    axes[0].imshow(rgb_np)
    axes[0].set_title(f"RGB\n{get_image_info(rgb)}")
    axes[0].axis('off')
    
    # Coarse mask
    axes[1].imshow(coarse_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Coarse Mask\n{get_image_info(coarse_mask)}")
    axes[1].axis('off')
    
    # Refined mask
    axes[2].imshow(refined_np, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f"Refined Mask\n{get_image_info(refined_mask)}")
    axes[2].axis('off')
    
    fig.suptitle(f"Sample: {sample_id}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_batch(samples, sample_ids, save_path=None):
    """Visualize multiple samples in a grid.
    
    Args:
        samples: List of (rgb, coarse, refined) tuples
        sample_ids: List of sample IDs
        save_path: Optional path to save figure
    """
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    
    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, ((rgb, coarse, refined), sample_id) in enumerate(zip(samples, sample_ids)):
        # Convert to numpy
        rgb_np = tensor_to_numpy(rgb)
        coarse_np = tensor_to_numpy(coarse)
        refined_np = tensor_to_numpy(refined)
        
        # RGB
        axes[i, 0].imshow(rgb_np)
        if i == 0:
            axes[i, 0].set_title("RGB")
        axes[i, 0].set_ylabel(f"{sample_id}\n{get_image_info(rgb)}", fontsize=10)
        axes[i, 0].axis('off')
        
        # Coarse mask
        axes[i, 1].imshow(coarse_np, cmap='gray', vmin=0, vmax=1)
        if i == 0:
            axes[i, 1].set_title("Coarse Mask")
        axes[i, 1].axis('off')
        
        # Refined mask
        axes[i, 2].imshow(refined_np, cmap='gray', vmin=0, vmax=1)
        if i == 0:
            axes[i, 2].set_title("Refined Mask")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main debugging function."""
    args = parse_args()
    
    print("=" * 70)
    print("DATASET DEBUGGING")
    print("=" * 70)
    print(f"Metadata directory: {args.metadata_dir}")
    print(f"Split: {args.split}")
    print(f"Source: {args.source}")
    print(f"Load images: {args.load_images}")
    print(f"Apply transforms: {args.apply_transforms}")
    if args.apply_transforms:
        print(f"  - Augment: {args.augment}")
        print(f"  - Image size: {args.image_size}")
    print()
    
    try:
        # Build transform if requested
        transform = None
        if args.apply_transforms:
            train_mode = (args.split == 'train') and args.augment
            transform = build_transforms(
                train=train_mode,
                augment=args.augment,
                image_size=args.image_size
            )
            print(f"✓ Built transform pipeline (train={train_mode}, augment={args.augment})")
        
        # Load dataset
        dataset = SurgicalMaskRefinementDataset(
            metadata_dir=args.metadata_dir,
            split=args.split,
            source=args.source,
            load_images=args.load_images,
            return_paths=True,
            apply_transforms=args.apply_transforms,
            transform=transform,
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"✓ Total samples: {len(dataset)}")
        print()
        
        # Print source distribution
        source_counts = dataset.get_source_counts()
        print("Source distribution:")
        for source, count in sorted(source_counts.items()):
            percentage = (count / len(dataset)) * 100
            print(f"  - {source}: {count} ({percentage:.1f}%)")
        print()
        
        # Inspect samples
        num_to_inspect = min(args.num_samples, len(dataset))
        print(f"Inspecting first {num_to_inspect} samples:")
        print("-" * 70)
        
        samples_for_viz = []
        
        for i in range(num_to_inspect):
            sample = dataset[i]
            
            print(f"\nSample {i + 1}:")
            print(f"  ID: {sample['id']}")
            print(f"  File stem: {sample['file_stem']}")
            print(f"  Source: {sample['source']}")
            
            if args.load_images:
                rgb = sample['rgb']
                coarse = sample['coarse_mask']
                refined = sample['refined_mask']
                
                # Check if we have PIL Images or tensors
                if torch.is_tensor(rgb):
                    # Tensor mode
                    print(f"  RGB: shape={tuple(rgb.shape)}, dtype={rgb.dtype}")
                    print(f"  Coarse mask: shape={tuple(coarse.shape)}, dtype={coarse.dtype}")
                    print(f"  Refined mask: shape={tuple(refined.shape)}, dtype={refined.dtype}")
                    
                    # Check value ranges
                    print(f"  RGB value range: [{rgb.min():.3f}, {rgb.max():.3f}]")
                    print(f"  Coarse mask value range: [{coarse.min():.3f}, {coarse.max():.3f}]")
                    print(f"  Refined mask value range: [{refined.min():.3f}, {refined.max():.3f}]")
                    
                    # Check unique values in masks (should be {0.0, 1.0})
                    coarse_unique = torch.unique(coarse).cpu().numpy()
                    refined_unique = torch.unique(refined).cpu().numpy()
                    print(f"  Coarse mask unique values: {coarse_unique}")
                    print(f"  Refined mask unique values: {refined_unique}")
                    
                    # Verify binary masks
                    if not (set(coarse_unique.tolist()).issubset({0.0, 1.0})):
                        print(f"  ⚠ Warning: Coarse mask has non-binary values!")
                    if not (set(refined_unique.tolist()).issubset({0.0, 1.0})):
                        print(f"  ⚠ Warning: Refined mask has non-binary values!")
                    
                else:
                    # PIL Image mode
                    print(f"  RGB: mode={rgb.mode}, size={rgb.size}")
                    print(f"  Coarse mask: mode={coarse.mode}, size={coarse.size}")
                    print(f"  Refined mask: mode={refined.mode}, size={refined.size}")
                    
                    # Check if modes are as expected
                    if rgb.mode != 'RGB':
                        print(f"  ⚠ Warning: RGB image mode is {rgb.mode}, expected RGB")
                    if coarse.mode != 'L':
                        print(f"  ⚠ Warning: Coarse mask mode is {coarse.mode}, expected L")
                    if refined.mode != 'L':
                        print(f"  ⚠ Warning: Refined mask mode is {refined.mode}, expected L")
                    
                    # Check if sizes match
                    if rgb.size != coarse.size or rgb.size != refined.size:
                        print(f"  ⚠ Warning: Image sizes do not match!")
                    
                    # Convert to numpy for value inspection
                    rgb_np = np.array(rgb)
                    coarse_np = np.array(coarse)
                    refined_np = np.array(refined)
                    
                    print(f"  RGB value range: [{rgb_np.min()}, {rgb_np.max()}]")
                    print(f"  Coarse mask value range: [{coarse_np.min()}, {coarse_np.max()}]")
                    print(f"  Refined mask value range: [{refined_np.min()}, {refined_np.max()}]")
                    print(f"  Coarse mask unique values: {len(np.unique(coarse_np))}")
                    print(f"  Refined mask unique values: {len(np.unique(refined_np))}")
                
                samples_for_viz.append((rgb, coarse, refined))
            
            if 'rgb_path' in sample:
                print(f"  RGB path: {sample['rgb_path']}")
                print(f"  Coarse path: {sample['coarse_mask_path']}")
                print(f"  Refined path: {sample['refined_mask_path']}")
        
        print()
        print("=" * 70)
        print("✓ Dataset debugging complete!")
        
        # Visualize if requested
        if args.visualize and args.load_images and samples_for_viz:
            print("\nGenerating visualization...")
            sample_ids = [dataset[i]['id'] for i in range(num_to_inspect)]
            visualize_batch(
                samples_for_viz,
                sample_ids,
                save_path=args.save_viz
            )
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("\nMake sure you have run:")
        print("  1. python data/build_index.py --dataset_root <path>")
        print("  2. python data/splits.py --metadata_file data/metadata/all_samples.json")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
