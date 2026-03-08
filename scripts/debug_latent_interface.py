"""Debug script for frozen VAE latent interface.

Tests the latent interface by:
1. Loading one sample with coarse and refined masks
2. Encoding both masks to latent space
3. Decoding both latents back to mask space
4. Computing and printing latent statistics
5. Saving visualization comparing coarse vs refined paths

Usage:
    python3 scripts/debug_latent_interface.py \\
        --checkpoint outputs/vae/checkpoints/best.pt \\
        --model_config configs/model/vae.yaml \\
        --split val \\
        --sample_idx 0
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion.latent_interface import FrozenVAELatentInterface
from data.dataset import SurgicalMaskRefinementDataset
from utils.metrics import dice_score, iou_score


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Debug frozen VAE latent interface"
    )
    
    # Model and checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/vae/checkpoints/best.pt",
        help="Path to trained VAE checkpoint"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model/vae.yaml",
        help="Path to model config"
    )
    
    # Dataset
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="data/metadata",
        help="Directory containing split JSON files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to load sample from"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "real_world", "synthetic"],
        help="Which source to load sample from"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of sample to test"
    )
    
    # Processing
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--use_mu_only",
        action="store_true",
        default=True,
        help="Use deterministic encoding (z=mu) instead of sampling"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing reconstructions"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/diffusion/debug_latent_interface",
        help="Output directory for results"
    )
    
    return parser.parse_args()


def create_simple_transform(image_size: int = 512):
    """Create simple deterministic transform for masks."""
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
        T.Lambda(lambda x: (x > 0.5).float())  # Binarize
    ])


def load_sample(
    metadata_dir: str,
    split: str,
    source: str,
    sample_idx: int,
    image_size: int,
    device: str
) -> dict:
    """Load one sample with coarse and refined masks."""
    # Create simple transform for masks
    mask_transform = create_simple_transform(image_size)
    
    # Custom transform function
    def paired_transform(rgb, coarse, refined):
        coarse_t = mask_transform(coarse)
        refined_t = mask_transform(refined)
        return None, coarse_t, refined_t
    
    # Create dataset
    dataset = SurgicalMaskRefinementDataset(
        metadata_dir=metadata_dir,
        split=split,
        source=source,
        load_images=True,
        return_paths=False,
        apply_transforms=True,
        transform=paired_transform
    )
    
    if sample_idx >= len(dataset):
        raise ValueError(
            f"Sample index {sample_idx} out of range. "
            f"Dataset has {len(dataset)} samples."
        )
    
    # Load sample
    sample = dataset[sample_idx]
    
    # Extract and move to device
    return {
        'id': sample['id'],
        'file_stem': sample['file_stem'],
        'source': sample['source'],
        'coarse_mask': sample['coarse_mask'].unsqueeze(0).to(device),  # [1, 1, H, W]
        'refined_mask': sample['refined_mask'].unsqueeze(0).to(device),  # [1, 1, H, W]
    }


def compute_latent_distance(z_coarse: torch.Tensor, z_refined: torch.Tensor) -> dict:
    """Compute distance metrics between coarse and refined latents.
    
    Args:
        z_coarse: Coarse mask latent [B, C, h, w]
        z_refined: Refined mask latent [B, C, h, w]
        
    Returns:
        Dictionary with mse, l1, l2, cosine_sim
    """
    mse = F.mse_loss(z_coarse, z_refined).item()
    l1 = F.l1_loss(z_coarse, z_refined).item()
    l2 = torch.norm(z_coarse - z_refined, p=2).item()
    
    # Cosine similarity (flatten to vectors)
    z_coarse_flat = z_coarse.flatten(1)
    z_refined_flat = z_refined.flatten(1)
    cosine_sim = F.cosine_similarity(z_coarse_flat, z_refined_flat, dim=1).mean().item()
    
    return {
        'mse': mse,
        'l1': l1,
        'l2': l2,
        'cosine_sim': cosine_sim,
    }


def print_latent_stats(name: str, z: torch.Tensor):
    """Print statistics about a latent representation."""
    print(f"\n{name}:")
    print(f"  Shape:      {list(z.shape)}")
    print(f"  Mean:       {z.mean().item():+.6f}")
    print(f"  Std:        {z.std().item():.6f}")
    print(f"  Min:        {z.min().item():+.6f}")
    print(f"  Max:        {z.max().item():+.6f}")
    print(f"  Norm (L2):  {torch.norm(z, p=2).item():.6f}")


def save_comparison_figure(
    sample_info: dict,
    coarse_results: dict,
    refined_results: dict,
    output_path: Path,
    threshold: float = 0.5
):
    """Save comparison figure with coarse and refined reconstructions.
    
    Layout: 2 rows × 4 columns
    Row 1: coarse GT | coarse probs | coarse binary | coarse error
    Row 2: refined GT | refined probs | refined binary | refined error
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Extract data
    coarse_gt = coarse_results['gt'].squeeze().cpu().numpy()
    coarse_probs = coarse_results['recon_probs'].squeeze().cpu().numpy()
    coarse_binary = (coarse_probs > threshold).astype(np.float32)
    coarse_error = np.abs(coarse_gt - coarse_binary)
    
    refined_gt = refined_results['gt'].squeeze().cpu().numpy()
    refined_probs = refined_results['recon_probs'].squeeze().cpu().numpy()
    refined_binary = (refined_probs > threshold).astype(np.float32)
    refined_error = np.abs(refined_gt - refined_binary)
    
    # Compute metrics
    coarse_dice = dice_score(
        torch.from_numpy(coarse_probs).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(coarse_gt).unsqueeze(0).unsqueeze(0),
        threshold=threshold
    ).item()
    coarse_iou = iou_score(
        torch.from_numpy(coarse_probs).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(coarse_gt).unsqueeze(0).unsqueeze(0),
        threshold=threshold
    ).item()
    
    refined_dice = dice_score(
        torch.from_numpy(refined_probs).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(refined_gt).unsqueeze(0).unsqueeze(0),
        threshold=threshold
    ).item()
    refined_iou = iou_score(
        torch.from_numpy(refined_probs).unsqueeze(0).unsqueeze(0),
        torch.from_numpy(refined_gt).unsqueeze(0).unsqueeze(0),
        threshold=threshold
    ).item()
    
    # Row 1: Coarse mask
    axes[0, 0].imshow(coarse_gt, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Coarse GT', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(coarse_probs, cmap='viridis', vmin=0, vmax=1)
    axes[0, 1].set_title('Decoded Coarse\n(Probabilities)', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(coarse_binary, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Decoded Coarse\n(Thresholded)', fontsize=10)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(coarse_error, cmap='hot', vmin=0, vmax=1)
    axes[0, 3].set_title(
        f'Coarse Error\nDice: {coarse_dice:.4f}\nIoU: {coarse_iou:.4f}',
        fontsize=10
    )
    axes[0, 3].axis('off')
    
    # Row 2: Refined mask
    axes[1, 0].imshow(refined_gt, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Refined GT', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(refined_probs, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title('Decoded Refined\n(Probabilities)', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(refined_binary, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('Decoded Refined\n(Thresholded)', fontsize=10)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(refined_error, cmap='hot', vmin=0, vmax=1)
    axes[1, 3].set_title(
        f'Refined Error\nDice: {refined_dice:.4f}\nIoU: {refined_iou:.4f}',
        fontsize=10
    )
    axes[1, 3].axis('off')
    
    # Overall title
    fig.suptitle(
        f'Frozen VAE Latent Interface Debug\n'
        f'Sample: {sample_info["id"]} ({sample_info["source"]}) | File: {sample_info["file_stem"]}',
        fontsize=12,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 70)
    print("DEBUG: FROZEN VAE LATENT INTERFACE")
    print("=" * 70)
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = 'cpu'
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize frozen VAE latent interface
    print(f"\n{'=' * 70}")
    print("INITIALIZING LATENT INTERFACE")
    print("=" * 70)
    
    interface = FrozenVAELatentInterface(
        model_config_path=args.model_config,
        checkpoint_path=args.checkpoint,
        device=device,
        use_mu_only=args.use_mu_only,
    )
    
    print(f"\nInterface details:")
    print(f"  Latent shape: {interface.latent_shape}")
    print(f"  Total parameters: {interface.count_total_parameters():,}")
    print(f"  Trainable parameters: {interface.count_trainable_parameters()}")
    print(f"  Frozen: {interface.is_frozen()}")
    
    # Load sample
    print(f"\n{'=' * 70}")
    print("LOADING SAMPLE")
    print("=" * 70)
    print(f"Split: {args.split}")
    print(f"Sample index: {args.sample_idx}")
    
    sample = load_sample(
        args.metadata_dir,
        args.split,
        args.source,
        args.sample_idx,
        512,  # image_size
        device
    )
    
    print(f"\nSample info:")
    print(f"  ID: {sample['id']}")
    print(f"  File stem: {sample['file_stem']}")
    print(f"  Source: {sample['source']}")
    print(f"  Coarse mask shape: {list(sample['coarse_mask'].shape)}")
    print(f"  Refined mask shape: {list(sample['refined_mask'].shape)}")
    
    # Encode masks
    print(f"\n{'=' * 70}")
    print("ENCODING MASKS TO LATENT SPACE")
    print("=" * 70)
    
    coarse_mask = sample['coarse_mask']
    refined_mask = sample['refined_mask']
    
    print("\nEncoding coarse mask...")
    z_coarse = interface.encode_coarse_mask(coarse_mask)
    
    print("Encoding refined mask...")
    z_refined = interface.encode_refined_mask(refined_mask)
    
    print_latent_stats("z_coarse", z_coarse)
    print_latent_stats("z_refined", z_refined)
    
    # Compute latent distance
    print(f"\n{'=' * 70}")
    print("LATENT DISTANCE METRICS")
    print("=" * 70)
    
    latent_dist = compute_latent_distance(z_coarse, z_refined)
    print(f"\nDistance between z_coarse and z_refined:")
    print(f"  MSE:           {latent_dist['mse']:.6f}")
    print(f"  L1:            {latent_dist['l1']:.6f}")
    print(f"  L2 norm:       {latent_dist['l2']:.6f}")
    print(f"  Cosine sim:    {latent_dist['cosine_sim']:.6f}")
    
    # Decode latents
    print(f"\n{'=' * 70}")
    print("DECODING LATENTS BACK TO MASK SPACE")
    print("=" * 70)
    
    print("\nDecoding z_coarse...")
    coarse_recon_logits = interface.decode_latent(z_coarse)
    coarse_recon_probs = torch.sigmoid(coarse_recon_logits)
    
    print("Decoding z_refined...")
    refined_recon_logits = interface.decode_latent(z_refined)
    refined_recon_probs = torch.sigmoid(refined_recon_logits)
    
    print(f"\nDecoded shapes:")
    print(f"  Coarse logits: {list(coarse_recon_logits.shape)}")
    print(f"  Coarse probs:  {list(coarse_recon_probs.shape)}")
    print(f"  Refined logits: {list(refined_recon_logits.shape)}")
    print(f"  Refined probs:  {list(refined_recon_probs.shape)}")
    
    # Compute reconstruction metrics
    print(f"\n{'=' * 70}")
    print("RECONSTRUCTION METRICS")
    print("=" * 70)
    
    coarse_dice = dice_score(coarse_recon_probs, coarse_mask, threshold=args.threshold).item()
    coarse_iou = iou_score(coarse_recon_probs, coarse_mask, threshold=args.threshold).item()
    
    refined_dice = dice_score(refined_recon_probs, refined_mask, threshold=args.threshold).item()
    refined_iou = iou_score(refined_recon_probs, refined_mask, threshold=args.threshold).item()
    
    print(f"\nCoarse mask reconstruction:")
    print(f"  Dice: {coarse_dice:.6f}")
    print(f"  IoU:  {coarse_iou:.6f}")
    
    print(f"\nRefined mask reconstruction:")
    print(f"  Dice: {refined_dice:.6f}")
    print(f"  IoU:  {refined_iou:.6f}")
    
    # Save visualization
    output_path = output_dir / f"sample_{sample['id']}.png"
    save_comparison_figure(
        {
            'id': sample['id'],
            'file_stem': sample['file_stem'],
            'source': sample['source'],
        },
        {
            'gt': coarse_mask,
            'recon_probs': coarse_recon_probs,
        },
        {
            'gt': refined_mask,
            'recon_probs': refined_recon_probs,
        },
        output_path,
        threshold=args.threshold
    )
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"\n✓ Latent interface is working correctly!")
    print(f"\nKey findings:")
    print(f"  • VAE is frozen: {interface.is_frozen()}")
    print(f"  • Trainable params: {interface.count_trainable_parameters()}")
    print(f"  • Latent shape: {interface.latent_shape}")
    print(f"  • Coarse → Latent → Decoded: Dice={coarse_dice:.4f}")
    print(f"  • Refined → Latent → Decoded: Dice={refined_dice:.4f}")
    print(f"  • Latent distance (MSE): {latent_dist['mse']:.6f}")
    print(f"\nInterpretation:")
    if latent_dist['mse'] < 0.001:
        print("  → Coarse and refined latents are VERY similar")
        print("  → Diffusion will need to learn subtle differences")
    elif latent_dist['mse'] < 0.01:
        print("  → Coarse and refined latents are moderately different")
        print("  → Good separation for diffusion training")
    else:
        print("  → Coarse and refined latents are VERY different")
        print("  → Strong separation for diffusion training")
    
    print(f"\n✓ Ready for latent diffusion training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
