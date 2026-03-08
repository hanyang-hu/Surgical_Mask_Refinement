"""Quick diagnostic script for trained mask-only VAE.

Tests whether the trained VAE can reconstruct both coarse and refined masks.
This is a sanity check before building latent diffusion.

Usage:
    python3 scripts/quick_test_vae_coarse_vs_refined.py \\
        --checkpoint outputs/vae/checkpoints/best.pt \\
        --model_config configs/model/vae.yaml \\
        --split val \\
        --sample_idx 0
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae import MaskVAE
from data.dataset import SurgicalMaskRefinementDataset
from utils.checkpoint import load_checkpoint
from utils.metrics import dice_score, iou_score, binary_cross_entropy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Quick test for VAE reconstruction on coarse vs refined masks"
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
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing reconstructions"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/vae/quick_tests",
        help="Output directory for results"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_config: dict, checkpoint_path: str, device: str) -> MaskVAE:
    """Create VAE model and load checkpoint."""
    model = MaskVAE(
        in_channels=model_config['model']['in_channels'],
        base_channels=model_config['model']['base_channels'],
        channel_multipliers=model_config['model']['channel_multipliers'],
        latent_channels=model_config['model']['latent_channels'],
        num_res_blocks=model_config['model'].get('num_res_blocks', 1),
        norm=model_config['model'].get('norm', 'batch'),
        activation=model_config['model'].get('activation', 'silu'),
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        strict=True
    )
    
    epoch = checkpoint.get('epoch', 'unknown')
    best_loss = checkpoint.get('best_val_loss', 'unknown')
    print(f"  Checkpoint epoch: {epoch}")
    print(f"  Best val loss: {best_loss}")
    
    model = model.to(device)
    model.eval()
    
    return model


def create_simple_transform(image_size: int):
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
    """Load one sample with coarse and refined masks.
    
    Returns:
        Dictionary with:
        - id: Sample ID
        - file_stem: Filename stem
        - source: Source type
        - coarse_mask: Tensor [1, 1, H, W]
        - refined_mask: Tensor [1, 1, H, W]
    """
    # Create simple transform for masks
    mask_transform = create_simple_transform(image_size)
    
    # Custom transform function that applies same transform to both masks
    def paired_transform(rgb, coarse, refined):
        # We don't use RGB for VAE, but transform it anyway for consistency
        coarse_t = mask_transform(coarse)
        refined_t = mask_transform(refined)
        return None, coarse_t, refined_t  # Return None for RGB
    
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


@torch.no_grad()
def reconstruct_mask(model: MaskVAE, mask: torch.Tensor) -> dict:
    """Run VAE encoding and decoding on a mask.
    
    Args:
        model: Trained VAE model
        mask: Input mask [1, 1, H, W]
        
    Returns:
        Dictionary with:
        - recon_logits: Reconstruction logits [1, 1, H, W]
        - recon_probs: Reconstruction probabilities [1, 1, H, W]
        - mu: Latent mean [1, C, H, W]
        - logvar: Latent log variance [1, C, H, W]
        - z: Sampled latent [1, C, H, W]
    """
    outputs = model(mask)
    
    recon_logits = outputs['recon_logits']
    recon_probs = torch.sigmoid(recon_logits)
    
    return {
        'recon_logits': recon_logits,
        'recon_probs': recon_probs,
        'mu': outputs['mu'],
        'logvar': outputs['logvar'],
        'z': outputs['z'],
    }


def compute_metrics(pred_probs: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute reconstruction metrics.
    
    Args:
        pred_probs: Predicted probabilities [1, 1, H, W]
        target: Ground truth binary mask [1, 1, H, W]
        threshold: Threshold for binarization
        
    Returns:
        Dictionary with dice, iou, bce, fg_count_gt, fg_count_pred
    """
    # Compute metrics
    dice = dice_score(pred_probs, target, threshold=threshold).item()
    iou = iou_score(pred_probs, target, threshold=threshold).item()
    bce = binary_cross_entropy(pred_probs, target, from_logits=False).item()
    
    # Count foreground pixels
    fg_count_gt = target.sum().item()
    pred_binary = (pred_probs > threshold).float()
    fg_count_pred = pred_binary.sum().item()
    
    return {
        'dice': dice,
        'iou': iou,
        'bce': bce,
        'fg_count_gt': int(fg_count_gt),
        'fg_count_pred': int(fg_count_pred),
    }


def compute_latent_stats(mu: torch.Tensor, z: torch.Tensor) -> dict:
    """Compute statistics of latent representations.
    
    Args:
        mu: Latent mean [1, C, H, W]
        z: Sampled latent [1, C, H, W]
        
    Returns:
        Dictionary with mu_mean, mu_std, z_mean, z_std
    """
    return {
        'mu_mean': mu.mean().item(),
        'mu_std': mu.std().item(),
        'z_mean': z.mean().item(),
        'z_std': z.std().item(),
    }


def save_comparison_figure(
    sample_info: dict,
    coarse_results: dict,
    refined_results: dict,
    coarse_metrics: dict,
    refined_metrics: dict,
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
    
    # Row 1: Coarse mask
    axes[0, 0].imshow(coarse_gt, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Coarse GT\n({coarse_metrics["fg_count_gt"]} fg pixels)', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(coarse_probs, cmap='viridis', vmin=0, vmax=1)
    axes[0, 1].set_title('Coarse Reconstruction\n(Probabilities)', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(coarse_binary, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Coarse Binary\n({coarse_metrics["fg_count_pred"]} fg pixels)', fontsize=10)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(coarse_error, cmap='hot', vmin=0, vmax=1)
    axes[0, 3].set_title(
        f'Coarse Error\nDice: {coarse_metrics["dice"]:.4f}\nIoU: {coarse_metrics["iou"]:.4f}',
        fontsize=10
    )
    axes[0, 3].axis('off')
    
    # Row 2: Refined mask
    axes[1, 0].imshow(refined_gt, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Refined GT\n({refined_metrics["fg_count_gt"]} fg pixels)', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(refined_probs, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title('Refined Reconstruction\n(Probabilities)', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(refined_binary, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Refined Binary\n({refined_metrics["fg_count_pred"]} fg pixels)', fontsize=10)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(refined_error, cmap='hot', vmin=0, vmax=1)
    axes[1, 3].set_title(
        f'Refined Error\nDice: {refined_metrics["dice"]:.4f}\nIoU: {refined_metrics["iou"]:.4f}',
        fontsize=10
    )
    axes[1, 3].axis('off')
    
    # Overall title
    fig.suptitle(
        f'VAE Reconstruction Test: Coarse vs Refined\n'
        f'Sample: {sample_info["id"]} ({sample_info["source"]}) | File: {sample_info["file_stem"]}',
        fontsize=12,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")


def print_results(
    sample_info: dict,
    coarse_metrics: dict,
    refined_metrics: dict,
    coarse_latent_stats: dict,
    refined_latent_stats: dict
):
    """Print results to console."""
    print("\n" + "=" * 70)
    print("SAMPLE INFO")
    print("=" * 70)
    print(f"ID:         {sample_info['id']}")
    print(f"File stem:  {sample_info['file_stem']}")
    print(f"Source:     {sample_info['source']}")
    
    print("\n" + "=" * 70)
    print("COARSE MASK RECONSTRUCTION")
    print("=" * 70)
    print(f"Dice:                 {coarse_metrics['dice']:.6f}")
    print(f"IoU:                  {coarse_metrics['iou']:.6f}")
    print(f"BCE:                  {coarse_metrics['bce']:.6f}")
    print(f"FG pixels (GT):       {coarse_metrics['fg_count_gt']}")
    print(f"FG pixels (Pred):     {coarse_metrics['fg_count_pred']}")
    
    print("\n" + "=" * 70)
    print("REFINED MASK RECONSTRUCTION")
    print("=" * 70)
    print(f"Dice:                 {refined_metrics['dice']:.6f}")
    print(f"IoU:                  {refined_metrics['iou']:.6f}")
    print(f"BCE:                  {refined_metrics['bce']:.6f}")
    print(f"FG pixels (GT):       {refined_metrics['fg_count_gt']}")
    print(f"FG pixels (Pred):     {refined_metrics['fg_count_pred']}")
    
    print("\n" + "=" * 70)
    print("LATENT STATISTICS")
    print("=" * 70)
    print("\nCoarse latent:")
    print(f"  mu:  mean={coarse_latent_stats['mu_mean']:+.4f}, std={coarse_latent_stats['mu_std']:.4f}")
    print(f"  z:   mean={coarse_latent_stats['z_mean']:+.4f}, std={coarse_latent_stats['z_std']:.4f}")
    
    print("\nRefined latent:")
    print(f"  mu:  mean={refined_latent_stats['mu_mean']:+.4f}, std={refined_latent_stats['mu_std']:.4f}")
    print(f"  z:   mean={refined_latent_stats['z_mean']:+.4f}, std={refined_latent_stats['z_std']:.4f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    # Analyze results
    coarse_good = coarse_metrics['dice'] > 0.90
    refined_good = refined_metrics['dice'] > 0.95
    
    if refined_good and coarse_good:
        print("✓ EXCELLENT: Both coarse and refined masks reconstruct well.")
        print("  → VAE latent space can represent both mask types")
        print("  → Ready for latent diffusion training")
    elif refined_good and not coarse_good:
        print("⚠ WARNING: Refined reconstructs well but coarse is poor.")
        print("  → VAE was trained only on refined masks")
        print("  → May struggle with coarse masks in diffusion")
        print("  → Consider: ensure diffusion targets refined latents")
    elif not refined_good and coarse_good:
        print("⚠ WARNING: Coarse reconstructs well but refined is poor.")
        print("  → This is unusual - VAE should be best at refined masks")
        print("  → Check: was VAE trained on refined masks?")
    else:
        print("✗ POOR: Both reconstructions are suboptimal.")
        print("  → VAE quality may be insufficient")
        print("  → Consider: more training or architecture changes")
    
    print("\nNote: VAE was trained on refined masks, so perfect coarse")
    print("      reconstruction is not expected. The key question is:")
    print("      Can the decoder handle coarse-mask-like latents?")
    
    print("\n" + "=" * 70)


def main():
    """Main function."""
    args = parse_args()
    
    print("=" * 70)
    print("QUICK VAE TEST: COARSE VS REFINED RECONSTRUCTION")
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
    
    # Load model config
    print(f"\nLoading model config from: {args.model_config}")
    model_config = load_config(args.model_config)
    image_size = model_config['input']['image_size']
    
    # Create model and load checkpoint
    model = create_model(model_config, args.checkpoint, device)
    
    # Load sample
    print(f"\nLoading sample {args.sample_idx} from {args.split} split...")
    sample = load_sample(
        args.metadata_dir,
        args.split,
        args.source,
        args.sample_idx,
        image_size,
        device
    )
    
    print(f"  Sample ID: {sample['id']}")
    print(f"  File stem: {sample['file_stem']}")
    print(f"  Source: {sample['source']}")
    
    # Extract masks
    coarse_mask = sample['coarse_mask']
    refined_mask = sample['refined_mask']
    
    print(f"\nMask shapes:")
    print(f"  Coarse:  {coarse_mask.shape}")
    print(f"  Refined: {refined_mask.shape}")
    
    # Reconstruct coarse mask
    print("\nReconstructing coarse mask...")
    coarse_results = reconstruct_mask(model, coarse_mask)
    coarse_results['gt'] = coarse_mask
    
    # Reconstruct refined mask
    print("Reconstructing refined mask...")
    refined_results = reconstruct_mask(model, refined_mask)
    refined_results['gt'] = refined_mask
    
    # Compute metrics
    print("\nComputing metrics...")
    coarse_metrics = compute_metrics(
        coarse_results['recon_probs'],
        coarse_mask,
        threshold=args.threshold
    )
    refined_metrics = compute_metrics(
        refined_results['recon_probs'],
        refined_mask,
        threshold=args.threshold
    )
    
    # Compute latent statistics
    coarse_latent_stats = compute_latent_stats(
        coarse_results['mu'],
        coarse_results['z']
    )
    refined_latent_stats = compute_latent_stats(
        refined_results['mu'],
        refined_results['z']
    )
    
    # Save visualization
    output_path = output_dir / f"coarse_vs_refined_sample_{sample['id']}.png"
    save_comparison_figure(
        {
            'id': sample['id'],
            'file_stem': sample['file_stem'],
            'source': sample['source'],
        },
        coarse_results,
        refined_results,
        coarse_metrics,
        refined_metrics,
        output_path,
        threshold=args.threshold
    )
    
    # Print results
    print_results(
        {
            'id': sample['id'],
            'file_stem': sample['file_stem'],
            'source': sample['source'],
        },
        coarse_metrics,
        refined_metrics,
        coarse_latent_stats,
        refined_latent_stats
    )


if __name__ == "__main__":
    main()
