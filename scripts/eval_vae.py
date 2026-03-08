"""VAE Reconstruction Evaluation Script.

Evaluates trained VAE on validation/test splits:
- Loads trained checkpoint
- Runs inference on refined masks
- Computes reconstruction metrics (Dice, IoU, BCE)
- Saves visualizations and summary report
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import json
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae import MaskVAE
from data.dataset import VAEDataset
from utils.checkpoint import load_checkpoint
from utils.metrics import dice_score, iou_score, binary_cross_entropy
import torchvision.transforms as T


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained VAE on reconstruction task"
    )
    
    # Model and checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained VAE checkpoint (e.g., outputs/vae/checkpoints/best.pt)"
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
        choices=["val", "test"],
        help="Which split to evaluate"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "real_world", "synthetic"],
        help="Which source to evaluate"
    )
    
    # Processing
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for debugging)"
    )
    
    # Visualization
    parser.add_argument(
        "--num_visualizations",
        type=int,
        default=20,
        help="Number of random samples to visualize"
    )
    parser.add_argument(
        "--num_worst_cases",
        type=int,
        default=10,
        help="Number of worst reconstruction cases to save"
    )
    parser.add_argument(
        "--num_best_cases",
        type=int,
        default=5,
        help="Number of best reconstruction cases to save"
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Save individual visualization images instead of grids"
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for evaluation results"
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


def create_dataset(metadata_dir: str, split: str, source: str, image_size: int):
    """Create evaluation dataset."""
    # Simple mask-only transform
    mask_transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
        T.Lambda(lambda x: (x > 0.5).float())  # Binarize
    ])
    
    dataset = VAEDataset(
        metadata_dir=metadata_dir,
        split=split,
        source=source,
        mask_type='refined',
        apply_transforms=True,
        mask_transform=mask_transform
    )
    
    return dataset


@torch.no_grad()
def evaluate_model(
    model: MaskVAE,
    dataloader: DataLoader,
    device: str,
    max_samples: int = None
) -> dict:
    """Run inference and collect metrics.
    
    Returns:
        Dictionary containing:
        - per_sample_metrics: list of dicts with per-sample metrics
        - aggregate_metrics: dict with dataset-level statistics
    """
    model.eval()
    
    per_sample_metrics = []
    samples_processed = 0
    
    print("\nRunning inference...")
    pbar = tqdm(dataloader, desc="Evaluating")
    
    for batch_idx, batch in enumerate(pbar):
        # Extract data
        if isinstance(batch, dict):
            x = batch['mask'].to(device)
            sample_ids = batch.get('id', [f"sample_{i}" for i in range(len(x))])
            sources = batch.get('source', ['unknown'] * len(x))
        else:
            x = batch.to(device)
            sample_ids = [f"sample_{samples_processed + i}" for i in range(len(x))]
            sources = ['unknown'] * len(x)
        
        # Forward pass
        outputs = model(x)
        recon_logits = outputs['recon_logits']
        
        # Convert to probabilities
        recon_probs = torch.sigmoid(recon_logits)
        
        # Compute metrics per sample
        dice_scores = dice_score(recon_probs, x, threshold=0.5)
        iou_scores = iou_score(recon_probs, x, threshold=0.5)
        bce_losses = binary_cross_entropy(recon_probs, x, from_logits=False)
        
        # Store per-sample results
        for i in range(len(x)):
            per_sample_metrics.append({
                'sample_id': sample_ids[i] if isinstance(sample_ids[i], str) else str(sample_ids[i]),
                'source': sources[i] if isinstance(sources[i], str) else str(sources[i]),
                'dice': dice_scores[i].item(),
                'iou': iou_scores[i].item(),
                'bce': bce_losses[i].item(),
                'input': x[i].cpu(),
                'recon_probs': recon_probs[i].cpu(),
                'recon_logits': recon_logits[i].cpu(),
            })
            
            samples_processed += 1
            if max_samples and samples_processed >= max_samples:
                break
        
        # Update progress bar
        pbar.set_postfix({
            'dice': f"{dice_scores.mean().item():.4f}",
            'iou': f"{iou_scores.mean().item():.4f}",
        })
        
        if max_samples and samples_processed >= max_samples:
            break
    
    # Compute aggregate statistics
    aggregate_metrics = compute_aggregate_metrics(per_sample_metrics)
    
    return {
        'per_sample_metrics': per_sample_metrics,
        'aggregate_metrics': aggregate_metrics
    }


def compute_aggregate_metrics(per_sample_metrics: list) -> dict:
    """Compute aggregate statistics from per-sample metrics."""
    # Overall metrics
    all_dice = [m['dice'] for m in per_sample_metrics]
    all_iou = [m['iou'] for m in per_sample_metrics]
    all_bce = [m['bce'] for m in per_sample_metrics]
    
    aggregate = {
        'num_samples': len(per_sample_metrics),
        'mean_dice': float(np.mean(all_dice)),
        'std_dice': float(np.std(all_dice)),
        'mean_iou': float(np.mean(all_iou)),
        'std_iou': float(np.std(all_iou)),
        'mean_bce': float(np.mean(all_bce)),
        'std_bce': float(np.std(all_bce)),
    }
    
    # Per-source metrics
    sources = set(m['source'] for m in per_sample_metrics)
    by_source = {}
    
    for source in sources:
        source_metrics = [m for m in per_sample_metrics if m['source'] == source]
        if source_metrics:
            source_dice = [m['dice'] for m in source_metrics]
            source_iou = [m['iou'] for m in source_metrics]
            source_bce = [m['bce'] for m in source_metrics]
            
            by_source[source] = {
                'num_samples': len(source_metrics),
                'mean_dice': float(np.mean(source_dice)),
                'std_dice': float(np.std(source_dice)),
                'mean_iou': float(np.mean(source_iou)),
                'std_iou': float(np.std(source_iou)),
                'mean_bce': float(np.mean(source_bce)),
                'std_bce': float(np.std(source_bce)),
            }
    
    aggregate['by_source'] = by_source
    
    return aggregate


def save_visualization(
    input_mask: torch.Tensor,
    recon_probs: torch.Tensor,
    save_path: Path,
    sample_id: str = None
):
    """Save side-by-side visualization of reconstruction.
    
    Args:
        input_mask: Input mask [1, H, W]
        recon_probs: Reconstruction probabilities [1, H, W]
        save_path: Path to save image
        sample_id: Optional sample identifier for caption
    """
    # Threshold reconstruction
    recon_binary = (recon_probs > 0.5).float()
    
    # Compute error map (absolute difference)
    error_map = torch.abs(input_mask - recon_binary)
    
    # Stack: [input, recon_probs, recon_binary, error]
    comparison = torch.cat([
        input_mask,
        recon_probs,
        recon_binary,
        error_map
    ], dim=2)  # Concatenate along width: [1, H, W*4]
    
    # Save
    vutils.save_image(comparison, save_path, normalize=False, padding=2, pad_value=1.0)


def save_grid_visualization(
    samples: list,
    save_path: Path,
    nrow: int = 4
):
    """Save grid of multiple reconstructions.
    
    Args:
        samples: List of (input, recon_probs) tuples
        save_path: Path to save grid
        nrow: Number of samples per row
    """
    comparisons = []
    
    for input_mask, recon_probs in samples:
        recon_binary = (recon_probs > 0.5).float()
        error_map = torch.abs(input_mask - recon_binary)
        
        # Stack horizontally for this sample
        comparison = torch.cat([input_mask, recon_probs, recon_binary, error_map], dim=2)
        comparisons.append(comparison)
    
    # Make grid
    grid = vutils.make_grid(comparisons, nrow=nrow, padding=4, pad_value=1.0)
    vutils.save_image(grid, save_path, normalize=False)


def save_worst_best_cases(
    per_sample_metrics: list,
    output_dir: Path,
    num_worst: int = 10,
    num_best: int = 5
):
    """Save visualizations of worst and best reconstruction cases."""
    # Sort by Dice score
    sorted_metrics = sorted(per_sample_metrics, key=lambda x: x['dice'])
    
    # Worst cases
    worst_dir = output_dir / 'worst_cases'
    worst_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving {num_worst} worst cases...")
    for i, sample in enumerate(sorted_metrics[:num_worst]):
        save_path = worst_dir / f"worst_{i:04d}_dice{sample['dice']:.3f}.png"
        save_visualization(
            sample['input'],
            sample['recon_probs'],
            save_path,
            sample['sample_id']
        )
    
    # Best cases
    best_dir = output_dir / 'best_cases'
    best_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving {num_best} best cases...")
    for i, sample in enumerate(sorted_metrics[-num_best:]):
        save_path = best_dir / f"best_{i:04d}_dice{sample['dice']:.3f}.png"
        save_visualization(
            sample['input'],
            sample['recon_probs'],
            save_path,
            sample['sample_id']
        )


def save_random_visualizations(
    per_sample_metrics: list,
    output_dir: Path,
    num_vis: int = 20
):
    """Save random sample visualizations."""
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(per_sample_metrics), min(num_vis, len(per_sample_metrics)), replace=False)
    
    print(f"\nSaving {len(indices)} random visualizations...")
    for i, idx in enumerate(indices):
        sample = per_sample_metrics[idx]
        save_path = vis_dir / f"sample_{i:04d}_dice{sample['dice']:.3f}.png"
        save_visualization(
            sample['input'],
            sample['recon_probs'],
            save_path,
            sample['sample_id']
        )


def save_summary_report(
    aggregate_metrics: dict,
    output_dir: Path,
    checkpoint_path: str,
    split: str,
    source: str
):
    """Save evaluation summary as JSON."""
    summary = {
        'checkpoint': checkpoint_path,
        'split': split,
        'source': source,
        **aggregate_metrics
    }
    
    # Save JSON
    json_path = output_dir / 'summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {json_path}")
    
    # Also save human-readable text
    txt_path = output_dir / 'summary.txt'
    with open(txt_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("VAE RECONSTRUCTION EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Source: {source}\n\n")
        f.write("-" * 70 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Num samples: {aggregate_metrics['num_samples']}\n")
        f.write(f"Mean Dice:   {aggregate_metrics['mean_dice']:.4f} ± {aggregate_metrics['std_dice']:.4f}\n")
        f.write(f"Mean IoU:    {aggregate_metrics['mean_iou']:.4f} ± {aggregate_metrics['std_iou']:.4f}\n")
        f.write(f"Mean BCE:    {aggregate_metrics['mean_bce']:.4f} ± {aggregate_metrics['std_bce']:.4f}\n")
        
        if 'by_source' in aggregate_metrics and aggregate_metrics['by_source']:
            f.write("\n" + "-" * 70 + "\n")
            f.write("METRICS BY SOURCE\n")
            f.write("-" * 70 + "\n")
            for source_name, source_metrics in aggregate_metrics['by_source'].items():
                f.write(f"\n{source_name}:\n")
                f.write(f"  Num samples: {source_metrics['num_samples']}\n")
                f.write(f"  Mean Dice:   {source_metrics['mean_dice']:.4f} ± {source_metrics['std_dice']:.4f}\n")
                f.write(f"  Mean IoU:    {source_metrics['mean_iou']:.4f} ± {source_metrics['std_iou']:.4f}\n")
                f.write(f"  Mean BCE:    {source_metrics['mean_bce']:.4f} ± {source_metrics['std_bce']:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"Summary saved to: {txt_path}")


def print_summary(aggregate_metrics: dict):
    """Print evaluation summary to console."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\nOverall Performance:")
    print(f"  Samples:    {aggregate_metrics['num_samples']}")
    print(f"  Mean Dice:  {aggregate_metrics['mean_dice']:.4f} ± {aggregate_metrics['std_dice']:.4f}")
    print(f"  Mean IoU:   {aggregate_metrics['mean_iou']:.4f} ± {aggregate_metrics['std_iou']:.4f}")
    print(f"  Mean BCE:   {aggregate_metrics['mean_bce']:.4f} ± {aggregate_metrics['std_bce']:.4f}")
    
    if 'by_source' in aggregate_metrics and aggregate_metrics['by_source']:
        print(f"\nPer-Source Performance:")
        for source, metrics in aggregate_metrics['by_source'].items():
            print(f"  {source}:")
            print(f"    Samples:    {metrics['num_samples']}")
            print(f"    Mean Dice:  {metrics['mean_dice']:.4f} ± {metrics['std_dice']:.4f}")
            print(f"    Mean IoU:   {metrics['mean_iou']:.4f} ± {metrics['std_iou']:.4f}")
    
    print("\n" + "=" * 70 + "\n")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 70)
    print("VAE RECONSTRUCTION EVALUATION")
    print("=" * 70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Source: {args.source}")
    print(f"Device: {args.device}")
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = 'cpu'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model config
    print(f"\nLoading model config from: {args.model_config}")
    model_config = load_config(args.model_config)
    
    # Create model and load checkpoint
    model = create_model(model_config, args.checkpoint, device)
    
    # Create dataset
    print(f"\nCreating {args.split} dataset...")
    dataset = create_dataset(
        args.metadata_dir,
        args.split,
        args.source,
        model_config['input']['image_size']
    )
    print(f"Dataset size: {len(dataset)}")
    
    if args.max_samples:
        print(f"Limiting to {args.max_samples} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Run evaluation
    results = evaluate_model(model, dataloader, device, args.max_samples)
    per_sample_metrics = results['per_sample_metrics']
    aggregate_metrics = results['aggregate_metrics']
    
    # Print summary
    print_summary(aggregate_metrics)
    
    # Save summary report
    save_summary_report(
        aggregate_metrics,
        output_dir,
        args.checkpoint,
        args.split,
        args.source
    )
    
    # Save visualizations
    if args.num_visualizations > 0:
        save_random_visualizations(
            per_sample_metrics,
            output_dir,
            args.num_visualizations
        )
    
    # Save worst/best cases
    if args.num_worst_cases > 0 or args.num_best_cases > 0:
        save_worst_best_cases(
            per_sample_metrics,
            output_dir,
            args.num_worst_cases,
            args.num_best_cases
        )
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  Summary: {output_dir / 'summary.json'}")
    print(f"  Visualizations: {output_dir / 'visualizations'}")
    print(f"  Worst cases: {output_dir / 'worst_cases'}")
    print(f"  Best cases: {output_dir / 'best_cases'}")
    print()


if __name__ == "__main__":
    main()
