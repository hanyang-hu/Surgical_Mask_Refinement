"""Debug script for latent diffusion baseline.

Tests the full forward path of latent diffusion:
1. Load coarse and refined masks
2. Encode to latent space using frozen VAE
3. Sample timesteps and add noise
4. Run latent U-Net forward pass
5. Compute epsilon-prediction loss

Usage:
    python3 scripts/debug_latent_diffusion_baseline.py \\
        --checkpoint outputs/vae/checkpoints/best.pt \\
        --vae_config configs/model/vae.yaml \\
        --diffusion_config configs/model/diffusion.yaml \\
        --batch_size 2
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np
import torchvision.transforms as T

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion.latent_interface import FrozenVAELatentInterface
from models.diffusion.scheduler import LatentDiffusionScheduler
from models.diffusion.unet import LatentDiffusionUNet
from models.diffusion.losses import DiffusionLoss, diffusion_epsilon_loss
from data.dataset import SurgicalMaskRefinementDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Debug latent diffusion baseline"
    )
    
    # VAE checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/vae/checkpoints/best.pt",
        help="Path to trained VAE checkpoint"
    )
    parser.add_argument(
        "--vae_config",
        type=str,
        default="configs/model/vae.yaml",
        help="Path to VAE config"
    )
    
    # Diffusion config
    parser.add_argument(
        "--diffusion_config",
        type=str,
        default="configs/model/diffusion.yaml",
        help="Path to diffusion config"
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
        help="First sample index"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of samples to test"
    )
    
    # Processing
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_simple_transform(image_size: int = 512):
    """Create simple deterministic transform for masks."""
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
        T.Lambda(lambda x: (x > 0.5).float())  # Binarize
    ])


def load_batch_samples(
    metadata_dir: str,
    split: str,
    source: str,
    start_idx: int,
    batch_size: int,
    image_size: int,
    device: str
) -> dict:
    """Load a batch of samples with coarse and refined masks."""
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
    
    # Load batch
    coarse_masks = []
    refined_masks = []
    ids = []
    
    for i in range(start_idx, min(start_idx + batch_size, len(dataset))):
        sample = dataset[i]
        coarse_masks.append(sample['coarse_mask'])
        refined_masks.append(sample['refined_mask'])
        ids.append(sample['id'])
    
    # Stack into batches
    coarse_batch = torch.stack(coarse_masks).to(device)  # [B, 1, H, W]
    refined_batch = torch.stack(refined_masks).to(device)  # [B, 1, H, W]
    
    return {
        'coarse_mask': coarse_batch,
        'refined_mask': refined_batch,
        'ids': ids,
    }


def print_separator(title: str):
    """Print separator line with title."""
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def print_tensor_stats(name: str, tensor: torch.Tensor):
    """Print statistics about a tensor."""
    print(f"  {name}:")
    print(f"    Shape:  {list(tensor.shape)}")
    print(f"    Mean:   {tensor.mean().item():+.6f}")
    print(f"    Std:    {tensor.std().item():.6f}")
    print(f"    Min:    {tensor.min().item():+.6f}")
    print(f"    Max:    {tensor.max().item():+.6f}")


def main():
    """Main function."""
    args = parse_args()
    
    print_separator("DEBUG: LATENT DIFFUSION BASELINE")
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        device = 'cpu'
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    
    # Load configs
    print_separator("LOADING CONFIGURATIONS")
    vae_config = load_config(args.vae_config)
    diffusion_config = load_config(args.diffusion_config)
    
    print(f"VAE config: {args.vae_config}")
    print(f"Diffusion config: {args.diffusion_config}")
    
    # Initialize frozen VAE latent interface
    print_separator("INITIALIZING FROZEN VAE INTERFACE")
    
    vae_interface = FrozenVAELatentInterface(
        model_config_path=args.vae_config,
        checkpoint_path=args.checkpoint,
        device=device,
        use_mu_only=True,  # Deterministic encoding
    )
    
    print(f"\nVAE Interface:")
    print(f"  Latent shape: {vae_interface.latent_shape}")
    print(f"  Total parameters: {vae_interface.count_total_parameters():,}")
    print(f"  Trainable parameters: {vae_interface.count_trainable_parameters()}")
    print(f"  Frozen: {vae_interface.is_frozen()}")
    
    # Initialize diffusion scheduler
    print_separator("INITIALIZING DIFFUSION SCHEDULER")
    
    scheduler = LatentDiffusionScheduler(
        num_train_timesteps=diffusion_config['scheduler']['num_train_timesteps'],
        beta_schedule=diffusion_config['scheduler']['beta_schedule'],
        beta_start=diffusion_config['scheduler']['beta_start'],
        beta_end=diffusion_config['scheduler']['beta_end'],
        device=device,
    )
    
    print(f"\nScheduler: {scheduler}")
    
    # Initialize latent U-Net
    print_separator("INITIALIZING LATENT U-NET")
    
    model = LatentDiffusionUNet(
        in_channels=diffusion_config['model']['in_channels'],
        out_channels=diffusion_config['model']['out_channels'],
        base_channels=diffusion_config['model']['base_channels'],
        channel_multipliers=diffusion_config['model']['channel_multipliers'],
        num_res_blocks=diffusion_config['model']['num_res_blocks'],
        time_embed_dim=diffusion_config['model']['time_embed_dim'],
        norm=diffusion_config['model']['norm'],
        activation=diffusion_config['model']['activation'],
        dropout=diffusion_config['model']['dropout'],
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nLatent U-Net:")
    print(f"  Input channels: {diffusion_config['model']['in_channels']}")
    print(f"  Output channels: {diffusion_config['model']['out_channels']}")
    print(f"  Base channels: {diffusion_config['model']['base_channels']}")
    print(f"  Channel multipliers: {diffusion_config['model']['channel_multipliers']}")
    print(f"  Time embed dim: {diffusion_config['model']['time_embed_dim']}")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")
    
    # Load batch of samples
    print_separator("LOADING BATCH SAMPLES")
    print(f"Split: {args.split}")
    print(f"Sample indices: {args.sample_idx} to {args.sample_idx + args.batch_size - 1}")
    
    batch = load_batch_samples(
        args.metadata_dir,
        args.split,
        args.source,
        args.sample_idx,
        args.batch_size,
        512,  # image_size
        device
    )
    
    print(f"\nLoaded batch:")
    print(f"  Sample IDs: {batch['ids']}")
    print(f"  Coarse mask shape: {list(batch['coarse_mask'].shape)}")
    print(f"  Refined mask shape: {list(batch['refined_mask'].shape)}")
    
    # Encode masks to latent space
    print_separator("ENCODING MASKS TO LATENT SPACE")
    
    with torch.no_grad():
        z_coarse = vae_interface.encode_coarse_mask(batch['coarse_mask'])
        z_refined = vae_interface.encode_refined_mask(batch['refined_mask'])
    
    print(f"\nLatent representations:")
    print_tensor_stats("z_coarse", z_coarse)
    print_tensor_stats("z_refined", z_refined)
    
    # Sample timesteps
    print_separator("SAMPLING TIMESTEPS AND ADDING NOISE")
    
    t = scheduler.sample_timesteps(args.batch_size, device=device)
    print(f"\nSampled timesteps: {t.cpu().numpy()}")
    
    # Sample noise
    noise = torch.randn_like(z_refined)
    print_tensor_stats("noise", noise)
    
    # Add noise to z_refined (forward diffusion)
    z_t = scheduler.q_sample(z_refined, t, noise)
    print_tensor_stats("z_t (noisy latent)", z_t)
    
    # Forward pass through U-Net
    print_separator("FORWARD PASS THROUGH U-NET")
    
    model.eval()  # Set to eval for testing
    with torch.no_grad():
        eps_pred = model(z_t, t, z_coarse)
    
    print(f"\nU-Net forward pass:")
    print(f"  Input shape (z_t): {list(z_t.shape)}")
    print(f"  Input shape (z_coarse): {list(z_coarse.shape)}")
    print(f"  Timesteps shape: {list(t.shape)}")
    print(f"  Output shape (eps_pred): {list(eps_pred.shape)}")
    
    print_tensor_stats("eps_pred", eps_pred)
    
    # Compute loss
    print_separator("COMPUTING DIFFUSION LOSS")
    
    loss = diffusion_epsilon_loss(eps_pred, noise)
    
    print(f"\nLoss:")
    print(f"  Epsilon MSE loss: {loss.item():.6f}")
    
    # Additional metrics
    with torch.no_grad():
        # Cosine similarity between predicted and target noise
        eps_pred_flat = eps_pred.flatten(1)
        noise_flat = noise.flatten(1)
        cos_sim = F.cosine_similarity(eps_pred_flat, noise_flat, dim=1).mean()
        
        # L1 loss
        l1_loss = F.l1_loss(eps_pred, noise)
        
        # Per-sample MSE
        mse_per_sample = F.mse_loss(eps_pred, noise, reduction='none').mean(dim=[1, 2, 3])
    
    print(f"  Epsilon L1 loss: {l1_loss.item():.6f}")
    print(f"  Cosine similarity: {cos_sim.item():.6f}")
    print(f"  Per-sample MSE: {mse_per_sample.cpu().numpy()}")
    
    # Test x0 prediction
    print_separator("TESTING X0 PREDICTION FROM EPSILON")
    
    with torch.no_grad():
        x0_pred = scheduler.predict_x0_from_eps(z_t, t, eps_pred)
        x0_pred_from_true_eps = scheduler.predict_x0_from_eps(z_t, t, noise)
    
    print(f"\nPredicted x0 from model epsilon:")
    print_tensor_stats("x0_pred", x0_pred)
    
    print(f"\nPredicted x0 from true epsilon:")
    print_tensor_stats("x0_pred_from_true_eps", x0_pred_from_true_eps)
    
    print(f"\nOriginal z_refined:")
    print_tensor_stats("z_refined", z_refined)
    
    # Compare x0 predictions
    x0_mse_model = F.mse_loss(x0_pred, z_refined).item()
    x0_mse_true = F.mse_loss(x0_pred_from_true_eps, z_refined).item()
    
    print(f"\nX0 reconstruction error:")
    print(f"  MSE (using model epsilon): {x0_mse_model:.6f}")
    print(f"  MSE (using true epsilon): {x0_mse_true:.6f}")
    
    # Summary
    print_separator("SUMMARY")
    print(f"\n✓ Latent diffusion baseline is working!")
    print(f"\nKey findings:")
    print(f"  • VAE interface frozen: {vae_interface.is_frozen()}")
    print(f"  • Latent shape: {vae_interface.latent_shape}")
    print(f"  • U-Net parameters: {num_params:,} (all trainable)")
    print(f"  • Forward pass successful")
    print(f"  • Loss computed: {loss.item():.6f}")
    print(f"  • Epsilon prediction quality: {cos_sim.item():.4f} cosine similarity")
    
    print(f"\nInterpretation:")
    if cos_sim.item() > 0.5:
        print("  → Model (untrained) has some alignment with target noise")
        print("  → This is expected for random initialization")
    else:
        print("  → Model (untrained) has low alignment with target noise")
        print("  → This is normal - model needs training")
    
    print(f"\n✓ Ready for latent diffusion training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
