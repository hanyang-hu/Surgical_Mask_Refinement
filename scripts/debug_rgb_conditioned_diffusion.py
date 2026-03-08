"""Debug script for RGB-conditioned latent diffusion model.

Verifies that one forward-loss pass works correctly with:
- VAE latent encoding (frozen)
- Diffusion scheduler
- RGB-conditioned U-Net with cross-attention
- Epsilon-prediction loss

Usage:
    python3 scripts/debug_rgb_conditioned_diffusion.py \\
        --vae_checkpoint outputs/vae/checkpoints/best.pt \\
        --vae_config configs/model/vae.yaml \\
        --diffusion_config configs/model/diffusion_rgb.yaml \\
        --metadata_dir data/metadata \\
        --token_dir outputs/clip_tokens \\
        --split val \\
        --source all \\
        --batch_size 2 \\
        --device cuda
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae import MaskVAE
from models.diffusion.unet import RGBConditionedLatentDiffusionUNet
from models.diffusion.scheduler import LatentDiffusionScheduler
from models.diffusion.latent_interface import FrozenVAELatentInterface
from data.dataset import SurgicalMaskRefinementDataset
from utils.checkpoint import load_checkpoint
import torchvision.transforms as T


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Debug RGB-conditioned latent diffusion model"
    )
    
    # Model checkpoints and configs
    parser.add_argument(
        "--vae_checkpoint",
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
    parser.add_argument(
        "--diffusion_config",
        type=str,
        default="configs/model/diffusion_rgb.yaml",
        help="Path to RGB diffusion config"
    )
    
    # Dataset
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="data/metadata",
        help="Directory containing split JSON files"
    )
    parser.add_argument(
        "--token_dir",
        type=str,
        default="outputs/clip_tokens",
        help="Directory containing precomputed CLIP tokens"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to load"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "real_world", "synthetic"],
        help="Which source to load"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for testing"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    
    # Random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_frozen_vae_interface(
    vae_config_path: str,
    vae_checkpoint: str,
    device: str,
) -> FrozenVAELatentInterface:
    """Create frozen VAE interface."""
    print(f"Loading VAE checkpoint from: {vae_checkpoint}")
    
    # Create frozen interface
    vae_interface = FrozenVAELatentInterface(
        model_config_path=vae_config_path,
        checkpoint_path=vae_checkpoint,
        device=device,
        use_mu_only=True,  # Deterministic encoding
    )
    
    return vae_interface


def create_rgb_diffusion_model(
    diffusion_config: dict,
    device: str,
) -> RGBConditionedLatentDiffusionUNet:
    """Create RGB-conditioned diffusion model."""
    model_cfg = diffusion_config['model']
    rgb_cfg = diffusion_config['rgb_condition']
    
    model = RGBConditionedLatentDiffusionUNet(
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels'],
        base_channels=model_cfg['base_channels'],
        channel_multipliers=model_cfg['channel_multipliers'],
        num_res_blocks=model_cfg['num_res_blocks'],
        time_embed_dim=model_cfg['time_embed_dim'],
        norm=model_cfg['norm'],
        activation=model_cfg['activation'],
        dropout=model_cfg['dropout'],
        rgb_token_dim=rgb_cfg['token_dim'],
        rgb_projected_dim=rgb_cfg['projected_dim'],
        rgb_num_heads=rgb_cfg['num_heads'],
        rgb_dropout=rgb_cfg['dropout'],
    )
    
    model = model.to(device)
    
    return model


def create_scheduler(diffusion_config: dict, device: str) -> LatentDiffusionScheduler:
    """Create diffusion scheduler."""
    sched_cfg = diffusion_config['scheduler']
    
    scheduler = LatentDiffusionScheduler(
        num_train_timesteps=sched_cfg['num_train_timesteps'],
        beta_schedule=sched_cfg['beta_schedule'],
        beta_start=sched_cfg['beta_start'],
        beta_end=sched_cfg['beta_end'],
    )
    
    # Move scheduler buffers to device
    scheduler = scheduler.to(device)
    
    return scheduler


def create_transforms():
    """Create deterministic transforms for masks."""
    mask_transform = T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
        T.Lambda(lambda x: (x > 0.5).float())  # Binarize
    ])
    
    rgb_transform = T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])
    
    def paired_transform(rgb, coarse_mask, refined_mask):
        """Apply transforms to all three images."""
        rgb_t = rgb_transform(rgb)
        coarse_t = mask_transform(coarse_mask)
        refined_t = mask_transform(refined_mask)
        return rgb_t, coarse_t, refined_t
    
    return paired_transform


def create_dataset(
    metadata_dir: str,
    split: str,
    source: str,
    paired_transform,
):
    """Create dataset (without token loading - we'll load tokens separately)."""
    dataset = SurgicalMaskRefinementDataset(
        metadata_dir=metadata_dir,
        split=split,
        source=source,
        load_images=True,
        return_paths=False,
        apply_transforms=True,
        transform=paired_transform,
    )
    
    return dataset


def create_dataloader(dataset, batch_size: int, shuffle: bool = False):
    """Create dataloader."""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Keep simple for debugging
        pin_memory=False,
    )
    
    return loader


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("=" * 70)
    print("RGB-CONDITIONED LATENT DIFFUSION DEBUG")
    print("=" * 70)
    print()
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()
    
    # Load configs
    print("Loading configs...")
    vae_config = load_config(args.vae_config)
    diffusion_config = load_config(args.diffusion_config)
    print(f"  VAE config: {args.vae_config}")
    print(f"  Diffusion config: {args.diffusion_config}")
    print()
    
    # Create frozen VAE interface
    print("Creating frozen VAE interface...")
    vae_interface = create_frozen_vae_interface(
        vae_config_path=args.vae_config,
        vae_checkpoint=args.vae_checkpoint,
        device=device,
    )
    num_trainable = vae_interface.count_trainable_parameters()
    num_total = vae_interface.count_total_parameters()
    print(f"  VAE parameters: {num_trainable:,} trainable / {num_total:,} total")
    print()
    
    # Create diffusion scheduler
    print("Creating diffusion scheduler...")
    scheduler = create_scheduler(diffusion_config, device=device)
    print(f"  Timesteps: {scheduler.num_train_timesteps}")
    print(f"  Beta schedule: {diffusion_config['scheduler']['beta_schedule']}")
    print()
    
    # Create RGB-conditioned diffusion model
    print("Creating RGB-conditioned diffusion model...")
    model = create_rgb_diffusion_model(
        diffusion_config=diffusion_config,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Diffusion model parameters: {num_trainable_params:,} trainable / {num_params:,} total")
    print()
    
    # Create dataset and dataloader
    print("Creating dataset...")
    paired_transform = create_transforms()
    dataset = create_dataset(
        metadata_dir=args.metadata_dir,
        split=args.split,
        source=args.source,
        paired_transform=paired_transform,
    )
    print(f"  Loaded {len(dataset)} samples from {args.split} split (source: {args.source})")
    
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"  Batch size: {args.batch_size}")
    print()
    
    # Load one batch
    print("Loading one batch...")
    batch = next(iter(dataloader))
    coarse_mask = batch['coarse_mask'].to(device)
    refined_mask = batch['refined_mask'].to(device)
    
    # Manually load RGB tokens for this batch
    print("Loading RGB tokens...")
    token_dir = Path(args.token_dir)
    rgb_tokens_list = []
    for i in range(len(batch['id'])):
        sample_id = batch['id'][i]
        source = batch['source'][i]
        file_stem = batch['file_stem'][i]
        
        # Token path: token_dir / split / source / file_stem.pt
        token_path = token_dir / args.split / source / f"{file_stem}.pt"
        if not token_path.exists():
            raise FileNotFoundError(f"Token file not found: {token_path}")
        token_data = torch.load(token_path, map_location=device)
        tokens = token_data['tokens']  # Extract tokens from dictionary
        rgb_tokens_list.append(tokens)
    rgb_tokens = torch.stack(rgb_tokens_list, dim=0)  # [B, N, D]
    
    print(f"  coarse_mask shape:  {coarse_mask.shape}")
    print(f"  refined_mask shape: {refined_mask.shape}")
    print(f"  rgb_tokens shape:   {rgb_tokens.shape}")
    print()
    
    # Encode masks to latents
    print("Encoding masks to latents...")
    with torch.no_grad():
        z_coarse = vae_interface.encode_coarse_mask(coarse_mask)
        z_refined = vae_interface.encode_refined_mask(refined_mask)
    
    print(f"  z_coarse shape:  {z_coarse.shape}")
    print(f"  z_refined shape: {z_refined.shape}")
    print(f"  z_coarse mean:   {z_coarse.mean().item():.6f}, std: {z_coarse.std().item():.6f}")
    print(f"  z_refined mean:  {z_refined.mean().item():.6f}, std: {z_refined.std().item():.6f}")
    print()
    
    # Sample timesteps
    print("Sampling timesteps...")
    batch_size = z_refined.shape[0]
    timesteps = scheduler.sample_timesteps(batch_size, device=device)
    print(f"  timesteps: {timesteps.tolist()}")
    print()
    
    # Sample noise
    print("Sampling noise...")
    noise = torch.randn_like(z_refined)
    print(f"  noise shape: {noise.shape}")
    print(f"  noise mean:  {noise.mean().item():.6f}, std: {noise.std().item():.6f}")
    print()
    
    # Add noise to refined latent
    print("Adding noise to refined latent...")
    z_t = scheduler.q_sample(z_refined, timesteps, noise)
    print(f"  z_t shape: {z_t.shape}")
    print(f"  z_t mean:  {z_t.mean().item():.6f}, std: {z_t.std().item():.6f}")
    print()
    
    # Forward pass through RGB-conditioned model
    print("Forward pass through RGB-conditioned model...")
    model.eval()  # Set to eval mode for debugging
    with torch.no_grad():
        eps_pred = model(z_t, timesteps, z_coarse, rgb_tokens)
    
    print(f"  eps_pred shape: {eps_pred.shape}")
    print(f"  eps_pred mean:  {eps_pred.mean().item():.6f}, std: {eps_pred.std().item():.6f}")
    print()
    
    # Compute loss
    print("Computing epsilon-prediction loss...")
    loss = F.mse_loss(eps_pred, noise)
    print(f"  MSE loss: {loss.item():.6f}")
    print()
    
    # Verify shapes
    print("Verifying shapes...")
    assert eps_pred.shape == noise.shape, "eps_pred and noise shapes must match"
    assert eps_pred.shape == z_refined.shape, "eps_pred and z_refined shapes must match"
    print("  ✓ All shapes match correctly")
    print()
    
    # Additional diagnostics
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print()
    
    print("Input shapes:")
    print(f"  coarse_mask:  {coarse_mask.shape}")
    print(f"  refined_mask: {refined_mask.shape}")
    print(f"  rgb_tokens:   {rgb_tokens.shape}")
    print()
    
    print("Latent shapes:")
    print(f"  z_coarse:  {z_coarse.shape}")
    print(f"  z_refined: {z_refined.shape}")
    print(f"  z_t:       {z_t.shape}")
    print()
    
    print("Model output:")
    print(f"  eps_pred: {eps_pred.shape}")
    print(f"  loss:     {loss.item():.6f}")
    print()
    
    print("Model configuration:")
    print(f"  RGB token dim:      {diffusion_config['rgb_condition']['token_dim']}")
    print(f"  RGB projected dim:  {diffusion_config['rgb_condition']['projected_dim']}")
    print(f"  RGB num heads:      {diffusion_config['rgb_condition']['num_heads']}")
    print(f"  Cross-attn at mid:  {diffusion_config['rgb_condition']['cross_attention_at_mid']}")
    print()
    
    print("=" * 70)
    print("DEBUG SUCCESSFUL!")
    print("=" * 70)
    print()
    print("RGB-conditioned diffusion model is working correctly.")
    print("Ready for training implementation (Step 12).")
    print()


if __name__ == "__main__":
    main()
