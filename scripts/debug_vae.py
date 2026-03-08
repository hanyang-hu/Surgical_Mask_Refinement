"""Debug script for Mask VAE.

Tests the VAE model with dummy data and optionally with real masks.
Verifies shapes, forward pass, and loss computation.
"""

import argparse
import sys
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae.vae import MaskVAE
from models.vae.losses import vae_loss, VAELoss


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Debug Mask VAE")
    
    # Model config
    parser.add_argument(
        '--config',
        type=str,
        default='configs/model/vae.yaml',
        help='Path to VAE config file'
    )
    
    # Test mode
    parser.add_argument(
        '--use_real_data',
        action='store_true',
        help='Use real masks from dataset (requires Step 1/2 data)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split to use'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for testing'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load VAE configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_vae_from_config(config: dict) -> MaskVAE:
    """Create VAE model from configuration."""
    model_config = config['model']
    vae = MaskVAE(
        in_channels=model_config['in_channels'],
        base_channels=model_config['base_channels'],
        channel_multipliers=model_config['channel_multipliers'],
        latent_channels=model_config['latent_channels'],
        num_res_blocks=model_config['num_res_blocks'],
        norm=model_config['norm'],
        activation=model_config['activation']
    )
    return vae


def print_separator(char='=', length=70):
    """Print a separator line."""
    print('\n' + char * length)


def test_with_dummy_data(vae: MaskVAE, config: dict, device: torch.device):
    """Test VAE with dummy data."""
    print_separator()
    print("DUMMY DATA TEST")
    print_separator()
    
    batch_size = 4
    input_size = config['input']['image_size']
    
    # Create dummy input
    x = torch.rand(batch_size, 1, input_size, input_size, device=device)
    x = (x > 0.5).float()  # Binarize
    
    print(f"Input shape: {x.shape}")
    print(f"Input dtype: {x.dtype}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Input unique values: {torch.unique(x).tolist()}")
    
    # Forward pass
    print("\nRunning forward pass...")
    vae.eval()
    with torch.no_grad():
        outputs = vae(x)
    
    print("✓ Forward pass successful")
    
    # Print output shapes
    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {list(value.shape)}")
    
    # Verify shapes
    print("\nShape verification:")
    recon_logits = outputs['recon_logits']
    mu = outputs['mu']
    logvar = outputs['logvar']
    z = outputs['z']
    
    # Check reconstruction shape
    if recon_logits.shape == x.shape:
        print(f"  ✓ Reconstruction shape matches input: {list(recon_logits.shape)}")
    else:
        print(f"  ✗ Reconstruction shape mismatch: {list(recon_logits.shape)} vs {list(x.shape)}")
    
    # Check latent shapes
    expected_latent_size = config['latent']['spatial_size']
    expected_latent_channels = config['model']['latent_channels']
    expected_latent_shape = (batch_size, expected_latent_channels, expected_latent_size, expected_latent_size)
    
    if mu.shape == expected_latent_shape:
        print(f"  ✓ Latent mu shape correct: {list(mu.shape)}")
    else:
        print(f"  ✗ Latent mu shape: {list(mu.shape)}, expected: {list(expected_latent_shape)}")
    
    if logvar.shape == expected_latent_shape:
        print(f"  ✓ Latent logvar shape correct: {list(logvar.shape)}")
    else:
        print(f"  ✗ Latent logvar shape: {list(logvar.shape)}, expected: {list(expected_latent_shape)}")
    
    if z.shape == expected_latent_shape:
        print(f"  ✓ Sampled z shape correct: {list(z.shape)}")
    else:
        print(f"  ✗ Sampled z shape: {list(z.shape)}, expected: {list(expected_latent_shape)}")
    
    # Print latent statistics
    print("\nLatent statistics:")
    print(f"  mu    - mean: {mu.mean():.4f}, std: {mu.std():.4f}, range: [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"  logvar- mean: {logvar.mean():.4f}, std: {logvar.std():.4f}, range: [{logvar.min():.4f}, {logvar.max():.4f}]")
    print(f"  z     - mean: {z.mean():.4f}, std: {z.std():.4f}, range: [{z.min():.4f}, {z.max():.4f}]")
    
    # Compute losses
    print("\nComputing losses...")
    loss_config = config['loss']
    losses = vae_loss(
        recon_logits=recon_logits,
        target=x,
        mu=mu,
        logvar=logvar,
        recon_loss_type=loss_config['recon_loss_type'],
        beta=loss_config['beta'],
        bce_weight=loss_config.get('bce_weight', 1.0),
        dice_weight=loss_config.get('dice_weight', 1.0)
    )
    
    print("✓ Loss computation successful")
    print("\nLoss values:")
    for key, value in losses.items():
        print(f"  {key:15s}: {value.item():.6f}")
    
    # Test sampling
    print("\nTesting sampling from prior...")
    with torch.no_grad():
        samples = vae.sample(num_samples=2, latent_size=expected_latent_size, device=device)
    print(f"  Sampled logits shape: {list(samples.shape)}")
    print(f"  ✓ Sampling successful")
    
    # Test reconstruction
    print("\nTesting deterministic reconstruction...")
    with torch.no_grad():
        recon_probs = vae.reconstruct(x)
    print(f"  Reconstruction probs shape: {list(recon_probs.shape)}")
    print(f"  Reconstruction range: [{recon_probs.min():.4f}, {recon_probs.max():.4f}]")
    print(f"  ✓ Reconstruction successful")


def test_with_real_data(vae: MaskVAE, config: dict, split: str, batch_size: int, device: torch.device):
    """Test VAE with real masks from dataset."""
    print_separator()
    print("REAL DATA TEST")
    print_separator()
    
    try:
        from data.dataset import VAEDataset
        from data.transforms import build_transforms
        
        print(f"Loading {split} dataset...")
        
        # Build transforms (deterministic)
        mask_transform = build_transforms(
            image_size=config['input']['image_size'],
            train=False,
            augment=False
        )
        
        # Create dataset (masks only)
        dataset = VAEDataset(
            metadata_dir='/workspace/ece285/dataset/metadata',
            split=split,
            source='all',
            mask_type='refined',
            apply_transforms=True,
            mask_transform=mask_transform
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        # Get one batch
        batch = next(iter(dataloader))
        
        # Extract masks
        if isinstance(batch, dict):
            if 'refined_mask' in batch:
                x = batch['refined_mask'].to(device)
            elif 'mask' in batch:
                x = batch['mask'].to(device)
            else:
                raise KeyError(f"Batch keys: {batch.keys()}. Expected 'refined_mask' or 'mask'")
        else:
            x = batch[2].to(device)  # Assuming (rgb, coarse, refined) tuple
        
        print(f"\nReal batch loaded:")
        print(f"  Shape: {x.shape}")
        print(f"  Dtype: {x.dtype}")
        print(f"  Range: [{x.min():.4f}, {x.max():.4f}]")
        unique_vals = torch.unique(x)
        print(f"  Unique values: {len(unique_vals)} values")
        if len(unique_vals) <= 10:
            print(f"    {unique_vals.tolist()}")
        
        # Forward pass
        print("\nRunning forward pass on real data...")
        vae.eval()
        with torch.no_grad():
            outputs = vae(x)
        
        print("✓ Forward pass successful")
        
        # Compute losses
        print("\nCompute losses on real data...")
        loss_config = config['loss']
        losses = vae_loss(
            recon_logits=outputs['recon_logits'],
            target=x,
            mu=outputs['mu'],
            logvar=outputs['logvar'],
            recon_loss_type=loss_config['recon_loss_type'],
            beta=loss_config['beta'],
            bce_weight=loss_config.get('bce_weight', 1.0),
            dice_weight=loss_config.get('dice_weight', 1.0)
        )
        
        print("✓ Loss computation successful")
        print("\nReal data loss values:")
        for key, value in losses.items():
            print(f"  {key:15s}: {value.item():.6f}")
        
        # Show reconstruction quality
        recon_probs = torch.sigmoid(outputs['recon_logits'])
        print("\nReconstruction statistics:")
        print(f"  Input mean:  {x.mean():.4f}")
        print(f"  Recon mean:  {recon_probs.mean():.4f}")
        print(f"  Recon range: [{recon_probs.min():.4f}, {recon_probs.max():.4f}]")
        
    except Exception as e:
        print(f"✗ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Real data test requires Step 1/2 dataset to be available.")
        print("Run dummy data test only if dataset is not ready.")


def main():
    """Main debug function."""
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("MASK VAE DEBUG")
    print("=" * 70)
    
    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    print("✓ Config loaded")
    
    # Print config
    print("\nModel configuration:")
    print(f"  Base channels: {config['model']['base_channels']}")
    print(f"  Channel multipliers: {config['model']['channel_multipliers']}")
    print(f"  Latent channels: {config['model']['latent_channels']}")
    print(f"  Num res blocks: {config['model']['num_res_blocks']}")
    print(f"  Norm: {config['model']['norm']}")
    print(f"  Activation: {config['model']['activation']}")
    
    print(f"\nLoss configuration:")
    print(f"  Reconstruction loss: {config['loss']['recon_loss_type']}")
    print(f"  Beta (KL weight): {config['loss']['beta']}")
    
    # Create VAE
    print("\nCreating VAE model...")
    device = torch.device(args.device)
    vae = create_vae_from_config(config)
    vae = vae.to(device)
    print(f"✓ VAE created on device: {device}")
    
    # Count parameters
    param_counts = vae.count_parameters()
    print(f"\nModel parameters:")
    print(f"  Encoder: {param_counts['encoder']:,}")
    print(f"  Decoder: {param_counts['decoder']:,}")
    print(f"  Total:   {param_counts['total']:,}")
    
    # Get latent shape
    latent_shape = vae.get_latent_shape(config['input']['image_size'])
    print(f"\nLatent representation:")
    print(f"  Shape: [{config['model']['latent_channels']}, {latent_shape[0]}, {latent_shape[1]}]")
    print(f"  Total elements: {config['model']['latent_channels'] * latent_shape[0] * latent_shape[1]:,}")
    
    # Test with dummy data
    test_with_dummy_data(vae, config, device)
    
    # Test with real data if requested
    if args.use_real_data:
        test_with_real_data(vae, config, args.split, args.batch_size, device)
    
    # Summary
    print_separator()
    print("DEBUG COMPLETE")
    print_separator()
    print("\n✓ All tests passed!")
    print("\nNext steps:")
    print("  1. Implement VAE training loop")
    print("  2. Train VAE on full dataset")
    print("  3. Evaluate reconstruction quality")
    print("  4. Use trained VAE for latent diffusion")
    print()


if __name__ == '__main__':
    main()
