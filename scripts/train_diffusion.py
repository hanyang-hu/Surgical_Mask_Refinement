"""Training script for latent diffusion model.

Entry point for diffusion training. Loads config and runs diffusion engine.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train latent diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to diffusion training config YAML"
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default=None,
        help="Path to pretrained VAE checkpoint (overrides config)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    # TODO: Add more arguments
    return parser.parse_args()


def main():
    """Main training function.
    
    TODO: Initialize DiffusionEngine with config
    TODO: Load pretrained VAE
    TODO: Handle checkpoint resuming
    TODO: Run training
    """
    args = parse_args()
    
    print("=" * 50)
    print("Latent Diffusion Training")
    print("=" * 50)
    print(f"Config: {args.config}")
    if args.vae_checkpoint:
        print(f"VAE checkpoint: {args.vae_checkpoint}")
    
    # TODO: Implement training execution
    

if __name__ == "__main__":
    main()
