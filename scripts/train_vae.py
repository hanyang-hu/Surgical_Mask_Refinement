"""Training script for mask VAE.

Entry point for VAE training. Loads config and runs VAE training.
"""

import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import sys
import logging
import random
import numpy as np
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vae import MaskVAE, VAELoss
from data.dataset import VAEDataset
from data.transforms import build_transforms
from trainers.vae_trainer import VAETrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train mask VAE")
    
    # Config files
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train/vae_train.yaml",
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model/vae.yaml",
        help="Path to model config YAML"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on"
    )
    
    # Overfit mode for debugging
    parser.add_argument(
        "--overfit_small",
        action="store_true",
        help="Enable overfit mode on small subset for debugging"
    )
    parser.add_argument(
        "--overfit_num_samples",
        type=int,
        default=16,
        help="Number of samples for overfit mode"
    )
    
    # Training overrides
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate"
    )
    
    # Checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    
    # WandB
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="WandB run name"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable WandB logging"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(model_config: dict, device: str) -> MaskVAE:
    """Create VAE model from config."""
    model = MaskVAE(
        in_channels=model_config['model']['in_channels'],
        base_channels=model_config['model']['base_channels'],
        channel_multipliers=model_config['model']['channel_multipliers'],
        latent_channels=model_config['model']['latent_channels'],
        num_res_blocks=model_config['model'].get('num_res_blocks', 1),
        norm=model_config['model'].get('norm', 'batch'),
        activation=model_config['model'].get('activation', 'silu'),
    )
    
    # Count parameters
    param_counts = model.count_parameters()
    logger.info(f"Model created:")
    logger.info(f"  Encoder parameters: {param_counts['encoder']:,}")
    logger.info(f"  Decoder parameters: {param_counts['decoder']:,}")
    logger.info(f"  Total parameters: {param_counts['total']:,}")
    
    return model.to(device)


def create_datasets(train_config: dict, overfit_mode: bool = False, overfit_num: int = 16):
    """Create train and validation datasets."""
    data_config = train_config['data']
    
    # Build simple mask-only transforms
    # For VAE training, we only need to resize and convert to tensor
    import torchvision.transforms as T
    
    mask_transform = T.Compose([
        T.Resize((data_config['image_size'], data_config['image_size']), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
        T.Lambda(lambda x: (x > 0.5).float())  # Binarize
    ])
    
    # Create datasets
    train_dataset = VAEDataset(
        metadata_dir=data_config['metadata_dir'],
        split=data_config['split_train'],
        source=data_config['source'],
        mask_type=data_config.get('mask_type', 'refined'),
        apply_transforms=True,
        mask_transform=mask_transform
    )
    
    val_dataset = VAEDataset(
        metadata_dir=data_config['metadata_dir'],
        split=data_config['split_val'],
        source=data_config['source'],
        mask_type=data_config.get('mask_type', 'refined'),
        apply_transforms=True,
        mask_transform=mask_transform  # No augmentation for val
    )
    
    # Overfit mode: use small subset
    if overfit_mode:
        logger.info(f"OVERFIT MODE: Using only {overfit_num} samples")
        train_indices = list(range(min(overfit_num, len(train_dataset))))
        val_indices = list(range(min(overfit_num // 2, len(val_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    logger.info(f"Datasets created:")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int
) -> tuple:
    """Create dataloaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_optimizer(model: torch.nn.Module, train_config: dict) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    opt_config = train_config.get('optimizer', {})
    opt_type = opt_config.get('type', 'adam').lower()
    
    lr = train_config['train']['learning_rate']
    weight_decay = train_config['train'].get('weight_decay', 0.0)
    
    if opt_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=opt_config.get('betas', (0.9, 0.999)),
            eps=opt_config.get('eps', 1e-8)
        )
    elif opt_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=opt_config.get('betas', (0.9, 0.999)),
            eps=opt_config.get('eps', 1e-8)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    logger.info(f"Optimizer: {opt_type}, LR: {lr}, Weight decay: {weight_decay}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, train_config: dict):
    """Create learning rate scheduler if configured."""
    scheduler_config = train_config.get('scheduler', {})
    
    if not scheduler_config.get('use_scheduler', False):
        return None
    
    scheduler_type = scheduler_config.get('type', 'cosine').lower()
    epochs = train_config['train']['epochs']
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=scheduler_config.get('eta_min', 0)
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, not using scheduler")
        return None
    
    logger.info(f"Scheduler: {scheduler_type}")
    return scheduler


def init_wandb(train_config: dict, model_config: dict, run_name: str = None):
    """Initialize Weights & Biases."""
    wandb_config = train_config.get('wandb', {})
    
    if not wandb_config.get('use_wandb', True):
        logger.info("WandB disabled by config")
        return False
    
    # Check if wandb is available
    try:
        import wandb
    except ImportError:
        logger.error("WandB not installed. Install with: pip install wandb")
        raise
    
    # Initialize wandb
    project = wandb_config.get('project', 'surgical-mask-vae')
    name = run_name or wandb_config.get('run_name', None)
    mode = wandb_config.get('mode', 'online')
    
    wandb.init(
        project=project,
        name=name,
        config={
            'train': train_config['train'],
            'data': train_config['data'],
            'model': model_config['model'],
            'loss': model_config['loss'],
        },
        mode=mode
    )
    
    logger.info(f"WandB initialized: project={project}, name={name}, mode={mode}")
    return True


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 70)
    print("MASK VAE TRAINING")
    print("=" * 70)
    
    # Load configs
    logger.info(f"Loading training config from {args.train_config}")
    train_config = load_config(args.train_config)
    
    logger.info(f"Loading model config from {args.model_config}")
    model_config = load_config(args.model_config)
    
    # Apply CLI overrides
    if args.epochs is not None:
        train_config['train']['epochs'] = args.epochs
        logger.info(f"Override epochs: {args.epochs}")
    
    if args.batch_size is not None:
        train_config['train']['batch_size'] = args.batch_size
        logger.info(f"Override batch_size: {args.batch_size}")
    
    if args.learning_rate is not None:
        train_config['train']['learning_rate'] = args.learning_rate
        logger.info(f"Override learning_rate: {args.learning_rate}")
    
    if args.output_dir is not None:
        train_config['checkpoint']['output_dir'] = args.output_dir
        logger.info(f"Override output_dir: {args.output_dir}")
    
    if args.no_wandb:
        train_config['wandb']['use_wandb'] = False
        logger.info("WandB disabled by CLI")
    
    # Set seed
    seed = train_config['train'].get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    logger.info(f"Device: {device}")
    
    # Initialize WandB
    use_wandb = init_wandb(train_config, model_config, args.run_name) if not args.no_wandb else False
    
    # Create model
    logger.info("Creating model...")
    model = create_model(model_config, device)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(
        train_config,
        overfit_mode=args.overfit_small,
        overfit_num=args.overfit_num_samples
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=train_config['train']['batch_size'],
        num_workers=train_config['train']['num_workers']
    )
    
    # Create criterion
    logger.info("Creating loss function...")
    criterion = VAELoss(
        recon_loss_type=model_config['loss']['recon_loss_type'],
        beta=model_config['loss']['beta'],
        bce_weight=model_config['loss'].get('bce_weight', 1.0),
        dice_weight=model_config['loss'].get('dice_weight', 1.0)
    )
    logger.info(f"  Recon loss: {model_config['loss']['recon_loss_type']}")
    logger.info(f"  Beta (KL weight): {model_config['loss']['beta']}")
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = create_optimizer(model, train_config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, train_config)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=train_config,
        device=device,
        scheduler=scheduler,
        use_wandb=use_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint_wrapper(args.resume)
    
    # Start training
    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    
    try:
        trainer.fit()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        trainer.save_checkpoint_wrapper('interrupted.pt')
        logger.info("Saved interrupted checkpoint")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        if use_wandb:
            wandb.finish()
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    logger.info(f"Reconstructions saved to: {trainer.vis_dir}")


if __name__ == "__main__":
    main()
