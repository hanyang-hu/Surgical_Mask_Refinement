"""Precompute and save frozen CLIP RGB tokens for the entire dataset.

This script:
- Loads the dataset with deterministic preprocessing (no augmentation)
- Loads the frozen CLIP tokenizer from Step 3
- Extracts CLIP tokens in batches
- Saves one .pt file per sample with metadata
- Supports skip/overwrite control and verification
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
import yaml
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SurgicalMaskRefinementDataset
from data.transforms import build_transforms
from models.rgb import FrozenCLIPVisionTokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute CLIP tokens for dataset"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="data/metadata",
        help="Directory containing split JSON files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Which split to process (default: train)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "real_world", "synthetic"],
        help="Which source to process (default: all)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/clip_tokens",
        help="Output directory for saved tokens"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing token files"
    )
    parser.add_argument(
        "--save_spatial_map",
        action="store_true",
        help="Save spatial feature maps in addition to tokens"
    )
    parser.add_argument(
        "--save_preprocessed",
        action="store_true",
        help="Save preprocessed images (usually not needed)"
    )
    
    # Model arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/rgb_tokenizer.yaml",
        help="Path to tokenizer config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for token extraction"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for debugging)"
    )
    
    # Verification arguments
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing token files, don't process"
    )
    parser.add_argument(
        "--verify_samples",
        type=int,
        default=3,
        help="Number of samples to verify (default: 3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_output_path(
    output_dir: Path,
    split: str,
    source: str,
    file_stem: str
) -> Path:
    """Construct output path for a sample.
    
    Args:
        output_dir: Base output directory
        split: Split name (train/val/test)
        source: Source name (real_world/synthetic)
        file_stem: File stem (e.g., '1672')
        
    Returns:
        Path to save token file
    """
    # Structure: output_dir/split/source/file_stem.pt
    output_path = output_dir / split / source / f"{file_stem}.pt"
    return output_path


def save_token_file(
    output_path: Path,
    sample_id: str,
    file_stem: str,
    source: str,
    split: str,
    model_name: str,
    tokens: torch.Tensor,
    spatial_map: Optional[torch.Tensor] = None,
    preprocessed: Optional[torch.Tensor] = None,
):
    """Save token data to file.
    
    Args:
        output_path: Path to save file
        sample_id: Sample identifier
        file_stem: Original filename stem
        source: Source type
        split: Split name
        model_name: CLIP model name
        tokens: Token tensor [N, C]
        spatial_map: Optional spatial map [C, H, W]
        preprocessed: Optional preprocessed image [3, H, W]
    """
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build data dictionary
    data = {
        'id': sample_id,
        'file_stem': file_stem,
        'source': source,
        'split': split,
        'model_name': model_name,
        'tokens': tokens.cpu(),  # Move to CPU for saving
    }
    
    # Add optional data
    if spatial_map is not None:
        data['spatial_map'] = spatial_map.cpu()
    
    if preprocessed is not None:
        data['preprocessed'] = preprocessed.cpu()
    
    # Save to disk
    torch.save(data, output_path)


def process_split(
    split: str,
    source: str,
    metadata_dir: Path,
    output_dir: Path,
    tokenizer: FrozenCLIPVisionTokenizer,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
    save_spatial_map: bool,
    save_preprocessed: bool,
    max_samples: Optional[int],
    verbose: bool,
) -> Dict[str, int]:
    """Process one split of the dataset.
    
    Args:
        split: Split name (train/val/test)
        source: Source filter (all/real_world/synthetic)
        metadata_dir: Metadata directory path
        output_dir: Output directory path
        tokenizer: CLIP tokenizer instance
        batch_size: Batch size for processing
        num_workers: Number of DataLoader workers
        overwrite: Whether to overwrite existing files
        save_spatial_map: Whether to save spatial maps
        save_preprocessed: Whether to save preprocessed images
        max_samples: Maximum samples to process
        verbose: Verbose output
        
    Returns:
        Dictionary with processing statistics
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING SPLIT: {split.upper()}")
    print(f"{'='*70}")
    
    # Build deterministic eval transform (no augmentation)
    transform = build_transforms(
        train=False,
        augment=False,
        image_size=512
    )
    
    # Load dataset
    dataset = SurgicalMaskRefinementDataset(
        metadata_dir=metadata_dir,
        split=split,
        source=source,
        load_images=True,
        return_paths=False,
        apply_transforms=True,
        transform=transform
    )
    
    # Limit samples if requested
    if max_samples is not None and max_samples < len(dataset):
        print(f"Limiting to {max_samples} samples for debugging")
        # Use subset
        indices = list(range(max_samples))
        dataset.samples = [dataset.samples[i] for i in indices]
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Source filter: {source}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic order
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False
    )
    
    # Statistics
    stats = {
        'processed': 0,
        'skipped': 0,
        'failed': 0,
    }
    
    # Process batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Processing {split}"):
            # Get batch data
            rgb = batch['rgb'].to(tokenizer.device)
            ids = batch['id']
            file_stems = batch['file_stem']
            sources = batch['source']
            
            # Check which samples need processing
            batch_indices = []
            output_paths = []
            
            for i, (sample_id, file_stem, source_name) in enumerate(zip(ids, file_stems, sources)):
                output_path = get_output_path(
                    output_dir,
                    split,
                    source_name,
                    file_stem
                )
                output_paths.append(output_path)
                
                # Check if file exists
                if output_path.exists() and not overwrite:
                    stats['skipped'] += 1
                    if verbose:
                        print(f"  Skipping existing: {output_path}")
                    continue
                
                batch_indices.append(i)
            
            # Skip batch if all files exist
            if len(batch_indices) == 0:
                continue
            
            try:
                # Extract tokens for entire batch (more efficient)
                output = tokenizer(rgb)
                
                tokens_batch = output['tokens']  # [B, 196, 768]
                spatial_map_batch = output.get('spatial_map', None)  # [B, 768, 14, 14]
                preprocessed_batch = output.get('preprocessed', None)  # [B, 3, 224, 224]
                
                # Save each sample in batch
                for i in range(len(ids)):
                    # Skip if not in processing list
                    if i not in batch_indices:
                        continue
                    
                    try:
                        # Extract single sample tensors
                        tokens = tokens_batch[i]  # [196, 768]
                        spatial_map = spatial_map_batch[i] if save_spatial_map and spatial_map_batch is not None else None
                        preprocessed = preprocessed_batch[i] if save_preprocessed and preprocessed_batch is not None else None
                        
                        # Save to file
                        save_token_file(
                            output_path=output_paths[i],
                            sample_id=ids[i],
                            file_stem=file_stems[i],
                            source=sources[i],
                            split=split,
                            model_name=tokenizer.model_name,
                            tokens=tokens,
                            spatial_map=spatial_map,
                            preprocessed=preprocessed,
                        )
                        
                        stats['processed'] += 1
                        
                        if verbose:
                            print(f"  Saved: {output_paths[i]}")
                            
                    except Exception as e:
                        print(f"  ✗ Failed to save {ids[i]}: {e}")
                        stats['failed'] += 1
                        
            except Exception as e:
                print(f"  ✗ Batch processing failed: {e}")
                stats['failed'] += len(batch_indices)
    
    # Print statistics
    print()
    print(f"Split '{split}' completed:")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total: {stats['processed'] + stats['skipped'] + stats['failed']}")
    
    return stats


def verify_saved_tokens(
    output_dir: Path,
    num_samples: int = 3
):
    """Verify a few saved token files.
    
    Args:
        output_dir: Output directory containing saved tokens
        num_samples: Number of samples to verify
    """
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")
    
    # Find all .pt files
    token_files = list(output_dir.rglob("*.pt"))
    
    if len(token_files) == 0:
        print("No token files found!")
        return
    
    print(f"Found {len(token_files)} token files")
    print(f"Verifying {min(num_samples, len(token_files))} random samples...")
    print()
    
    # Sample random files
    sample_files = random.sample(token_files, min(num_samples, len(token_files)))
    
    for i, file_path in enumerate(sample_files, 1):
        print(f"Sample {i}: {file_path.relative_to(output_dir)}")
        
        try:
            # Load token file
            data = torch.load(file_path)
            
            # Print metadata
            print(f"  ID: {data.get('id', 'N/A')}")
            print(f"  Source: {data.get('source', 'N/A')}")
            print(f"  Split: {data.get('split', 'N/A')}")
            print(f"  Model: {data.get('model_name', 'N/A')}")
            
            # Print tensor info
            if 'tokens' in data:
                tokens = data['tokens']
                print(f"  Tokens shape: {tokens.shape}")
                print(f"  Tokens dtype: {tokens.dtype}")
                print(f"  Tokens range: [{tokens.min():.3f}, {tokens.max():.3f}]")
            
            if 'spatial_map' in data:
                spatial_map = data['spatial_map']
                print(f"  Spatial map shape: {spatial_map.shape}")
            
            if 'preprocessed' in data:
                preprocessed = data['preprocessed']
                print(f"  Preprocessed shape: {preprocessed.shape}")
            
            # Verify expected shape
            expected_shape = (196, 768)
            if 'tokens' in data:
                actual_shape = tuple(data['tokens'].shape)
                if actual_shape == expected_shape:
                    print(f"  ✓ Shape matches expected {expected_shape}")
                else:
                    print(f"  ⚠ Shape {actual_shape} doesn't match expected {expected_shape}")
            
            print()
            
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            print()


def main():
    """Main precomputation function."""
    args = parse_args()
    
    print("="*70)
    print("CLIP TOKEN PRECOMPUTATION")
    print("="*70)
    print()
    
    # Setup paths
    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)
    
    print("Configuration:")
    print(f"  Metadata dir: {metadata_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Split: {args.split}")
    print(f"  Source: {args.source}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Overwrite: {args.overwrite}")
    print(f"  Save spatial map: {args.save_spatial_map}")
    print(f"  Save preprocessed: {args.save_preprocessed}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")
    print()
    
    # Verify-only mode
    if args.verify_only:
        verify_saved_tokens(output_dir, args.verify_samples)
        return
    
    # Load tokenizer config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config['device'] = args.device
    
    # Convert dtype string to torch dtype if needed
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    if 'dtype' in config and config['dtype'] in dtype_map:
        config['dtype'] = dtype_map[config['dtype']]
    
    # Create tokenizer
    print("Initializing CLIP tokenizer...")
    tokenizer = FrozenCLIPVisionTokenizer(
        model_name=config.get('model_name', 'openai/clip-vit-base-patch16'),
        freeze=config.get('freeze', True),
        clip_input_size=config.get('clip_input_size', 224),
        remove_cls_token=config.get('remove_cls_token', True),
        return_spatial_map=config.get('return_spatial_map', True),
        device=config.get('device'),
        dtype=config.get('dtype', torch.float32),
    )
    print()
    
    # Determine which splits to process
    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]
    
    # Process each split
    total_stats = {
        'processed': 0,
        'skipped': 0,
        'failed': 0,
    }
    
    for split in splits:
        stats = process_split(
            split=split,
            source=args.source,
            metadata_dir=metadata_dir,
            output_dir=output_dir,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
            save_spatial_map=args.save_spatial_map,
            save_preprocessed=args.save_preprocessed,
            max_samples=args.max_samples,
            verbose=args.verbose,
        )
        
        # Accumulate stats
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Print overall summary
    print()
    print("="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"Total processed: {total_stats['processed']}")
    print(f"Total skipped: {total_stats['skipped']}")
    print(f"Total failed: {total_stats['failed']}")
    print(f"Total samples: {total_stats['processed'] + total_stats['skipped'] + total_stats['failed']}")
    print()
    
    # Verify a few samples
    if total_stats['processed'] > 0:
        verify_saved_tokens(output_dir, args.verify_samples)
    
    print("="*70)
    print("✓ PRECOMPUTATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
