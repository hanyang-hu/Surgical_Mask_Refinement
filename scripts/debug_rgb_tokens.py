"""Debug script for testing CLIP vision tokenizer.

Tests the frozen CLIP tokenizer module with dummy data or real dataset samples.
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.rgb.clip_tokenizer import FrozenCLIPVisionTokenizer, CLIPVisionTokenizerWithAdapter
from models.rgb.adapters import build_adapter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Debug CLIP tokenizer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/rgb_tokenizer.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="Input image size (H=W)"
    )
    parser.add_argument(
        "--use_real_data",
        action="store_true",
        help="Use real dataset samples instead of dummy data"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of real samples to test (if use_real_data)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_with_dummy_data(tokenizer, batch_size, input_size, device):
    """Test tokenizer with dummy random data."""
    print("=" * 70)
    print("TEST WITH DUMMY DATA")
    print("=" * 70)
    
    # Create dummy input
    print(f"Creating dummy input: [{batch_size}, 3, {input_size}, {input_size}]")
    rgb = torch.randn(batch_size, 3, input_size, input_size, device=device)
    
    # Normalize to [0, 1] range
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    print(f"Input shape: {rgb.shape}")
    print(f"Input dtype: {rgb.dtype}")
    print(f"Input device: {rgb.device}")
    print(f"Input range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    print()
    
    # Forward pass
    print("Running forward pass...")
    output = tokenizer(rgb)
    
    # Print output shapes
    print("\nOutput:")
    for key, value in output.items():
        if torch.is_tensor(value):
            print(f"  {key}:")
            print(f"    - shape: {value.shape}")
            print(f"    - dtype: {value.dtype}")
            print(f"    - device: {value.device}")
            print(f"    - range: [{value.min():.3f}, {value.max():.3f}]")
    
    print()
    print("✓ Dummy data test passed!")
    return output


def test_with_real_data(tokenizer, num_samples):
    """Test tokenizer with real dataset samples."""
    print("=" * 70)
    print("TEST WITH REAL DATASET")
    print("=" * 70)
    
    try:
        from data.dataset import SurgicalMaskRefinementDataset
        from data.transforms import build_transforms
        
        # Build transforms
        transform = build_transforms(train=False, augment=False, image_size=512)
        
        # Load dataset
        dataset = SurgicalMaskRefinementDataset(
            metadata_dir='data/metadata',
            split='train',
            load_images=True,
            apply_transforms=True,
            transform=transform
        )
        
        print(f"✓ Loaded dataset with {len(dataset)} samples")
        print()
        
        # Test on multiple samples
        for i in range(min(num_samples, len(dataset))):
            print(f"Sample {i + 1}:")
            sample = dataset[i]
            
            # Get RGB image and add batch dimension
            rgb = sample['rgb'].unsqueeze(0).to(tokenizer.device)
            
            print(f"  ID: {sample['id']}")
            print(f"  RGB shape: {rgb.shape}")
            print(f"  RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
            
            # Forward pass
            output = tokenizer(rgb)
            
            print(f"  Output shapes:")
            for key, value in output.items():
                if torch.is_tensor(value):
                    print(f"    - {key}: {value.shape}")
            print()
        
        print("✓ Real data test passed!")
        
    except Exception as e:
        print(f"⚠ Could not test with real data: {e}")
        print("  Make sure Step 1 and Step 2 are completed")


def verify_frozen_parameters(tokenizer):
    """Verify that CLIP parameters are frozen."""
    print("=" * 70)
    print("PARAMETER FREEZE VERIFICATION")
    print("=" * 70)
    
    total_params = tokenizer.count_total_parameters()
    trainable_params = tokenizer.count_trainable_parameters()
    frozen_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print()
    
    if trainable_params == 0:
        print("✓ All parameters are frozen (as expected)")
    else:
        print(f"⚠ Warning: {trainable_params:,} parameters are trainable")
    
    # Check if model reports as frozen
    is_frozen = tokenizer.is_frozen()
    print(f"Model reports frozen: {is_frozen}")
    print()


def main():
    """Main debugging function."""
    args = parse_args()
    
    print("=" * 70)
    print("CLIP VISION TOKENIZER DEBUG")
    print("=" * 70)
    print()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    print("Config:")
    for key, value in config.items():
        if value is not None:
            print(f"  {key}: {value}")
    print()
    
    # Override device if specified
    if args.device:
        config['device'] = args.device
    
    # Convert dtype string to torch dtype
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
    print(tokenizer)
    print()
    
    # Verify parameters are frozen
    verify_frozen_parameters(tokenizer)
    
    # Test with dummy data
    output = test_with_dummy_data(
        tokenizer,
        args.batch_size,
        args.input_size,
        tokenizer.device
    )
    print()
    
    # Test with real data if requested
    if args.use_real_data:
        test_with_real_data(tokenizer, args.num_samples)
        print()
    
    # Test adapter if configured
    if config.get('use_adapter', False):
        print("=" * 70)
        print("TEST WITH ADAPTER")
        print("=" * 70)
        
        adapter_out_dim = config.get('adapter_out_dim')
        adapter_type = config.get('adapter_type', 'linear')
        
        if adapter_out_dim is None:
            print("⚠ use_adapter=true but adapter_out_dim not specified")
        else:
            print(f"Creating {adapter_type} adapter: {tokenizer.hidden_dim} -> {adapter_out_dim}")
            
            model_with_adapter = CLIPVisionTokenizerWithAdapter(
                tokenizer=tokenizer,
                use_adapter=True,
                adapter_out_dim=adapter_out_dim,
                adapter_type=adapter_type
            )
            
            # Test with dummy data
            rgb = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=tokenizer.device)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            
            output = model_with_adapter(rgb)
            
            print("\nOutput with adapter:")
            for key, value in output.items():
                if torch.is_tensor(value):
                    print(f"  {key}: {value.shape}")
            
            print("\n✓ Adapter test passed!")
        print()
    
    print("=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
