"""Frozen CLIP vision tokenizer for RGB feature extraction.

This module loads a pretrained CLIP vision encoder, freezes its weights,
and extracts visual patch tokens from RGB images.
"""

from typing import Dict, Optional, Union, Tuple
from pathlib import Path
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor


class FrozenCLIPVisionTokenizer(nn.Module):
    """Frozen CLIP vision encoder for RGB token extraction.
    
    This module:
    - Loads a pretrained CLIP vision model
    - Freezes all weights (no training)
    - Accepts RGB tensors in [0, 1] range
    - Internally resizes and normalizes for CLIP
    - Extracts visual patch tokens
    - Optionally returns spatial feature maps
    
    Args:
        model_name: Hugging Face model name or local path
            Default: "openai/clip-vit-base-patch16"
        freeze: Whether to freeze CLIP weights (default: True)
        clip_input_size: Input size for CLIP (default: 224)
        remove_cls_token: Remove CLS token from output (default: True)
        return_spatial_map: Return tokens as spatial map (default: True)
        device: Device to load model on (default: "cuda" if available)
        dtype: Model dtype (default: torch.float32)
        
    Example:
        >>> tokenizer = FrozenCLIPVisionTokenizer()
        >>> rgb = torch.randn(4, 3, 512, 512)  # From Step 2 dataset
        >>> output = tokenizer(rgb)
        >>> print(output['tokens'].shape)      # [4, 196, 768]
        >>> print(output['spatial_map'].shape) # [4, 768, 14, 14]
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        freeze: bool = True,
        clip_input_size: int = 224,
        remove_cls_token: bool = True,
        return_spatial_map: bool = True,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize frozen CLIP vision tokenizer."""
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        self.clip_input_size = clip_input_size
        self.remove_cls_token = remove_cls_token
        self.return_spatial_map = return_spatial_map
        self.dtype = dtype
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading CLIP vision model: {model_name}")
        
        # Load pretrained CLIP vision model
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.vision_model.to(device=device, dtype=dtype)
        
        # Get model config for later use
        self.config = self.vision_model.config
        self.hidden_dim = self.config.hidden_size
        self.patch_size = self.config.patch_size
        
        # Calculate patch grid size
        self.num_patches_per_side = clip_input_size // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        
        # Load image processor for preprocessing
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # Extract CLIP normalization parameters
        self.register_buffer(
            "clip_mean",
            torch.tensor(self.image_processor.image_mean, dtype=dtype, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(self.image_processor.image_std, dtype=dtype, device=device).view(1, 3, 1, 1)
        )
        
        # Freeze model if requested
        if freeze:
            self._freeze_model()
        
        # Set to eval mode
        self.vision_model.eval()
        
        print(f"✓ CLIP model loaded successfully")
        print(f"  - Model: {model_name}")
        print(f"  - Device: {device}")
        print(f"  - Hidden dim: {self.hidden_dim}")
        print(f"  - Patch size: {self.patch_size}")
        print(f"  - Num patches: {self.num_patches} ({self.num_patches_per_side}x{self.num_patches_per_side})")
        print(f"  - Frozen: {freeze}")
        print(f"  - Trainable params: {self.count_trainable_parameters()}")
        
    def _freeze_model(self):
        """Freeze all CLIP vision model parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        print("✓ All CLIP parameters frozen")
        
    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_total_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def is_frozen(self) -> bool:
        """Check if all CLIP parameters are frozen."""
        return all(not p.requires_grad for p in self.vision_model.parameters())
    
    def preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        """Preprocess RGB images for CLIP.
        
        Applies resizing and normalization expected by CLIP.
        
        Args:
            rgb: RGB tensor [B, 3, H, W] in range [0, 1]
            
        Returns:
            Preprocessed tensor [B, 3, 224, 224] normalized for CLIP
        """
        # Ensure correct device and dtype
        rgb = rgb.to(device=self.device, dtype=self.dtype)
        
        # Resize to CLIP input size if needed
        if rgb.shape[-2:] != (self.clip_input_size, self.clip_input_size):
            rgb = torch.nn.functional.interpolate(
                rgb,
                size=(self.clip_input_size, self.clip_input_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Apply CLIP normalization
        # Input is in [0, 1], CLIP expects normalized with mean/std
        rgb = (rgb - self.clip_mean) / self.clip_std
        
        return rgb
    
    def extract_patch_tokens(
        self,
        last_hidden_state: torch.Tensor,
        remove_cls: bool = True
    ) -> torch.Tensor:
        """Extract patch tokens from CLIP output.
        
        Args:
            last_hidden_state: CLIP output [B, N+1, C] where N is num patches
            remove_cls: Whether to remove CLS token (first token)
            
        Returns:
            Patch tokens [B, N, C] if remove_cls=True, else [B, N+1, C]
        """
        if remove_cls:
            # Remove CLS token (first token)
            tokens = last_hidden_state[:, 1:, :]
        else:
            tokens = last_hidden_state
        
        return tokens
    
    def tokens_to_spatial_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert token sequence to spatial feature map.
        
        Args:
            tokens: Patch tokens [B, N, C] where N = H*W
            
        Returns:
            Spatial map [B, C, H, W]
        """
        B, N, C = tokens.shape
        H = W = self.num_patches_per_side
        
        # Verify token count matches expected patch grid
        assert N == H * W, (
            f"Token count {N} doesn't match patch grid {H}x{W}={H*W}"
        )
        
        # Reshape to spatial map [B, H, W, C] -> [B, C, H, W]
        spatial_map = tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return spatial_map
    
    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract CLIP visual tokens from RGB images.
        
        Args:
            rgb: RGB images [B, 3, H, W] in range [0, 1]
                Typically [B, 3, 512, 512] from Step 2 dataset
                
        Returns:
            Dictionary containing:
                - 'tokens': Patch token sequence [B, N, C]
                    where N = num_patches (196 for ViT-B/16 at 224x224)
                    and C = hidden_dim (768 for ViT-B/16)
                - 'spatial_map': Spatial feature map [B, C, H, W] (if return_spatial_map=True)
                    where H = W = num_patches_per_side (14 for ViT-B/16)
                - 'preprocessed': Preprocessed input [B, 3, 224, 224] (for debugging)
        """
        # Ensure model is in eval mode
        self.vision_model.eval()
        
        # Preprocess RGB for CLIP
        preprocessed = self.preprocess(rgb)
        
        # Extract features with CLIP
        outputs = self.vision_model(
            pixel_values=preprocessed,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get last hidden state [B, N+1, C]
        last_hidden_state = outputs.last_hidden_state
        
        # Extract patch tokens (optionally remove CLS)
        tokens = self.extract_patch_tokens(
            last_hidden_state,
            remove_cls=self.remove_cls_token
        )
        
        # Build output dictionary
        result = {
            'tokens': tokens,
            'preprocessed': preprocessed,
        }
        
        # Optionally add spatial map
        if self.return_spatial_map:
            spatial_map = self.tokens_to_spatial_map(tokens)
            result['spatial_map'] = spatial_map
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FrozenCLIPVisionTokenizer(\n"
            f"  model_name={self.model_name},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_patches={self.num_patches},\n"
            f"  patch_grid={self.num_patches_per_side}x{self.num_patches_per_side},\n"
            f"  frozen={self.freeze},\n"
            f"  trainable_params={self.count_trainable_parameters()},\n"
            f"  device={self.device}\n"
            f")"
        )


class CLIPVisionTokenizerWithAdapter(nn.Module):
    """CLIP tokenizer with optional adapter/projection layer.
    
    This combines the frozen CLIP tokenizer with an optional lightweight
    adapter for projecting features to a different dimension.
    
    Args:
        tokenizer: FrozenCLIPVisionTokenizer instance
        use_adapter: Whether to use adapter (default: False)
        adapter_out_dim: Output dimension for adapter (default: None)
        adapter_type: Type of adapter ('linear' or 'conv1x1')
        
    Example:
        >>> tokenizer = FrozenCLIPVisionTokenizer()
        >>> model = CLIPVisionTokenizerWithAdapter(
        ...     tokenizer=tokenizer,
        ...     use_adapter=True,
        ...     adapter_out_dim=256
        ... )
    """
    
    def __init__(
        self,
        tokenizer: FrozenCLIPVisionTokenizer,
        use_adapter: bool = False,
        adapter_out_dim: Optional[int] = None,
        adapter_type: str = "linear",
    ):
        """Initialize tokenizer with adapter."""
        super().__init__()
        
        self.tokenizer = tokenizer
        self.use_adapter = use_adapter
        self.adapter_out_dim = adapter_out_dim
        self.adapter_type = adapter_type
        
        # Create adapter if requested
        if use_adapter:
            if adapter_out_dim is None:
                raise ValueError("adapter_out_dim must be specified when use_adapter=True")
            
            if adapter_type == "linear":
                # Linear projection for token sequences
                self.adapter = nn.Linear(tokenizer.hidden_dim, adapter_out_dim)
            elif adapter_type == "conv1x1":
                # 1x1 conv for spatial maps
                self.adapter = nn.Conv2d(tokenizer.hidden_dim, adapter_out_dim, 1)
            else:
                raise ValueError(f"Unknown adapter_type: {adapter_type}")
            
            print(f"✓ Created {adapter_type} adapter: {tokenizer.hidden_dim} -> {adapter_out_dim}")
        else:
            self.adapter = None
    
    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract tokens and optionally apply adapter.
        
        Args:
            rgb: RGB images [B, 3, H, W] in range [0, 1]
            
        Returns:
            Dictionary with tokens and optional adapted features
        """
        # Extract CLIP tokens
        output = self.tokenizer(rgb)
        
        # Apply adapter if enabled
        if self.use_adapter:
            if self.adapter_type == "linear":
                # Project token sequences
                adapted_tokens = self.adapter(output['tokens'])
                output['adapted_tokens'] = adapted_tokens
            elif self.adapter_type == "conv1x1" and 'spatial_map' in output:
                # Project spatial map
                adapted_map = self.adapter(output['spatial_map'])
                output['adapted_spatial_map'] = adapted_map
        
        return output
