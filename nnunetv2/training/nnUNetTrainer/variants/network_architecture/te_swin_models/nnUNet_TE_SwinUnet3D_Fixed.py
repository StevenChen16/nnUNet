"""
Fixed version of nnUNet-compatible TE-Swin UNet3D model implementation.
This version addresses dimension compatibility issues with small input sizes.
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple

# Import SwinUnet3D components
from .SwinUnet_3D import (SwinBlock3D, Encoder, Decoder, Norm, Converge, 
                         PatchMerging3D, FinalExpand3D)

# Import our texture-specific modules 
from .TextureAttentionModule import TextureAttentionModule
from .MultiScaleTexturePyramid import MultiScaleTexturePyramid
from .ShapeTextureFusion import ShapeTextureFusion
from .TemporalModules import TemporalAttentionModule, SlicePropagationModule


class DimensionValidator:
    """Helper class to validate and fix dimension compatibility issues."""
    
    @staticmethod
    def validate_input_size(input_shape: Tuple[int, int, int], 
                           window_size: int, 
                           downscaling_factors: Tuple[int, ...]) -> Tuple[bool, str]:
        """
        Validate if input dimensions are compatible with the network.
        
        Args:
            input_shape: (D, H, W) input dimensions
            window_size: Size of attention window
            downscaling_factors: Downscaling factors for each stage
            
        Returns:
            Tuple of (is_valid, message)
        """
        total_downscaling = np.prod(downscaling_factors)
        
        # Check if dimensions are divisible by window_size
        for i, dim in enumerate(input_shape):
            if dim % window_size != 0:
                return False, f"Dimension {i} ({dim}) must be divisible by window_size ({window_size})"
        
        # Check if dimensions remain positive after all downscaling
        current_dims = list(input_shape)
        for stage, factor in enumerate(downscaling_factors):
            current_dims = [d // factor for d in current_dims]
            if min(current_dims) <= 0:
                return False, f"After stage {stage+1} downscaling, dimensions become {current_dims}"
        
        # Check if final dimensions are valid for attention
        final_dims = [d // total_downscaling for d in input_shape]
        if min(final_dims) <= 0:
            return False, f"Final dimensions {final_dims} are too small"
            
        return True, "Dimensions are compatible"
    
    @staticmethod
    def suggest_compatible_size(target_size: Tuple[int, int, int], 
                               window_size: int, 
                               downscaling_factors: Tuple[int, ...]) -> Tuple[int, int, int]:
        """
        Suggest a compatible input size close to the target size.
        
        Args:
            target_size: Desired input size
            window_size: Window size for attention
            downscaling_factors: Downscaling factors
            
        Returns:
            Compatible input size
        """
        total_downscaling = np.prod(downscaling_factors)
        min_size = window_size * total_downscaling
        
        compatible_size = []
        for dim in target_size:
            # Find the smallest multiple of window_size that's >= dim and >= min_size
            adjusted_dim = max(dim, min_size)
            # Round up to nearest multiple of window_size
            adjusted_dim = ((adjusted_dim + window_size - 1) // window_size) * window_size
            compatible_size.append(adjusted_dim)
            
        return tuple(compatible_size)


class nnUNet_TE_SwinUnet3D_Fixed(nn.Module):
    """
    Fixed nnUNet-compatible Texture-Enhanced Swin UNet3D.
    
    This version includes dimension validation and adaptive sizing.
    """
    
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 96,
        layers: Tuple[int, ...] = (2, 2, 4, 2),
        heads: Tuple[int, ...] = (3, 6, 9, 12),
        head_dim: int = 32,
        window_size: Union[int, List[int]] = 7,
        downscaling_factors: Tuple[int, ...] = (4, 2, 2, 2),
        relative_pos_embedding: bool = True,
        dropout: float = 0.0,
        stl_channels: int = 32,
        deep_supervision: bool = True,
        adaptive_sizing: bool = True
    ):
        """
        Initialize fixed TE-Swin UNet3D for nnUNet compatibility.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            hidden_dim: Base hidden dimension
            layers: Number of layers in each stage
            heads: Number of attention heads in each stage  
            head_dim: Dimension of each attention head
            window_size: Size of attention window
            downscaling_factors: Downscaling factors for each stage
            relative_pos_embedding: Whether to use relative position embedding
            dropout: Dropout rate
            stl_channels: Second-to-last channels
            deep_supervision: Whether to enable deep supervision
            adaptive_sizing: Whether to enable adaptive input sizing
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.adaptive_sizing = adaptive_sizing
        self.dsf = downscaling_factors
        self.window_size = window_size
        
        # Validate parameters
        self._validate_parameters()
        
        # Create encoder blocks with proper error handling
        try:
            self.enc12 = Encoder(
                in_dims=input_channels, 
                hidden_dimension=hidden_dim, 
                layers=layers[0],
                downscaling_factor=downscaling_factors[0], 
                num_heads=heads[0],
                head_dim=head_dim, 
                window_size=window_size, 
                dropout=dropout,
                relative_pos_embedding=relative_pos_embedding
            )
            
            self.enc3 = Encoder(
                in_dims=hidden_dim, 
                hidden_dimension=hidden_dim * 2,
                layers=layers[1],
                downscaling_factor=downscaling_factors[1], 
                num_heads=heads[1],
                head_dim=head_dim, 
                window_size=window_size, 
                dropout=dropout,
                relative_pos_embedding=relative_pos_embedding
            )
            
            self.enc4 = Encoder(
                in_dims=hidden_dim * 2, 
                hidden_dimension=hidden_dim * 4,
                layers=layers[2],
                downscaling_factor=downscaling_factors[2], 
                num_heads=heads[2],
                head_dim=head_dim, 
                window_size=window_size, 
                dropout=dropout,
                relative_pos_embedding=relative_pos_embedding
            )
            
            self.enc5 = Encoder(
                in_dims=hidden_dim * 4, 
                hidden_dimension=hidden_dim * 8,
                layers=layers[3],
                downscaling_factor=downscaling_factors[3], 
                num_heads=heads[3],
                head_dim=head_dim, 
                window_size=window_size, 
                dropout=dropout,
                relative_pos_embedding=relative_pos_embedding
            )
        except Exception as e:
            print(f"Error creating encoder blocks: {e}")
            raise
        
        # Texture and temporal modules with error handling
        try:
            # Texture attention modules for each encoder level
            self.texture_attention_modules = nn.ModuleList([
                TextureAttentionModule(dim=hidden_dim),
                TextureAttentionModule(dim=hidden_dim * 2),
                TextureAttentionModule(dim=hidden_dim * 4),
                TextureAttentionModule(dim=hidden_dim * 8)
            ])
            
            # Multi-scale texture pyramid
            self.texture_pyramid = MultiScaleTexturePyramid(
                dims=[hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8],
                debug_output=False
            )
            
            # Shape-texture fusion modules
            self.fusion_modules = nn.ModuleList([
                ShapeTextureFusion(dim=hidden_dim),
                ShapeTextureFusion(dim=hidden_dim * 2),
                ShapeTextureFusion(dim=hidden_dim * 4),
                ShapeTextureFusion(dim=hidden_dim * 8)
            ])
            
            # Temporal attention modules - treating Z axis as time dimension
            self.temporal_attention_modules = nn.ModuleList([
                TemporalAttentionModule(dim=hidden_dim, num_heads=max(1, heads[0]//3)),
                TemporalAttentionModule(dim=hidden_dim * 2, num_heads=max(1, heads[1]//3)),
                TemporalAttentionModule(dim=hidden_dim * 4, num_heads=max(1, heads[2]//3)),
                TemporalAttentionModule(dim=hidden_dim * 8, num_heads=max(1, heads[3]//3))
            ])
            
            # Slice propagation module for bidirectional information flow
            self.slice_propagation = SlicePropagationModule(hidden_dim=hidden_dim * 8)
        except Exception as e:
            print(f"Error creating texture/temporal modules: {e}")
            raise
        
        # Create decoder blocks
        try:
            self.dec4 = Decoder(
                in_dims=hidden_dim * 8, 
                out_dims=hidden_dim * 4,
                layers=layers[2],
                up_scaling_factor=downscaling_factors[3], 
                num_heads=heads[2],
                head_dim=head_dim, 
                window_size=window_size, 
                dropout=dropout,
                relative_pos_embedding=relative_pos_embedding
            )
            
            self.dec3 = Decoder(
                in_dims=hidden_dim * 4, 
                out_dims=hidden_dim * 2,
                layers=layers[1],
                up_scaling_factor=downscaling_factors[2], 
                num_heads=heads[1],
                head_dim=head_dim, 
                window_size=window_size, 
                dropout=dropout,
                relative_pos_embedding=relative_pos_embedding
            )
            
            self.dec12 = Decoder(
                in_dims=hidden_dim * 2, 
                out_dims=hidden_dim,
                layers=layers[0],
                up_scaling_factor=downscaling_factors[1], 
                num_heads=heads[0],
                head_dim=head_dim, 
                window_size=window_size, 
                dropout=dropout,
                relative_pos_embedding=relative_pos_embedding
            )
        except Exception as e:
            print(f"Error creating decoder blocks: {e}")
            raise
        
        # Skip connections and final layers
        self.converge4 = Converge(hidden_dim * 4)
        self.converge3 = Converge(hidden_dim * 2)
        self.converge12 = Converge(hidden_dim)
        
        # Final layers
        self.final = FinalExpand3D(
            in_dim=hidden_dim, 
            out_dim=stl_channels,
            up_scaling_factor=downscaling_factors[0]
        )
        
        self.out = nn.Conv3d(stl_channels, num_classes, kernel_size=1)
        
        # Deep supervision heads
        if self.deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                nn.Conv3d(hidden_dim * 4, num_classes, kernel_size=1),
                nn.Conv3d(hidden_dim * 2, num_classes, kernel_size=1),
                nn.Conv3d(hidden_dim, num_classes, kernel_size=1)
            ])
        
        # Initialize weights
        self.init_weight()
        
    def _validate_parameters(self):
        """Validate model parameters."""
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        
        if any(factor <= 0 for factor in self.dsf):
            raise ValueError(f"All downscaling factors must be positive, got {self.dsf}")
        
        total_downscaling = np.prod(self.dsf)
        min_input_size = self.window_size * total_downscaling
        
        print(f"Model requirements:")
        print(f"  - window_size: {self.window_size}")
        print(f"  - downscaling_factors: {self.dsf}")
        print(f"  - total_downscaling: {total_downscaling}")
        print(f"  - minimum_input_size: {min_input_size} per dimension")
        
    def validate_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Validate and potentially adjust input tensor dimensions.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            
        Returns:
            Validated/adjusted input tensor
        """
        input_shape = x.shape[2:]  # D, H, W
        
        # Check compatibility
        is_valid, message = DimensionValidator.validate_input_size(
            input_shape, self.window_size, self.dsf
        )
        
        if not is_valid:
            if self.adaptive_sizing:
                # Suggest compatible size
                compatible_size = DimensionValidator.suggest_compatible_size(
                    input_shape, self.window_size, self.dsf
                )
                
                print(f"Warning: Input size {input_shape} is not compatible: {message}")
                print(f"Resizing to compatible size: {compatible_size}")
                
                # Resize input
                x = F.interpolate(
                    x, 
                    size=compatible_size, 
                    mode='trilinear', 
                    align_corners=False
                )
            else:
                raise ValueError(f"Input size {input_shape} is not compatible: {message}")
        
        return x
        
    def forward(self, x):
        """
        Forward pass through the fixed TE-Swin UNet3D.
        
        Args:
            x (Tensor): Input volume [B, C, D, H, W]
            
        Returns:
            Tensor or List[Tensor]: Segmentation prediction(s)
        """
        # Validate input dimensions
        original_size = x.shape[2:]
        x = self.validate_input(x)
        
        # Store outputs for deep supervision
        deep_supervision_outputs = []
        
        try:
            # Encoder pathway with texture attention
            encoder_features = []
            texture_features = []
            
            # Stage 1-2
            x = self.enc12(x)
            x = self.temporal_attention_modules[0](x)
            texture_feat = self.texture_attention_modules[0](x)
            encoder_features.append(x)
            texture_features.append(texture_feat)
            
            # Stage 3
            x = self.enc3(x)
            x = self.temporal_attention_modules[1](x)
            texture_feat = self.texture_attention_modules[1](x)
            encoder_features.append(x)
            texture_features.append(texture_feat)
            
            # Stage 4
            x = self.enc4(x)
            x = self.temporal_attention_modules[2](x)
            texture_feat = self.texture_attention_modules[2](x)
            encoder_features.append(x)
            texture_features.append(texture_feat)
            
            # Stage 5 (bottleneck)
            x = self.enc5(x)
            x = self.temporal_attention_modules[3](x)
            texture_feat = self.texture_attention_modules[3](x)
            encoder_features.append(x)
            texture_features.append(texture_feat)
            
            # Apply slice propagation
            x = self.slice_propagation(x)
            
            # Create multi-scale texture pyramid
            texture_pyramid = self.texture_pyramid(texture_features)
            
            # Decoder pathway with shape-texture fusion
            
            # Stage 4 decoder
            x = self.dec4(x)
            shape_feat = encoder_features[2]
            texture_feat = texture_pyramid[2]
            fused_feat = self.fusion_modules[2](shape_feat, texture_feat)
            x = self.converge4(x, fused_feat)
            
            if self.deep_supervision:
                ds_out = self.deep_supervision_heads[0](x)
                deep_supervision_outputs.append(ds_out)
            
            # Stage 3 decoder
            x = self.dec3(x)
            shape_feat = encoder_features[1]
            texture_feat = texture_pyramid[1]
            fused_feat = self.fusion_modules[1](shape_feat, texture_feat)
            x = self.converge3(x, fused_feat)
            
            if self.deep_supervision:
                ds_out = self.deep_supervision_heads[1](x)
                deep_supervision_outputs.append(ds_out)
            
            # Stage 1-2 decoder
            x = self.dec12(x)
            shape_feat = encoder_features[0]
            texture_feat = texture_pyramid[0]
            fused_feat = self.fusion_modules[0](shape_feat, texture_feat)
            x = self.converge12(x, fused_feat)
            
            if self.deep_supervision:
                ds_out = self.deep_supervision_heads[2](x)
                deep_supervision_outputs.append(ds_out)
            
            # Final upsampling and output head
            x = self.final(x)
            x = self.out(x)
            
            # Resize back to original size if needed
            if x.shape[2:] != original_size:
                x = F.interpolate(
                    x, 
                    size=original_size, 
                    mode='trilinear', 
                    align_corners=False
                )
                
                # Also resize deep supervision outputs
                for i, ds_out in enumerate(deep_supervision_outputs):
                    deep_supervision_outputs[i] = F.interpolate(
                        ds_out, 
                        size=original_size, 
                        mode='trilinear', 
                        align_corners=False
                    )
            
            # Return outputs
            if self.deep_supervision and len(deep_supervision_outputs) > 0:
                # Add main output to the beginning
                return [x] + deep_supervision_outputs
            else:
                return x
                
        except Exception as e:
            print(f"Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def init_weight(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Factory functions for different model variants
def create_te_swinunet_t_3d_fixed(input_channels: int = 1, num_classes: int = 2, **kwargs):
    """Create a tiny fixed TE-Swin UNet3D model."""
    # Safe parameters for tiny model
    default_kwargs = {
        'hidden_dim': 32,
        'layers': (1, 1, 2, 1),
        'heads': (2, 4, 6, 8),
        'head_dim': 16,
        'window_size': 4,  # Smaller window for compatibility
        'downscaling_factors': (2, 2, 2, 2),  # Conservative downscaling
        'deep_supervision': True,
        'adaptive_sizing': True
    }
    default_kwargs.update(kwargs)
    
    return nnUNet_TE_SwinUnet3D_Fixed(
        input_channels=input_channels,
        num_classes=num_classes,
        **default_kwargs
    )


def create_te_swinunet_s_3d_fixed(input_channels: int = 1, num_classes: int = 2, **kwargs):
    """Create a small fixed TE-Swin UNet3D model."""
    # Balanced parameters for small model
    default_kwargs = {
        'hidden_dim': 48,
        'layers': (2, 2, 4, 2),
        'heads': (3, 6, 9, 12),
        'head_dim': 24,
        'window_size': 7,  # Standard window size
        'downscaling_factors': (4, 2, 2, 2),  # Standard downscaling
        'deep_supervision': True,
        'adaptive_sizing': True
    }
    default_kwargs.update(kwargs)
    
    return nnUNet_TE_SwinUnet3D_Fixed(
        input_channels=input_channels,
        num_classes=num_classes,
        **default_kwargs
    )


def create_te_swinunet_b_3d_fixed(input_channels: int = 1, num_classes: int = 2, **kwargs):
    """Create a base fixed TE-Swin UNet3D model."""
    # Full parameters for base model
    default_kwargs = {
        'hidden_dim': 96,
        'layers': (2, 2, 8, 2),
        'heads': (4, 8, 16, 32),
        'head_dim': 32,
        'window_size': 7,
        'downscaling_factors': (4, 2, 2, 2),
        'deep_supervision': True,
        'adaptive_sizing': True
    }
    default_kwargs.update(kwargs)
    
    return nnUNet_TE_SwinUnet3D_Fixed(
        input_channels=input_channels,
        num_classes=num_classes,
        **default_kwargs
    )
