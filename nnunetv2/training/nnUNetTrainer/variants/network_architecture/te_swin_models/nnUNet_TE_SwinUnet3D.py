"""
nnUNet-compatible TE-Swin UNet3D model implementation.
This version integrates the "MRI as GIF" concept into the nnUNet framework.
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


class nnUNet_TE_SwinUnet3D(nn.Module):
    """
    nnUNet-compatible Texture-Enhanced Swin UNet3D.
    
    This model combines the nnUNet framework with our "MRI as GIF" approach,
    treating MRI slices as temporal sequences like video frames.
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
        deep_supervision: bool = True
    ):
        """
        Initialize TE-Swin UNet3D for nnUNet compatibility.
        
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
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.dsf = downscaling_factors
        self.window_size = window_size
        
        # Encoder blocks with texture and temporal attention
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
        # Reduce number of heads to avoid potential issues
        self.temporal_attention_modules = nn.ModuleList([
            TemporalAttentionModule(dim=hidden_dim, num_heads=max(1, heads[0]//3)),
            TemporalAttentionModule(dim=hidden_dim * 2, num_heads=max(1, heads[1]//3)),
            TemporalAttentionModule(dim=hidden_dim * 4, num_heads=max(1, heads[2]//3)),
            TemporalAttentionModule(dim=hidden_dim * 8, num_heads=max(1, heads[3]//3))
        ])
        
        # Slice propagation module for bidirectional information flow
        self.slice_propagation = SlicePropagationModule(hidden_dim=hidden_dim * 8)
        
        # Decoder blocks
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
        
        # Enhanced skip connections
        self.converge4 = Converge(hidden_dim * 4)
        self.converge3 = Converge(hidden_dim * 2)
        self.converge12 = Converge(hidden_dim)
        
        # Final layers
        self.final = FinalExpand3D(
            in_dim=hidden_dim, 
            out_dim=stl_channels,
            up_scaling_factor=downscaling_factors[0]
        )
        
        # Main output
        self.seg_head = nn.Conv3d(stl_channels, num_classes, kernel_size=1)
        
        # Deep supervision outputs if enabled
        if self.deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                nn.Conv3d(hidden_dim * 8, num_classes, kernel_size=1),
                nn.Conv3d(hidden_dim * 4, num_classes, kernel_size=1),
                nn.Conv3d(hidden_dim * 2, num_classes, kernel_size=1),
                nn.Conv3d(hidden_dim, num_classes, kernel_size=1)
            ])
        
        # Initialize weights
        self.init_weights()
        
    def forward(self, x):
        """
        Forward pass through the TE-Swin UNet3D.
        
        Args:
            x (Tensor): Input volume [B, C, D, H, W]
            
        Returns:
            Tensor or List[Tensor]: Segmentation prediction(s)
        """
        # Check input dimensions for compatibility
        input_shape = x.shape[2:]  # D, H, W
        valid, message = self.validate_dimensions(input_shape)
        if not valid:
            # Try to make compatible by padding
            x = self._pad_to_compatible_size(x)
        
        # Store features for deep supervision
        deep_supervision_outputs = []
        
        # Encoder pathway with texture and temporal attention
        encoder_features = []  # Original features from encoder
        texture_features = []  # Texture-enhanced features
        
        # Stage 1-2
        x = self.enc12(x)
        # Apply temporal attention - treating Z-axis as time axis  
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
        
        # Apply slice propagation - similar to video forward/backward propagation
        x = self.slice_propagation(x)
        
        # Deep supervision from bottleneck
        if self.deep_supervision:
            ds_out = F.interpolate(self.deep_supervision_heads[0](x), 
                                 size=input_shape, mode='trilinear', align_corners=False)
            deep_supervision_outputs.append(ds_out)
        
        # Create multi-scale texture pyramid
        texture_pyramid = self.texture_pyramid(texture_features)
        
        # Decoder pathway with shape-texture fusion
        
        # Stage 4 decoder
        x = self.dec4(x)
        shape_feat = encoder_features[2]  # Stage 4 features
        texture_feat = texture_pyramid[2]
        fused_feat = self.fusion_modules[2](shape_feat, texture_feat)
        x = self.converge4(x, fused_feat)
        
        if self.deep_supervision:
            ds_out = F.interpolate(self.deep_supervision_heads[1](x), 
                                 size=input_shape, mode='trilinear', align_corners=False)
            deep_supervision_outputs.append(ds_out)
        
        # Stage 3 decoder
        x = self.dec3(x)
        shape_feat = encoder_features[1]  # Stage 3 features
        texture_feat = texture_pyramid[1]
        fused_feat = self.fusion_modules[1](shape_feat, texture_feat)
        x = self.converge3(x, fused_feat)
        
        if self.deep_supervision:
            ds_out = F.interpolate(self.deep_supervision_heads[2](x), 
                                 size=input_shape, mode='trilinear', align_corners=False)
            deep_supervision_outputs.append(ds_out)
        
        # Stage 1-2 decoder
        x = self.dec12(x)
        shape_feat = encoder_features[0]  # Stage 1-2 features
        texture_feat = texture_pyramid[0]
        fused_feat = self.fusion_modules[0](shape_feat, texture_feat)
        x = self.converge12(x, fused_feat)
        
        if self.deep_supervision:
            ds_out = F.interpolate(self.deep_supervision_heads[3](x), 
                                 size=input_shape, mode='trilinear', align_corners=False)
            deep_supervision_outputs.append(ds_out)
        
        # Final upsampling and output head
        x = self.final(x)
        main_output = self.seg_head(x)
        
        # Return outputs based on training mode
        if self.deep_supervision and self.training:
            # Return list of outputs for deep supervision
            all_outputs = [main_output] + deep_supervision_outputs
            return all_outputs
        else:
            return main_output
    
    def validate_dimensions(self, input_shape):
        """Validate that dimensions are compatible with the model architecture."""
        window_size = self.window_size
        if isinstance(window_size, int):
            window_size = [window_size] * 3
        
        # Check basic compatibility
        for i, dim in enumerate(input_shape):
            if dim % window_size[i] != 0:
                return False, f"Input dimension {i} (size {dim}) must be divisible by window_size {window_size[i]}"
            # Check compatibility with downscaling
            total_downscale = 1
            for dsf in self.dsf:
                total_downscale *= dsf
            if dim % total_downscale != 0:
                return False, f"Input dimension {i} (size {dim}) must be divisible by total downscale factor {total_downscale}"
        
        return True, "All dimensions are compatible"
    
    def _pad_to_compatible_size(self, x):
        """Pad input to make it compatible with the model architecture."""
        window_size = self.window_size
        if isinstance(window_size, int):
            window_size = [window_size] * 3
        
        b, c, d, h, w = x.shape
        
        # Calculate required padding for each dimension
        total_downscale = 1
        for dsf in self.dsf:
            total_downscale *= dsf
        
        target_d = ((d + total_downscale - 1) // total_downscale) * total_downscale
        target_h = ((h + total_downscale - 1) // total_downscale) * total_downscale
        target_w = ((w + total_downscale - 1) // total_downscale) * total_downscale
        
        # Further ensure compatibility with window size
        target_d = ((target_d + window_size[0] - 1) // window_size[0]) * window_size[0]
        target_h = ((target_h + window_size[1] - 1) // window_size[1]) * window_size[1]
        target_w = ((target_w + window_size[2] - 1) // window_size[2]) * window_size[2]
        
        if target_d != d or target_h != h or target_w != w:
            pad_d = target_d - d
            pad_h = target_h - h
            pad_w = target_w - w
            
            padding = (
                0, pad_w,     # W dimension
                0, pad_h,     # H dimension  
                0, pad_d      # D dimension
            )
            x = F.pad(x, padding, mode='constant', value=0)
            
        return x
    
    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def create_te_swinunet_s_3d(input_channels: int, num_classes: int, **kwargs):
    """
    Create a small TE-Swin UNet3D model compatible with nnUNet.
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        **kwargs: Additional arguments
        
    Returns:
        nnUNet_TE_SwinUnet3D: Initialized model
    """
    # Set default parameters, allowing kwargs to override
    default_params = {
        'hidden_dim': 96,
        'layers': (2, 2, 4, 2),
        'heads': (3, 6, 12, 24),  # Adjusted to ensure divisibility 
        'head_dim': 32,
        'window_size': 4,  # Smaller, more compatible window size
        'downscaling_factors': (2, 2, 2, 2),  # Smaller downscaling factors
    }
    
    # Update with any provided kwargs
    default_params.update(kwargs)
    
    return nnUNet_TE_SwinUnet3D(
        input_channels=input_channels,
        num_classes=num_classes,
        **default_params
    )


def create_te_swinunet_t_3d(input_channels: int, num_classes: int, **kwargs):
    """
    Create a tiny TE-Swin UNet3D model compatible with nnUNet.
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        **kwargs: Additional arguments
        
    Returns:
        nnUNet_TE_SwinUnet3D: Initialized model
    """
    # Set default parameters, allowing kwargs to override
    default_params = {
        'hidden_dim': 48,
        'layers': (2, 2, 2, 2),
        'heads': (3, 6, 12, 24),  # Adjusted to ensure divisibility
        'head_dim': 16,
        'window_size': 4,  # Smaller, more compatible window size
        'downscaling_factors': (2, 2, 2, 2),  # Smaller downscaling factors
    }
    
    # Update with any provided kwargs
    default_params.update(kwargs)
    
    return nnUNet_TE_SwinUnet3D(
        input_channels=input_channels,
        num_classes=num_classes,
        **default_params
    )


def create_te_swinunet_b_3d(input_channels: int, num_classes: int, **kwargs):
    """
    Create a base TE-Swin UNet3D model compatible with nnUNet.
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        **kwargs: Additional arguments
        
    Returns:
        nnUNet_TE_SwinUnet3D: Initialized model
    """
    # Set default parameters, allowing kwargs to override
    default_params = {
        'hidden_dim': 128,
        'layers': (2, 2, 8, 2),
        'heads': (4, 8, 16, 32),
        'head_dim': 32,
        'window_size': 7,  # Standard window size for base model
        'downscaling_factors': (4, 2, 2, 2),
    }
    
    # Update with any provided kwargs
    default_params.update(kwargs)
    
    return nnUNet_TE_SwinUnet3D(
        input_channels=input_channels,
        num_classes=num_classes,
        **default_params
    )
