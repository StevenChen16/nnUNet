import torch
import torch.nn as nn
import torch.nn.functional as F

class ShapeTextureFusion(nn.Module):
    """
    Shape-Texture Fusion module for TE-Swin UNet3D.
    
    This module intelligently combines shape information (spatial structure)
    and texture information (appearance patterns) streams using an adaptive
    fusion mechanism that learns the optimal weights for each component.
    """
    def __init__(self, dim):
        """
        Initialize the Shape-Texture Fusion module.
        
        Args:
            dim (int): Feature dimension (channel count)
        """
        super(ShapeTextureFusion, self).__init__()
        
        # Feature transformation for shape and texture streams
        self.shape_conv = nn.Conv3d(dim, dim, kernel_size=1)
        # Adaptive texture conversion to handle potential channel mismatch
        self.texture_adapt = True  # Flag to check if texture adapter is needed during forward pass
        self.texture_conv = nn.Conv3d(dim, dim, kernel_size=1)
        
        # Fusion weight generation network
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(dim*2, dim//2, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(dim//2, 2, kernel_size=1),  # 2 channels for shape and texture weights
            nn.Softmax(dim=1)  # Ensure weights sum to 1
        )
        
    def forward(self, shape_feat, texture_feat):
        """
        Forward pass through the Shape-Texture Fusion module.
        
        Args:
            shape_feat (Tensor): Feature map from shape stream [B, C, D, H, W]
            texture_feat (Tensor): Feature map from texture stream [B, C', D, H, W] (C' might differ from C)
            
        Returns:
            Tensor: Fused feature map [B, C, D, H, W]
        """
        # Get dimensions
        shape_channels = shape_feat.size(1)
        texture_channels = texture_feat.size(1)
        
        # Check for channel mismatch and create adaptive conv layer if needed
        if shape_channels != texture_channels:
            # Create an on-the-fly adapter to match texture channels to shape channels
            texture_adapter = nn.Conv3d(texture_channels, shape_channels, kernel_size=1).to(shape_feat.device)
            # Initialize with identity-like weights to minimize initial transformation
            nn.init.kaiming_normal_(texture_adapter.weight)
            if texture_adapter.bias is not None:
                nn.init.zeros_(texture_adapter.bias)
            # Adapt texture features to match shape feature channels
            texture_feat = texture_adapter(texture_feat)
            print(f"[ShapeTextureFusion] Adapted texture features from {texture_channels} to {shape_channels} channels")
        
        # Transform features with 1x1x1 convolutions
        shape_feat = self.shape_conv(shape_feat)
        texture_feat = self.texture_conv(texture_feat)
        
        # Concatenate features along channel dimension
        concat_feat = torch.cat([shape_feat, texture_feat], dim=1)
        
        # Generate fusion weights
        weights = self.fusion_conv(concat_feat)  # [B, 2, D, H, W]
        
        # Split weights for shape and texture
        alpha = weights[:, 0:1]  # Shape weight [B, 1, D, H, W]
        beta = weights[:, 1:2]   # Texture weight [B, 1, D, H, W]
        
        # Apply weighted fusion
        # Each weight is broadcast across all channels
        fused_feat = alpha * shape_feat + beta * texture_feat
        
        return fused_feat
