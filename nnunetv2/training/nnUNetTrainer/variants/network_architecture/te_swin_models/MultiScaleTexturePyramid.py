import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleTexturePyramid(nn.Module):
    """
    Multi-scale Texture Pyramid (MTP) module for TE-Swin UNet3D.
    
    This module captures texture features at different scales in a top-down pathway,
    similar to Feature Pyramid Networks (FPN) but specialized for texture extraction.
    It enables the network to handle tumors of different sizes effectively.
    """
    def __init__(self, dims, debug_output=False):
        """
        Initialize the Multi-scale Texture Pyramid module.
        
        Args:
            dims (list): List of dimensions for each level of the feature pyramid
            debug_output (bool, optional): Whether to print debug information. Defaults to False.
        """
        super(MultiScaleTexturePyramid, self).__init__()
        
        self.dims = dims
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        # Debug flag to control print statements
        self.debug_output = debug_output
        
        # Store the expected output dimensions for each level
        self.expected_output_dims = dims.copy()
        
        # Use a fixed channel dimension for FPN internal processing
        # This is crucial for compatibility between different model variants
        self.fpn_channels = dims[0]  # Use the first dimension as reference
        
        # Lateral convolutions - transform each level's features to the same channel dimension
        self.lateral_convs = nn.ModuleList([
            nn.Conv3d(dim, self.fpn_channels, kernel_size=1)
            for dim in dims
        ])
        
        # Refinement convolutions - smooth the combined features and output to same dimension
        self.smooth_convs = nn.ModuleList([
            nn.Conv3d(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)
            for _ in dims
        ])
        
        # Output projections to transform back to the original dimensions if needed
        self.output_projs = nn.ModuleList([
            nn.Conv3d(self.fpn_channels, dim, kernel_size=1) if dim != self.fpn_channels else nn.Identity()
            for dim in dims
        ])
        
        # Dynamic adapters to handle variant-specific dimensions
        # These will be created on-the-fly if needed
        
    def forward(self, features):
        """
        Forward pass through the Multi-scale Texture Pyramid.
        
        Args:
            features (list): List of feature maps from the encoder at different scales
                            [level 0 -> level n-1], where level 0 is the lowest level
                            
        Returns:
            list: Pyramidal texture features at different scales
                 [level 0 -> level n-1]
        """
        # Debugging information
        if self.debug_output:
            print(f"[MultiScaleTexturePyramid] Processing {len(features)} feature levels")
            for i, feat in enumerate(features):
                print(f"[MultiScaleTexturePyramid] Level {i} input shape: {feat.shape}")
        
        # Check if the number of feature levels matches what we expected
        if len(features) != len(self.dims):
            if self.debug_output:
                print(f"Warning: Expected {len(self.dims)} feature levels but got {len(features)}")
        
        # Check channel dimensions and create level-specific lateral and smooth convs if needed
        for i, feat in enumerate(features):
            actual_dim = feat.size(1)
            expected_dim = self.dims[i] if i < len(self.dims) else actual_dim
            
            # Update lateral convs if needed
            lateral_key = f"dynamic_lateral_{i}"
            if actual_dim != expected_dim and not hasattr(self, lateral_key):
                setattr(self, lateral_key, nn.Conv3d(
                    actual_dim, self.dims[0], kernel_size=1 # All lateral convs output dims[0] channels
                ).to(feat.device))
                # Initialize with proper weights
                getattr(self, lateral_key).weight.data.normal_(0, 0.01)
                if getattr(self, lateral_key).bias is not None:
                    getattr(self, lateral_key).bias.data.zero_()
                if self.debug_output:
                    print(f"[MultiScaleTexturePyramid] Created dynamic lateral conv for level {i}: {actual_dim} -> {self.dims[0]} channels")
        
        # Top-down pathway (from deepest to shallowest)
        pyramid_features = []
        
        # Start with the deepest (highest semantic) level
        top_level_idx = len(features) - 1
        
        # Use dynamic lateral conv if available
        lateral_key = f"dynamic_lateral_{top_level_idx}"
        if hasattr(self, lateral_key):
            top_feature = getattr(self, lateral_key)(features[top_level_idx])
        else:
            top_feature = self.lateral_convs[top_level_idx](features[top_level_idx])
            
        # Process through smooth conv
        top_processed = self.smooth_convs[top_level_idx](top_feature)
        
        # Apply output projection for the top level
        if top_level_idx < len(self.output_projs):
            top_processed = self.output_projs[top_level_idx](top_processed)
        
        # Add to pyramid
        pyramid_features.append(top_processed)
        if self.debug_output:
            print(f"[MultiScaleTexturePyramid] Top level output shape: {top_processed.shape}")
        
        # Process other levels in top-down order
        for i in range(len(features)-2, -1, -1):
            # Upsample higher-level features
            upsampled = self.upsample(pyramid_features[0])
            if self.debug_output:
                print(f"[MultiScaleTexturePyramid] Level {i} upsampled shape: {upsampled.shape}")
            
            # Get lateral features from the current level - use dynamic if available
            lateral_key = f"dynamic_lateral_{i}"
            if hasattr(self, lateral_key):
                lateral = getattr(self, lateral_key)(features[i])
            else:
                lateral = self.lateral_convs[i](features[i])
                
            if self.debug_output:
                print(f"[MultiScaleTexturePyramid] Level {i} lateral shape: {lateral.shape}")
            
            # Check if channel dimensions match for addition
            if upsampled.size(1) != lateral.size(1):
                if self.debug_output:
                    print(f"[MultiScaleTexturePyramid] Channel mismatch: upsampled={upsampled.size(1)}, lateral={lateral.size(1)}")
                # Create adapter to match upsampled to lateral
                adapter_key = f"upsampled_adapter_{i}"
                if not hasattr(self, adapter_key):
                    setattr(self, adapter_key, nn.Conv3d(
                        upsampled.size(1), lateral.size(1), kernel_size=1
                    ).to(upsampled.device))
                    # Initialize adapter
                    getattr(self, adapter_key).weight.data.normal_(0, 0.01)
                    if getattr(self, adapter_key).bias is not None:
                        getattr(self, adapter_key).bias.data.zero_()
                    if self.debug_output:
                        print(f"[MultiScaleTexturePyramid] Created upsampled adapter: {upsampled.size(1)} -> {lateral.size(1)} channels")
                
                # Apply adapter
                upsampled = getattr(self, adapter_key)(upsampled)
            
            # Combine features
            fused = upsampled + lateral
            
            # Smooth the combined features
            smooth = self.smooth_convs[i](fused)
            if self.debug_output:
                print(f"[MultiScaleTexturePyramid] Level {i} output shape: {smooth.shape}")
            
            # Apply output projection to transform back to the expected dimensions
            if i < len(self.output_projs):
                smooth = self.output_projs[i](smooth)
                if self.debug_output:
                    print(f"[MultiScaleTexturePyramid] Level {i} after projection: {smooth.shape}")
            
            # Add to the front of the pyramid features list 
            # (so they're ordered from lowest to highest resolution)
            pyramid_features.insert(0, smooth)
        
        return pyramid_features
