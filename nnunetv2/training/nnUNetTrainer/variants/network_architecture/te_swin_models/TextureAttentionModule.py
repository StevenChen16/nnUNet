import torch
import torch.nn as nn
import torch.nn.functional as F

class TextureAttentionModule(nn.Module):
    """
    Texture-Aware Attention Module (TAAM) - a core component of TE-Swin UNet3D.
    
    This module is responsible for:
    1. Extracting texture features from input feature maps
    2. Generating spatial and channel attention maps based on texture features
    3. Applying these attention maps to enhance texture-relevant information
    """
    def __init__(self, dim):
        """
        Initialize the Texture Attention Module.
        
        Args:
            dim (int): Input feature dimension (channel count)
        """
        super(TextureAttentionModule, self).__init__()
        
        # Store the expected input dimension for later checking
        self.expected_dim = dim
        
        # Texture feature extractor - specialized convolution layers
        # Uses dilated convolutions to capture larger texture patterns
        self.texture_extractor = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, dilation=1),
            nn.InstanceNorm3d(dim),
            nn.LeakyReLU(0.2),
            nn.Conv3d(dim, dim, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm3d(dim),
            nn.LeakyReLU(0.2)
        )
        
        # Spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(dim, max(dim//4, 1), kernel_size=3, padding=1),  # Ensure at least 1 channel
            nn.LeakyReLU(0.2),
            nn.Conv3d(max(dim//4, 1), 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, max(dim//4, 1), kernel_size=1),  # Ensure at least 1 channel
            nn.LeakyReLU(0.2),
            nn.Conv3d(max(dim//4, 1), dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Flag to track if we've created an adapter
        self.has_adapter = False
        # Will be created on-the-fly if needed
        self.input_adapter = None
        
    def forward(self, x):
        """
        Forward pass through the Texture Attention Module.
        
        Args:
            x (Tensor): Input feature map [B, C, D, H, W]
            
        Returns:
            Tensor: Texture-enhanced feature map with same shape as input
        """
        original_x = x
        input_channels = x.size(1)
        
        # Check if the input dimensions match what we expect
        if input_channels != self.expected_dim:
            # If this is the first time we're seeing this mismatch, create an adapter
            if not self.has_adapter:
                self.has_adapter = True
                self.input_adapter = nn.Conv3d(
                    input_channels, self.expected_dim, kernel_size=1,
                ).to(x.device)
                
                # Initialize with identity-like weights
                nn.init.kaiming_normal_(self.input_adapter.weight)
                if self.input_adapter.bias is not None:
                    nn.init.zeros_(self.input_adapter.bias)
                
                print(f"[TextureAttentionModule] Created input adapter: {input_channels} -> {self.expected_dim} channels")
            
            # Apply the adapter to match expected channels
            x = self.input_adapter(x)
        
        # Extract texture features
        texture_feat = self.texture_extractor(x)
        
        # Generate spatial attention map
        spatial_attn = self.spatial_attention(texture_feat)
        
        # Generate channel attention map
        channel_attn = self.channel_attention(texture_feat)
        
        # Apply attention - combine spatial and channel attention
        # Spatial attention is broadcast across channels
        # Channel attention is broadcast across spatial dimensions
        attn = spatial_attn * channel_attn
        
        # If we used an adapter, we need to convert back to the original channels
        if input_channels != self.expected_dim:
            # Create an output adapter on-the-fly
            if not hasattr(self, 'output_adapter') or self.output_adapter is None:
                self.output_adapter = nn.Conv3d(
                    self.expected_dim, input_channels, kernel_size=1,
                ).to(x.device)
                
                # Initialize with identity-like weights
                nn.init.kaiming_normal_(self.output_adapter.weight)
                if self.output_adapter.bias is not None:
                    nn.init.zeros_(self.output_adapter.bias)
                
                print(f"[TextureAttentionModule] Created output adapter: {self.expected_dim} -> {input_channels} channels")
            
            # Apply attention to feature map
            enhanced = x * attn
            # Convert back to original channel dimension
            enhanced = self.output_adapter(enhanced)
            # Residual connection with original input
            out = enhanced + original_x
        else:
            # Standard residual connection - add enhanced features to original
            out = x * attn + original_x
        
        return out
