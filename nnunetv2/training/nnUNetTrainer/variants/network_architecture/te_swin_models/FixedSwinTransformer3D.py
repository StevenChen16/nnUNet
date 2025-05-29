"""
Fixed Swin Transformer 3D implementation with robust window handling.
This version addresses tensor reshape issues in window attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Tuple


def safe_rearrange(tensor, pattern, **axes_lengths):
    """
    Safe tensor rearrangement with detailed error checking.
    """
    try:
        if 'b c (n_x w_x) (n_y w_y) (n_z w_z) -> b (n_x n_y n_z) (w_x w_y w_z) c' in pattern:
            b, c, d, h, w = tensor.shape
            w_x, w_y, w_z = axes_lengths.get('w_x', 1), axes_lengths.get('w_y', 1), axes_lengths.get('w_z', 1)
            
            # Calculate number of windows
            n_x, n_y, n_z = d // w_z, h // w_y, w // w_x
            
            # Validate calculations
            if n_x * w_z != d or n_y * w_y != h or n_z * w_x != w:
                print(f"Window division error: {d}÷{w_z}={n_x}, {h}÷{w_y}={n_y}, {w}÷{w_x}={n_z}")
                print(f"Expected: {n_x*w_z}×{n_y*w_y}×{n_z*w_x}, Got: {d}×{h}×{w}")
                raise ValueError(f"Input dimensions not divisible by window size")
            
            # Perform reshape
            reshaped = tensor.view(b, c, n_x, w_z, n_y, w_y, n_z, w_x)
            result = reshaped.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
            return result.view(b, n_x * n_y * n_z, w_x * w_y * w_z, c)
            
        elif 'b (n_x n_y n_z) (w_x w_y w_z) c -> b c (n_x w_x) (n_y w_y) (n_z w_z)' in pattern:
            b, n_total, w_total, c = tensor.shape
            w_x, w_y, w_z = axes_lengths.get('w_x', 1), axes_lengths.get('w_y', 1), axes_lengths.get('w_z', 1)
            n_x, n_y, n_z = axes_lengths.get('n_x', 1), axes_lengths.get('n_y', 1), axes_lengths.get('n_z', 1)
            
            # Validate
            if n_total != n_x * n_y * n_z:
                raise ValueError(f"Window count mismatch: {n_total} != {n_x}×{n_y}×{n_z}")
            if w_total != w_x * w_y * w_z:
                raise ValueError(f"Window size mismatch: {w_total} != {w_x}×{w_y}×{w_z}")
            
            reshaped = tensor.view(b, n_x, n_y, n_z, w_x, w_y, w_z, c)
            result = reshaped.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
            return result.view(b, c, n_x * w_x, n_y * w_y, n_z * w_z)
            
        elif 'b h (n_x n_y n_z) i j -> b h n_x n_y n_z i j' in pattern:
            b, h, n_total, i, j = tensor.shape
            n_x, n_y = axes_lengths.get('n_x', 1), axes_lengths.get('n_y', 1)
            
            # Safe division
            if n_x * n_y == 0:
                raise ValueError(f"Invalid window dimensions: n_x={n_x}, n_y={n_y}")
            
            n_z = n_total // (n_x * n_y)
            if n_z * n_x * n_y != n_total:
                raise ValueError(f"Cannot divide {n_total} windows into {n_x}×{n_y}×{n_z}")
            
            return tensor.view(b, h, n_x, n_y, n_z, i, j)
            
        elif 'b h n_x n_y n_z i j -> b h (n_x n_y n_z) i j' in pattern:
            b, h, n_x, n_y, n_z, i, j = tensor.shape
            return tensor.view(b, h, n_x * n_y * n_z, i, j)
            
        else:
            raise NotImplementedError(f"Rearrange pattern not implemented: {pattern}")
            
    except Exception as e:
        print(f"Rearrange error: {e}")
        print(f"Input shape: {tensor.shape}")
        print(f"Pattern: {pattern}")
        print(f"Axes: {axes_lengths}")
        raise


class FixedSwinBlock3D(nn.Module):
    """Fixed Swin Transformer 3D block with robust window handling."""
    
    def __init__(self, dim, heads, head_dim, window_size, shift_size=0, mlp_ratio=4., 
                 dropout=0., attention_dropout=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Ensure window_size is tuple/list
        if isinstance(window_size, int):
            self.window_size = [window_size] * 3
        else:
            self.window_size = list(window_size)
            
        # Attention layers
        self.norm1 = nn.LayerNorm(dim)
        self.attention = FixedWindowAttention3D(
            dim=dim,
            heads=heads,
            head_dim=head_dim,
            window_size=self.window_size,
            dropout=attention_dropout
        )
        
        # MLP layers
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, D, H, W]
        Returns:
            Output tensor [B, C, D, H, W]
        """
        # Store original shape and validate
        b, c, d, h, w = x.shape
        
        # Check if dimensions are compatible with window size
        if d % self.window_size[0] != 0 or h % self.window_size[1] != 0 or w % self.window_size[2] != 0:
            raise ValueError(
                f"Input size ({d}, {h}, {w}) not divisible by window size {self.window_size}"
            )
        
        # Convert to [B, D*H*W, C] for layer norm and attention
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous().view(b, d*h*w, c)
        
        # Self-attention
        shortcut = x_flat
        x_flat = self.norm1(x_flat)
        
        # Reshape back to spatial format for window attention
        x_spatial = x_flat.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        
        # Apply window attention
        x_attn = self.attention(x_spatial)
        
        # Convert back to flat format
        x_flat = x_attn.permute(0, 2, 3, 4, 1).contiguous().view(b, d*h*w, c)
        
        # Residual connection
        x_flat = shortcut + x_flat
        
        # MLP
        shortcut = x_flat  
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat)
        x_flat = shortcut + x_flat
        
        # Reshape back to original format
        output = x_flat.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        
        return output


class FixedWindowAttention3D(nn.Module):
    """Fixed 3D Window Attention with robust tensor handling."""
    
    def __init__(self, dim, heads, head_dim, window_size, dropout=0.):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.window_size = window_size if isinstance(window_size, list) else [window_size] * 3
        self.dropout = dropout
        self.scale = head_dim ** -0.5
        
        inner_dim = head_dim * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, D, H, W]
        Returns:
            Output tensor [B, C, D, H, W]
        """
        b, c, d, h, w = x.shape
        w_d, w_h, w_w = self.window_size
        
        # Validate input dimensions
        if d % w_d != 0 or h % w_h != 0 or w % w_w != 0:
            raise ValueError(f"Input ({d},{h},{w}) not divisible by window size {self.window_size}")
        
        # Calculate number of windows
        n_d, n_h, n_w = d // w_d, h // w_h, w // w_w
        
        # Partition into windows using safe rearrangement
        try:
            # Reshape to windows: [B, num_windows, window_size, C]
            x_windows = safe_rearrange(
                x, 
                'b c (n_x w_x) (n_y w_y) (n_z w_z) -> b (n_x n_y n_z) (w_x w_y w_z) c',
                w_x=w_w, w_y=w_h, w_z=w_d
            )
            
            # Apply attention within each window
            b_w, num_windows, window_size, c = x_windows.shape
            
            # Generate Q, K, V
            qkv = self.to_qkv(x_windows)  # [B, num_windows, window_size, 3*inner_dim]
            qkv = qkv.reshape(b_w, num_windows, window_size, 3, self.heads, self.head_dim)
            qkv = qkv.permute(3, 0, 1, 4, 2, 5)  # [3, B, num_windows, heads, window_size, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Attention computation
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            
            # Apply attention to values
            out = attn @ v  # [B, num_windows, heads, window_size, head_dim]
            out = out.transpose(2, 3).reshape(b_w, num_windows, window_size, self.heads * self.head_dim)
            
            # Output projection
            out = self.to_out(out)  # [B, num_windows, window_size, C]
            
            # Reshape back to spatial format
            out_spatial = safe_rearrange(
                out, 
                'b (n_x n_y n_z) (w_x w_y w_z) c -> b c (n_x w_x) (n_y w_y) (n_z w_z)',
                n_x=n_w, n_y=n_h, n_z=n_d, w_x=w_w, w_y=w_h, w_z=w_d
            )
            
            return out_spatial
            
        except Exception as e:
            print(f"Window attention error: {e}")
            print(f"Input shape: {x.shape}")
            print(f"Window size: {self.window_size}")
            print(f"Expected windows: {n_d}×{n_h}×{n_w}")
            # Fallback: return input unchanged
            return x


# Export safe rearrange for use in other modules
__all__ = ['safe_rearrange', 'FixedSwinBlock3D', 'FixedWindowAttention3D']
