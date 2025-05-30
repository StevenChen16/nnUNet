"""
nnUNet-compatible TE-Swin UNet3D model implementation.
核心修复：解决输出尺寸不匹配问题，确保输出与target尺寸一致
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


class SimpleDecoderProxy:
    """
    简单的decoder代理类 - 避免循环引用
    不存储对parent模型的引用，而是通过传入的回调函数操作
    """
    def __init__(self, get_deep_supervision_func, set_deep_supervision_func):
        self._get_ds = get_deep_supervision_func
        self._set_ds = set_deep_supervision_func
        
    @property 
    def deep_supervision(self):
        """获取深度监督状态"""
        return self._get_ds()
        
    @deep_supervision.setter
    def deep_supervision(self, value):
        """设置深度监督状态"""
        self._set_ds(value)
        print(f"🔧 TE-Swin UNet3D deep supervision set to: {value}")


class nnUNet_TE_SwinUnet3D(nn.Module):
    """
    nnUNet-compatible Texture-Enhanced Swin UNet3D.
    核心修复：解决输出尺寸不匹配问题
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
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision  # 使用私有变量避免属性冲突
        self.dsf = downscaling_factors
        self.window_size = window_size
        
        # 确保layers都是偶数
        layers = tuple(max(2, (layer // 2) * 2) for layer in layers)
        print(f"🔧 Adjusted layers to ensure even numbers: {layers}")
        
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
        safe_heads = [max(1, h//4) for h in heads]  # 减少head数量避免问题
        self.temporal_attention_modules = nn.ModuleList([
            TemporalAttentionModule(dim=hidden_dim, num_heads=safe_heads[0]),
            TemporalAttentionModule(dim=hidden_dim * 2, num_heads=safe_heads[1]),
            TemporalAttentionModule(dim=hidden_dim * 4, num_heads=safe_heads[2]),
            TemporalAttentionModule(dim=hidden_dim * 8, num_heads=safe_heads[3])
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
        if self._deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                nn.Conv3d(hidden_dim * 8, num_classes, kernel_size=1),
                nn.Conv3d(hidden_dim * 4, num_classes, kernel_size=1),
                nn.Conv3d(hidden_dim * 2, num_classes, kernel_size=1),
                nn.Conv3d(hidden_dim, num_classes, kernel_size=1)
            ])
        
        # ✅ nnUNet兼容性：使用安全的代理模式，避免循环引用
        self.decoder = SimpleDecoderProxy(
            get_deep_supervision_func=lambda: self._deep_supervision,
            set_deep_supervision_func=self._set_deep_supervision
        )
        
        # ✅ 不要自引用！创建一个标记即可
        self._has_encoder = True  # 用标记代替self.encoder = self
        
        # ✅ 添加其他nnUNet可能需要的属性
        self.do_ds = deep_supervision
        
        # Initialize weights
        self.init_weights()
        
        print(f"✅ TE-Swin UNet3D initialized with nnUNet compatibility (no circular references)")
        print(f"   - Input channels: {input_channels}")
        print(f"   - Output classes: {num_classes}")  
        print(f"   - Deep supervision: {deep_supervision}")
        print(f"   - Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _set_deep_supervision(self, value):
        """安全的深度监督设置方法"""
        self._deep_supervision = value
        self.do_ds = value
        print(f"🔧 Deep supervision internally set to: {value}")
    
    @property
    def deep_supervision(self):
        """深度监督属性getter"""
        return self._deep_supervision
    
    @deep_supervision.setter  
    def deep_supervision(self, value):
        """深度监督属性setter"""
        self._set_deep_supervision(value)
        
    def forward(self, x):
        """
        Forward pass through the TE-Swin UNet3D.
        核心修复：确保输出尺寸与原始输入尺寸一致
        """
        # 记录原始输入尺寸
        original_shape = x.shape[2:]  # D, H, W
        b, c = x.shape[:2]
        
        print(f"🔍 DEBUG: Input shape: {x.shape}, Original spatial shape: {original_shape}")
        
        # 检查并padding输入以满足架构需求
        padded_x, padding_info = self._ensure_compatible_size(x)
        x = padded_x
        
        if padding_info['applied']:
            print(f"🔍 DEBUG: Applied padding, new shape: {x.shape}")
        
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
        
        print(f"🔍 DEBUG: After encoder, bottleneck shape: {x.shape}")
        
        # Create multi-scale texture pyramid
        texture_pyramid = self.texture_pyramid(texture_features)
        
        # Store decoder features for deep supervision
        decoder_features = []
        
        # Decoder pathway with shape-texture fusion
        
        # Stage 4 decoder
        x = self.dec4(x)
        shape_feat = encoder_features[2]  # Stage 4 features
        texture_feat = texture_pyramid[2]
        fused_feat = self.fusion_modules[2](shape_feat, texture_feat)
        x = self.converge4(x, fused_feat)
        decoder_features.append(x)
        print(f"🔍 DEBUG: After dec4, shape: {x.shape}")
        
        # Stage 3 decoder
        x = self.dec3(x)
        shape_feat = encoder_features[1]  # Stage 3 features
        texture_feat = texture_pyramid[1]
        fused_feat = self.fusion_modules[1](shape_feat, texture_feat)
        x = self.converge3(x, fused_feat)
        decoder_features.append(x)
        print(f"🔍 DEBUG: After dec3, shape: {x.shape}")
        
        # Stage 1-2 decoder
        x = self.dec12(x)
        shape_feat = encoder_features[0]  # Stage 1-2 features
        texture_feat = texture_pyramid[0]
        fused_feat = self.fusion_modules[0](shape_feat, texture_feat)
        x = self.converge12(x, fused_feat)
        decoder_features.append(x)
        print(f"🔍 DEBUG: After dec12, shape: {x.shape}")
        
        # Final upsampling and output head
        x = self.final(x)
        print(f"🔍 DEBUG: After final upsampling, shape: {x.shape}")
        
        main_output = self.seg_head(x)
        print(f"🔍 DEBUG: Main output shape before processing: {main_output.shape}")
        
        # 核心修复：确保主输出匹配原始输入尺寸
        if padding_info['applied']:
            main_output = self._crop_to_original(main_output, original_shape)
            print(f"🔍 DEBUG: Main output after cropping: {main_output.shape}")
        
        # 最终尺寸检查和调整
        expected_shape = (b, self.num_classes) + original_shape
        if main_output.shape != expected_shape:
            print(f"🔍 DEBUG: Main output shape mismatch, interpolating...")
            main_output = F.interpolate(main_output, size=original_shape, mode='trilinear', align_corners=False)
            print(f"🔍 DEBUG: Main output after final interpolation: {main_output.shape}")
        
        # 处理深度监督输出（如果启用）
        if self._deep_supervision and self.training:
            print("🔍 DEBUG: Processing deep supervision outputs...")
            
            deep_supervision_outputs = []
            
            # 使用decoder features和bottleneck feature来生成深度监督输出
            ds_source_features = [
                encoder_features[3],  # bottleneck (最低分辨率)
                decoder_features[0],  # dec4 output
                decoder_features[1],  # dec3 output  
                decoder_features[2],  # dec12 output
            ]
            
            for i, feat in enumerate(ds_source_features):
                ds_out = self.deep_supervision_heads[i](feat)
                print(f"🔍 DEBUG: DS output {i} before interpolation: {ds_out.shape}")
                
                # 先裁剪（如果需要）
                if padding_info['applied']:
                    ds_out = self._crop_to_original(ds_out, original_shape)
                    print(f"🔍 DEBUG: DS output {i} after cropping: {ds_out.shape}")
                
                # 然后插值到原始尺寸
                if ds_out.shape[2:] != original_shape:
                    ds_out = F.interpolate(ds_out, size=original_shape, mode='trilinear', align_corners=False)
                    print(f"🔍 DEBUG: DS output {i} after interpolation: {ds_out.shape}")
                
                deep_supervision_outputs.append(ds_out)
            
            print(f"🔍 DEBUG: Returning {len(deep_supervision_outputs) + 1} outputs for deep supervision")
            for i, ds_out in enumerate(deep_supervision_outputs):
                print(f"🔍 DEBUG: Final DS output {i} shape: {ds_out.shape}")
            
            # Return list of outputs for deep supervision
            all_outputs = [main_output] + deep_supervision_outputs
            return all_outputs
        else:
            print(f"🔍 DEBUG: Returning single output: {main_output.shape}")
            return main_output
    
    def _ensure_compatible_size(self, x):
        """
        确保输入尺寸与模型架构兼容，返回padding后的tensor和padding信息
        """
        b, c, d, h, w = x.shape
        original_size = (d, h, w)
        
        window_size = self.window_size
        if isinstance(window_size, int):
            window_size = [window_size] * 3
        
        # 计算总的下采样倍数
        total_downscale = 1
        for dsf in self.dsf:
            total_downscale *= dsf
        
        # 计算目标尺寸
        target_d = ((d + total_downscale - 1) // total_downscale) * total_downscale
        target_h = ((h + total_downscale - 1) // total_downscale) * total_downscale  
        target_w = ((w + total_downscale - 1) // total_downscale) * total_downscale
        
        # 确保与window_size兼容
        target_d = ((target_d + window_size[0] - 1) // window_size[0]) * window_size[0]
        target_h = ((target_h + window_size[1] - 1) // window_size[1]) * window_size[1]
        target_w = ((target_w + window_size[2] - 1) // window_size[2]) * window_size[2]
        
        padding_info = {
            'applied': False,
            'original_size': original_size,
            'target_size': (target_d, target_h, target_w)
        }
        
        if target_d != d or target_h != h or target_w != w:
            # 需要padding
            pad_d = target_d - d
            pad_h = target_h - h  
            pad_w = target_w - w
            
            # F.pad的参数顺序是从最后一个维度开始：(left, right, top, bottom, front, back)
            padding = (0, pad_w, 0, pad_h, 0, pad_d)
            x_padded = F.pad(x, padding, mode='constant', value=0)
            
            padding_info.update({
                'applied': True,
                'padding': (pad_d, pad_h, pad_w)
            })
            
            return x_padded, padding_info
        else:
            return x, padding_info
    
    def _crop_to_original(self, output, original_shape):
        """
        将输出裁剪回原始尺寸
        """
        d_orig, h_orig, w_orig = original_shape
        return output[:, :, :d_orig, :h_orig, :w_orig]
    
    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# 模型创建函数
def create_te_swinunet_s_3d(input_channels: int, num_classes: int, **kwargs):
    """Create a small TE-Swin UNet3D model compatible with nnUNet."""
    default_params = {
        'hidden_dim': 96,
        'layers': (2, 2, 4, 2),
        'heads': (3, 6, 12, 24),
        'head_dim': 32,
        'window_size': 4,  # 兼容性设置
        'downscaling_factors': (2, 2, 2, 2),  # 兼容性设置
    }
    
    default_params.update(kwargs)
    
    return nnUNet_TE_SwinUnet3D(
        input_channels=input_channels,
        num_classes=num_classes,
        **default_params
    )


def create_te_swinunet_t_3d(input_channels: int, num_classes: int, **kwargs):
    """Create a tiny TE-Swin UNet3D model compatible with nnUNet."""
    default_params = {
        'hidden_dim': 48,
        'layers': (2, 2, 2, 2),  # 全部偶数
        'heads': (3, 6, 12, 24),
        'head_dim': 16,
        'window_size': 4,  # 兼容性设置
        'downscaling_factors': (2, 2, 2, 2),  # 兼容性设置
    }
    
    default_params.update(kwargs)
    
    return nnUNet_TE_SwinUnet3D(
        input_channels=input_channels,
        num_classes=num_classes,
        **default_params
    )


def create_te_swinunet_b_3d(input_channels: int, num_classes: int, **kwargs):
    """Create a base TE-Swin UNet3D model compatible with nnUNet."""
    default_params = {
        'hidden_dim': 128,
        'layers': (2, 2, 8, 2),  # 确保全部偶数
        'heads': (4, 8, 16, 32),
        'head_dim': 32,
        'window_size': 4,  # 兼容性设置  
        'downscaling_factors': (2, 2, 2, 2),  # 兼容性设置
    }
    
    default_params.update(kwargs)
    
    return nnUNet_TE_SwinUnet3D(
        input_channels=input_channels,
        num_classes=num_classes,
        **default_params
    )