import torch
import torch.nn as nn
import torch.nn.functional as F

class TextureAttentionModule(nn.Module):
    """
    修复后的纹理注意力模块 - 解决动态模块创建和设备不一致问题
    """
    def __init__(self, dim, adaptive_channels=None):
        """
        Initialize the Texture Attention Module.
        
        Args:
            dim (int): 预期的输入特征维度
            adaptive_channels (list, optional): 可能的输入通道数列表，用于预创建适配器
        """
        super(TextureAttentionModule, self).__init__()
        
        self.expected_dim = dim
        
        # ✅ 纹理特征提取器 - 使用更稳定的架构
        self.texture_extractor = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, dilation=1),
            nn.GroupNorm(min(8, dim), dim),  # 使用GroupNorm代替InstanceNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(dim, dim, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(min(8, dim), dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # ✅ 空间注意力机制
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(dim, max(dim//4, 1), kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(max(dim//4, 1), 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # ✅ 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, max(dim//4, 1), kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(max(dim//4, 1), dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # ✅ 预创建可能需要的适配器，避免在forward中动态创建
        self.input_adapters = nn.ModuleDict()
        self.output_adapters = nn.ModuleDict()
        
        # 如果提供了可能的通道数，预创建适配器
        if adaptive_channels is not None:
            for channels in adaptive_channels:
                if channels != dim:
                    # 输入适配器
                    self.input_adapters[str(channels)] = nn.Conv3d(
                        channels, dim, kernel_size=1, bias=False
                    )
                    # 输出适配器
                    self.output_adapters[str(channels)] = nn.Conv3d(
                        dim, channels, kernel_size=1, bias=False
                    )
                    
                    # ✅ 使用更好的初始化
                    nn.init.kaiming_normal_(self.input_adapters[str(channels)].weight)
                    nn.init.kaiming_normal_(self.output_adapters[str(channels)].weight)
        
        # ✅ 残差连接的权重
        self.residual_weight = nn.Parameter(torch.ones(1))
        
    def _get_or_create_adapter(self, channels, adapter_type='input'):
        """
        获取或创建适配器，但仅在必要时创建
        """
        adapter_dict = self.input_adapters if adapter_type == 'input' else self.output_adapters
        key = str(channels)
        
        if key not in adapter_dict:
            # 只有在预创建的适配器中没有找到时才创建新的
            if adapter_type == 'input':
                adapter = nn.Conv3d(channels, self.expected_dim, kernel_size=1, bias=False)
                target_dim = self.expected_dim
            else:
                adapter = nn.Conv3d(self.expected_dim, channels, kernel_size=1, bias=False)
                target_dim = channels
                
            # ✅ 使用更好的初始化
            nn.init.kaiming_normal_(adapter.weight)
            
            # ✅ 确保适配器在正确的设备上
            device = next(self.parameters()).device
            adapter = adapter.to(device)
            
            adapter_dict[key] = adapter
            
        return adapter_dict[key]
        
    def forward(self, x):
        """
        修复后的前向传播，避免动态模块创建
        """
        original_x = x
        input_channels = x.size(1)
        
        # ✅ 如果通道数不匹配，使用适配器
        if input_channels != self.expected_dim:
            input_adapter = self._get_or_create_adapter(input_channels, 'input')
            x = input_adapter(x)
        
        # 提取纹理特征
        texture_feat = self.texture_extractor(x)
        
        # 生成注意力图
        spatial_attn = self.spatial_attention(texture_feat)
        channel_attn = self.channel_attention(texture_feat)
        
        # ✅ 组合注意力 - 使用更稳定的方式
        combined_attn = spatial_attn * channel_attn
        enhanced_feat = x * combined_attn
        
        # ✅ 如果使用了输入适配器，需要输出适配器转换回原始维度
        if input_channels != self.expected_dim:
            output_adapter = self._get_or_create_adapter(input_channels, 'output')
            enhanced_feat = output_adapter(enhanced_feat)
        
        # ✅ 带权重的残差连接
        output = enhanced_feat + self.residual_weight * original_x
        
        return output
    
    def extra_repr(self):
        """提供模块的额外表示信息"""
        return f'expected_dim={self.expected_dim}, adapters={len(self.input_adapters)}'


class MultiScaleTextureAttention(nn.Module):
    """
    多尺度纹理注意力模块 - 用于处理不同尺度的纹理特征
    """
    def __init__(self, dims):
        """
        初始化多尺度纹理注意力模块
        
        Args:
            dims (list): 各个尺度的特征维度
        """
        super().__init__()
        
        # ✅ 为每个尺度创建纹理注意力模块
        self.texture_modules = nn.ModuleList([
            TextureAttentionModule(dim) for dim in dims
        ])
        
    def forward(self, features):
        """
        对多尺度特征应用纹理注意力
        
        Args:
            features (list): 多尺度特征列表
            
        Returns:
            list: 增强后的多尺度特征
        """
        enhanced_features = []
        
        for i, (feat, texture_module) in enumerate(zip(features, self.texture_modules)):
            try:
                enhanced = texture_module(feat)
                enhanced_features.append(enhanced)
            except Exception as e:
                print(f"⚠️  TextureAttention failed at scale {i}: {e}")
                # 如果失败，使用原始特征
                enhanced_features.append(feat)
                
        return enhanced_features


# ✅ 工厂函数，用于创建预配置的纹理注意力模块
def create_texture_attention_for_te_swin(variant='small'):
    """
    为TE-Swin模型创建预配置的纹理注意力模块
    
    Args:
        variant (str): 'tiny', 'small', 'base'
        
    Returns:
        list: 纹理注意力模块列表
    """
    if variant == 'tiny':
        dims = [48, 96, 192, 384]
    elif variant == 'base':
        dims = [128, 256, 512, 1024]
    else:  # small
        dims = [96, 192, 384, 768]
    
    # 为每个层级创建纹理注意力模块
    modules = []
    for dim in dims:
        # 预定义可能的通道数，避免动态创建
        possible_channels = [dim//2, dim, dim*2]
        module = TextureAttentionModule(dim, adaptive_channels=possible_channels)
        modules.append(module)
    
    return modules