"""
安全无循环引用的 nnUNet Trainer with TE-Swin UNet3D architecture.
主要修复：移除可能导致循环引用的代码，简化错误处理
"""
import torch
import torch.nn as nn
from typing import Tuple, Union, List
from tqdm import tqdm
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

# Import our custom model
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.te_swin_models.nnUNet_TE_SwinUnet3D import (
    create_te_swinunet_s_3d, 
    create_te_swinunet_t_3d, 
    create_te_swinunet_b_3d
)


class nnUNetTrainer_TE_SwinUnet3D(nnUNetTrainer):
    """
    安全版本的 nnUNet Trainer using TE-Swin UNet3D architecture.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        """Initialize TE-Swin UNet3D trainer."""
        
        # ✅ 立即禁用torch.compile，防止动态尺寸问题
        import os
        import torch._dynamo
        os.environ['NNUNET_COMPILE'] = '0'
        torch._dynamo.config.suppress_errors = True
        print("🔧 Disabled torch.compile for TE-Swin UNet3D compatibility")
        
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                        dataset_json=dataset_json, device=device)
        
        # Model variant - 使用tiny变体，参数更少，更稳定
        self.model_variant = 't'
        
    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        """网络架构构建方法"""
        self.print_to_log_file(f"Building TE-SwinUnet3D-{self.model_variant} variant")
        self.print_to_log_file(f"Input channels: {num_input_channels}, Output channels: {num_output_channels}")
        self.print_to_log_file(f"Deep supervision: {enable_deep_supervision}")
        
        # ✅ 使用兼容性优化的参数
        common_params = {
            'input_channels': num_input_channels,
            'num_classes': num_output_channels,
            'deep_supervision': enable_deep_supervision,
            'window_size': 4,  # 兼容性优化
            'downscaling_factors': (2, 2, 2, 2)  # 兼容性优化
        }
        
        if self.model_variant == 't':
            network = create_te_swinunet_t_3d(**common_params)
        elif self.model_variant == 'b':
            network = create_te_swinunet_b_3d(**common_params)
        else:  # Default to 's' (small) model
            network = create_te_swinunet_s_3d(**common_params)
            
        self.print_to_log_file(f"✅ Successfully created TE-SwinUnet3D-{self.model_variant}")
        self.print_to_log_file(f"Model parameters: {sum(p.numel() for p in network.parameters()):,}")
        
        # ✅ 简化的设备移动
        current_device = next(network.parameters()).device
        if current_device != self.device:
            self.print_to_log_file(f"Moving model from {current_device} to {self.device}")
            network = network.to(self.device)
            self.print_to_log_file(f"✅ Model moved to device: {self.device}")
        else:
            self.print_to_log_file(f"✅ Model already on correct device: {self.device}")
            
        # ✅ 验证nnUNet兼容性属性
        if hasattr(network, 'decoder'):
            self.print_to_log_file("✅ Model has decoder attribute for nnUNet compatibility")
        else:
            self.print_to_log_file("⚠️  Model missing decoder attribute")
        
        return network
    
    def initialize(self):
        """重写初始化方法，确保正确的设备管理"""
        # 调用父类初始化
        super().initialize()
        
        # ✅ 确保网络在正确设备上
        if hasattr(self, 'network') and self.network is not None:
            self.network = self.network.to(self.device)
            self.print_to_log_file(f"✅ Network confirmed on device: {next(self.network.parameters()).device}")
            
        # ✅ 设置网络为训练模式
        if hasattr(self, 'network'):
            self.network.train()
    
    def set_deep_supervision_enabled(self, enabled: bool):
        """深度监督设置方法"""
        # 获取实际的模型（处理DDP和compile包装）
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
            
        # 处理torch.compile包装
        if hasattr(mod, '_orig_mod'):
            mod = mod._orig_mod
        
        # ✅ 多种方式尝试设置深度监督
        success = False
        
        # 方式1：通过decoder设置（nnUNet标准方式）
        if hasattr(mod, 'decoder') and hasattr(mod.decoder, 'deep_supervision'):
            mod.decoder.deep_supervision = enabled
            self.print_to_log_file(f"✅ Set via decoder.deep_supervision: {enabled}")
            success = True
        
        # 方式2：直接设置模型属性
        if hasattr(mod, 'deep_supervision'):
            mod.deep_supervision = enabled
            self.print_to_log_file(f"✅ Set via model.deep_supervision: {enabled}")
            success = True
        
        # 方式3：设置do_ds标志
        if hasattr(mod, 'do_ds'):
            mod.do_ds = enabled
            self.print_to_log_file(f"✅ Set via model.do_ds: {enabled}")
            success = True
            
        if not success:
            self.print_to_log_file(f"⚠️  Could not set deep supervision - continuing anyway")
            
    def on_train_start(self):
        """训练开始时的设备检查"""
        # ✅ 基本的设备状态检查
        if hasattr(self, 'network') and self.network is not None:
            network_device = next(self.network.parameters()).device
            self.print_to_log_file(f"🔍 Network device: {network_device}")
            self.print_to_log_file(f"🔍 Expected device: {self.device}")
            
            if network_device != self.device:
                self.print_to_log_file(f"⚠️  Device mismatch detected!")
                    
        # ✅ GPU内存状态检查
        if torch.cuda.is_available():
            self.print_to_log_file(f"🔍 GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            self.print_to_log_file(f"🔍 GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        print("🚀 Starting TE-Swin UNet3D training with 'MRI as GIF' approach...")
        
        # ✅ 调用父类方法
        super().on_train_start()
            
    def train_step(self, batch):
        """修复的训练步骤 - 直接强制修复尺寸不匹配"""
        # ✅ 正确处理数据 - 可能是列表或张量
        data = batch['data']
        target = batch['target']
        
        # 处理数据设备移动
        if isinstance(data, list):
            data = [d.to(self.device, non_blocking=True) if hasattr(d, 'to') else d for d in data]
        else:
            if hasattr(data, 'device') and data.device != self.device:
                data = data.to(self.device, non_blocking=True)
                
        # 处理目标设备移动  
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) if hasattr(t, 'to') else t for t in target]
        else:
            if hasattr(target, 'device') and target.device != self.device:
                target = target.to(self.device, non_blocking=True)
        
        # ✅ 直接检查并强制修复尺寸不匹配问题
        if hasattr(data, 'shape') and hasattr(target, 'shape'):
            # 记录原始形状用于调试
            orig_data_shape = data.shape
            orig_target_shape = target.shape
            
            # 从错误信息看：data是[2, 3, 64, 192, 192], target是[2, 56, 192, 192]
            # 所以data的空间维度是[64, 192, 192], target的空间维度是[56, 192, 192]
            
            if len(data.shape) == 5 and len(target.shape) == 4:  # 常见情况
                data_spatial = data.shape[2:]  # [64, 192, 192]
                target_spatial = target.shape[1:]  # [56, 192, 192]
                
                if data_spatial != target_spatial:
                    self.print_to_log_file(f"🔧 FIXING size mismatch: data{data_spatial} vs target{target_spatial}")
                    
                    # 计算需要的padding - 只在target更小时padding
                    need_padding = False
                    pad_list = []
                    
                    for i in range(len(target_spatial)):
                        if data_spatial[i] > target_spatial[i]:
                            pad_amount = data_spatial[i] - target_spatial[i]
                            need_padding = True
                        else:
                            pad_amount = 0
                        pad_list.append(pad_amount)
                    
                    if need_padding:
                        # PyTorch padding: (W_pad, W_pad, H_pad, H_pad, D_pad, D_pad)
                        # 我们需要在尾部padding，所以格式是(0, W_pad, 0, H_pad, 0, D_pad)
                        pytorch_padding = []
                        for pad_val in reversed(pad_list):  # PyTorch要求从最后一个维度开始
                            pytorch_padding.extend([0, pad_val])
                        
                        self.print_to_log_file(f"🔧 Applying padding: {pytorch_padding}")
                        target = torch.nn.functional.pad(target, pytorch_padding, mode='constant', value=0)
                        self.print_to_log_file(f"🔧 Target shape after padding: {target.shape}")
                        
        # 最终检查
        if hasattr(data, 'shape') and hasattr(target, 'shape'):
            self.print_to_log_file(f"🔍 FINAL: data{data.shape} target{target.shape}")
                
        batch['data'] = data
        batch['target'] = target
        
        # 调用父类的训练步骤
        return super().train_step(batch)

    def validation_step(self, batch):
        """修复的验证步骤 - 正确处理数据格式"""
        # ✅ 正确处理数据 - 可能是列表或张量
        data = batch['data']
        target = batch['target']
        
        # 处理数据设备移动
        if isinstance(data, list):
            data = [d.to(self.device, non_blocking=True) if hasattr(d, 'to') else d for d in data]
        else:
            if hasattr(data, 'device') and data.device != self.device:
                data = data.to(self.device, non_blocking=True)
                
        # 处理目标设备移动
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) if hasattr(t, 'to') else t for t in target]
        else:
            if hasattr(target, 'device') and target.device != self.device:
                target = target.to(self.device, non_blocking=True)
                
        batch['data'] = data
        batch['target'] = target
        
        return super().validation_step(batch)


class nnUNetTrainer_TE_SwinUnet3D_tiny(nnUNetTrainer_TE_SwinUnet3D):
    """Tiny variant - 最安全的选择"""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                         dataset_json=dataset_json, device=device)
        self.model_variant = 't'


class nnUNetTrainer_TE_SwinUnet3D_small(nnUNetTrainer_TE_SwinUnet3D):
    """Small variant"""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                         dataset_json=dataset_json, device=device)
        self.model_variant = 's'


class nnUNetTrainer_TE_SwinUnet3D_base(nnUNetTrainer_TE_SwinUnet3D):
    """Base variant"""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                         dataset_json=dataset_json, device=device)
        self.model_variant = 'b'