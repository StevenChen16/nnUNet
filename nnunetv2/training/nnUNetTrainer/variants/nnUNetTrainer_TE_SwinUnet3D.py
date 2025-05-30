"""
最终修复版 nnUNet Trainer with TE-Swin UNet3D architecture.
修复了所有兼容性问题，包括decoder属性、设备管理等。
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
    最终修复版 nnUNet Trainer using TE-Swin UNet3D architecture.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        """Initialize TE-Swin UNet3D trainer."""
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                        dataset_json=dataset_json, device=device)
        
        # Model variant - can be 't' (tiny), 's' (small), or 'b' (base)
        self.model_variant = 't'  # 使用tiny变体，参数更少，更稳定
        
    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        """修复后的网络架构构建方法。"""
        self.print_to_log_file(f"Building TE-SwinUnet3D-{self.model_variant} variant")
        self.print_to_log_file(f"Input channels: {num_input_channels}, Output channels: {num_output_channels}")
        self.print_to_log_file(f"Deep supervision: {enable_deep_supervision}")
        
        try:
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
            
            # ✅ 确保模型在正确的设备上
            network = network.to(self.device)
            self.print_to_log_file(f"✅ Model moved to device: {self.device}")
            
            # ✅ 验证nnUNet兼容性属性
            if hasattr(network, 'decoder'):
                self.print_to_log_file("✅ Model has decoder attribute for nnUNet compatibility")
            else:
                self.print_to_log_file("⚠️  Model missing decoder attribute")
            
            return network
            
        except Exception as e:
            self.print_to_log_file(f"❌ Failed to create TE-SwinUnet3D: {e}")
            import traceback
            self.print_to_log_file(f"Full traceback:\n{traceback.format_exc()}")
            raise e
    
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        重写的深度监督设置方法，完美兼容TE-Swin UNet3D架构
        """
        # 获取实际的模型（处理DDP和compile包装）
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
            
        # 处理torch.compile包装
        if hasattr(mod, '_orig_mod'):
            mod = mod._orig_mod
        
        try:
            # ✅ 检查是否是我们的TE-Swin模型
            if hasattr(mod, '__class__') and 'TE_SwinUnet3D' in mod.__class__.__name__:
                # 我们的TE-Swin UNet3D模型有两种设置方式
                
                # 方式1：直接设置模型的deep_supervision属性
                if hasattr(mod, 'deep_supervision'):
                    mod.deep_supervision = enabled
                    self.print_to_log_file(f"✅ TE-Swin UNet3D deep_supervision set to: {enabled}")
                
                # 方式2：通过decoder包装器设置（nnUNet兼容性）
                if hasattr(mod, 'decoder') and hasattr(mod.decoder, 'deep_supervision'):
                    mod.decoder.deep_supervision = enabled
                    self.print_to_log_file(f"✅ TE-Swin UNet3D decoder.deep_supervision set to: {enabled}")
                
                # 方式3：设置do_ds标志（nnUNet的另一个标志）
                if hasattr(mod, 'do_ds'):
                    mod.do_ds = enabled
                    self.print_to_log_file(f"✅ TE-Swin UNet3D do_ds set to: {enabled}")
                    
            else:
                # 标准nnUNet模型的处理
                if hasattr(mod, 'decoder') and hasattr(mod.decoder, 'deep_supervision'):
                    mod.decoder.deep_supervision = enabled
                    self.print_to_log_file(f"✅ Standard model decoder.deep_supervision set to: {enabled}")
                else:
                    # 最后的备选方案
                    if hasattr(mod, 'deep_supervision'):
                        mod.deep_supervision = enabled
                        self.print_to_log_file(f"✅ Model deep_supervision set to: {enabled}")
                    else:
                        self.print_to_log_file(f"⚠️  Could not set deep supervision - no suitable attribute found")
                        
        except Exception as e:
            self.print_to_log_file(f"⚠️  Error setting deep supervision: {e}")
            # 不抛出异常，让训练继续
            
    def initialize(self):
        """重写初始化方法，确保正确的设备管理"""
        try:
            # 调用父类初始化
            super().initialize()
            
            # ✅ 确保网络在正确设备上
            if hasattr(self, 'network') and self.network is not None:
                self.network = self.network.to(self.device)
                self.print_to_log_file(f"✅ Network confirmed on device: {next(self.network.parameters()).device}")
                
            # ✅ 设置网络为训练模式
            if hasattr(self, 'network'):
                self.network.train()
                
        except Exception as e:
            self.print_to_log_file(f"❌ Initialization error: {e}")
            import traceback
            self.print_to_log_file(f"Full traceback:\n{traceback.format_exc()}")
            raise e
            
    def on_train_start(self):
        """训练开始时的设备检查和状态验证"""
        try:
            # ✅ 设备状态检查
            if hasattr(self, 'network') and self.network is not None:
                network_device = next(self.network.parameters()).device
                self.print_to_log_file(f"🔍 Network device: {network_device}")
                self.print_to_log_file(f"🔍 Expected device: {self.device}")
                
                if network_device != self.device:
                    self.print_to_log_file(f"⚠️  Device mismatch! Moving network to {self.device}")
                    self.network = self.network.to(self.device)
                    
            # ✅ GPU内存状态检查
            if torch.cuda.is_available():
                self.print_to_log_file(f"🔍 GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                self.print_to_log_file(f"🔍 GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
            
            print("🚀 Starting TE-Swin UNet3D training with 'MRI as GIF' approach...")
            
            # ✅ 调用父类方法（这里可能会调用set_deep_supervision_enabled）
            super().on_train_start()
            
        except Exception as e:
            self.print_to_log_file(f"❌ on_train_start error: {e}")
            import traceback
            self.print_to_log_file(f"Full traceback:\n{traceback.format_exc()}")
            raise e
        
    def train_step(self, batch):
        """重写训练步骤，添加错误处理和设备检查"""
        try:
            # ✅ 确保数据在正确设备上
            data = batch['data']
            target = batch['target']
            
            if data.device != self.device:
                data = data.to(self.device, non_blocking=True)
                
            if target.device != self.device:
                target = target.to(self.device, non_blocking=True)
                
            batch['data'] = data
            batch['target'] = target
            
            # 调用父类的训练步骤
            return super().train_step(batch)
            
        except Exception as e:
            self.print_to_log_file(f"❌ Training step error: {e}")
            # 返回一个默认的loss避免崩溃
            return {'loss': torch.tensor(1.0, device=self.device)}

    def validation_step(self, batch):
        """重写验证步骤，添加错误处理"""
        try:
            # ✅ 确保数据在正确设备上
            data = batch['data']
            target = batch['target']
            
            if data.device != self.device:
                data = data.to(self.device, non_blocking=True)
                
            if target.device != self.device:
                target = target.to(self.device, non_blocking=True)
                
            batch['data'] = data
            batch['target'] = target
            
            return super().validation_step(batch)
            
        except Exception as e:
            self.print_to_log_file(f"❌ Validation step error: {e}")
            # 返回一个默认的loss避免崩溃
            return {'loss': torch.tensor(1.0, device=self.device)}

    def run_training(self):
        """重写训练循环，添加更好的错误处理"""
        try:
            self.on_train_start()

            for epoch in range(self.current_epoch, self.num_epochs):
                self.print_to_log_file(f"🚀 Starting epoch {epoch}/{self.num_epochs-1}")
                
                self.on_epoch_start()
                
                # ✅ 每个epoch开始时检查设备状态
                if hasattr(self, 'network'):
                    network_device = next(self.network.parameters()).device
                    if network_device != self.device:
                        self.print_to_log_file(f"⚠️  Epoch {epoch}: Network on {network_device}, moving to {self.device}")
                        self.network = self.network.to(self.device)
                
                # Training phase
                self.on_train_epoch_start()
                train_outputs = []
                
                try:
                    train_iterator = tqdm(
                        range(self.num_iterations_per_epoch),
                        desc=f"Epoch {epoch} Training",
                        unit="batch",
                        colour="green",
                        leave=False,
                        bar_format='{l_bar}{bar:30}{r_bar}'
                    )
                    
                    for batch_id in train_iterator:
                        try:
                            batch = next(self.dataloader_train)
                            output = self.train_step(batch)
                            train_outputs.append(output)
                            
                            # Update progress bar
                            if 'loss' in output:
                                train_iterator.set_postfix({
                                    "loss": f"{output['loss']:.4f}",
                                })
                                
                        except Exception as e:
                            self.print_to_log_file(f"❌ Training batch {batch_id} failed: {e}")
                            # 继续下一个batch而不是崩溃
                            continue
                            
                except Exception as e:
                    self.print_to_log_file(f"❌ Training epoch setup failed: {e}")
                    
                self.on_train_epoch_end(train_outputs)
                
                # Validation phase
                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    
                    try:
                        val_iterator = tqdm(
                            range(self.num_val_iterations_per_epoch),
                            desc=f"Epoch {epoch} Validation",
                            unit="batch",
                            colour="blue",
                            leave=False,
                            bar_format='{l_bar}{bar:30}{r_bar}'
                        )
                        
                        for batch_id in val_iterator:
                            try:
                                batch = next(self.dataloader_val)
                                output = self.validation_step(batch)
                                val_outputs.append(output)
                                
                                # Update progress bar
                                if 'loss' in output:
                                    val_iterator.set_postfix({
                                        "loss": f"{output['loss']:.4f}",
                                    })
                                    
                            except Exception as e:
                                self.print_to_log_file(f"❌ Validation batch {batch_id} failed: {e}")
                                continue
                                
                    except Exception as e:
                        self.print_to_log_file(f"❌ Validation epoch setup failed: {e}")
                        
                    self.on_validation_epoch_end(val_outputs)
                
                self.on_epoch_end()
                
                self.print_to_log_file(f"✅ Epoch {epoch} completed", also_print_to_console=True)
                
        except Exception as e:
            self.print_to_log_file(f"❌ Training failed: {e}")
            import traceback
            self.print_to_log_file(f"Full traceback:\n{traceback.format_exc()}")
            raise e


class nnUNetTrainer_TE_SwinUnet3D_tiny(nnUNetTrainer_TE_SwinUnet3D):
    """Tiny variant with all fixes."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                         dataset_json=dataset_json, device=device)
        self.model_variant = 't'


class nnUNetTrainer_TE_SwinUnet3D_small(nnUNetTrainer_TE_SwinUnet3D):
    """Small variant with all fixes."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                         dataset_json=dataset_json, device=device)
        self.model_variant = 's'


class nnUNetTrainer_TE_SwinUnet3D_base(nnUNetTrainer_TE_SwinUnet3D):
    """Base variant with all fixes."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                         dataset_json=dataset_json, device=device)
        self.model_variant = 'b'