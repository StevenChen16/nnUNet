"""
修复后的 nnUNet Trainer with TE-Swin UNet3D architecture.
修复了参数名称不匹配和设备管理问题。
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
    修复后的 nnUNet Trainer using TE-Swin UNet3D architecture.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        """
        Initialize TE-Swin UNet3D trainer.
        """
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                        dataset_json=dataset_json, device=device)
        
        # Model variant - can be 't' (tiny), 's' (small), or 'b' (base)
        self.model_variant = 's'  # Default to small model
        
    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        """
        修复后的网络架构构建方法。
        主要修复：参数名称匹配问题
        """
        self.print_to_log_file(f"Building TE-SwinUnet3D-{self.model_variant} variant")
        self.print_to_log_file(f"Input channels: {num_input_channels}, Output channels: {num_output_channels}")
        self.print_to_log_file(f"Deep supervision: {enable_deep_supervision}")
        
        # ✅ 修复：使用正确的参数名称
        try:
            if self.model_variant == 't':
                network = create_te_swinunet_t_3d(
                    input_channels=num_input_channels,  # ✅ 正确参数名
                    num_classes=num_output_channels,    # ✅ 正确参数名
                    deep_supervision=enable_deep_supervision,  # ✅ 正确参数名
                    window_size=4,  # 兼容性优化
                    downscaling_factors=(2, 2, 2, 2)  # 兼容性优化
                )
            elif self.model_variant == 'b':
                network = create_te_swinunet_b_3d(
                    input_channels=num_input_channels,
                    num_classes=num_output_channels,
                    deep_supervision=enable_deep_supervision,
                    window_size=4,  # 兼容性优化
                    downscaling_factors=(2, 2, 2, 2)  # 兼容性优化
                )
            else:  # Default to 's' (small) model
                network = create_te_swinunet_s_3d(
                    input_channels=num_input_channels,
                    num_classes=num_output_channels,
                    deep_supervision=enable_deep_supervision,
                    window_size=4,  # 兼容性优化
                    downscaling_factors=(2, 2, 2, 2)  # 兼容性优化
                )
                
            self.print_to_log_file(f"✅ Successfully created TE-SwinUnet3D-{self.model_variant}")
            self.print_to_log_file(f"Model parameters: {sum(p.numel() for p in network.parameters()):,}")
            
            # ✅ 确保模型在正确的设备上
            network = network.to(self.device)
            self.print_to_log_file(f"✅ Model moved to device: {self.device}")
            
            return network
            
        except Exception as e:
            self.print_to_log_file(f"❌ Failed to create TE-SwinUnet3D: {e}")
            raise e
        
    def initialize(self):
        """
        重写初始化方法，确保正确的设备管理
        """
        # 调用父类初始化
        super().initialize()
        
        # ✅ 确保网络在正确设备上
        if hasattr(self, 'network') and self.network is not None:
            self.network = self.network.to(self.device)
            self.print_to_log_file(f"✅ Network confirmed on device: {next(self.network.parameters()).device}")
            
        # ✅ 设置网络为训练模式
        if hasattr(self, 'network'):
            self.network.train()
            
    def on_train_start(self):
        """
        训练开始时的设备检查
        """
        super().on_train_start()
        
        # ✅ 详细的设备状态检查
        if hasattr(self, 'network') and self.network is not None:
            network_device = next(self.network.parameters()).device
            self.print_to_log_file(f"🔍 Network device: {network_device}")
            self.print_to_log_file(f"🔍 Expected device: {self.device}")
            
            if network_device != self.device:
                self.print_to_log_file(f"⚠️  Device mismatch detected! Moving network to {self.device}")
                self.network = self.network.to(self.device)
                
        # ✅ GPU内存状态检查
        if torch.cuda.is_available():
            self.print_to_log_file(f"🔍 GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            self.print_to_log_file(f"🔍 GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        print("🚀 Starting TE-Swin UNet3D training with 'MRI as GIF' approach...")
        
    def train_step(self, batch):
        """
        重写训练步骤，添加设备检查
        """
        # ✅ 确保数据在正确设备上
        data = batch['data']
        target = batch['target']
        
        if data.device != self.device:
            self.print_to_log_file(f"⚠️  Moving data from {data.device} to {self.device}")
            data = data.to(self.device, non_blocking=True)
            
        if target.device != self.device:
            self.print_to_log_file(f"⚠️  Moving target from {target.device} to {self.device}")
            target = target.to(self.device, non_blocking=True)
            
        # 调用父类的训练步骤
        batch['data'] = data
        batch['target'] = target
        
        return super().train_step(batch)

    def validation_step(self, batch):
        """
        重写验证步骤，添加设备检查
        """
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

    def run_training(self):
        """
        重写训练循环，添加进度条和设备监控
        """
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            
            # ✅ 每个epoch开始时检查设备状态
            if hasattr(self, 'network'):
                network_device = next(self.network.parameters()).device
                if network_device != self.device:
                    self.print_to_log_file(f"⚠️  Epoch {epoch}: Network on {network_device}, moving to {self.device}")
                    self.network = self.network.to(self.device)
            
            # Training phase with progress bar
            self.on_train_epoch_start()
            train_outputs = []
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
                    
                    # Update progress bar with current loss
                    if 'loss' in output:
                        train_iterator.set_postfix({
                            "loss": f"{output['loss']:.4f}",
                            "device": str(next(self.network.parameters()).device)
                        })
                        
                except Exception as e:
                    self.print_to_log_file(f"❌ Training step {batch_id} failed: {e}")
                    raise e
                    
            self.on_train_epoch_end(train_outputs)
            
            # Validation phase with progress bar
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
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
                        
                        # Update progress bar with current loss
                        if 'loss' in output:
                            val_iterator.set_postfix({
                                "loss": f"{output['loss']:.4f}",
                                "device": str(next(self.network.parameters()).device)
                            })
                            
                    except Exception as e:
                        self.print_to_log_file(f"❌ Validation step {batch_id} failed: {e}")
                        raise e
                        
                self.on_validation_epoch_end(val_outputs)
            
            self.on_epoch_end()
            
            # Print completion message for the epoch
            self.print_to_log_file(f"✅ Epoch {epoch} completed", also_print_to_console=True)


class nnUNetTrainer_TE_SwinUnet3D_tiny(nnUNetTrainer_TE_SwinUnet3D):
    """Tiny variant with fixed initialization."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                         dataset_json=dataset_json, device=device)
        self.model_variant = 't'


class nnUNetTrainer_TE_SwinUnet3D_base(nnUNetTrainer_TE_SwinUnet3D):
    """Base variant with fixed initialization."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                         dataset_json=dataset_json, device=device)
        self.model_variant = 'b'