"""
nnUNet Trainer for TE-Swin UNet3D
"""
import torch
from typing import Union, Tuple, List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.te_swin_models.nnUNet_TE_SwinUnet3D import (
    create_te_swinunet_s_3d,
    create_te_swinunet_t_3d,
    create_te_swinunet_b_3d
)


class nnUNetTrainer_TE_SwinUnet3D(nnUNetTrainer):
    """
    nnUNet Trainer for TE-Swin UNet3D that resolves output size mismatch issues.
    """
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device('cuda')
    ):
        """
        Initialize the TE-Swin UNet3D trainer.
        """
        # 调用父类初始化
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        print("🎯 TE-Swin UNet3D Trainer initialized")
        print(f"   - Configuration: {configuration}")
        print(f"   - Fold: {fold}")
        print(f"   - Device: {device}")
        print(f"   - Deep supervision: {self.enable_deep_supervision}")
        
    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """
        Build the TE-Swin UNet3D network architecture.
        """
        print("🔧 Building TE-Swin UNet3D architecture...")
        
        # 根据内存情况选择模型大小
        if hasattr(self, 'configuration_manager'):
            patch_size = self.configuration_manager.patch_size
            batch_size = self.configuration_manager.batch_size
        else:
            # 从plans中获取配置信息
            config = self.plans['configurations'][self.configuration]
            patch_size = config.get('patch_size', [128, 128, 128])
            batch_size = config.get('batch_size', 2)
        
        print(f"🔧 Configuration details:")
        print(f"   - Patch size: {patch_size}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Input channels: {num_input_channels}")
        print(f"   - Output channels: {num_output_channels}")
        print(f"   - Deep supervision: {enable_deep_supervision}")
        
        # 根据patch size和batch size智能选择模型变体
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2] if len(patch_size) == 3 else patch_size[0] * patch_size[1]
        memory_requirement = patch_volume * batch_size
        
        if memory_requirement < 500000:  # 小内存需求
            print("🔧 Using TINY variant (low memory)")
            model = create_te_swinunet_t_3d(
                input_channels=num_input_channels,
                num_classes=num_output_channels,
                deep_supervision=enable_deep_supervision,
                # 针对小patch size的参数
                hidden_dim=32,
                layers=(2, 2, 2, 2),
                heads=(2, 4, 6, 8),
                window_size=2,  # 更小的window size
                downscaling_factors=(2, 2, 2, 2)
            )
        elif memory_requirement < 2000000:  # 中等内存需求
            print("🔧 Using SMALL variant (balanced)")
            model = create_te_swinunet_s_3d(
                input_channels=num_input_channels,
                num_classes=num_output_channels,
                deep_supervision=enable_deep_supervision,
                # 平衡的参数设置
                hidden_dim=48,
                layers=(2, 2, 4, 2),
                heads=(3, 6, 9, 12),
                window_size=4,
                downscaling_factors=(2, 2, 2, 2)
            )
        else:  # 大内存需求
            print("🔧 Using BASE variant (high performance)")
            model = create_te_swinunet_b_3d(
                input_channels=num_input_channels,
                num_classes=num_output_channels,
                deep_supervision=enable_deep_supervision,
                # 高性能参数设置
                hidden_dim=96,
                layers=(2, 2, 6, 2),
                heads=(4, 8, 12, 16),
                window_size=4,
                downscaling_factors=(2, 2, 2, 2)
            )
        
        print(f"✅ TE-Swin UNet3D model created successfully")
        print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Model type: {type(model).__name__}")
        
        return model
    
    def configure_optimizers(self):
        """
        Configure optimizers with settings for TE-Swin UNet3D.
        """
        print("🔧 Configuring optimizers for TE-Swin UNet3D...")
        
        # 使用AdamW优化器，对Transformer架构更友好
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=3e-5,  # 适中的权重衰减
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 使用余弦退火学习率调度器
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
            eta_min=self.initial_lr * 0.01  # 最小学习率为初始学习率的1%
        )
        
        print(f"✅ Optimizers configured:")
        print(f"   - Optimizer: AdamW")
        print(f"   - Initial LR: {self.initial_lr}")
        print(f"   - Weight decay: 3e-5")
        print(f"   - LR Scheduler: CosineAnnealingLR")
        
        return optimizer, lr_scheduler
    
    def train_step(self, batch: dict) -> dict:
        """
        Execute a single training step.
        """
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        output = self.network(data)
        
        # Loss computation
        l = self.loss(output, target)
        
        # Backward pass
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        
        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        """
        Execute a single validation step.
        """
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        # Forward pass
        self.network.eval()
        with torch.no_grad():
            output = self.network(data)
        
        # Loss computation
        l = self.loss(output, target)
        
        return {'loss': l.detach().cpu().numpy()}


class nnUNetTrainer_TE_SwinUnet3D_tiny(nnUNetTrainer_TE_SwinUnet3D):
    """
    Tiny variant for limited GPU memory.
    """
    
    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """
        Build the tiny TE-Swin UNet3D for memory-constrained environments.
        """
        print("🔧 Building TINY TE-Swin UNet3D for limited memory...")
        
        model = create_te_swinunet_t_3d(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            # 极小的参数配置
            hidden_dim=24,
            layers=(2, 2, 2, 2),
            heads=(2, 4, 6, 8),
            window_size=2,
            downscaling_factors=(2, 2, 2, 2)
        )
        
        print(f"✅ TINY TE-Swin UNet3D created (Parameters: {sum(p.numel() for p in model.parameters()):,})")
        return model


class nnUNetTrainer_TE_SwinUnet3D_small(nnUNetTrainer_TE_SwinUnet3D):
    """
    Small variant for balanced performance and memory usage.
    """
    
    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """
        Build the small TE-Swin UNet3D for balanced performance.
        """
        print("🔧 Building SMALL TE-Swin UNet3D for balanced performance...")
        
        model = create_te_swinunet_s_3d(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            # 平衡的参数配置
            hidden_dim=48,
            layers=(2, 2, 4, 2),
            heads=(3, 6, 9, 12),
            window_size=4,
            downscaling_factors=(2, 2, 2, 2)
        )
        
        print(f"✅ SMALL TE-Swin UNet3D created (Parameters: {sum(p.numel() for p in model.parameters()):,})")
        return model


class nnUNetTrainer_TE_SwinUnet3D_base(nnUNetTrainer_TE_SwinUnet3D):
    """
    Base variant for high-performance training.
    """
    
    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """
        Build the base TE-Swin UNet3D for high performance.
        """
        print("🔧 Building BASE TE-Swin UNet3D for high performance...")
        
        model = create_te_swinunet_b_3d(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            # 高性能参数配置
            hidden_dim=96,
            layers=(2, 2, 6, 2),
            heads=(4, 8, 12, 16),
            window_size=4,
            downscaling_factors=(2, 2, 2, 2)
        )
        
        print(f"✅ BASE TE-Swin UNet3D created (Parameters: {sum(p.numel() for p in model.parameters()):,})")
        return model