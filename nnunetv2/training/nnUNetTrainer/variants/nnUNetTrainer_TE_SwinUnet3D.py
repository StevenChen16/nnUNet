"""
nnUNet Trainer with TE-Swin UNet3D architecture.
This integrates our "MRI as GIF" approach into the nnUNet framework.
"""
import torch
import torch.nn as nn
from typing import Tuple, Union, List
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
    nnUNet Trainer using TE-Swin UNet3D architecture.
    
    This trainer replaces the standard nnUNet architecture with our texture-enhanced
    Swin UNet3D that treats MRI sequences as video-like temporal data.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        """
        Initialize TE-Swin UNet3D trainer.
        
        Args:
            plans: nnUNet plans dictionary
            configuration: Configuration name
            fold: Cross-validation fold
            dataset_json: Dataset configuration
            unpack_dataset: Whether to unpack dataset
            device: Training device
        """
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # Model variant - can be 't' (tiny), 's' (small), or 'b' (base)
        self.model_variant = 's'  # Default to small model
        
    @staticmethod  
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """
        Build the TE-Swin UNet3D network architecture.
        
        Args:
            plans_manager: Plans manager instance
            dataset_json: Dataset configuration
            configuration_manager: Configuration manager instance
            num_input_channels: Number of input channels
            enable_deep_supervision: Whether to enable deep supervision
            
        Returns:
            torch.nn.Module: TE-Swin UNet3D model
        """
        # Get number of output classes
        num_classes = len(dataset_json['labels'].keys())
        
        # Create TE-Swin UNet3D model (small variant by default)
        model = create_te_swinunet_s_3d(
            input_channels=num_input_channels,
            num_classes=num_classes,
            deep_supervision=enable_deep_supervision,
            # Use patch size from configuration if available
            window_size=7,  # Fixed window size for now
            dropout=0.0,
        )
        
        print(f"Created TE-Swin UNet3D with {num_input_channels} input channels, "
              f"{num_classes} output classes, deep supervision: {enable_deep_supervision}")
        
        return model

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        Enable/disable deep supervision for TE-Swin UNet3D.
        
        Args:
            enabled: Whether to enable deep supervision
        """
        # Get the actual model (handle DDP wrapping)
        if self.is_ddp:
            model = self.network.module
        else:
            model = self.network
            
        # Handle torch.compile wrapping
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
            
        # Set deep supervision flag
        if hasattr(model, 'deep_supervision'):
            model.deep_supervision = enabled
            print(f"Set deep supervision to {enabled}")
        else:
            print("Warning: Model does not have deep_supervision attribute")

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler for TE-Swin UNet3D.
        
        Returns:
            Tuple: (optimizer, lr_scheduler)
        """
        # Use Adam optimizer for Transformer-based architectures
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Use the same LR scheduler as the base trainer
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        
        return optimizer, lr_scheduler

    def on_train_start(self):
        """
        Called at the start of training.
        """
        print("Starting TE-Swin UNet3D training with 'MRI as GIF' approach...")
        super().on_train_start()


class nnUNetTrainer_TE_SwinUnet3D_small(nnUNetTrainer_TE_SwinUnet3D):
    """Small variant of TE-Swin UNet3D trainer."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.model_variant = 's'
        

class nnUNetTrainer_TE_SwinUnet3D_tiny(nnUNetTrainer_TE_SwinUnet3D):
    """Tiny variant of TE-Swin UNet3D trainer."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.model_variant = 't'
        
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """Build tiny TE-Swin UNet3D."""
        num_classes = len(dataset_json['labels'].keys())
        
        model = create_te_swinunet_t_3d(
            input_channels=num_input_channels,
            num_classes=num_classes,
            deep_supervision=enable_deep_supervision,
            window_size=7,
            dropout=0.0,
        )
        
        print(f"Created tiny TE-Swin UNet3D with {num_input_channels} input channels, "
              f"{num_classes} output classes")
        
        return model


class nnUNetTrainer_TE_SwinUnet3D_base(nnUNetTrainer_TE_SwinUnet3D):
    """Base variant of TE-Swin UNet3D trainer."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.model_variant = 'b'
        
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """Build base TE-Swin UNet3D."""
        num_classes = len(dataset_json['labels'].keys())
        
        model = create_te_swinunet_b_3d(
            input_channels=num_input_channels,
            num_classes=num_classes,
            deep_supervision=enable_deep_supervision,
            window_size=7,
            dropout=0.0,
        )
        
        print(f"Created base TE-Swin UNet3D with {num_input_channels} input channels, "
              f"{num_classes} output classes")
        
        return model
