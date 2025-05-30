"""
nnUNet Trainer with TE-Swin UNet3D architecture.
This integrates our "MRI as GIF" approach into the nnUNet framework.
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
    nnUNet Trainer using TE-Swin UNet3D architecture.
    
    This trainer replaces the standard nnUNet architecture with our texture-enhanced
    Swin UNet3D that treats MRI sequences as video-like temporal data.
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        """
        Initialize TE-Swin UNet3D trainer.
        
        Args:
            plans: nnUNet plans dictionary
            configuration: Configuration name
            fold: Cross-validation fold
            dataset_json: Dataset configuration
            device: Training device
        """
        # Only pass parameters that exist in the parent class's signature to avoid KeyError
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json, device=device)
        
        # Model variant - can be 't' (tiny), 's' (small), or 'b' (base)
        self.model_variant = 's'  # Default to small model
        
    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        """Build TE-Swin UNet3D architecture.
        
        We override this method to use our custom TE-Swin UNet3D models instead of the
        standard nnUNet architecture.
        
        Args:
            architecture_class_name: Not used, we override with our model
            arch_init_kwargs: Not used, we override with our model
            arch_init_kwargs_req_import: Not used, we override with our model
            num_input_channels: Number of input channels
            num_output_channels: Number of output channels
            enable_deep_supervision: Whether to use deep supervision
            
        Returns:
            TE-Swin UNet3D model
        """
        self.print_to_log_file(f"Using TE-SwinUnet3D-{self.model_variant} variant")
        
        # Use different model creation function based on model variant
        if self.model_variant == 't':
            network = create_te_swinunet_t_3d(
                in_channels=num_input_channels,
                out_channels=num_output_channels,
                enable_deep_supervision=enable_deep_supervision
            )
        elif self.model_variant == 'b':
            network = create_te_swinunet_b_3d(
                in_channels=num_input_channels,
                out_channels=num_output_channels,
                enable_deep_supervision=enable_deep_supervision
            )
        else:  # Default to 's' (small) model
            network = create_te_swinunet_s_3d(
                in_channels=num_input_channels,
                out_channels=num_output_channels,
                enable_deep_supervision=enable_deep_supervision
            )
        
        return network
        
    def run_training(self):
        """Run training with progress bars for each epoch.
        
        Override the parent class method to add progress bars using tqdm.
        """
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            
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
                batch = next(self.dataloader_train)
                output = self.train_step(batch)
                train_outputs.append(output)
                # Update progress bar with current loss
                if 'loss' in output:
                    train_iterator.set_postfix({"loss": f"{output['loss']:.4f}"})
                    
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
                    batch = next(self.dataloader_val)
                    output = self.validation_step(batch)
                    val_outputs.append(output)
                    # Update progress bar with current loss
                    if 'loss' in output:
                        val_iterator.set_postfix({"loss": f"{output['loss']:.4f}"})
                        
                self.on_validation_epoch_end(val_outputs)
            
            self.on_epoch_end()
            
            # Print completion message for the epoch
            self.print_to_log_file(f"Epoch {epoch} completed", also_print_to_console=True)

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
                 device: torch.device = torch.device('cuda')):
        # Use named parameters to avoid issues with parameter passing
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                         dataset_json=dataset_json, device=device)
        self.model_variant = 's'
        

class nnUNetTrainer_TE_SwinUnet3D_tiny(nnUNetTrainer_TE_SwinUnet3D):
    """Tiny variant of TE-Swin UNet3D trainer."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        # Use named parameters to avoid issues with parameter passing
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                         dataset_json=dataset_json, device=device)
        self.model_variant = 't'
        
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """Build tiny TE-Swin UNet3D."""
        
        model = create_te_swinunet_t_3d(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            window_size=7,
            dropout=0.0,
        )
        
        print(f"Created tiny TE-Swin UNet3D with {num_input_channels} input channels, "
              f"{num_output_channels} output classes")
        
        return model


class nnUNetTrainer_TE_SwinUnet3D_base(nnUNetTrainer_TE_SwinUnet3D):
    """Base variant of TE-Swin UNet3D trainer."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 device: torch.device = torch.device('cuda')):
        # Use named parameters to avoid issues with parameter passing
        super().__init__(plans=plans, configuration=configuration, fold=fold, 
                         dataset_json=dataset_json, device=device)
        self.model_variant = 'b'
        
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """Build base TE-Swin UNet3D."""
        
        model = create_te_swinunet_b_3d(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            window_size=7,
            dropout=0.0,
        )
        
        print(f"Created base TE-Swin UNet3D with {num_input_channels} input channels, "
              f"{num_output_channels} output classes")
        
        return model
