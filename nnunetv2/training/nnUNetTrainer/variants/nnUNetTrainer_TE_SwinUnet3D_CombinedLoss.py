"""
Custom loss functions for TE-Swin UNet3D in nnUNet framework.
Integrates our advanced loss functions with nnUNet's loss handling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.te_swin_models.nnUNet_TE_SwinUnet3D import create_te_swinunet_s_3d
from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainer_TE_SwinUnet3D import nnUNetTrainer_TE_SwinUnet3D


class DiceLoss(nn.Module):
    """
    Generalized Dice Loss for multi-class segmentation.
    """
    def __init__(self, smooth=1e-5, reduction='mean', weight=None):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, true):
        """
        Args:
            logits: [B, C, D, H, W] tensor of raw model outputs
            true: [B, D, H, W] tensor of class indices
        
        Returns:
            Generalized Dice Loss value
        """
        # Convert from raw logits to probabilities
        prob = torch.softmax(logits, dim=1)  # [B, C, D, H, W]
        
        # Convert to one-hot encoding
        true_onehot = F.one_hot(true, num_classes=prob.shape[1])  # [B, D, H, W, C]
        true_onehot = true_onehot.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]
        
        # Dimensions to sum over
        dims = (0, 2, 3, 4)  # Sum over batch, depth, height, width
        
        # Calculate class weights if not provided
        if self.weight is None:
            # Calculate weights inversely proportional to class volume
            weight = 1.0 / (torch.sum(true_onehot, dims, keepdim=True) + self.smooth)
        else:
            weight = self.weight.view(1, -1, 1, 1, 1).to(true_onehot.device)
        
        # Numerator and denominator for Dice coefficient
        intersection = torch.sum(weight * prob * true_onehot, dims)
        cardinality = torch.sum(weight * (prob + true_onehot), dims)
        
        # Dice coefficient per class
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Mean or sum over classes depending on reduction
        if self.reduction == 'mean':
            dice = dice_per_class.mean()
        else:  # 'sum'
            dice = dice_per_class.sum()
            
        # Return loss value (1 - dice)
        return 1.0 - dice


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for better segmentation of object boundaries.
    """
    def __init__(self, theta0=3, theta=2, smooth=1e-5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        self.smooth = smooth
        
    def forward(self, logits, true):
        """
        Args:
            logits: [B, C, D, H, W] tensor of raw model outputs
            true: [B, D, H, W] tensor of class indices
        
        Returns:
            Boundary Loss value
        """
        # Convert logits to probabilities and true to one-hot
        prob = torch.softmax(logits, dim=1)  # [B, C, D, H, W]
        true_onehot = F.one_hot(true, num_classes=prob.shape[1])  # [B, D, H, W, C]
        true_onehot = true_onehot.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]
        
        # Initialize boundary loss
        batch_size, n_classes = prob.shape[0], prob.shape[1]
        boundary_loss = 0.0
        
        # For each class (except background=0), compute boundary loss
        for c in range(1, n_classes):  # Skip background class
            # Reference segmentation (V)
            v = true_onehot[:, c:c+1, :, :, :]  # [B, 1, D, H, W]
            
            # Predicted probability (U)
            u = prob[:, c:c+1, :, :, :]  # [B, 1, D, H, W]
            
            # Create boundary map using morphological operations approximation
            # Use 3D max pooling to approximate distance from boundary
            kernel_size = 2 * self.theta0 + 1
            padding = self.theta0
            
            # Complement of segmentation: 1-V
            v_complement = 1 - v
            
            # Approximate distance map from boundary
            pooled = F.max_pool3d(
                v_complement, 
                kernel_size=kernel_size, 
                stride=1, 
                padding=padding
            )
            
            # Create weight map based on distance from boundary
            weight_map = torch.exp(-torch.pow(pooled, 2) / (self.theta ** 2))
            
            # Compute weighted dice-like metrics
            boundary_u = u * weight_map
            boundary_v = v * weight_map
            
            # Compute precision (weighted dice)
            intersection_p = torch.sum(boundary_u * boundary_v)
            cardinality_p = torch.sum(boundary_u) + torch.sum(boundary_v)
            precision = (2.0 * intersection_p + self.smooth) / (cardinality_p + self.smooth)
            
            # Compute recall (weighted dice on complements)
            boundary_u_comp = (1 - u) * weight_map
            boundary_v_comp = (1 - v) * weight_map
            intersection_r = torch.sum(boundary_u_comp * boundary_v_comp)
            cardinality_r = torch.sum(boundary_u_comp) + torch.sum(boundary_v_comp)
            recall = (2.0 * intersection_r + self.smooth) / (cardinality_r + self.smooth)
            
            # Compute boundary F1 loss
            if precision + recall > 0:
                f1 = 2.0 * precision * recall / (precision + recall)
                boundary_loss += 1.0 - f1
            else:
                boundary_loss += 1.0
        
        # Average over classes (excluding background)
        if n_classes > 1:
            boundary_loss = boundary_loss / (n_classes - 1)
        
        return boundary_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function integrating Dice, Cross-Entropy, and Boundary losses.
    """
    def __init__(self, dice_weight=0.33, ce_weight=0.33, boundary_weight=0.33, 
                 class_weights=None):
        super().__init__()
        self.dice_loss = DiceLoss(weight=class_weights)
        self.boundary_loss = BoundaryLoss()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.class_weights = class_weights
        
    def forward(self, logits, true):
        """
        Args:
            logits: [B, C, D, H, W] tensor of raw model outputs
            true: [B, D, H, W] tensor of class indices
        
        Returns:
            Combined loss value
        """
        dice_val = self.dice_loss(logits, true)
        ce_val = F.cross_entropy(logits, true, weight=self.class_weights)
        boundary_val = self.boundary_loss(logits, true)
        
        # Combine losses with weights
        total_loss = (
            self.dice_weight * dice_val +
            self.ce_weight * ce_val +
            self.boundary_weight * boundary_val
        )
        
        return total_loss


class nnUNetTrainer_TE_SwinUnet3D_CombinedLoss(nnUNetTrainer_TE_SwinUnet3D):
    """
    TE-Swin UNet3D trainer with custom combined loss function.
    """
    
    def _build_loss(self):
        """
        Build custom combined loss function for TE-Swin UNet3D.
        
        Returns:
            Loss function for training
        """
        # For Task 2 (binary), use weighted loss due to class imbalance
        if len(self.dataset_json['labels'].keys()) == 2:  # Binary segmentation
            class_weights = torch.tensor([0.1, 0.9], device=self.device)  # Background, Tumor
        else:
            class_weights = None
            
        # Create combined loss
        loss = CombinedLoss(
            dice_weight=0.4,
            ce_weight=0.4,
            boundary_weight=0.2,
            class_weights=class_weights
        )
        
        # Enable deep supervision if needed
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = torch.tensor([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0  # Set lowest resolution weight to 0
            weights = weights / weights.sum()  # Normalize
            
            # Wrap with deep supervision
            loss = DeepSupervisionWrapper(loss, weights)
        
        return loss

    def on_train_start(self):
        """Called at the start of training."""
        print("Starting TE-Swin UNet3D training with combined loss function...")
        super().on_train_start()
