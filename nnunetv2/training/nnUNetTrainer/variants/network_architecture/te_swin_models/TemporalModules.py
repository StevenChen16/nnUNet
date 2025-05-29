import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttentionModule(nn.Module):
    """Handles temporal relationships between MRI slices, similar to frame relationships in videos"""
    def __init__(self, dim, num_heads=8, window_size=4):
        super().__init__()
        self.window_size = window_size
        
        # Ensure num_heads divides dim evenly
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        
        self.num_heads = num_heads
        
        # Simplified implementation, focusing on Z-axis (time axis) attention
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, x):
        # x shape: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        
        x_perm = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        
        x_reshape = x_perm.reshape(B, D, H*W, C)
        
        output = []
        for i in range(0, H*W, 256):  # Process in batches to save memory
            end_i = min(i + 256, H*W)
            # Extract all spatial locations for the current batch
            x_batch = x_reshape[:, :, i:end_i, :]  # [B, D, batch_size, C]
            
            # Reshape to sequence format for processing
            x_batch = x_batch.permute(0, 2, 1, 3)  # [B, batch_size, D, C]
            x_batch = x_batch.reshape(B * (end_i-i), D, C)  # [B*batch_size, D, C]
            
            # Apply self-attention
            x_norm = self.norm(x_batch)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
            
            # Residual connection
            x_batch = x_batch + attn_out
            
            # Reshape back to original shape
            x_batch = x_batch.reshape(B, end_i-i, D, C)
            output.append(x_batch)
            
        # Combine all batch results
        x_out = torch.cat(output, dim=1)  # [B, H*W, D, C]
        x_out = x_out.permute(0, 2, 1, 3)  # [B, D, H*W, C]
        
        # Reshape back to original dimensions
        x_out = x_out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        
        return x_out


class SliceGRU(nn.Module):
    """Used for information transmission between slices, similar to ConvLSTM"""
    def __init__(self, dim):
        super().__init__()
        self.update_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.reset_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.out_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x, h_prev):
        # x: current slice [B, C, H, W]
        # h_prev: previous state [B, C, H, W]
        
        # Ensure x and h_prev are on the same device
        if x.device != h_prev.device:
            h_prev = h_prev.to(x.device)
            
        # Feature fusion
        combined = torch.cat([x, h_prev], dim=1)
        
        # Update gate and reset gate
        update = self.update_gate(combined)
        reset = self.reset_gate(combined)
        
        # Candidate hidden state
        combined_reset = torch.cat([x, reset * h_prev], dim=1)
        candidate = self.out_gate(combined_reset)
        
        # New hidden state
        h_new = (1 - update) * h_prev + update * candidate
        
        return h_new


class SlicePropagationModule(nn.Module):
    """Implements information transmission between slices, similar to video frame propagation"""
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Slice flow network
        self.slice_flow = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim*2, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.GroupNorm(8, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(hidden_dim*2, hidden_dim, kernel_size=(3,1,1), padding=(1,0,0)),
        )
        
        # Recurrent network for slice information transmission
        self.slice_gru = SliceGRU(hidden_dim)
        
    def forward(self, x):
        # x shape: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        
        # First extract slice flow features
        slice_features = self.slice_flow(x)
        
        # Then transmit information through the recurrent network (from top to bottom and then from bottom to top)
        forward_states = []
        h_state = torch.zeros(B, C, H, W).to(x.device)
        
        # Top-down propagation
        for d in range(D):
            current = x[:, :, d]  # current slice [B, C, H, W]
            h_state = self.slice_gru(current, h_state)
            forward_states.append(h_state)
            
        # Bottom-up propagation
        backward_states = []
        h_state = torch.zeros(B, C, H, W).to(x.device)
        
        for d in range(D-1, -1, -1):
            current = x[:, :, d]  # current slice
            h_state = self.slice_gru(current, h_state)
            backward_states.insert(0, h_state)
            
        # Fuse bidirectional features
        enhanced_features = []
        for d in range(D):
            enhanced = x[:, :, d] + forward_states[d] + backward_states[d]
            enhanced_features.append(enhanced)
            
        # Rebuild output volume
        output = torch.stack(enhanced_features, dim=2)  # [B, C, D, H, W]
        return output + slice_features  # residual connection


class TemporalConsistencyLoss(nn.Module):
    """Ensures temporal consistency between adjacent slices"""
    def __init__(self, weight=0.2):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred, target):
        # Calculate temporal consistency loss
        temp_loss = 0.0
        
        # Only calculate loss when there are multiple slices
        if pred.size(2) > 1:
            for d in range(1, pred.size(2)):
                # Get adjacent slices
                pred_curr = pred[:, :, d]
                pred_prev = pred[:, :, d-1]
                
                # Calculate difference
                temp_diff = F.mse_loss(pred_curr, pred_prev)
                
                # Check if target has changed (adaptive constraint)
                if target.dim() == 4:  # [B, D, H, W] format
                    target_curr = target[:, d]
                    target_prev = target[:, d-1]
                else:  # [B, C, D, H, W] format
                    target_curr = target[:, :, d]
                    target_prev = target[:, :, d-1]
                    
                # Create consistency mask - only enforce consistency in regions with same label
                if target.dim() == 4:
                    consistency_mask = (target_curr == target_prev).float()
                else:
                    consistency_mask = torch.isclose(target_curr, target_prev).float()
                    
                # Apply mask
                masked_diff = temp_diff * consistency_mask
                temp_loss += masked_diff.mean()
                
        return self.weight * temp_loss
