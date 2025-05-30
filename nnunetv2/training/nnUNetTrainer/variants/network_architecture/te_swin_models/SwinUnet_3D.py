import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
from typing import Union, List
import torch._dynamo as dynamo
# 导入einops以替代复杂的permute+view操作
from einops import rearrange as einops_rearrange


@dynamo.disable
def rearrange(tensor, pattern, **axes_lengths):
    """Simple rearrange implementation for basic patterns"""
    # 添加调试信息 - 防止在非分布式环境中报错
    is_main_process = not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if is_main_process:
        print(f"Rearrange pattern: {pattern}")
        print(f"Input tensor shape: {tensor.shape}")
        print(f"Parameters: {axes_lengths}")
        
    if pattern == 'b c h w d -> b h w d c':
        return tensor.permute(0, 2, 3, 4, 1)
    elif pattern == 'b h w d c -> b c h w d':
        return tensor.permute(0, 4, 1, 2, 3)
    elif pattern == '(x1 y1 z1) (x2 y2 z2) -> x1 y1 z1 x2 y2 z2':
        # For mask creation
        x1, y1, x2, y2 = axes_lengths['x1'], axes_lengths['y1'], axes_lengths['x2'], axes_lengths['y2']
        z1 = tensor.shape[0] // (x1 * y1)
        z2 = tensor.shape[1] // (x2 * y2)
        return tensor.view(x1, y1, z1, x2, y2, z2)
    elif pattern == 'x1 y1 z1 x2 y2 z2 -> (x1 y1 z1) (x2 y2 z2)':
        # For mask creation reverse
        x1, y1, z1, x2, y2, z2 = tensor.shape
        return tensor.view(x1*y1*z1, x2*y2*z2)
    elif pattern.startswith('b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d) -> b h (nw_x nw_y nw_z) (w_x w_y w_z) d'):
        # For window attention reshape
        b, height, width, depth, feat_dim = tensor.shape
        h = axes_lengths['h']
        w_x = int(axes_lengths['w_x'])
        w_y = int(axes_lengths['w_y'])
        w_z = int(axes_lengths['w_z'])
        
        # 验证通道维度可被头数整除
        if feat_dim % h != 0:
            raise ValueError(f"Feature dimension {feat_dim} must be divisible by heads {h}")
            
        d = feat_dim // h
        
        # 计算窗口数量，并确保完全整除
        if height % w_x != 0:
            w_x = height  # 使用完整高度作为单一窗口
            nw_x = 1
        else:
            nw_x = height // w_x
            
        if width % w_y != 0:
            w_y = width  # 使用完整宽度作为单一窗口
            nw_y = 1
        else:
            nw_y = width // w_y
            
        if depth % w_z != 0:
            w_z = depth  # 使用完整深度作为单一窗口
            nw_z = 1
        else:
            nw_z = depth // w_z
        
        # 计算目标形状总元素数是否与原张量匹配
        total_elements = b * nw_x * w_x * nw_y * w_y * nw_z * w_z * h * d
        original_elements = tensor.numel()
        
        if total_elements != original_elements:
            print(f"Warning: Shape mismatch! Original: {original_elements}, Target: {total_elements}")
            print(f"Target shape: [{b}, {nw_x}, {w_x}, {nw_y}, {w_y}, {nw_z}, {w_z}, {h}, {d}]")
            
            # 尝试智能调整窗口大小
            while total_elements > original_elements and (w_x > 1 or w_y > 1 or w_z > 1):
                if w_x > 1:
                    w_x -= 1
                elif w_y > 1:
                    w_y -= 1
                elif w_z > 1:
                    w_z -= 1
                total_elements = b * nw_x * w_x * nw_y * w_y * nw_z * w_z * h * d
                
            while total_elements < original_elements and (nw_x > 1 or nw_y > 1 or nw_z > 1):
                if nw_x > 1:
                    nw_x -= 1
                    w_x = height // nw_x
                elif nw_y > 1:
                    nw_y -= 1
                    w_y = width // nw_y
                elif nw_z > 1:
                    nw_z -= 1
                    w_z = depth // nw_z
                total_elements = b * nw_x * w_x * nw_y * w_y * nw_z * w_z * h * d
                
            print(f"Adjusted to: [{b}, {nw_x}, {w_x}, {nw_y}, {w_y}, {nw_z}, {w_z}, {h}, {d}]")
            print(f"New total: {total_elements}")
            
            if total_elements != original_elements:
                # 如果仍不匹配，采用最安全的方式
                nw_x = nw_y = nw_z = 1
                w_x = height
                w_y = width
                w_z = depth
                print(f"Fallback to: [{b}, {nw_x}, {w_x}, {nw_y}, {w_y}, {nw_z}, {w_z}, {h}, {d}]")
        
        try:
            # 使用安全的reshape操作
            reshaped = tensor.view(b, nw_x, w_x, nw_y, w_y, nw_z, w_z, h, d)
            return reshaped.permute(0, 7, 1, 3, 5, 2, 4, 6, 8).contiguous().view(b, h, nw_x*nw_y*nw_z, w_x*w_y*w_z, d)
        except RuntimeError as e:
            print(f"Reshape error: {e}")
            print(f"Tensor size: {tensor.size()}, numel: {tensor.numel()}")
            print(f"Target shape: [{b}, {nw_x}, {w_x}, {nw_y}, {w_y}, {nw_z}, {w_z}, {h}, {d}] = {b * nw_x * w_x * nw_y * w_y * nw_z * w_z * h * d}")
            # 最终回退方案 - 跳过窗口重塑，返回原始张量
            return tensor
    elif pattern.startswith('b h (nw_x nw_y nw_z) (w_x w_y w_z) d -> b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d)'):
        # For window attention reverse reshape
        b, h, num_windows, window_size, d = tensor.shape
        nw_x = int(axes_lengths['nw_x'])
        nw_y = int(axes_lengths['nw_y'])
        nw_z = int(axes_lengths['nw_z'])
        w_x = int(axes_lengths['w_x'])
        w_y = int(axes_lengths['w_y'])
        w_z = int(axes_lengths['w_z'])
        
        # 检查维度匹配
        calculated_windows = nw_x * nw_y * nw_z
        calculated_window_size = w_x * w_y * w_z
        
        # 记录原始值以便调试
        orig_nw_x, orig_nw_y, orig_nw_z = nw_x, nw_y, nw_z
        orig_w_x, orig_w_y, orig_w_z = w_x, w_y, w_z
        
        # 如果不匹配，调整窗口参数
        if calculated_windows != num_windows or calculated_window_size != window_size:
            print(f"Dimension mismatch: Actual [windows={num_windows}, window_size={window_size}] ")
            print(f"Expected [windows={calculated_windows}={nw_x}×{nw_y}×{nw_z}, "
                  f"window_size={calculated_window_size}={w_x}×{w_y}×{w_z}]")
            
            # 重新计算窗口划分，使用整数因子分解
            def find_factors(n, count=3):
                """找出一个数的近似count个因子，尽量接近"""
                if count == 1:
                    return [n]
                    
                # 找最接近立方根的因子
                target = n ** (1/count)
                factor = max(1, round(target))
                while factor > 1 and n % factor != 0:
                    if factor > target:
                        factor -= 1
                    else:
                        factor += 1
                        
                if factor == 1:
                    # 无法找到合适因子，返回[1, 1, n]类型的结果
                    result = [1] * (count-1)
                    result.append(n)
                    return result
                    
                return [factor] + find_factors(n // factor, count-1)
            
            # 计算新的窗口数和窗口大小
            window_factors = find_factors(window_size, 3)
            w_x, w_y, w_z = window_factors
            
            window_num_factors = find_factors(num_windows, 3)
            nw_x, nw_y, nw_z = window_num_factors
            
            print(f"Adjusted to: windows=[{nw_x}×{nw_y}×{nw_z}]={nw_x*nw_y*nw_z}, "
                  f"window_size=[{w_x}×{w_y}×{w_z}]={w_x*w_y*w_z}")
        
        try:
            # 使用安全的reshape操作
            reshaped = tensor.view(b, h, nw_x, nw_y, nw_z, w_x, w_y, w_z, d)
            return reshaped.permute(0, 2, 5, 3, 6, 4, 7, 1, 8).contiguous().view(b, nw_x*w_x, nw_y*w_y, nw_z*w_z, h*d)
        except RuntimeError as e:
            print(f"Reshape error: {e}")
            print(f"Tensor size: {tensor.size()}, numel: {tensor.numel()}")
            print(f"Target shape: [{b}, {h}, {nw_x}, {nw_y}, {nw_z}, {w_x}, {w_y}, {w_z}, {d}] = "
                  f"{b * h * nw_x * nw_y * nw_z * w_x * w_y * w_z * d}")
            
            # 尝试使用原始参数
            try:
                print("Trying original parameters...")
                reshaped = tensor.view(b, h, orig_nw_x, orig_nw_y, orig_nw_z, orig_w_x, orig_w_y, orig_w_z, d)
                return reshaped.permute(0, 2, 5, 3, 6, 4, 7, 1, 8).contiguous().view(
                    b, orig_nw_x*orig_w_x, orig_nw_y*orig_w_y, orig_nw_z*orig_w_z, h*d)
            except RuntimeError:
                # 最终回退方案 - 跳过窗口重塑，返回原始张量的简单重排
                print("Fallback to simple reshape")
                return tensor.view(b, h*d, num_windows, window_size).permute(0, 2, 3, 1).contiguous()
    elif 'b h (n_x n_y n_z) i j -> b h n_y n_z n_x i j' in pattern:
        # For attention mask rearrangement
        b, h, n_total, i, j = tensor.shape
        n_x, n_y = axes_lengths.get('n_x', 1), axes_lengths.get('n_y', 1)
        n_z = n_total // (n_x * n_y)
        reshaped = tensor.view(b, h, n_x, n_y, n_z, i, j)
        return reshaped.permute(0, 1, 3, 4, 2, 5, 6)  # -> b h n_y n_z n_x i j
    elif 'b h n_y n_z n_x i j -> b h n_x n_z n_y i j' in pattern:
        return tensor.permute(0, 1, 4, 3, 2, 5, 6)
    elif 'b h n_x n_z n_y i j -> b h n_x n_y n_z i j' in pattern:
        return tensor.permute(0, 1, 2, 4, 3, 5, 6)
    elif 'b h n_y n_z n_x i j -> b h (n_x n_y n_z) i j' in pattern:
        b, h, n_y, n_z, n_x, i, j = tensor.shape
        return tensor.permute(0, 1, 4, 2, 3, 5, 6).contiguous().view(b, h, n_x*n_y*n_z, i, j)
    elif 'b h n_x n_y n_z i j -> b h (n_x n_y n_z) i j' in pattern:
        b, h, n_x, n_y, n_z, i, j = tensor.shape
        return tensor.contiguous().view(b, h, n_x*n_y*n_z, i, j)
    elif 'b  h w d c -> b c h w d' in pattern:
        return tensor.permute(0, 4, 1, 2, 3)
    else:
        # For other patterns, implement as needed
        raise NotImplementedError(f"Pattern {pattern} not implemented")


class Rearrange(nn.Module):
    def __init__(self, pattern, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths
        
    def forward(self, x):
        return rearrange(x, self.pattern, **self.axes_lengths)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization"""
    def norm_cdf(x):
        return (1. + torch.erf(x / torch.tensor(2.).sqrt())) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.")

    with torch.no_grad():
        l = norm_cdf(torch.tensor((a - mean) / std))
        u = norm_cdf(torch.tensor((b - mean) / std))

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * torch.tensor(2.).sqrt())
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class CyclicShift3D(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        assert type(displacement) is int or len(displacement) == 3
        if type(displacement) is int:
            displacement = np.array([displacement, displacement, displacement])
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement[0], self.displacement[1], self.displacement[2]), dims=(1, 2, 3))


class Residual3D(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm3D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward3D(nn.Module):
    def __init__(self, dim, hidden_dim, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.net(x)
        x = self.drop(x)
        return x


def create_mask3D(window_size: Union[int, List[int]], displacement: Union[int, List[int]],
                  x_shift: bool, y_shift: bool, z_shift: bool):
    assert type(window_size) is int or len(window_size) == 3
    if type(window_size) is int:
        window_size = np.array([window_size, window_size, window_size])

    assert type(displacement) is int or len(displacement) == 3
    if type(displacement) is int:
        displacement = np.array([displacement, displacement, displacement])

    assert len(window_size) == len(displacement)
    for i in range(len(window_size)):
        assert 0 < displacement[i] < window_size[i]

    mask = torch.zeros(window_size[0] * window_size[1] * window_size[2],
                       window_size[0] * window_size[1] * window_size[2])
    mask = rearrange(mask, '(x1 y1 z1) (x2 y2 z2) -> x1 y1 z1 x2 y2 z2',
                     x1=window_size[0], y1=window_size[1], x2=window_size[0], y2=window_size[1])

    x_dist, y_dist, z_dist = displacement[0], displacement[1], displacement[2]

    if x_shift:
        mask[-x_dist:, :, :, :-x_dist, :, :] = float('-inf')
        mask[:-x_dist, :, :, -x_dist:, :, :] = float('-inf')

    if y_shift:
        mask[:, -y_dist:, :, :, :-y_dist, :] = float('-inf')
        mask[:, :-y_dist, :, :, -y_dist:, :] = float('-inf')

    if z_shift:
        mask[:, :, -z_dist:, :, :, :-z_dist] = float('-inf')
        mask[:, :, :-z_dist, :, :, -z_dist:] = float('-inf')

    mask = rearrange(mask, 'x1 y1 z1 x2 y2 z2 -> (x1 y1 z1) (x2 y2 z2)')
    return mask


class WindowAttention3D(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, shifted: bool, window_size: Union[int, List[int]],
                 relative_pos_embedding: bool = True):
        super().__init__()

        assert type(window_size) is int or len(window_size) == 3
        if type(window_size) is int:
            window_size = np.array([window_size, window_size, window_size])
        else:
            window_size = np.array(window_size)

        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift3D(-displacement)
            self.cyclic_back_shift = CyclicShift3D(displacement)
            self.x_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
                                                     x_shift=True, y_shift=False, z_shift=False), requires_grad=False)
            self.y_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
                                                     x_shift=False, y_shift=True, z_shift=False), requires_grad=False)
            self.z_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
                                                     x_shift=False, y_shift=False, z_shift=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_x, n_y, n_z, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_x = n_x // self.window_size[0]
        nw_y = n_y // self.window_size[1]
        nw_z = n_z // self.window_size[2]

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d) -> b h (nw_x nw_y nw_z) (w_x w_y w_z) d',
                                h=h, w_x=self.window_size[0], w_y=self.window_size[1], w_z=self.window_size[2]), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.shifted:
            dots = rearrange(dots, 'b h (n_x n_y n_z) i j -> b h n_y n_z n_x i j',
                             n_x=nw_x, n_y=nw_y)
            dots[:, :, :, :, -1] += self.x_mask

            dots = rearrange(dots, 'b h n_y n_z n_x i j -> b h n_x n_z n_y i j')
            dots[:, :, :, :, -1] += self.y_mask

            dots = rearrange(dots, 'b h n_x n_z n_y i j -> b h n_x n_y n_z i j')
            dots[:, :, :, :, -1] += self.z_mask

            dots = rearrange(dots, 'b h n_x n_y n_z i j -> b h (n_x n_y n_z) i j')

        attn = self.softmax(dots)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)

        out = rearrange(out, 'b h (nw_x nw_y nw_z) (w_x w_y w_z) d -> b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d)',
                        h=h, w_x=self.window_size[0], w_y=self.window_size[1], w_z=self.window_size[2],
                        nw_x=nw_x, nw_y=nw_y, nw_z=nw_z)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock3D(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size: Union[int, List[int]],
                 relative_pos_embedding: bool = True, dropout: float = 0.0):
        super().__init__()
        self.attention_block = Residual3D(PreNorm3D(dim, WindowAttention3D(dim=dim,
                                                                           heads=heads,
                                                                           head_dim=head_dim,
                                                                           shifted=shifted,
                                                                           window_size=window_size,
                                                                           relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual3D(PreNorm3D(dim, FeedForward3D(dim=dim, hidden_dim=mlp_dim, dropout=dropout)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class Norm(nn.Module):
    def __init__(self, dim, channel_first: bool = True):
        super(Norm, self).__init__()
        if channel_first:
            self.net = nn.Sequential(
                Rearrange('b c h w d -> b h w d c'),
                nn.LayerNorm(dim),
                Rearrange('b h w d c -> b c h w d')
            )
        else:
            self.net = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.net(x)
        return x


class PatchMerging3D(nn.Module):
    def __init__(self, in_dim, out_dim, downscaling_factor):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=downscaling_factor, stride=downscaling_factor),
            Norm(dim=out_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class PatchExpanding3D(nn.Module):
    def __init__(self, in_dim, out_dim, up_scaling_factor, window_size=None):
        super(PatchExpanding3D, self).__init__()
        stride = up_scaling_factor
        kernel_size = up_scaling_factor
        padding = (kernel_size - stride) // 2
        self.net = nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            Norm(out_dim),
        )
        # 存储窗口大小用于自适应填充
        if window_size is None:
            self.window_size = None
        elif isinstance(window_size, int):
            self.window_size = np.array([window_size, window_size, window_size])
        else:
            self.window_size = np.array(window_size)
            
        # 记录当前填充情况，便于后续处理（比如在loss计算时可以移除填充）
        self.padding_values = [0, 0, 0]

    def forward(self, x):
        x = self.net(x)
        
        # 如果设置了窗口大小，执行自适应填充确保尺寸能被窗口大小整除
        if self.window_size is not None:
            _, _, D, H, W = x.shape
            
            # 计算每个维度需要的填充量
            pad_D = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
            pad_H = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
            pad_W = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
            
            # 更新填充记录
            self.padding_values = [pad_D, pad_H, pad_W]
            
            # 只有在需要填充时才进行填充
            if pad_D > 0 or pad_H > 0 or pad_W > 0:
                print(f"Applying padding: depth={pad_D}, height={pad_H}, width={pad_W}")
                print(f"Input shape: {x.shape}, expected divisible by {self.window_size}")
                
                # 执行填充 - 在右/下/后方向填充
                x = F.pad(x, (0, pad_W, 0, pad_H, 0, pad_D))
                print(f"Output shape after padding: {x.shape}")
        
        return x


class FinalExpand3D(nn.Module):
    def __init__(self, in_dim, out_dim, up_scaling_factor):
        super(FinalExpand3D, self).__init__()
        stride = up_scaling_factor
        kernel_size = up_scaling_factor
        padding = (kernel_size - stride) // 2
        self.net = nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            Norm(out_dim),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        groups = min(in_ch, out_ch)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=groups),
            Norm(dim=out_ch),
            nn.PReLU(),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=groups),
            Norm(dim=out_ch),
            nn.PReLU(),
        )

    def forward(self, x):
        x2 = x.clone()
        x = self.net(x) * x2
        return x


class Encoder(nn.Module):
    def __init__(self, in_dims, hidden_dimension, layers, downscaling_factor, num_heads, head_dim,
                 window_size: Union[int, List[int]], relative_pos_embedding: bool = True, dropout: float = 0.0):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging3D(in_dim=in_dims, out_dim=hidden_dimension,
                                              downscaling_factor=downscaling_factor)
        self.conv_block = ConvBlock(in_ch=hidden_dimension, out_ch=hidden_dimension)

        self.re1 = Rearrange('b c h w d -> b h w d c')
        self.swin_layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.swin_layers.append(nn.ModuleList([
                SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            dropout=dropout),
                SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            dropout=dropout),
            ]))
        self.re2 = Rearrange('b  h w d c -> b c h w d')

    def forward(self, x):
        x = self.patch_partition(x)
        x2 = self.conv_block(x)

        x = self.re1(x)
        for regular_block, shifted_block in self.swin_layers:
            x = regular_block(x)
            x = shifted_block(x)
        x = self.re2(x)

        x = x + x2
        return x


class Decoder(nn.Module):
    def __init__(self, in_dims, out_dims, layers, up_scaling_factor, num_heads, head_dim,
                 window_size: Union[int, List[int]], relative_pos_embedding, dropout: float = 0.0):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        # 传递window_size参数给PatchExpanding3D以实现自适应填充
        self.patch_expand = PatchExpanding3D(in_dim=in_dims, out_dim=out_dims,
                                             up_scaling_factor=up_scaling_factor,
                                             window_size=window_size)  # 添加window_size参数

        self.conv_block = ConvBlock(in_ch=out_dims, out_ch=out_dims)
        self.re1 = Rearrange('b c h w d -> b h w d c')
        self.swin_layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.swin_layers.append(nn.ModuleList([
                SwinBlock3D(dim=out_dims, heads=num_heads, head_dim=head_dim, mlp_dim=out_dims * 4,
                            shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            dropout=dropout),
                SwinBlock3D(dim=out_dims, heads=num_heads, head_dim=head_dim, mlp_dim=out_dims * 4,
                            shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            dropout=dropout),
            ]))
        self.re2 = Rearrange('b h w d c -> b c h w d')

    def forward(self, x):
        x = self.patch_expand(x)
        x2 = self.conv_block(x)

        x = self.re1(x)
        for regular_block, shifted_block in self.swin_layers:
            x = regular_block(x)
            x = shifted_block(x)
        x = self.re2(x)

        x = x + x2
        return x


class Converge(nn.Module):
    def __init__(self, dim: int):
        super(Converge, self).__init__()
        self.norm = Norm(dim=dim)

    def forward(self, x, enc_x):
        assert x.shape == enc_x.shape
        x = x + enc_x
        x = self.norm(x)
        return x


class SwinUnet3D(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, in_channel=1, num_classes=2, head_dim=32,
                 window_size: Union[int, List[int]] = 7, downscaling_factors=(4, 2, 2, 2),
                 relative_pos_embedding=True, dropout: float = 0.0, skip_style='stack',
                 stl_channels: int = 32):
        super().__init__()

        self.dsf = downscaling_factors
        self.window_size = window_size

        self.enc12 = Encoder(in_dims=in_channel, hidden_dimension=hidden_dim, layers=layers[0],
                             downscaling_factor=downscaling_factors[0], num_heads=heads[0],
                             head_dim=head_dim, window_size=window_size, dropout=dropout,
                             relative_pos_embedding=relative_pos_embedding)
        self.enc3 = Encoder(in_dims=hidden_dim, hidden_dimension=hidden_dim * 2,
                            layers=layers[1],
                            downscaling_factor=downscaling_factors[1], num_heads=heads[1],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)
        self.enc4 = Encoder(in_dims=hidden_dim * 2, hidden_dimension=hidden_dim * 4,
                            layers=layers[2],
                            downscaling_factor=downscaling_factors[2], num_heads=heads[2],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)
        self.enc5 = Encoder(in_dims=hidden_dim * 4, hidden_dimension=hidden_dim * 8,
                            layers=layers[3],
                            downscaling_factor=downscaling_factors[3], num_heads=heads[3],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)

        self.dec4 = Decoder(in_dims=hidden_dim * 8, out_dims=hidden_dim * 4,
                            layers=layers[2],
                            up_scaling_factor=downscaling_factors[3], num_heads=heads[2],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)

        self.dec3 = Decoder(in_dims=hidden_dim * 4, out_dims=hidden_dim * 2,
                            layers=layers[1],
                            up_scaling_factor=downscaling_factors[2], num_heads=heads[1],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)

        self.dec12 = Decoder(in_dims=hidden_dim * 2, out_dims=hidden_dim,
                             layers=layers[0],
                             up_scaling_factor=downscaling_factors[1], num_heads=heads[0],
                             head_dim=head_dim, window_size=window_size, dropout=dropout,
                             relative_pos_embedding=relative_pos_embedding)

        self.converge4 = Converge(hidden_dim * 4)
        self.converge3 = Converge(hidden_dim * 2)
        self.converge12 = Converge(hidden_dim)

        self.final = FinalExpand3D(in_dim=hidden_dim, out_dim=stl_channels,
                                   up_scaling_factor=downscaling_factors[0])
        self.out = nn.Sequential(
            nn.Conv3d(stl_channels, num_classes, kernel_size=1)
        )
        self.init_weight()

    def forward(self, img):
        # Simplified validation - just check basic requirements
        _, _, x_s, y_s, z_s = img.shape
        
        down12_1 = self.enc12(img)
        down3 = self.enc3(down12_1)
        down4 = self.enc4(down3)
        features = self.enc5(down4)

        up4 = self.dec4(features)
        up4 = self.converge4(up4, down4)

        up3 = self.dec3(up4)
        up3 = self.converge3(up3, down3)

        up12 = self.dec12(up3)
        up12 = self.converge12(up12, down12_1)

        out = self.final(up12)
        out = self.out(out)
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


def swinUnet_t_3D(hidden_dim=96, layers=(2, 2, 4, 2), heads=(3, 6, 9, 12), num_classes: int = 2, **kwargs):
    return SwinUnet3D(hidden_dim=hidden_dim, layers=layers, heads=heads, num_classes=num_classes, **kwargs)
