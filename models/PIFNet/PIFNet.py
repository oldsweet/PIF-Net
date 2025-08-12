import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.ops import DeformConv2d

from .mamba import VMamba
# ========= 1. Large Kernel Attention Module =========
class LargeKernelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_conv1 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.dw_conv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, dilation=3, groups=dim)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.dw_conv1(x)
        x = self.dw_conv2(x)
        x = self.pw_conv(x)
        return x * identity

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


# ========= 2. Squeeze-and-Excitation (SE) Module =========
class SEModule(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(self.pool(x))
        return x * scale

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


# ========= 3. FC-Norm-Attn-Norm Block =========
class FCNormAttnNorm(nn.Module):
    def __init__(self, dim, fc_ratio=1.0, norm_type='bn', lora_alpha=0.1):
        super().__init__()
        hidden_dim = int(dim * fc_ratio)
        self.alpha = lora_alpha

        # Channel transformation layers
        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # Normalization layers (currently only norm1 and norm2 used; can be extended)
        if norm_type == 'ln':
            self.norm1 = nn.GroupNorm(1, hidden_dim)
            self.norm2 = nn.GroupNorm(1, hidden_dim)
            self.norm3 = nn.GroupNorm(1, hidden_dim)
            self.norm4 = nn.GroupNorm(1, hidden_dim)
        else:
            self.norm1 = nn.BatchNorm2d(hidden_dim)
            self.norm2 = nn.BatchNorm2d(hidden_dim)

        # Attention modules
        self.lka = LargeKernelAttention(hidden_dim)
        self.se = SEModule(hidden_dim)

        # Frozen output projection layer acting as the backbone convolution
        self.out_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.out_proj.weight.requires_grad = False  # Freeze weights

        # Trainable LoRA branches with initialized weights
        self.lora_A = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        init.kaiming_uniform_(self.lora_A.weight)
        init.zeros_(self.lora_B.weight)

    def forward(self, x, x_f):
        residual = x
        x = self.conv1(x)
        x = self.lka(x) + self.se(x)
        x = self.conv2(x)
        self.lka.freeze()
        x = self.conv3(x)
        x = self.lka(x) + self.se(x + x_f)
        
        lora_feat = self.lora_B(self.lora_A(x_f)) * self.alpha
        
        x = lora_feat + x
        x = self.conv4(x)
        self.lka.unfreeze()
        self.se.unfreeze()
        return x + residual


# ========= 4. MobileNetV2 Inverted Residual Block =========
class InvertedResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.bottleneck(x)


# ========= 5. InBlock: Interaction module between two streams =========
class InBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            VMamba(dim),  # θ_φ
            VMamba(dim),  # θ_ρ
            VMamba(dim),  # θ_η
        ])

    def forward(self, z1, z2):
        z2 = z2 + self.blocks[0](z1)  # φ(z1)
        z1 = z1 * torch.exp(self.blocks[1](z2)) + self.blocks[2](z2)  # ρ(z2), η(z2)
        return z1, z2


# ========= 6. 2D Haar Wavelet Decomposition =========
def dwt_init(x):
    """
    2D Haar wavelet decomposition
    Input:
        x: (B, C, H, W)
    Output:
        LL, HL, LH, HH - low-frequency and high-frequency subbands
    """
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    LL = x1 + x2 + x3 + x4
    HL = -x1 - x2 + x3 + x4
    LH = -x1 + x2 - x3 + x4
    HH = x1 - x2 - x3 + x4
    return LL, HL, LH, HH


# ========= 7. 2D Haar Wavelet Inverse Transform =========
def iwt_init(x):
    """
    2D Haar wavelet inverse transform
    Input:
        x: (B, 4*C, H/2, W/2)
    Output:
        Reconstructed image (B, C, H, W)
    """
    r = 2
    B, C4, H, W = x.size()
    C = C4 // 4

    x1 = x[:, :C, :, :] / 2
    x2 = x[:, C:2*C, :, :] / 2
    x3 = x[:, 2*C:3*C, :, :] / 2
    x4 = x[:, 3*C:4*C, :, :] / 2

    out = torch.zeros([B, C, H * r, W * r], dtype=x.dtype, device=x.device)

    out[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    out[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    out[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    out[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return out


# ========= 8. Simple Convolution Head Module =========
class Head(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


# ========= 9. Main Model Structure: MoEFusion =========
class PIFNet(nn.Module):
    def __init__(self, args, dim=64):
        super().__init__()
        self.hsi_bands = args.hsi_bands
        self.msi_bands = args.msi_bands
        self.upscale = args.upscale
        self.dim = dim

        # Input encoders
        self.hsi_head = Head(self.hsi_bands, dim)
        self.msi_head = Head(self.msi_bands, dim)
        self.prior_conv = nn.Conv2d(dim, dim, 3, 1, 1)

        # Output reconstruction head
        self.tail = Head(dim * 2, self.hsi_bands)

        # Frequency information compression and restoration
        self.conv_proj = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.conv_proj_trans = nn.Conv2d(dim, dim * 3, kernel_size=1)

        # Multiple InBlock frequency interaction modules
        self.inblocks = nn.ModuleList([InBlock(dim) for _ in range(4)])

        # Spatial attention modules
        self.spatial_blocks = nn.ModuleList([FCNormAttnNorm(dim, fc_ratio=1.0, norm_type='ln') for _ in range(4)])

        # Semantic alignment modules
        self.sem_blocks = nn.ModuleList([SemanticAlignBlock(dim, dim, kernel_size=3, upscale_factor=2) for _ in range(4)])

    def forward(self, hsi, msi):
        # Upsample HSI to MSI resolution first
        up_hsi = F.interpolate(hsi, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        hsi_feat = self.hsi_head(up_hsi)
        # print(msi.shape) 
        msi_feat = self.msi_head(msi)
        res_prior = F.relu(self.prior_conv(hsi_feat - msi_feat))  # Prior residual

        # Wavelet transform decomposition
        x_LL, x_HL, x_LH, x_HH = dwt_init(hsi_feat)
        z1 = x_LL
        z2 = self.conv_proj(torch.cat([x_HL, x_LH, x_HH], dim=1))

        # Frequency interaction blocks
        for i in range(4):
            z1, z2 = self.inblocks[i](z1, z2)

        # Restore frequency features
        z2 = self.conv_proj_trans(z2)
        feat = iwt_init(torch.cat([z1, z2], dim=1))

        # Spatial attention and semantic alignment
        for i in range(4):
            feat = self.spatial_blocks[i](feat, res_prior)
            feat = self.sem_blocks[i](feat)

        out = self.tail(torch.cat([feat, msi_feat], dim=1))

        return out


# ========= 10. Semantic Alignment Block =========
class SemanticAlignBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upscale_factor):
        super().__init__()
        self.deform_conv1 = DeformConv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.deform_conv2 = DeformConv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv_offset1 = nn.Conv2d(out_channels, 18, kernel_size=3, padding=1)
        self.conv_offset2 = nn.Conv2d(out_channels, 18, kernel_size=3, padding=1)
        self.upscale_factor = upscale_factor
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset1 = self.conv_offset1(x)
        x = self.deform_conv1(x, offset1)
        x = self.relu(x)
        offset2 = self.conv_offset2(x)
        x = self.deform_conv2(x, offset2)
        return x
