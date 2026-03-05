"""
DeepLabVLaura — Faithful implementation of the MTFCsp architecture from:

    Cué La Rosa et al., "Multi-task fully convolutional network for tree
    species mapping in dense forests using small training hyperspectral
    data", ISPRS J. Photogramm. Remote Sens., 2021.
    arXiv:2106.00799v2

Encoder:  ResNet-9 (1 initial 3×3 conv + 3 residual blocks = 9 conv ops)
Decoder1: DeepLabv3+ semantic segmentation (ASPP with rates 3/6/9)
Decoder2: Distance-map regression (two CBs + sigmoid)
"""

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Pre-activation residual block (BN → ELU → Conv) × 2.

    When *downsample* is True the first convolution uses stride 2 and a
    parallel 1×1 shortcut projection adapts the identity path.
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: bool) -> None:
        super().__init__()
        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.9)
        self.elu1 = nn.ELU(inplace=True)

        stride1 = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride1, padding=1)

        if downsample:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=2)

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.elu2 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.elu1(out)
        out = self.conv1(out)

        if self.downsample:
            identity = self.shortcut(x)

        out = self.bn2(out)
        out = self.elu2(out)
        out = self.conv2(out)

        return out + identity


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with rates **3, 6, 9** (paper §3.3.2).

    Five parallel branches: 1×1 conv, 3×3@r=3, 3×3@r=6, 3×3@r=9, image-pool.
    Outputs are concatenated → BN → ELU  (5 × *out_channels* channels).
    """

    def __init__(self, in_channels: int, out_channels: int = 128) -> None:
        super().__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.atrous_r3 = nn.Conv2d(in_channels, out_channels, 3, padding=3, dilation=3)
        self.atrous_r6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.atrous_r9 = nn.Conv2d(in_channels, out_channels, 3, padding=9, dilation=9)

        self.image_pool_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels * 5, momentum=0.9)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]

        b1 = self.conv1x1(x)
        b2 = self.atrous_r3(x)
        b3 = self.atrous_r6(x)
        b4 = self.atrous_r9(x)
        b5 = F.interpolate(
            self.image_pool_conv(x), size=size,
            mode="bilinear", align_corners=True,
        )

        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return self.elu(self.bn(out))


class ConvBlock(nn.Module):
    """CB as defined in the paper: 3×3 conv → BN → ELU → bilinear ×2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)


# ---------------------------------------------------------------------------
# Full architecture
# ---------------------------------------------------------------------------

class DeepLabVLaura(nn.Module):
    """MTFCsp network faithful to Cué La Rosa et al. (2021).

    Parameters
    ----------
    num_ch : int
        Number of input spectral channels.
    psize : int
        Spatial size of the square input patches (must be divisible by 4).
    num_class : int
        Number of semantic classes.
    dropout_rate : float
        Dropout probability before the classification head (paper: 0.65).
    """

    def __init__(
        self,
        num_ch: int,
        psize: int,
        num_class: int,
        dropout_rate: float = 0.65,
    ) -> None:
        super().__init__()

        if psize % 4 != 0:
            raise ValueError(f"psize must be divisible by 4, got {psize}")

        self.num_ch = num_ch
        self.psize = psize
        self.nb_class = num_class

        # ── Encoder (ResNet-9) ──────────────────────────────────────────
        # 9 conv ops: 1 initial + 3 blocks × 2 convs + 2 shortcut projections
        self.conv1 = nn.Conv2d(num_ch, 64, 3, stride=1, padding=1)
        self.block1 = ResidualBlock(64, 128, downsample=True)    # /2
        self.block2 = ResidualBlock(128, 256, downsample=True)   # /4
        self.block3 = ResidualBlock(256, 256, downsample=False)
        self.encoder_bn = nn.BatchNorm2d(256, momentum=0.9)
        self.encoder_elu = nn.ELU(inplace=True)

        # ── Classification decoder (DeepLabv3+) ────────────────────────
        self.aspp = ASPPModule(256, 128)                         # → 640 ch
        self.reduce = nn.Conv2d(640, 128, 1)
        self.reduce_bn = nn.BatchNorm2d(128, momentum=0.9)
        self.reduce_elu = nn.ELU(inplace=True)

        self.seg_cb1 = ConvBlock(128, 128)                       # ×2 → psize/2
        self.seg_cb2 = ConvBlock(128 + 128, 128)                 # ×2 → psize
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv_class = nn.Conv2d(128, num_class, 1)

        # ── Regression decoder (distance map) ──────────────────────────
        self.reg_cb1 = ConvBlock(256, 128)                       # ×2 → psize/2
        self.reg_cb2 = ConvBlock(128 + 128, 128)                 # ×2 → psize
        self.conv_reg = nn.Conv2d(128, 1, 3, padding=1)

    # ----- sub-networks ------------------------------------------------

    def encoder(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x_skip = x                  # 128 ch @ psize/2
        x = self.block2(x)
        x = self.block3(x)
        x = self.encoder_bn(x)
        x = self.encoder_elu(x)     # 256 ch @ psize/4
        return x, x_skip

    def decoder_class(self, x, x_skip):
        x = self.aspp(x)
        x = self.reduce(x)
        x = self.reduce_bn(x)
        x = self.reduce_elu(x)

        x = self.seg_cb1(x)                          # 128 ch @ psize/2
        x = torch.cat([x_skip, x], dim=1)            # 256 ch @ psize/2
        x = self.seg_cb2(x)                           # 128 ch @ psize

        x = self.dropout(x)
        x = self.conv_class(x)
        return x

    def decoder_aux(self, x, x_skip):
        x = self.reg_cb1(x)                           # 128 ch @ psize/2
        x = torch.cat([x_skip, x], dim=1)             # 256 ch @ psize/2
        x = self.reg_cb2(x)                            # 128 ch @ psize

        x = self.conv_reg(x)
        return x

    # ----- forward -----------------------------------------------------

    def forward(self, x):
        x, x_skip = self.encoder(x)
        return {
            "out": self.decoder_class(x, x_skip),
            "aux": self.decoder_aux(x, x_skip),
        }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = DeepLabVLaura(num_ch=25, psize=128, num_class=14)
    print(model)

    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")

    dummy = torch.randn(2, 25, 128, 128)
    with torch.no_grad():
        out = model(dummy)
    print(f"Classification head shape: {out['out'].shape}")   # (2, 14, 128, 128)
    print(f"Distance-map head shape:   {out['aux'].shape}")   # (2,  1, 128, 128)
