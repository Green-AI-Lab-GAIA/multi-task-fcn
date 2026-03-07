"""
DeepLabVLaura — Faithful implementation of the MTFCsp architecture from:

    Cué La Rosa et al., "Multi-task fully convolutional network for tree
    species mapping in dense forests using small training hyperspectral
    data", ISPRS J. Photogramm. Remote Sens., 2021.
    arXiv:2106.00799v2

Encoder:  ResNet-9 (default), ResNet-18, or ResNet-50 (selectable via encoder_name parameter)
Decoder1: DeepLabv3+ semantic segmentation (ASPP with rates 3/6/9)
Decoder2: Distance-map regression (two CBs + sigmoid)
"""

from typing import Literal

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
    A 1×1 shortcut is also used when in_channels != out_channels (even without downsampling).
    """

    def __init__(self, in_channels: int, out_channels: int, downsample: bool) -> None:
        super().__init__()
        self.downsample = downsample
        self.needs_projection = (in_channels != out_channels) or downsample

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.9)
        self.elu1 = nn.ELU(inplace=True)

        stride1 = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride1, padding=1)

        if self.needs_projection:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride1)

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.elu2 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.elu1(out)
        out = self.conv1(out)

        if self.needs_projection:
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
# Encoder definitions
# ---------------------------------------------------------------------------

def _make_stage(in_ch: int, out_ch: int, num_blocks: int, downsample_first: bool) -> nn.Sequential:
    """Create a stage with multiple basic residual blocks."""
    blocks = []
    blocks.append(ResidualBlock(in_ch, out_ch, downsample=downsample_first))
    for _ in range(1, num_blocks):
        blocks.append(ResidualBlock(out_ch, out_ch, downsample=False))
    return nn.Sequential(*blocks)


class BottleneckBlock(nn.Module):
    """Pre-activation bottleneck block: (BN→ELU→1x1) → (BN→ELU→3x3) → (BN→ELU→1x1).

    Follows the same pre-activation style as ResidualBlock but uses the
    bottleneck pattern from ResNet-50/101/152.
    Output channels = mid_channels * EXPANSION (4).
    """

    EXPANSION = 4

    def __init__(self, in_channels: int, mid_channels: int, downsample: bool) -> None:
        super().__init__()
        out_channels = mid_channels * self.EXPANSION
        stride = 2 if downsample else 1
        self.needs_projection = (in_channels != out_channels) or downsample

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.9)
        self.elu1 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)

        self.bn2 = nn.BatchNorm2d(mid_channels, momentum=0.9)
        self.elu2 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3,
                               stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(mid_channels, momentum=0.9)
        self.elu3 = nn.ELU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

        if self.needs_projection:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1,
                                      stride=stride, bias=False)

    def forward(self, x):
        identity = x

        out = self.elu1(self.bn1(x))
        if self.needs_projection:
            identity = self.shortcut(out)
        out = self.conv1(out)

        out = self.elu2(self.bn2(out))
        out = self.conv2(out)

        out = self.elu3(self.bn3(out))
        out = self.conv3(out)

        return out + identity


def _make_bottleneck_stage(in_ch: int, mid_ch: int, num_blocks: int,
                           downsample_first: bool) -> nn.Sequential:
    """Create a stage with multiple bottleneck blocks."""
    blocks = []
    blocks.append(BottleneckBlock(in_ch, mid_ch, downsample=downsample_first))
    out_ch = mid_ch * BottleneckBlock.EXPANSION
    for _ in range(1, num_blocks):
        blocks.append(BottleneckBlock(out_ch, mid_ch, downsample=False))
    return nn.Sequential(*blocks)


class ResNet9Encoder(nn.Module):
    """ResNet-9 encoder: 1 initial conv + 3 residual blocks.

    Output: (features @ psize/4 with 256 ch, skip @ psize/2 with 128 ch)
    """

    def __init__(self, num_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_ch, 64, 3, stride=1, padding=1)
        self.block1 = ResidualBlock(64, 128, downsample=True)    # /2
        self.block2 = ResidualBlock(128, 256, downsample=True)   # /4
        self.block3 = ResidualBlock(256, 256, downsample=False)
        self.bn = nn.BatchNorm2d(256, momentum=0.9)
        self.elu = nn.ELU(inplace=True)

        self.out_channels = 256
        self.skip_channels = 128

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x_skip = x                  # 128 ch @ psize/2
        x = self.block2(x)
        x = self.block3(x)
        x = self.bn(x)
        x = self.elu(x)             # 256 ch @ psize/4
        return x, x_skip


class ResNet18Encoder(nn.Module):
    """ResNet-18 encoder: 1 initial conv + 4 stages with [2,2,2,2] blocks.

    Follows the same pre-activation style (BN→ELU→Conv) as ResNet-9.
    Output: (features @ psize/4 with 512 ch, skip @ psize/2 with 128 ch)

    Architecture:
        conv1: num_ch → 64         @ psize
        stage1: 64 → 64, 2 blocks  @ psize     (no downsample)
        stage2: 64 → 128, 2 blocks @ psize/2   (downsample) ← skip connection
        stage3: 128 → 256, 2 blocks @ psize/4  (downsample)
        stage4: 256 → 512, 2 blocks @ psize/4  (no downsample, to keep psize/4)
    """

    def __init__(self, num_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_ch, 64, 3, stride=1, padding=1)
        self.stage1 = _make_stage(64, 64, num_blocks=2, downsample_first=False)   # psize
        self.stage2 = _make_stage(64, 128, num_blocks=2, downsample_first=True)   # psize/2
        self.stage3 = _make_stage(128, 256, num_blocks=2, downsample_first=True)  # psize/4
        self.stage4 = _make_stage(256, 512, num_blocks=2, downsample_first=False) # psize/4
        self.bn = nn.BatchNorm2d(512, momentum=0.9)
        self.elu = nn.ELU(inplace=True)

        self.out_channels = 512
        self.skip_channels = 128

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)          # 64 ch @ psize
        x = self.stage2(x)
        x_skip = x                  # 128 ch @ psize/2
        x = self.stage3(x)          # 256 ch @ psize/4
        x = self.stage4(x)          # 512 ch @ psize/4
        x = self.bn(x)
        x = self.elu(x)
        return x, x_skip


class ResNet50Encoder(nn.Module):
    """ResNet-50 encoder with bottleneck blocks [3, 4, 6, 3].

    Follows the same pre-activation style (BN→ELU→Conv) as the other encoders.
    Downsampling is limited to /4 to match the decoder expectations.
    Output: (features @ psize/4 with 2048 ch, skip @ psize/2 with 512 ch)

    Architecture:
        conv1:  num_ch → 64          @ psize
        stage1: 64  → 256,  3 blocks @ psize     (no downsample)
        stage2: 256 → 512,  4 blocks @ psize/2   (downsample) ← skip connection
        stage3: 512 → 1024, 6 blocks @ psize/4   (downsample)
        stage4: 1024→ 2048, 3 blocks @ psize/4   (no downsample, to keep psize/4)
    """

    def __init__(self, num_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_ch, 64, 3, stride=1, padding=1)
        self.stage1 = _make_bottleneck_stage(64,   64,  num_blocks=3, downsample_first=False)  # → 256
        self.stage2 = _make_bottleneck_stage(256,  128, num_blocks=4, downsample_first=True)   # → 512
        self.stage3 = _make_bottleneck_stage(512,  256, num_blocks=6, downsample_first=True)   # → 1024
        self.stage4 = _make_bottleneck_stage(1024, 512, num_blocks=3, downsample_first=False)  # → 2048
        self.bn = nn.BatchNorm2d(2048, momentum=0.9)
        self.elu = nn.ELU(inplace=True)

        self.out_channels = 2048
        self.skip_channels = 512

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)          # 256 ch @ psize
        x = self.stage2(x)
        x_skip = x                  # 512 ch @ psize/2
        x = self.stage3(x)          # 1024 ch @ psize/4
        x = self.stage4(x)          # 2048 ch @ psize/4
        x = self.bn(x)
        x = self.elu(x)
        return x, x_skip


ENCODERS = {
    "resnet9": ResNet9Encoder,
    "resnet18": ResNet18Encoder,
    "resnet50": ResNet50Encoder,
}


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
    encoder_name : {"resnet9", "resnet18", "resnet50"}
        Which encoder backbone to use. Default is "resnet9" (original paper).
    """

    def __init__(
        self,
        num_ch: int,
        psize: int,
        num_class: int,
        dropout_rate: float = 0.65,
        encoder_name: Literal["resnet9", "resnet18", "resnet50"] = "resnet9",
    ) -> None:
        super().__init__()

        if psize % 4 != 0:
            raise ValueError(f"psize must be divisible by 4, got {psize}")
        if encoder_name not in ENCODERS:
            raise ValueError(f"encoder_name must be one of {list(ENCODERS.keys())}, got {encoder_name}")

        self.num_ch = num_ch
        self.psize = psize
        self.nb_class = num_class
        self.encoder_name = encoder_name

        # ── Encoder ─────────────────────────────────────────────────────
        self._encoder = ENCODERS[encoder_name](num_ch)
        enc_out_ch = self._encoder.out_channels      # 256 for resnet9, 512 for resnet18
        skip_ch = self._encoder.skip_channels        # 128 for both

        # ── Classification decoder (DeepLabv3+) ────────────────────────
        self.aspp = ASPPModule(enc_out_ch, 128)                   # → 640 ch
        self.reduce = nn.Conv2d(640, 128, 1)
        self.reduce_bn = nn.BatchNorm2d(128, momentum=0.9)
        self.reduce_elu = nn.ELU(inplace=True)

        self.seg_cb1 = ConvBlock(128, 128)                        # ×2 → psize/2
        self.seg_cb2 = ConvBlock(128 + skip_ch, 128)              # ×2 → psize
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv_class = nn.Conv2d(128, num_class, 1)

        # ── Regression decoder (distance map) ──────────────────────────
        self.reg_cb1 = ConvBlock(enc_out_ch, 128)                 # ×2 → psize/2
        self.reg_cb2 = ConvBlock(128 + skip_ch, 128)              # ×2 → psize
        self.conv_reg = nn.Conv2d(128, 1, 3, padding=1)

    # ----- sub-networks ------------------------------------------------

    def encoder(self, x):
        return self._encoder(x)

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
    for enc_name in ["resnet9", "resnet18", "resnet50"]:
        print(f"\n{'='*60}")
        print(f"Testing encoder: {enc_name}")
        print('='*60)

        model = DeepLabVLaura(num_ch=25, psize=128, num_class=14, encoder_name=enc_name)

        total = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total:,}")

        dummy = torch.randn(2, 25, 128, 128)
        with torch.no_grad():
            out = model(dummy)
        print(f"Classification head shape: {out['out'].shape}")   # (2, 14, 128, 128)
        print(f"Distance-map head shape:   {out['aux'].shape}")   # (2,  1, 128, 128)
