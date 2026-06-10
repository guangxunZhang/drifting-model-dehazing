from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _channel_mults_from_depth(depth: int) -> Tuple[int, ...]:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    reference = (1, 2, 3, 4, 6)
    if depth <= len(reference):
        return reference[:depth]
    mults = list(reference)
    while len(mults) < depth:
        mults.append(mults[-1] * 2)
    return tuple(mults)


class ResBlock(nn.Module):
    """Pre-activation residual block using GroupNorm and SiLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.norm1 = nn.GroupNorm(_norm_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(_norm_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + identity


class Downsample(nn.Module):
    """2x spatial downsample via strided 3x3 convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """2x nearest-neighbor upsample followed by a 3x3 convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention over spatial tokens."""

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(_norm_groups(channels), channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        qkv = self.qkv(tokens)
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x + out


class UNetEncoder(nn.Module):
    """Multi-scale encoder returning skip features at each level."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 3, 4, 6),
        n_blocks: int = 2,
        attn_levels: Sequence[int] = (4,),
        num_heads: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.n_blocks = n_blocks
        self.attn_levels = set(attn_levels)
        self.num_heads = num_heads
        self.num_levels = len(self.channel_mults)
        self.skip_channels: List[int] = [
            base_channels * mult for mult in self.channel_mults
        ]

        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.level_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        prev_ch = base_channels
        for level, mult in enumerate(self.channel_mults):
            level_out = base_channels * mult
            blocks = nn.ModuleList()
            for block_idx in range(n_blocks):
                block_in = prev_ch if block_idx == 0 else level_out
                blocks.append(ResBlock(block_in, level_out))
            if level in self.attn_levels:
                blocks.append(AttentionBlock(level_out, num_heads=num_heads))
            self.level_blocks.append(blocks)
            prev_ch = level_out
            if level < self.num_levels - 1:
                self.downsamples.append(Downsample(level_out))
            else:
                self.downsamples.append(nn.Identity())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.stem(x)
        skips: List[torch.Tensor] = []
        for level in range(self.num_levels):
            for block in self.level_blocks[level]:
                h = block(h)
            skips.append(h)
            h = self.downsamples[level](h)
        return skips


class UNetDecoder(nn.Module):
    """Multi-scale decoder consuming encoder skip features."""

    def __init__(
        self,
        skip_channels: Sequence[int],
        n_blocks: int = 2,
        attn_levels: Sequence[int] = (4,),
        num_heads: int = 8,
    ):
        super().__init__()
        self.skip_channels = list(skip_channels)
        self.n_blocks = n_blocks
        self.attn_levels = set(attn_levels)
        self.num_heads = num_heads
        self.num_levels = len(self.skip_channels)
        self.upsamples = nn.ModuleList()
        self.level_blocks = nn.ModuleList()

        prev_ch = self.skip_channels[-1]
        for level in reversed(range(self.num_levels)):
            level_out = self.skip_channels[level]
            if level == self.num_levels - 1:
                self.upsamples.append(nn.Identity())
            else:
                self.upsamples.append(Upsample(prev_ch))

            blocks = nn.ModuleList()
            blocks.append(ResBlock(prev_ch + self.skip_channels[level], level_out))
            for _ in range(n_blocks - 1):
                blocks.append(ResBlock(level_out, level_out))
            if level in self.attn_levels:
                blocks.append(AttentionBlock(level_out, num_heads=num_heads))
            self.level_blocks.append(blocks)
            prev_ch = level_out

        self.out_channels = prev_ch

    def forward(self, bottom: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        if len(skips) != self.num_levels:
            raise ValueError(
                f"Decoder expected {self.num_levels} skip features, got {len(skips)}."
            )
        h = bottom
        for index, level in enumerate(reversed(range(self.num_levels))):
            h = self.upsamples[index](h)
            if h.shape[-2:] != skips[level].shape[-2:]:
                h = F.interpolate(h, size=skips[level].shape[-2:], mode="nearest")
            h = torch.cat([h, skips[level]], dim=1)
            for block in self.level_blocks[index]:
                h = block(h)
        return h


class ResidualUNetDehazer(nn.Module):
    """Residual image-to-image U-Net dehazer.

    The training harness calls models as ``model(z, x_hazy)``. This model ignores
    ``z`` and predicts the final clean image directly from ``x_hazy``.
    """

    outputs_final_prediction = True

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        depth: int = 5,
        channel_mults: Optional[Sequence[int]] = None,
        n_blocks: int = 2,
        attn_levels: Optional[Sequence[int]] = None,
        num_heads: int = 8,
    ):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("ResidualUNetDehazer requires matching input/output channels.")
        if base_channels < 1:
            raise ValueError("base_channels must be >= 1")
        if channel_mults is None:
            channel_mults = _channel_mults_from_depth(depth)
        if len(channel_mults) == 0:
            raise ValueError("channel_mults must contain at least one level.")
        if attn_levels is None:
            attn_levels = (len(channel_mults) - 1,)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = len(channel_mults)
        self.channel_mults = tuple(channel_mults)
        self.n_blocks = n_blocks
        self.attn_levels = tuple(attn_levels)
        self.num_heads = num_heads

        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=self.channel_mults,
            n_blocks=n_blocks,
            attn_levels=self.attn_levels,
            num_heads=num_heads,
        )
        bot_ch = self.encoder.skip_channels[-1]
        self.bot_block1 = ResBlock(bot_ch, bot_ch)
        self.bot_attn = AttentionBlock(bot_ch, num_heads=num_heads)
        self.bot_block2 = ResBlock(bot_ch, bot_ch)
        self.decoder = UNetDecoder(
            skip_channels=self.encoder.skip_channels,
            n_blocks=n_blocks,
            attn_levels=self.attn_levels,
            num_heads=num_heads,
        )
        self.out_norm = nn.GroupNorm(_norm_groups(base_channels), base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.encoder(x)

    def bottleneck(self, h: torch.Tensor) -> torch.Tensor:
        h = self.bot_block1(h)
        h = self.bot_attn(h)
        h = self.bot_block2(h)
        return h

    def decode(self, bottom: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        return self.decoder(bottom, skips)

    def forward(self, z_or_x_hazy: torch.Tensor, x_hazy: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_hazy = z_or_x_hazy if x_hazy is None else x_hazy
        if x_hazy.ndim != 4:
            raise RuntimeError("ResidualUNetDehazer expects a 4D image tensor.")
        if x_hazy.shape[1] != self.in_channels:
            raise RuntimeError(
                f"ResidualUNetDehazer expected {self.in_channels} channels, "
                f"got {x_hazy.shape[1]}."
            )
        skips = self.encode(x_hazy)
        h = self.bottleneck(skips[-1])
        h = self.decode(h, skips)
        h = F.silu(self.out_norm(h))
        residual = self.out_conv(h)
        return (x_hazy + residual).clamp(-1.0, 1.0)
