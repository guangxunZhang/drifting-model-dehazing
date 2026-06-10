"""Training script for U-Net-based dehazing on the Fridge condensation dataset
(real, paired clean / hazy images), trained with a feature-space conditional
drifting loss.
"""

import argparse
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from drifting import compute_V
from utils import EMA, WarmupLRScheduler, count_parameters, save_image_grid, set_seed


# 1. Paired (clean, hazy) fridge condensation dataset


_START_SUFFIX = "_start.jpg"
_STOP_SUFFIX = "_stop.jpg"


def _scan_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Find all (clean=start, hazy=stop) pairs in the dataset directory.
    """
    pairs: List[Tuple[Path, Path]] = []
    # Anything followed by _start.jpg
    for start_path in sorted(root.glob(f"*{_START_SUFFIX}")):
        stem = start_path.name[: -len(_START_SUFFIX)]
        stop_path = root / f"{stem}{_STOP_SUFFIX}"
        if stop_path.exists():
            pairs.append((start_path, stop_path))
    return pairs


def make_train_val_split(
    root: str,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Deterministic train / val split of the (start, stop) pairs."""
    pairs = _scan_pairs(Path(root))
    if not pairs:
        raise RuntimeError(f"No (start, stop) pairs found under {root}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    n_val = max(1, int(round(val_fraction * len(pairs))))
    val_pairs = [pairs[i] for i in perm[:n_val]]
    train_pairs = [pairs[i] for i in perm[n_val:]]
    return train_pairs, val_pairs


class FridgeCondensationDataset(Dataset):
    """Paired fridge dataset: returns ``(x_clean, x_hazy)`` in [-1, 1].
    """

    def __init__(
        self,
        root: str,
        img_size: int = 512,
        pairs: Optional[List[Tuple[Path, Path]]] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.img_size = img_size
        if pairs is None:
            pairs = _scan_pairs(self.root)
        if not pairs:
            raise RuntimeError(
                f"No (start, stop) pairs found under {self.root}. "
                "Expected files like XXXXX_X_start.jpg / XXXXX_X_stop.jpg."
            )
        self.pairs = pairs

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)

    def __getitem__(self, idx: int):
        start_path, stop_path = self.pairs[idx]
        x_clean = self._load(start_path)
        x_hazy = self._load(stop_path)
        return x_clean, x_hazy


# 2. U-Net image-to-image dehazer   f_theta(x_hazy) -> x_hat_clean

class ResBlock(nn.Module):
    """Pre-activation ResNet basic block (GroupNorm + SiLU)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        groups = min(8, in_channels)
        self.norm1 = nn.GroupNorm(groups, in_channels) # divides channels into groups and normalizes each group
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + identity


class Downsample(nn.Module):
    """2x spatial downsample via strided 3x3 conv (channels unchanged)."""

    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """2x spatial nearest-neighbour upsample + 3x3 conv (channels unchanged)."""

    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention over spatial tokens (HxW -> sequence).

    Used at the U-Net bottleneck (and optionally at the deepest encoder /
    decoder levels) to give the model a global view. Wraps `F.scaled_dot_product_attention`
    so it gets PyTorch's flash / memory-efficient attention path.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels)
        # Zero-init proj so the block starts as identity (residual path).
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w

        tokens = self.norm(x).flatten(2).transpose(1, 2)              # (B, N, C)
        qkv = self.qkv(tokens)                                          # (B, N, 3C)
        qkv = qkv.reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                                # (3, B, heads, N, hd)
        q, k, v = qkv.unbind(0)                                         # each (B, heads, N, hd)

        out = F.scaled_dot_product_attention(q, k, v)                   # (B, heads, N, hd)
        out = out.transpose(1, 2).reshape(b, n, c)                      # (B, N, C)
        out = self.proj(out)                                            # (B, N, C)
        out = out.transpose(1, 2).reshape(b, c, h, w)                   # (B, C, H, W)
        return x + out


class UNetEncoder(nn.Module):
    """Multi-scale encoder 
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 3, 4, 6),
        n_blocks: int = 2,
        attn_levels: Tuple[int, ...] = (4,),
        num_heads: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.n_blocks = n_blocks
        self.attn_levels = set(attn_levels)
        self.num_heads = num_heads
        self.num_levels = len(channel_mults)

        # Output channels at each level (also the channels of each saved skip).
        self.skip_channels: List[int] = [base_channels * m for m in channel_mults]

        # Stem: lift to base_channels at full resolution.
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.level_blocks: nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()

        prev_ch = base_channels
        for level, mult in enumerate(channel_mults):
            level_out = base_channels * mult
            blocks = nn.ModuleList()
            for b_idx in range(n_blocks):
                in_ch = prev_ch if b_idx == 0 else level_out
                blocks.append(ResBlock(in_ch, level_out, stride=1))
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
        # After the loop, h equals skips[-1] (last downsample is Identity).
        return skips


class UNetDecoder(nn.Module):
    """Multi-scale decoder.
    """

    def __init__(
        self,
        skip_channels: List[int],
        n_blocks: int = 2,
        attn_levels: Tuple[int, ...] = (4,),
        num_heads: int = 8,
    ):
        super().__init__()
        self.skip_channels = list(skip_channels)
        self.n_blocks = n_blocks
        self.attn_levels = set(attn_levels)
        self.num_heads = num_heads
        self.num_levels = len(skip_channels)

        self.upsamples: nn.ModuleList = nn.ModuleList()
        self.level_blocks: nn.ModuleList = nn.ModuleList()

        prev_ch = self.skip_channels[-1]
        for level in reversed(range(self.num_levels)):
            level_out = self.skip_channels[level]

            # No upsample at the deepest level (co-located with the bottleneck).
            if level == self.num_levels - 1:
                self.upsamples.append(nn.Identity())
            else:
                self.upsamples.append(Upsample(prev_ch))

            blocks = nn.ModuleList()
            # First decoder ResBlock takes (prev_ch + skip_ch) channels.
            blocks.append(ResBlock(prev_ch + self.skip_channels[level], level_out, stride=1))
            for _ in range(n_blocks - 1):
                blocks.append(ResBlock(level_out, level_out, stride=1))
            if level in self.attn_levels:
                blocks.append(AttentionBlock(level_out, num_heads=num_heads))
            self.level_blocks.append(blocks)
            prev_ch = level_out

        self.out_channels = prev_ch  # = skip_channels[0]

    def forward(self, bottom: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        if len(skips) != self.num_levels:
            raise ValueError(
                f"Decoder expected {self.num_levels} skip features, got {len(skips)}."
            )
        h = bottom
        for i, level in enumerate(reversed(range(self.num_levels))):
            h = self.upsamples[i](h)
            h = torch.cat([h, skips[level]], dim=1)
            for block in self.level_blocks[i]:
                h = block(h)
        return h


class UNetDehazer(nn.Module):
    """Image-to-image residual U-Net dehazer.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 3, 4, 6),
        n_blocks: int = 2,
        attn_levels: Tuple[int, ...] = (4,),
        num_heads: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.n_blocks = n_blocks
        self.attn_levels = tuple(attn_levels)
        self.num_heads = num_heads

        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            n_blocks=n_blocks,
            attn_levels=attn_levels,
            num_heads=num_heads,
        )

        # Bottleneck at the deepest spatial resolution.
        bot_ch = self.encoder.skip_channels[-1]
        self.bot_block1 = ResBlock(bot_ch, bot_ch)
        self.bot_attn = AttentionBlock(bot_ch, num_heads=num_heads)
        self.bot_block2 = ResBlock(bot_ch, bot_ch)

        self.decoder = UNetDecoder(
            skip_channels=self.encoder.skip_channels,
            n_blocks=n_blocks,
            attn_levels=attn_levels,
            num_heads=num_heads,
        )

        # Output head: zero-init conv -> initial residual is 0.
        self.out_norm = nn.GroupNorm(min(8, base_channels), base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # ----- Convenience accessors (handy for feature-space drift losses) -----

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run only the encoder and return the per-level skip pyramid."""
        return self.encoder(x)

    def bottleneck(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the bottleneck (ResBlock + Attn + ResBlock) to the deepest features."""
        h = self.bot_block1(h)
        h = self.bot_attn(h)
        h = self.bot_block2(h)
        return h

    def decode(self, bottom: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Run only the decoder, returning features at full input resolution."""
        return self.decoder(bottom, skips)

    def forward(self, x_hazy: torch.Tensor) -> torch.Tensor:
        skips = self.encode(x_hazy)
        h = self.bottleneck(skips[-1])
        h = self.decode(h, skips)
        h = F.silu(self.out_norm(h))
        residual = self.out_conv(h)
        return (x_hazy + residual).clamp(-1.0, 1.0)


# 3. Conditional drifting loss in encoder feature space

def compute_drifting_loss(
    x_gen: torch.Tensor,
    x_pos: torch.Tensor,
    feature_encoder: Optional[nn.Module],
    feature_encoder_target: Optional[nn.Module] = None,
    temperatures: Tuple[float, ...] = (0.05, 0.2, 0.5),
    feature_levels: Optional[Tuple[int, ...]] = None,
    use_pixel_space: bool = False,
) -> Tuple[torch.Tensor, dict]:

    """
    Conditional drifting loss with the same encoder in the model
    """
    device = x_gen.device

    # ----- Extract features -----
    if use_pixel_space or feature_encoder is None:
        # Pixel space: single "scale".
        feat_gen_list = [x_gen.flatten(start_dim=1)]
        feat_pos_list = [x_pos.flatten(start_dim=1)]
    else:
        target_encoder = (
            feature_encoder_target if feature_encoder_target is not None
            else feature_encoder
        )

        # Multi-scale feature maps from the U-Net encoder (list of (B, C, H, W)).
        feat_gen_maps = feature_encoder(x_gen)
        with torch.no_grad():
            feat_pos_maps = target_encoder(x_pos)

        if feature_levels is not None:
            feat_gen_maps = [feat_gen_maps[i] for i in feature_levels]
            feat_pos_maps = [feat_pos_maps[i] for i in feature_levels]

        # Global-average-pool each scale to get vectors (matches train.py).
        feat_gen_list = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feat_gen_maps]
        feat_pos_list = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feat_pos_maps]

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_drift_norm = 0.0
    num_losses = 0

    # ----- Loss per scale -----
    for feat_gen, feat_pos in zip(feat_gen_list, feat_pos_list):
        # L2-normalise (project to unit sphere).
        feat_gen_norm = F.normalize(feat_gen, p=2, dim=1)
        feat_pos_norm = F.normalize(feat_pos, p=2, dim=1)

        # Negatives = other generated samples (Algorithm 1: y_neg = x).
        feat_neg_norm = feat_gen_norm

        # Compute V at multiple temperatures, each normalised before summing.
        V_total = torch.zeros_like(feat_gen_norm)
        for tau in temperatures:
            V_tau = compute_V(
                feat_gen_norm,
                feat_pos_norm,
                feat_neg_norm,
                tau,
                mask_self=True,  # y_neg = x, so mask self
            )
            v_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
            V_tau = V_tau / (v_norm + 1e-8)
            V_total = V_total + V_tau

        # Loss: MSE(phi(x), stopgrad(phi(x) + V))
        target = (feat_gen_norm + V_total).detach()
        loss_scale = F.mse_loss(feat_gen_norm, target)

        total_loss = total_loss + loss_scale
        total_drift_norm += (V_total ** 2).mean().item() ** 0.5
        num_losses += 1

    if num_losses == 0:
        return (
            torch.tensor(0.0, device=device, requires_grad=True),
            {"loss": 0.0, "drift_norm": 0.0},
        )

    loss = total_loss / num_losses
    info = {
        "loss": loss.item(),
        "drift_norm": total_drift_norm / num_losses,
    }
    return loss, info



# 4. Training

def train(
    epochs: int = 200,
    batch_size: int = 4,
    img_size: int = 512,
    base_channels: int = 64,
    channel_mults: Tuple[int, ...] = (1, 2, 3, 4, 6),
    n_blocks: int = 2,
    attn_levels: Tuple[int, ...] = (4,),
    num_heads: int = 8,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 2.0,
    warmup_steps: int = 2000,
    ema_decay: float = 0.999,
    lambda_recon: float = 1.0,
    lambda_drift: float = 1.0,
    feature_levels: Optional[Tuple[int, ...]] = None,
    temperatures: Tuple[float, ...] = (0.05, 0.2, 0.5),
    pixel_drift: bool = False,
    val_fraction: float = 0.1,
    data_dir: str = "./condensation_data",
    output_dir: str = "./outputs/dehaze_fridge",
    num_workers: int = 2,
    log_interval: int = 40,
    sample_interval: int = 0,
    epoch_save_interval: int = 5,
    seed: int = 42,
    smoke_test: bool = False,
    max_steps: Optional[int] = None,
):
    """End-to-end training loop on the Fridge condensation dataset."""
    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples_dir = out / "samples"
    checkpoints_dir = out / "checkpoints"
    samples_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")

    # In smoke-test mode, log every step so the user sees activity quickly.
    effective_log_interval = 5 if smoke_test else log_interval

    train_pairs, val_pairs = make_train_val_split(
        data_dir, val_fraction=val_fraction, seed=seed
    )
    print(f"Pairs: {len(train_pairs)} train  /  {len(val_pairs)} val")

    train_set = FridgeCondensationDataset(
        root=data_dir, img_size=img_size, pairs=train_pairs
    )
    vis_set = FridgeCondensationDataset(
        root=data_dir, img_size=img_size, pairs=val_pairs
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = UNetDehazer(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        channel_mults=channel_mults,
        n_blocks=n_blocks,
        attn_levels=attn_levels,
        num_heads=num_heads,
    ).to(device)
    print(f"UNetDehazer params: {count_parameters(model):,}")

    ema = EMA(model, decay=ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
    )
    scheduler = WarmupLRScheduler(optimizer, warmup_steps=warmup_steps, base_lr=lr)

    # ----- Explicit encoders used by the drifting loss -----
    # feature_encoder         = the trainable U-Net encoder (model.encoder).
    # feature_encoder_target  = the EMA copy of that encoder (ema.shadow.encoder).
    feature_encoder = None if pixel_drift else model.encoder
    feature_encoder_target = None if pixel_drift else ema.shadow.encoder

    if pixel_drift:
        print(
            "Loss: total = "
            f"{lambda_recon} * L1 + {lambda_drift} * drift_pixel (no encoder)"
        )
    else:
        n_enc_levels = len(model.encoder.channel_mults)
        levels_str = (
            "all levels" if feature_levels is None
            else f"levels={list(feature_levels)}"
        )
        print(
            "Loss: total = "
            f"{lambda_recon} * L1 + {lambda_drift} * drift_feature\n"
            f"  feature_encoder        = model.encoder (trainable U-Net encoder, "
            f"{n_enc_levels} levels)\n"
            f"  feature_encoder_target = ema.shadow.encoder (EMA copy, no grads)\n"
            f"  using {levels_str}, temperatures={list(temperatures)}, "
            f"GAP -> flatten -> L2-normalise per scale"
        )

    def checkpoint_payload():
        return {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "config": {
                "img_size": img_size,
                "base_channels": base_channels,
                "channel_mults": list(channel_mults),
                "n_blocks": n_blocks,
                "attn_levels": list(attn_levels),
                "num_heads": num_heads,
                "in_channels": 3,
                "out_channels": 3,
                "lambda_recon": lambda_recon,
                "lambda_drift": lambda_drift,
                "feature_levels": (
                    None if feature_levels is None else list(feature_levels)
                ),
                "temperatures": list(temperatures),
                "pixel_drift": pixel_drift,
            },
        }

    def save_random_triple_grid(path: Path, n: int = 16):
        n_use = min(n, len(vis_set))
        indices = torch.randperm(len(vis_set))[:n_use]
        samples = [vis_set[int(idx)] for idx in indices]
        vis_clean = torch.stack([s[0] for s in samples], dim=0).to(device)
        vis_hazy = torch.stack([s[1] for s in samples], dim=0).to(device)
        save_triple_grid(ema.shadow, vis_clean, vis_hazy, path)

    global_step = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        running = {"loss": 0.0, "recon": 0.0, "drift": 0.0, "drift_norm": 0.0, "n": 0}

        for batch_idx, (x_clean, x_hazy) in enumerate(train_loader):
            x_clean = x_clean.to(device, non_blocking=True)
            x_hazy = x_hazy.to(device, non_blocking=True)

            x_hat = model(x_hazy)

            # Drifting loss in encoder feature space (or pixel space if pixel_drift).
            drift_loss, info = compute_drifting_loss(
                x_gen=x_hat,
                x_pos=x_clean,
                feature_encoder=feature_encoder,
                feature_encoder_target=feature_encoder_target,
                temperatures=temperatures,
                feature_levels=feature_levels,
                use_pixel_space=pixel_drift,
            )
            recon_loss = F.l1_loss(x_hat, x_clean)
            loss = lambda_recon * recon_loss + lambda_drift * drift_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            running["loss"] += loss.item()
            running["recon"] += recon_loss.item()
            running["drift"] += info["loss"]
            running["drift_norm"] += info["drift_norm"]
            running["n"] += 1
            global_step += 1

            if global_step % effective_log_interval == 0:
                n = max(running["n"], 1)
                print(
                    f"Epoch {epoch + 1}/{epochs} | step {global_step} | "
                    f"loss {running['loss'] / n:.4f} "
                    f"(recon {running['recon'] / n:.4f}, drift {running['drift'] / n:.4f}) | "
                    f"drift_norm {running['drift_norm'] / n:.4f} | "
                    f"grad {grad_norm:.2f} | lr {scheduler.get_lr():.6f}"
                )

            if sample_interval > 0 and global_step % sample_interval == 0:
                save_random_triple_grid(samples_dir / f"step_{global_step:06d}.png")

            if smoke_test and global_step >= 20:
                print("[smoke_test] early stop")
                save_random_triple_grid(samples_dir / "smoke.png")
                torch.save(checkpoint_payload(), checkpoints_dir / "latest.pt")
                return

            if max_steps is not None and global_step >= max_steps:
                print(f"[max_steps] reached {max_steps}, stopping")
                save_random_triple_grid(samples_dir / "latest.png")
                torch.save(checkpoint_payload(), checkpoints_dir / "latest.pt")
                return

        elapsed = time.time() - epoch_start
        n = max(running["n"], 1)
        print(
            f"Epoch {epoch + 1} done in {elapsed:.1f}s | "
            f"avg loss {running['loss'] / n:.4f} "
            f"(recon {running['recon'] / n:.4f}, drift {running['drift'] / n:.4f}) | "
            f"avg drift_norm {running['drift_norm'] / n:.4f}"
        )

        epoch_num = epoch + 1
        should_save_epoch = (
            epoch_save_interval > 0
            and (epoch_num % epoch_save_interval == 0 or epoch_num == epochs)
        )
        if should_save_epoch:
            epoch_name = f"epoch_{epoch_num:03d}"
            epoch_sample_path = samples_dir / f"{epoch_name}.png"
            save_random_triple_grid(epoch_sample_path)
            shutil.copyfile(epoch_sample_path, samples_dir / "latest.png")

            ckpt_payload = checkpoint_payload()
            torch.save(ckpt_payload, checkpoints_dir / f"{epoch_name}.pt")
            torch.save(ckpt_payload, checkpoints_dir / "latest.pt")
            print(f"  Saved samples/{epoch_name}.png and checkpoints/{epoch_name}.pt")

    torch.save(checkpoint_payload(), checkpoints_dir / "final.pt")
    print(f"Saved final checkpoint to {checkpoints_dir / 'final.pt'}")


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------


@torch.no_grad()
def save_triple_grid(
    model: nn.Module,
    x_clean: torch.Tensor,
    x_hazy: torch.Tensor,
    path: Path,
):
    """Save a grid with rows of [clean | hazy | dehazed]."""
    was_training = model.training
    model.eval()
    x_hat = model(x_hazy).clamp(-1.0, 1.0)

    n = x_clean.shape[0]
    stacked = torch.stack([x_clean, x_hazy, x_hat], dim=1)  # (N, 3, C, H, W)
    grid = stacked.reshape(n * 3, *x_clean.shape[1:])
    save_image_grid(grid, str(path), nrow=3)
    if was_training:
        model.train()


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------


def _parse_int_list(s: str) -> Tuple[int, ...]:
    """Parse comma-separated integers, e.g. '1,2,3,4,6' -> (1,2,3,4,6)."""
    s = s.strip()
    if not s:
        return ()
    return tuple(int(part) for part in s.split(","))


def _parse_float_list(s: str) -> Tuple[float, ...]:
    """Parse comma-separated floats, e.g. '0.05,0.2,0.5' -> (0.05, 0.2, 0.5)."""
    s = s.strip()
    if not s:
        return ()
    return tuple(float(part) for part in s.split(","))


def main():
    p = argparse.ArgumentParser(description="Fridge condensation U-Net dehazing training.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--base_channels", type=int, default=64,
                   help="Base channel count of the U-Net (multiplied by channel_mults at each level).")
    p.add_argument("--channel_mults", type=_parse_int_list, default=(1, 2, 3, 4, 6),
                   help="Comma-separated channel multipliers per encoder level. "
                        "Default '1,2,3,4,6' = 5 levels suitable for 512x512.")
    p.add_argument("--n_blocks", type=int, default=2,
                   help="Number of ResBlocks per encoder/decoder level.")
    p.add_argument("--attn_levels", type=_parse_int_list, default=(4,),
                   help="Comma-separated level indices that use self-attention "
                        "(plus the bottleneck always does). Default '4' = deepest level only.")
    p.add_argument("--num_heads", type=int, default=8,
                   help="Number of heads for self-attention blocks.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--lambda_recon", type=float, default=1.0,
                   help="Weight of the L1 reconstruction loss.")
    p.add_argument("--lambda_drift", type=float, default=1.0,
                   help="Weight of the drifting loss.")
    p.add_argument("--feature_levels", type=_parse_int_list, default=None,
                   help="Encoder skip-pyramid levels used by the feature-space "
                        "drift loss (comma-separated, negative indices count "
                        "from the deepest level). Default: all levels (matches "
                        "train.py). Examples: '-1' = deepest only; '-2,-1' = "
                        "two deepest.")
    p.add_argument("--temperatures", type=_parse_float_list,
                   default=(0.05, 0.2, 0.5),
                   help="Drifting field temperatures (sharper -> softer).")
    p.add_argument("--pixel_drift", action="store_true",
                   help="Ablation: compute the drifting loss in raw pixel space "
                        "(flattened RGB) instead of in encoder feature space. "
                        "No encoder is used in this mode.")
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "condensation_data"),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs" / "dehaze_fridge"),
    )
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_interval", type=int, default=40)
    p.add_argument("--sample_interval", type=int, default=0,
                   help="Step interval for extra sample grids (0 = only save per epoch).")
    p.add_argument("--epoch_save_interval", type=int, default=5,
                   help="Save samples/checkpoints every N epochs (0 = disable).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_test", action="store_true",
                   help="Run only ~20 steps and save one visualization for a quick sanity check.")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Optional hard cap on number of training steps.")
    args = p.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        n_blocks=args.n_blocks,
        attn_levels=args.attn_levels,
        num_heads=args.num_heads,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        lambda_recon=args.lambda_recon,
        lambda_drift=args.lambda_drift,
        feature_levels=args.feature_levels,
        temperatures=args.temperatures,
        pixel_drift=args.pixel_drift,
        val_fraction=args.val_fraction,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        sample_interval=args.sample_interval,
        epoch_save_interval=args.epoch_save_interval,
        seed=args.seed,
        smoke_test=args.smoke_test,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
