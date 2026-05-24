import argparse
import math
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from drifting import compute_V
from model import DiTBlock, FinalLayer, PatchEmbed, RotaryPositionEmbedding
from utils import EMA, WarmupLRScheduler, count_parameters, save_image_grid, set_seed


# ---------------------------------------------------------------------------
# 1. Artificial fog (atmospheric scattering)
# ---------------------------------------------------------------------------


def _make_depth_map(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Random smooth depth maps in [0, 1] for artificial fog."""
    d = torch.zeros(batch_size, 1, height, width, device=device)
    weight, total = 1.0, 0.0
    for res in (2, 4, 8):
        octave = torch.rand(batch_size, 1, res, res, device=device, generator=generator)
        d += weight * F.interpolate(octave, size=(height, width), mode="bilinear", align_corners=False)
        total += weight
        weight *= 0.5

    # Add broad fog blobs so the haze varies locally instead of looking flat.
    ys = torch.linspace(0.0, 1.0, height, device=device).view(1, 1, height, 1)
    xs = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, 1, width)
    n_blobs = int(torch.randint(2, 5, (1,), generator=generator).item())
    blobs = torch.zeros_like(d)
    for _ in range(n_blobs):
        cy = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.0, 1.0, generator=generator)
        cx = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.0, 1.0, generator=generator)
        sigma = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.2, 0.55, generator=generator)
        blobs += torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * sigma ** 2))

    d = 0.65 * (d / total) + 0.35 * (blobs / n_blobs)
    d_min = d.amin(dim=(2, 3), keepdim=True)
    d_max = d.amax(dim=(2, 3), keepdim=True)
    return (d - d_min) / (d_max - d_min + 1e-6)


def _gaussian_kernel1d(sigma: float, device: torch.device) -> torch.Tensor:
    """1D Gaussian kernel, radius = ceil(3 * sigma)."""
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(
    image: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-channel separable Gaussian blur with random anisotropy."""
    if sigma <= 0.0:
        return image
    c = image.shape[1]
    jitter = torch.empty(2, device=image.device).uniform_(0.75, 1.25, generator=generator)
    sigma_x = max(0.05, sigma * float(jitter[0].item()))
    sigma_y = max(0.05, sigma * float(jitter[1].item()))

    kx_1d = _gaussian_kernel1d(sigma_x, image.device)
    ky_1d = _gaussian_kernel1d(sigma_y, image.device)
    radius_x = (kx_1d.numel() - 1) // 2
    radius_y = (ky_1d.numel() - 1) // 2
    kx = kx_1d.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = ky_1d.view(1, 1, -1, 1).expand(c, 1, -1, 1)

    out = F.pad(image, (radius_x, radius_x, 0, 0), mode="reflect")
    out = F.conv2d(out, kx, groups=c)
    out = F.pad(out, (0, 0, radius_y, radius_y), mode="reflect")
    out = F.conv2d(out, ky, groups=c)
    return out


def apply_fog(
    image: torch.Tensor,
    beta_range: Tuple[float, float] = (2.2, 4.8),
    a_range: Tuple[float, float] = (0.9, 1.0),
    blur_sigma_range: Tuple[float, float] = (0.2, 1.0),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply artificial fog to a batch of images.

    The model is the standard atmospheric scattering equation
        I = J * t + A * (1 - t),    t = exp(-beta * d)
    followed by a per-image Gaussian blur that mimics small-angle scattering.

    Args:
        image:             Clean image batch in [-1, 1], shape (B, C, H, W).
        beta_range:        Per-image range for haze density beta (larger = denser).
        a_range:           Per-image range for atmospheric light A (closer to 1 = whiter).
        blur_sigma_range:  Per-image range for Gaussian blur sigma (pixels).
                           Set (0.0, 0.0) to disable.
        generator:         Optional torch.Generator for reproducible fog.

    Returns:
        Hazy image batch in [-1, 1], same shape as input.
    """
    b, _, h, w = image.shape
    device = image.device

    # Work in [0, 1] so the physical scattering model has its standard form.
    j = (image + 1.0) / 2.0

    beta = torch.empty(b, 1, 1, 1, device=device)
    beta.uniform_(beta_range[0], beta_range[1], generator=generator)

    a = torch.empty(b, 1, 1, 1, device=device)
    a.uniform_(a_range[0], a_range[1], generator=generator)

    d = _make_depth_map(b, h, w, device, generator=generator)
    t = torch.exp(-beta * d)

    i = j * t + a * (1.0 - t)

    # Per-image blur sigma: each image gets its own scattering softness.
    sigma_lo, sigma_hi = blur_sigma_range
    if sigma_hi > 0.0:
        sigmas = torch.empty(b, device=device).uniform_(sigma_lo, sigma_hi, generator=generator)
        i = torch.cat(
            [
                gaussian_blur(i[n : n + 1], float(sigmas[n].item()), generator=generator)
                for n in range(b)
            ],
            dim=0,
        )

    return (i * 2.0 - 1.0).clamp(-1.0, 1.0)


class HazyCIFAR10(Dataset):
    """CIFAR-10 wrapped with on-the-fly fog. Returns (x_clean, x_hazy, label)."""

    def __init__(
        self,
        root: str = "./data/cifar10",
        train: bool = True,
        img_size: int = 32,
        beta_range: Tuple[float, float] = (1.0, 2.8),
        a_range: Tuple[float, float] = (0.85, 1.0),
        blur_sigma_range: Tuple[float, float] = (0.0, 0.15),
        fixed_fog: bool = False,
    ):
        """
        Args:
            fixed_fog: If True, fog parameters are seeded by image index so the
                       same image always gets the same haze. Useful for the
                       test set to keep evaluation consistent.
        """
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])
        self.base = datasets.CIFAR10(root, train=train, download=True, transform=transform)
        self.beta_range = beta_range
        self.a_range = a_range
        self.blur_sigma_range = blur_sigma_range
        self.fixed_fog = fixed_fog

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x_clean, label = self.base[idx]  # x_clean: (3, H, W) in [-1, 1]
        x_clean = x_clean.unsqueeze(0)   # add batch dim for apply_fog
        gen = None
        if self.fixed_fog:
            gen = torch.Generator(device=x_clean.device).manual_seed(int(idx))
        x_hazy = apply_fog(
            x_clean,
            beta_range=self.beta_range,
            a_range=self.a_range,
            blur_sigma_range=self.blur_sigma_range,
            generator=gen,
        )
        return x_clean.squeeze(0), x_hazy.squeeze(0), int(label)


# 2. Conditional generator   f_theta(z, x_hazy) -> x_hat_clean


class HazyContextEncoder(nn.Module):
    """Small CNN
    """

    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # 8x8
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # 4x4
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x_hazy: torch.Tensor) -> torch.Tensor:
        return self.net(x_hazy)


class DehazingDiT(nn.Module):
    """DiT-style conditional generator for dehazing.

    Inputs:
        z:       Gaussian noise, shape (B, in_channels, H, W).
        x_hazy:  Hazy image, shape (B, in_channels, H, W).
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 512,
        depth: int = 14,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches = (img_size // patch_size) ** 2

        # Concat(z, x_hazy) -> 2 * in_channels at the input.
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=2 * in_channels,
            embed_dim=hidden_size,
        )

        self.num_register_tokens = num_register_tokens
        self.register_tokens = nn.Parameter(
            torch.randn(1, num_register_tokens, hidden_size) * 0.02
        )

        head_dim = hidden_size // num_heads
        self.rope = RotaryPositionEmbedding(
            dim=head_dim,
            max_seq_len=self.num_patches + num_register_tokens + 16,
        )

        # Global conditioning vector from x_hazy (analogue of class label).
        self.context_encoder = HazyContextEncoder(in_channels, hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(dim=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, use_qk_norm=True)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # adaLN-Zero: zero out the modulation projection so blocks start as identity.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        # Final linear: small random so initial outputs are non-zero.
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = self.img_size // p
        x = x.reshape(-1, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(-1, c, h * p, w * p)

    def forward(self, z: torch.Tensor, x_hazy: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]

        # Spatial cues: concat noise and hazy image along channels.
        x = torch.cat([z, x_hazy], dim=1)
        x = self.patch_embed(x)

        register = self.register_tokens.expand(b, -1, -1)
        x = torch.cat([register, x], dim=1)

        seq_len = x.shape[1]
        rope_cos, rope_sin = self.rope(x, seq_len)

        # Global cue: adaLN-Zero modulation from x_hazy context vector.
        c = self.context_encoder(x_hazy)

        for block in self.blocks:
            x = block(x, c, rope_cos, rope_sin)

        x = x[:, self.num_register_tokens:, :]
        x = self.final_layer(x, c)
        residual = self.unpatchify(x)
        return (x_hazy + residual).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# 3. Losses
# ---------------------------------------------------------------------------



def conditional_drift_loss(
    x_hat: torch.Tensor,
    x_clean: torch.Tensor,
    temperatures: List[float] = (0.05, 0.2, 0.5),
) -> torch.Tensor:
    """Per-example "conditional drifting" loss.

    For sample i the only positive is x_clean_i (the matched clean image) and
    the negatives are the other generated samples in the batch x_hat_{j != i}.
    Reduces to the same V used by the rest of the codebase, just with a
    singleton positive set per example.
    """
    feat_gen = x_hat.flatten(start_dim=1)
    feat_pos = x_clean.flatten(start_dim=1)

    feat_gen_norm = F.normalize(feat_gen, p=2, dim=1)
    feat_pos_norm = F.normalize(feat_pos, p=2, dim=1)

    v_total = torch.zeros_like(feat_gen_norm)
    for tau in temperatures:
        # Negatives = other generated samples (mask_self=True drops the diagonal).
        v_tau = compute_V(feat_gen_norm, feat_pos_norm, feat_gen_norm, tau, mask_self=True)
        v_norm = torch.sqrt(torch.mean(v_tau ** 2) + 1e-8)
        v_total = v_total + v_tau / (v_norm + 1e-8)

    target = (feat_gen_norm + v_total).detach()
    return F.mse_loss(feat_gen_norm, target)


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------


def train(
    epochs: int = 150,
    batch_size: int = 256,
    img_size: int = 32,
    hidden_size: int = 512,
    depth: int = 14,
    num_heads: int = 8,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 2.0,
    warmup_steps: int = 3000,
    ema_decay: float = 0.999,
    data_dir: str = "./data/cifar10",
    output_dir: str = "./outputs/dehaze_cifar",
    num_workers: int = 2,
    log_interval: int = 40,
    sample_interval: int = 0,
    epoch_save_interval: int = 5,
    seed: int = 42,
    smoke_test: bool = False,
    max_steps: Optional[int] = None,
):
    """End-to-end training loop."""
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

    # In smoke-test mode, log every step so the user sees activity quickly.
    effective_log_interval = 5 if smoke_test else log_interval

    train_set = HazyCIFAR10(root=data_dir, train=True, img_size=img_size)
    vis_set = HazyCIFAR10(root=data_dir, train=False, img_size=img_size, fixed_fog=False)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = DehazingDiT(
        img_size=img_size,
        in_channels=3,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
    ).to(device)
    print(f"DehazingDiT params: {count_parameters(model):,}")

    ema = EMA(model, decay=ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
    )
    scheduler = WarmupLRScheduler(optimizer, warmup_steps=warmup_steps, base_lr=lr)

    def checkpoint_payload():
        return {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "config": {
                "img_size": img_size,
                "hidden_size": hidden_size,
                "depth": depth,
                "num_heads": num_heads,
                "in_channels": 3,
            },
        }

    def save_random_triple_grid(path: Path, n: int = 16):
        indices = torch.randint(len(vis_set), (n,))
        samples = [vis_set[int(idx)] for idx in indices]
        vis_clean = torch.stack([sample[0] for sample in samples], dim=0).to(device)
        vis_hazy = torch.stack([sample[1] for sample in samples], dim=0).to(device)
        save_triple_grid(ema.shadow, vis_clean, vis_hazy, path)

    global_step = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        running = {"loss": 0.0, "n": 0}

        for batch_idx, (x_clean, x_hazy, _label) in enumerate(train_loader):
            x_clean = x_clean.to(device, non_blocking=True)
            x_hazy = x_hazy.to(device, non_blocking=True)

            z = torch.zeros_like(x_clean)
            x_hat = model(z, x_hazy)

            # recon_l1 = F.l1_loss(x_hat, x_clean)
            # recon_l2 = F.mse_loss(x_hat, x_clean)
            drift = conditional_drift_loss(x_hat, x_clean)
            # loss = recon_l1 + 0.2 * recon_l2 + 0.02 * drift
            loss = drift

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            running["loss"] += loss.item()
            running["n"] += 1
            global_step += 1

            if global_step % effective_log_interval == 0:
                avg = {k: v / max(running["n"], 1) for k, v in running.items() if k != "n"}
                print(
                    f"Epoch {epoch + 1}/{epochs} | step {global_step} | "
                    f"loss {avg['loss']:.4f} | grad {grad_norm:.2f} | "
                    f"lr {scheduler.get_lr():.6f}"
                )

            if sample_interval > 0 and global_step % sample_interval == 0:
                save_random_triple_grid(samples_dir / f"step_{global_step:06d}.png")

            if smoke_test and global_step >= 20:
                print("[smoke_test] early stop")
                # Final smoke-test image to confirm the visualisation path works.
                save_random_triple_grid(samples_dir / "smoke.png")
                torch.save(checkpoint_payload(), checkpoints_dir / "latest.pt")
                return

            if max_steps is not None and global_step >= max_steps:
                print(f"[max_steps] reached {max_steps}, stopping")
                save_random_triple_grid(samples_dir / "latest.png")
                torch.save(checkpoint_payload(), checkpoints_dir / "latest.pt")
                return

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1} done in {elapsed:.1f}s | "
            f"avg loss {running['loss'] / max(running['n'], 1):.4f}"
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
    z = torch.zeros_like(x_clean)
    x_hat = model(z, x_hazy).clamp(-1.0, 1.0)

    # Interleave: clean_i, hazy_i, dehazed_i, clean_{i+1}, hazy_{i+1}, ...
    n = x_clean.shape[0]
    stacked = torch.stack([x_clean, x_hazy, x_hat], dim=1)  # (N, 3, C, H, W)
    grid = stacked.reshape(n * 3, *x_clean.shape[1:])
    save_image_grid(grid, str(path), nrow=3)
    if was_training:
        model.train()


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="CIFAR-10 dehazing toy example.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--depth", type=int, default=14)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=3000)
    p.add_argument("--data_dir", type=str, default="./data/cifar10")
    p.add_argument("--output_dir", type=str, default="./outputs/dehaze_cifar")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--sample_interval", type=int, default= 400,
                   help="Optional step interval for extra samples (0 = only save per epoch).")
    p.add_argument("--epoch_save_interval", type=int, default=20,
                   help="Save epoch samples/checkpoints every N epochs (0 = disable).")
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
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
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
