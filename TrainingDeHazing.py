import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from haze.asm import apply_fog, make_flat_depth_map, resolve_fog_config
from haze.mcbm import apply_mcbm_fog
from losses import (
    ModelEncodeFeatureExtractor,
    compute_drift_loss,
    make_multiscale_feature_encoder,
)
from model import DiTBlock, FinalLayer, PatchEmbed, RotaryPositionEmbedding
from models.residual_unet import ResidualUNetDehazer
from unet import UNetDehazer
from utils import EMA, WarmupLRScheduler, count_parameters, save_image_grid, set_seed

"""
Synthetic dehazing toy runs:

MNIST:
  python TrainingDeHazing.py --dataset mnist --data_dir ./data/mnist --batch_size 128 --epochs 10 --save_dir ./outputs/dehaze

CIFAR-10:
  python TrainingDeHazing.py --dataset cifar10 --data_dir ./data --batch_size 64 --epochs 5 --save_dir ./outputs/cifar_dehaze
"""


MODEL_PRESETS: Dict[str, Dict[str, int]] = {
    "small": {"hidden_size": 192, "depth": 6, "num_heads": 4},
    "medium": {"hidden_size": 384, "depth": 10, "num_heads": 8},
    "large": {"hidden_size": 512, "depth": 14, "num_heads": 8},
}


def resolve_model_config(
    model_preset: str,
    hidden_size: int,
    depth: int,
    num_heads: int,
) -> Tuple[int, int, int]:
    if model_preset == "custom":
        return hidden_size, depth, num_heads
    preset = MODEL_PRESETS[model_preset]
    return preset["hidden_size"], preset["depth"], preset["num_heads"]


def make_noise(x_clean: torch.Tensor, noise_mode: str) -> torch.Tensor:
    if noise_mode == "random":
        return torch.randn_like(x_clean)
    if noise_mode == "zero":
        return torch.zeros_like(x_clean)
    raise ValueError(f"Unknown noise mode: {noise_mode}")


def apply_prediction_mode(
    raw_output: torch.Tensor,
    x_hazy: torch.Tensor,
    prediction_mode: str,
) -> torch.Tensor:
    if raw_output.shape != x_hazy.shape:
        raise RuntimeError(f"Raw model output shape {tuple(raw_output.shape)} != hazy {tuple(x_hazy.shape)}")
    if prediction_mode == "direct":
        return raw_output
    if prediction_mode == "residual":
        return (x_hazy + raw_output).clamp(-1.0, 1.0)
    raise ValueError(f"Unknown prediction mode: {prediction_mode}")


def model_outputs_final_prediction(model_type: str) -> bool:
    return model_type == "residual_unet"


def apply_selected_fog(
    image: torch.Tensor,
    fog_type: str,
    beta_range: Tuple[float, float],
    a_range: Tuple[float, float],
    blur_sigma_range: Tuple[float, float],
    depth_mode: str = "synthetic",
    depth: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    return_meta: bool = False,
) -> Any:
    if depth is None:
        if depth_mode == "synthetic":
            fog_depth = None
        elif depth_mode == "flat":
            b, _, h, w = image.shape
            fog_depth = make_flat_depth_map(b, h, w, image.device, image.dtype)
        elif depth_mode == "external":
            raise ValueError("depth_mode=external requires a depth tensor from the dataset.")
        else:
            raise ValueError(f"Unknown depth mode: {depth_mode}")
    else:
        fog_depth = depth

    if fog_type == "asm":
        return apply_fog(
            image,
            beta_range=beta_range,
            a_range=a_range,
            blur_sigma_range=blur_sigma_range,
            depth=fog_depth,
            generator=generator,
            return_meta=return_meta,
        )
    if fog_type == "mcbm":
        return apply_mcbm_fog(
            image,
            beta_range=beta_range,
            a_range=a_range,
            blur_sigma_range=blur_sigma_range,
            depth=fog_depth,
            generator=generator,
            return_meta=return_meta,
        )
    raise ValueError(f"Unknown fog type: {fog_type}")


class HazyMNIST(Dataset):
    """MNIST wrapped with on-the-fly fog. Returns (x_clean, x_hazy, label)."""

    def __init__(
        self,
        root: str = "./data/mnist",
        train: bool = True,
        img_size: int = 32,
        beta_range: Tuple[float, float] = (2.0, 4.5),
        a_range: Tuple[float, float] = (0.85, 1.0),
        blur_sigma_range: Tuple[float, float] = (0.5, 1.8),
        fog_type: str = "asm",
        depth_mode: str = "synthetic",
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
            transforms.Normalize([0.5], [0.5]),  # -> [-1, 1]
        ])
        self.base = datasets.MNIST(root, train=train, download=True, transform=transform)
        self.beta_range = beta_range
        self.a_range = a_range
        self.blur_sigma_range = blur_sigma_range
        self.fog_type = fog_type
        self.depth_mode = depth_mode
        self.fixed_fog = fixed_fog
        if self.depth_mode == "external":
            raise ValueError("external depth is only supported for folder dataset for now.")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x_clean, label = self.base[idx]  # x_clean: (1, H, W) in [-1, 1]
        x_clean = x_clean.unsqueeze(0)   # add batch dim for apply_fog
        gen = None
        if self.fixed_fog:
            gen = torch.Generator(device=x_clean.device).manual_seed(int(idx))
        x_hazy = apply_selected_fog(
            x_clean,
            fog_type=self.fog_type,
            beta_range=self.beta_range,
            a_range=self.a_range,
            blur_sigma_range=self.blur_sigma_range,
            depth_mode=self.depth_mode,
            generator=gen,
        )
        return x_clean.squeeze(0), x_hazy.squeeze(0), int(label)


class HazyCIFAR10(Dataset):
    """CIFAR-10 wrapped with on-the-fly fog. Returns (x_clean, x_hazy, label)."""

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        beta_range: Tuple[float, float] = (3.5, 7.0),
        a_range: Tuple[float, float] = (0.85, 1.0),
        blur_sigma_range: Tuple[float, float] = (0.5, 1.8),
        fog_type: str = "asm",
        depth_mode: str = "synthetic",
        fixed_fog: bool = False,
    ):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ])
        self.base = datasets.CIFAR10(root, train=train, download=True, transform=transform)
        self.beta_range = beta_range
        self.a_range = a_range
        self.blur_sigma_range = blur_sigma_range
        self.fog_type = fog_type
        self.depth_mode = depth_mode
        self.fixed_fog = fixed_fog
        if self.depth_mode == "external":
            raise ValueError("external depth is only supported for folder dataset for now.")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x_clean, label = self.base[idx]  # x_clean: (3, 32, 32) in [-1, 1]
        if x_clean.shape != (3, 32, 32):
            raise ValueError(f"CIFAR sample should be (3, 32, 32), got {tuple(x_clean.shape)}")
        x_clean = x_clean.unsqueeze(0)
        gen = None
        if self.fixed_fog:
            gen = torch.Generator(device=x_clean.device).manual_seed(int(idx))
        x_hazy = apply_selected_fog(
            x_clean,
            fog_type=self.fog_type,
            beta_range=self.beta_range,
            a_range=self.a_range,
            blur_sigma_range=self.blur_sigma_range,
            depth_mode=self.depth_mode,
            generator=gen,
        )
        return x_clean.squeeze(0), x_hazy.squeeze(0), int(label)


class RealRGBFolderDataset(Dataset):
    """RGB image folder wrapped with on-the-fly fog. Returns (x_clean, x_hazy, -1)."""

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    DEPTH_EXTENSIONS = (".png", ".jpg", ".jpeg", ".npy", ".pt")

    def __init__(
        self,
        root: str,
        train: bool = True,
        img_size: int = 128,
        beta_range: Tuple[float, float] = (3.5, 7.0),
        a_range: Tuple[float, float] = (0.85, 1.0),
        blur_sigma_range: Tuple[float, float] = (0.5, 1.8),
        fog_type: str = "asm",
        depth_mode: str = "synthetic",
        depth_dir: Optional[str] = None,
        fixed_fog: bool = False,
        recursive: bool = False,
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.train = train
        self.depth_mode = depth_mode
        self.depth_dir = Path(depth_dir) if depth_dir is not None else None
        if not self.root.exists():
            raise FileNotFoundError(f"Image folder does not exist: {self.root}")
        if self.depth_mode == "external":
            if self.depth_dir is None:
                raise ValueError("--depth_mode external requires --depth_dir.")
            if not self.depth_dir.exists():
                raise FileNotFoundError(f"Depth folder does not exist: {self.depth_dir}")
        iterator = self.root.rglob("*") if recursive else self.root.iterdir()
        self.paths = sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in self.IMG_EXTENSIONS
        )
        if not self.paths:
            mode = "recursively" if recursive else "non-recursively"
            extensions = ", ".join(sorted(self.IMG_EXTENSIONS))
            raise ValueError(
                f"No supported images found {mode} in {self.root}. "
                f"Supported extensions: {extensions}"
            )

        self.beta_range = beta_range
        self.a_range = a_range
        self.blur_sigma_range = blur_sigma_range
        self.fog_type = fog_type
        self.fixed_fog = fixed_fog

    def __len__(self) -> int:
        return len(self.paths)

    def _resolve_depth_path(self, image_path: Path) -> Path:
        if self.depth_dir is None:
            raise ValueError("--depth_mode external requires --depth_dir.")
        rel_path = image_path.relative_to(self.root)

        # Matching is intentionally simple and deterministic:
        # 1. mirror the RGB relative path under depth_dir, first with the same
        #    suffix and then with common depth suffixes;
        # 2. fall back to root-level files that share the RGB stem.
        candidates = [self.depth_dir / rel_path]
        candidates.extend(self.depth_dir / rel_path.with_suffix(ext) for ext in self.DEPTH_EXTENSIONS)
        candidates.extend(self.depth_dir / f"{image_path.stem}{ext}" for ext in self.DEPTH_EXTENSIONS)

        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique_candidates.append(candidate)
            if candidate.exists() and candidate.is_file():
                return candidate

        tried = ", ".join(str(path) for path in unique_candidates[:8])
        if len(unique_candidates) > 8:
            tried += ", ..."
        raise FileNotFoundError(
            f"No matching depth file found for {image_path}. "
            f"Looked under {self.depth_dir} using mirrored relative paths and same-stem files. "
            f"Tried: {tried}"
        )

    @staticmethod
    def _depth_tensor_from_array(depth: Any, source: Path) -> torch.Tensor:
        if isinstance(depth, dict):
            for key in ("depth", "depth_map", "map"):
                if key in depth:
                    depth = depth[key]
                    break
            else:
                raise ValueError(f"Depth file {source} is a dict without a depth/depth_map/map key.")
        if not torch.is_tensor(depth):
            depth = torch.as_tensor(depth)
        depth = depth.detach().float().cpu()
        if depth.ndim == 4 and depth.shape[0] == 1:
            depth = depth.squeeze(0)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        elif depth.ndim == 3:
            if depth.shape[0] in (1, 3, 4):
                pass
            elif depth.shape[-1] in (1, 3, 4):
                depth = depth.permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported depth shape in {source}: {tuple(depth.shape)}")
            if depth.shape[0] != 1:
                depth = depth[:3].mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported depth shape in {source}: {tuple(depth.shape)}")
        return depth

    def _load_external_depth(self, image_path: Path) -> torch.Tensor:
        depth_path = self._resolve_depth_path(image_path)
        suffix = depth_path.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg"):
            with Image.open(depth_path) as depth_image:
                return TF.to_tensor(depth_image.convert("L"))
        if suffix == ".npy":
            import numpy as np

            return self._depth_tensor_from_array(np.load(depth_path), depth_path)
        if suffix == ".pt":
            return self._depth_tensor_from_array(torch.load(depth_path, map_location="cpu"), depth_path)
        raise ValueError(f"Unsupported depth extension for {depth_path}; expected {self.DEPTH_EXTENSIONS}")

    @staticmethod
    def _normalize_depth_tensor(depth: torch.Tensor) -> torch.Tensor:
        depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=1.0, neginf=0.0)
        d_min = depth.amin(dim=(1, 2), keepdim=True)
        d_max = depth.amax(dim=(1, 2), keepdim=True)
        d_range = d_max - d_min
        already_normalized = (d_min >= 0.0) & (d_max <= 1.0)
        constant_map = d_range <= 1e-6
        scaled = (depth - d_min) / (d_range + 1e-6)
        clamped = depth.clamp(0.0, 1.0)
        return torch.where(already_normalized | constant_map, clamped, scaled.clamp(0.0, 1.0))

    def _transform_image_and_depth(
        self,
        image: Image.Image,
        depth: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        image = TF.resize(image, self.img_size, interpolation=InterpolationMode.BILINEAR)
        resized_width, resized_height = TF.get_image_size(image)
        if depth is not None:
            depth = TF.resize(
                depth,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

        if self.train:
            crop_i, crop_j, crop_h, crop_w = transforms.RandomCrop.get_params(
                image,
                (self.img_size, self.img_size),
            )
        else:
            crop_h = crop_w = self.img_size
            crop_i = max(0, int(round((resized_height - crop_h) / 2.0)))
            crop_j = max(0, int(round((resized_width - crop_w) / 2.0)))

        image = TF.crop(image, crop_i, crop_j, crop_h, crop_w)
        if depth is not None:
            depth = TF.crop(depth, crop_i, crop_j, crop_h, crop_w)
            depth = self._normalize_depth_tensor(depth)

        x_clean = TF.to_tensor(image)
        x_clean = TF.normalize(x_clean, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1, 1]
        return x_clean, depth

    def _load_transformed_sample(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        image_path = self.paths[idx]
        with Image.open(image_path) as image:
            depth = self._load_external_depth(image_path) if self.depth_mode == "external" else None
            return self._transform_image_and_depth(image.convert("RGB"), depth)

    def fog_depth_for_index(self, idx: int) -> Optional[torch.Tensor]:
        if self.depth_mode != "external":
            return None
        _, depth = self._load_transformed_sample(idx)
        return depth

    def __getitem__(self, idx: int):
        x_clean, depth = self._load_transformed_sample(idx)
        expected_shape = (3, self.img_size, self.img_size)
        if tuple(x_clean.shape) != expected_shape:
            raise ValueError(f"Folder sample should be {expected_shape}, got {tuple(x_clean.shape)}")
        if depth is not None and tuple(depth.shape) != (1, self.img_size, self.img_size):
            raise ValueError(
                f"External depth sample should be {(1, self.img_size, self.img_size)}, "
                f"got {tuple(depth.shape)}"
            )
        x_clean = x_clean.unsqueeze(0)
        fog_depth = depth.unsqueeze(0) if depth is not None else None
        gen = None
        if self.fixed_fog:
            gen = torch.Generator(device=x_clean.device).manual_seed(int(idx))
        x_hazy = apply_selected_fog(
            x_clean,
            fog_type=self.fog_type,
            beta_range=self.beta_range,
            a_range=self.a_range,
            blur_sigma_range=self.blur_sigma_range,
            depth_mode=self.depth_mode,
            depth=fog_depth,
            generator=gen,
        )
        return x_clean.squeeze(0), x_hazy.squeeze(0), -1


class FridgePairsDataset(Dataset):
    """Weakly paired fridge start/stop images with fog applied to the start image.

    Returns (x_stop, x_hazy_from_start, -1). The start/stop images may not be
    pixel-aligned, so this dataset is intended for feature-drift objectives.
    """

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
    START_SUFFIX = "_start"
    STOP_SUFFIX = "_stop"

    def __init__(
        self,
        root: str,
        train: bool = True,
        img_size: int = 128,
        beta_range: Tuple[float, float] = (3.5, 7.0),
        a_range: Tuple[float, float] = (0.85, 1.0),
        blur_sigma_range: Tuple[float, float] = (0.5, 1.8),
        fog_type: str = "asm",
        depth_mode: str = "synthetic",
        fixed_fog: bool = False,
        recursive: bool = False,
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.train = train
        self.beta_range = beta_range
        self.a_range = a_range
        self.blur_sigma_range = blur_sigma_range
        self.fog_type = fog_type
        self.depth_mode = depth_mode
        self.fixed_fog = fixed_fog
        if not self.root.exists():
            raise FileNotFoundError(f"Fridge pair folder does not exist: {self.root}")
        if self.depth_mode == "external":
            raise ValueError("external depth is only supported for folder dataset for now.")

        self.pairs, self.skipped_missing_stop = self._scan_pairs(recursive=recursive)
        mode = "recursively" if recursive else "non-recursively"
        print(
            f"[fridge_pairs] found {len(self.pairs)} start/stop pairs {mode} under "
            f"{self.root}; skipped {self.skipped_missing_stop} start images with no stop match."
        )
        if not self.pairs:
            extensions = ", ".join(sorted(self.IMG_EXTENSIONS))
            raise ValueError(
                f"No fridge start/stop pairs found in {self.root}. Expected files like "
                f"18127_0_start.jpg and 18127_0_stop.jpg. Supported extensions: {extensions}"
            )

    def _scan_pairs(self, recursive: bool) -> Tuple[List[Tuple[Path, Path]], int]:
        iterator = self.root.rglob("*") if recursive else self.root.iterdir()
        start_paths = sorted(
            path
            for path in iterator
            if (
                path.is_file()
                and path.suffix.lower() in self.IMG_EXTENSIONS
                and path.stem.lower().endswith(self.START_SUFFIX)
            )
        )
        pairs: List[Tuple[Path, Path]] = []
        skipped = 0
        for start_path in start_paths:
            prefix = start_path.stem[: -len(self.START_SUFFIX)]
            stop_stem = f"{prefix}{self.STOP_SUFFIX}"
            candidate_suffixes = [start_path.suffix]
            candidate_suffixes.extend(
                ext for ext in sorted(self.IMG_EXTENSIONS) if ext != start_path.suffix.lower()
            )
            stop_path = None
            for suffix in candidate_suffixes:
                candidate = start_path.with_name(f"{stop_stem}{suffix}")
                if candidate.exists() and candidate.is_file():
                    stop_path = candidate
                    break
            if stop_path is None:
                skipped += 1
                continue
            pairs.append((start_path, stop_path))
        return sorted(pairs, key=lambda pair: (str(pair[0]), str(pair[1]))), skipped

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_path, stop_path = self.pairs[idx]
        with Image.open(start_path) as start_image, Image.open(stop_path) as stop_image:
            start_image = TF.resize(
                start_image.convert("RGB"),
                self.img_size,
                interpolation=InterpolationMode.BILINEAR,
            )
            stop_image = TF.resize(
                stop_image.convert("RGB"),
                self.img_size,
                interpolation=InterpolationMode.BILINEAR,
            )
            start_width, start_height = TF.get_image_size(start_image)
            stop_width, stop_height = TF.get_image_size(stop_image)
            crop_limit_width = min(start_width, stop_width)
            crop_limit_height = min(start_height, stop_height)
            if self.train:
                max_i = max(0, crop_limit_height - self.img_size)
                max_j = max(0, crop_limit_width - self.img_size)
                crop_i = int(torch.randint(max_i + 1, (1,)).item()) if max_i > 0 else 0
                crop_j = int(torch.randint(max_j + 1, (1,)).item()) if max_j > 0 else 0
                crop_h = crop_w = self.img_size
            else:
                crop_h = crop_w = self.img_size
                crop_i = max(0, int(round((crop_limit_height - crop_h) / 2.0)))
                crop_j = max(0, int(round((crop_limit_width - crop_w) / 2.0)))

            start_image = TF.crop(start_image, crop_i, crop_j, crop_h, crop_w)
            stop_image = TF.crop(stop_image, crop_i, crop_j, crop_h, crop_w)
            x_start = TF.to_tensor(start_image)
            x_stop = TF.to_tensor(stop_image)
            x_start = TF.normalize(x_start, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            x_stop = TF.normalize(x_stop, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return x_start, x_stop

    def __getitem__(self, idx: int):
        x_start, x_stop = self._load_pair(idx)
        expected_shape = (3, self.img_size, self.img_size)
        if tuple(x_start.shape) != expected_shape or tuple(x_stop.shape) != expected_shape:
            raise ValueError(
                f"Fridge pair should produce {expected_shape}; "
                f"got start={tuple(x_start.shape)}, stop={tuple(x_stop.shape)}"
            )
        x_start = x_start.unsqueeze(0)
        gen = None
        if self.fixed_fog:
            gen = torch.Generator(device=x_start.device).manual_seed(int(idx))
        x_hazy = apply_selected_fog(
            x_start,
            fog_type=self.fog_type,
            beta_range=self.beta_range,
            a_range=self.a_range,
            blur_sigma_range=self.blur_sigma_range,
            depth_mode=self.depth_mode,
            generator=gen,
        )
        return x_stop, x_hazy.squeeze(0), -1


def build_hazy_dataset(
    dataset: str,
    data_dir: str,
    train: bool,
    img_size: int,
    fog_config: Dict[str, Tuple[float, float]],
    fog_type: str = "asm",
    depth_mode: str = "synthetic",
    depth_dir: Optional[str] = None,
    fixed_fog: bool = False,
    recursive: bool = False,
) -> Dataset:
    name = dataset.lower()
    if name == "mnist":
        return HazyMNIST(
            root=data_dir,
            train=train,
            img_size=img_size,
            beta_range=fog_config["beta_range"],
            a_range=fog_config["a_range"],
            blur_sigma_range=fog_config["blur_sigma_range"],
            fog_type=fog_type,
            depth_mode=depth_mode,
            fixed_fog=fixed_fog,
        )
    if name in ("cifar", "cifar10"):
        if img_size != 32:
            raise ValueError("CIFAR-10 dehazing uses fixed image size 32.")
        return HazyCIFAR10(
            root=data_dir,
            train=train,
            beta_range=fog_config["beta_range"],
            a_range=fog_config["a_range"],
            blur_sigma_range=fog_config["blur_sigma_range"],
            fog_type=fog_type,
            depth_mode=depth_mode,
            fixed_fog=fixed_fog,
        )
    if name == "folder":
        return RealRGBFolderDataset(
            root=data_dir,
            train=train,
            img_size=img_size,
            beta_range=fog_config["beta_range"],
            a_range=fog_config["a_range"],
            blur_sigma_range=fog_config["blur_sigma_range"],
            fog_type=fog_type,
            depth_mode=depth_mode,
            depth_dir=depth_dir,
            fixed_fog=fixed_fog,
            recursive=recursive,
        )
    if name == "fridge_pairs":
        return FridgePairsDataset(
            root=data_dir,
            train=train,
            img_size=img_size,
            beta_range=fog_config["beta_range"],
            a_range=fog_config["a_range"],
            blur_sigma_range=fog_config["blur_sigma_range"],
            fog_type=fog_type,
            depth_mode=depth_mode,
            fixed_fog=fixed_fog,
            recursive=recursive,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def dataset_channels(dataset: str) -> int:
    name = dataset.lower()
    if name == "mnist":
        return 1
    if name in ("cifar", "cifar10", "folder", "fridge_pairs"):
        return 3
    raise ValueError(f"Unknown dataset: {dataset}")


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
        in_channels: int = 1,
        hidden_size: int = 192,
        depth: int = 6,
        num_heads: int = 4,
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
        expected = (self.in_channels, self.img_size, self.img_size)
        if z.shape != x_hazy.shape:
            raise RuntimeError(f"z shape {tuple(z.shape)} must match x_hazy shape {tuple(x_hazy.shape)}")
        if tuple(z.shape[1:]) != expected:
            raise RuntimeError(f"DehazingDiT expected (*, {expected}), got {tuple(z.shape)}")

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
        return self.unpatchify(x)


def build_dehazing_model(
    model_type: str,
    img_size: int,
    in_channels: int,
    hidden_size: int,
    depth: int,
    num_heads: int,
    unet_base_channels: int,
    unet_depth: int,
) -> nn.Module:
    if model_type == "dit":
        return DehazingDiT(
            img_size=img_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
        )
    if model_type == "unet":
        return UNetDehazer(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            base_channels=unet_base_channels,
            depth=unet_depth,
        )
    if model_type == "residual_unet":
        return ResidualUNetDehazer(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=unet_base_channels,
            depth=unet_depth,
            num_heads=num_heads,
        )
    raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# 3. Losses
# ---------------------------------------------------------------------------


def build_drift_feature_extractor(
    drift_space: str,
    feature_encoder: str,
    model: nn.Module,
    in_channels: int,
    device: torch.device,
    freeze_feature_encoder: bool,
) -> Tuple[Optional[nn.Module], bool]:
    if drift_space == "pixel":
        return None, False
    if drift_space != "feature":
        raise ValueError(f"Unknown drift space: {drift_space}")

    if feature_encoder == "multiscale":
        extractor = make_multiscale_feature_encoder(in_channels).to(device)
        if freeze_feature_encoder:
            for param in extractor.parameters():
                param.requires_grad_(False)
            extractor.eval()
        return extractor, True

    if feature_encoder == "none":
        return ModelEncodeFeatureExtractor(model), False

    raise ValueError(f"Unknown feature encoder: {feature_encoder}")


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------


def train(
    dataset: str = "mnist",
    epochs: int = 10,
    batch_size: int = 128,
    img_size: int = 32,
    hidden_size: int = 192,
    depth: int = 6,
    num_heads: int = 4,
    model_preset: str = "small",
    model_type: str = "dit",
    unet_base_channels: int = 32,
    unet_depth: int = 3,
    fog_preset: Optional[str] = None,
    fog_type: str = "asm",
    depth_mode: str = "synthetic",
    depth_dir: Optional[str] = None,
    noise_mode: str = "random",
    prediction_mode: str = "direct",
    loss_mode: str = "drift",
    drift_positive_mode: str = "batch",
    drift_space: str = "pixel",
    feature_encoder: str = "none",
    freeze_feature_encoder: bool = True,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 2.0,
    warmup_steps: int = 500,
    ema_decay: float = 0.999,
    data_dir: str = "./data/mnist",
    output_dir: str = "./outputs/dehaze",
    device_name: str = "auto",
    num_workers: int = 2,
    log_interval: int = 40,
    sample_interval: int = 0,
    epoch_save_interval: int = 5,
    seed: int = 42,
    smoke_test: bool = False,
    max_steps: Optional[int] = None,
    recursive: bool = False,
    save_fog_debug: bool = False,
    fog_debug_dir: Optional[str] = None,
    fog_debug_max_images: int = 8,
):
    """End-to-end training loop."""
    set_seed(seed)
    name = dataset.lower()
    in_channels = dataset_channels(name)
    if model_type not in ("dit", "unet", "residual_unet"):
        raise ValueError(f"Unknown model type: {model_type}")
    if drift_space not in ("pixel", "feature"):
        raise ValueError(f"Unknown drift space: {drift_space}")
    if feature_encoder not in ("none", "multiscale"):
        raise ValueError(f"Unknown feature encoder: {feature_encoder}")
    effective_prediction_mode = (
        "direct" if model_outputs_final_prediction(model_type) else prediction_mode
    )
    uses_drift_loss = loss_mode in ("drift", "mixed")
    effective_drift_space = drift_space if uses_drift_loss else "pixel"
    effective_feature_encoder = feature_encoder if effective_drift_space == "feature" else "none"
    effective_freeze_feature_encoder = (
        freeze_feature_encoder if effective_drift_space == "feature" else True
    )
    if name in ("cifar", "cifar10"):
        img_size = 32
    if depth_mode == "external":
        if name != "folder":
            raise ValueError("external depth is only supported for folder dataset for now.")
        if depth_dir is None:
            raise ValueError("--depth_mode external requires --depth_dir.")
    if loss_mode == "supervised" and lambda_l1 == 0.0 and lambda_l2 == 0.0:
        raise ValueError(
            "loss_mode=supervised requires at least one supervised weight; "
            "set --lambda_l1 or --lambda_l2."
        )
    if name == "fridge_pairs":
        print(
            "[fridge_pairs] warning: start/stop images are weakly paired and may not be "
            "pixel-aligned. Prefer --loss_mode drift --drift_space feature with no pixel "
            "loss; PSNR/MSE are not strict paired reconstruction metrics unless alignment "
            "has been manually verified."
        )
    fog_preset, fog_config = resolve_fog_config(name, fog_preset)
    hidden_size, depth, num_heads = resolve_model_config(
        model_preset, hidden_size, depth, num_heads
    )

    if device_name == "auto":
        requested_device = None
    else:
        requested_device = torch.device(device_name)

    if requested_device is not None:
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if requested_device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        device = requested_device
    elif torch.cuda.is_available():
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
    print(f"Dataset: {dataset} | image size: {img_size} | channels: {in_channels}")
    print(f"Device: {device}")
    if model_type == "dit":
        print(
            "Model config: "
            f"preset={model_preset}, hidden_size={hidden_size}, depth={depth}, num_heads={num_heads}"
        )
    elif model_type == "residual_unet":
        print(
            "Model config: "
            f"type=residual_unet, input_channels={in_channels}, output_channels={in_channels}, "
            f"base_channels={unet_base_channels}, depth={unet_depth}, num_heads={num_heads}"
        )
    else:
        print(
            "Model config: "
            f"type=unet, input_channels={2 * in_channels}, output_channels={in_channels}, "
            f"base_channels={unet_base_channels}, depth={unet_depth}"
        )
    if prediction_mode != effective_prediction_mode:
        print(
            "Prediction mode: "
            f"requested={prediction_mode}, effective={effective_prediction_mode} "
            f"because {model_type} returns the final dehazed image."
        )
    print(
        "Ablation config: "
        f"fog={fog_preset}, fog_type={fog_type}, beta={fog_config['beta_range']}, "
        f"blur={fog_config['blur_sigma_range']}, depth_mode={depth_mode}, "
        f"noise={noise_mode}, prediction={effective_prediction_mode}, loss_mode={loss_mode}, "
        f"drift_space={effective_drift_space}, drift_positive_mode={drift_positive_mode}, "
        f"feature_encoder={effective_feature_encoder}, "
        f"freeze_feature_encoder={effective_freeze_feature_encoder}, "
        f"lambda_l1={lambda_l1}, lambda_l2={lambda_l2}"
    )

    run_config: Dict[str, Any] = {
        "dataset": name,
        "fog_preset": fog_preset,
        "fog_type": fog_type,
        "depth_mode": depth_mode,
        "depth_dir": depth_dir,
        "beta_range": fog_config["beta_range"],
        "a_range": fog_config["a_range"],
        "blur_sigma_range": fog_config["blur_sigma_range"],
        "noise_mode": noise_mode,
        "prediction_mode": effective_prediction_mode,
        "requested_prediction_mode": prediction_mode,
        "loss_mode": loss_mode,
        "drift_space": effective_drift_space,
        "drift_positive_mode": drift_positive_mode,
        "feature_encoder": effective_feature_encoder,
        "freeze_feature_encoder": effective_freeze_feature_encoder,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "model_type": model_type,
        "model_preset": model_preset,
        "hidden_size": hidden_size,
        "depth": depth,
        "num_heads": num_heads,
        "unet_base_channels": unet_base_channels,
        "unet_depth": unet_depth,
        "batch_size": batch_size,
        "epochs": epochs,
        "img_size": img_size,
        "in_channels": in_channels,
        "save_fog_debug": save_fog_debug,
        "fog_debug_dir": fog_debug_dir,
        "fog_debug_max_images": fog_debug_max_images,
    }

    # In smoke-test mode, log every step so the user sees activity quickly.
    effective_log_interval = 5 if smoke_test else log_interval

    train_set = build_hazy_dataset(
        name,
        data_dir,
        train=True,
        img_size=img_size,
        fog_config=fog_config,
        fog_type=fog_type,
        depth_mode=depth_mode,
        depth_dir=depth_dir,
        recursive=recursive,
    )
    vis_set = build_hazy_dataset(
        name,
        data_dir,
        train=False,
        img_size=img_size,
        fog_config=fog_config,
        fog_type=fog_type,
        depth_mode=depth_mode,
        depth_dir=depth_dir,
        fixed_fog=True,
        recursive=recursive,
    )
    if save_fog_debug:
        save_fog_debug_outputs(
            vis_set,
            device=device,
            fog_type=fog_type,
            depth_mode=depth_mode,
            fog_preset=fog_preset,
            fog_config=fog_config,
            output_dir=out,
            fog_debug_dir=fog_debug_dir,
            max_images=fog_debug_max_images,
            seed=seed,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = build_dehazing_model(
        model_type=model_type,
        img_size=img_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        unet_base_channels=unet_base_channels,
        unet_depth=unet_depth,
    ).to(device)
    print(f"{model.__class__.__name__} params: {count_parameters(model):,}")

    drift_feature_extractor: Optional[nn.Module] = None
    has_external_feature_encoder = False
    if uses_drift_loss and effective_drift_space == "feature":
        drift_feature_extractor, has_external_feature_encoder = build_drift_feature_extractor(
            drift_space=effective_drift_space,
            feature_encoder=effective_feature_encoder,
            model=model,
            in_channels=in_channels,
            device=device,
            freeze_feature_encoder=effective_freeze_feature_encoder,
        )
    train_feature_encoder = (
        drift_feature_extractor is not None
        and has_external_feature_encoder
        and not effective_freeze_feature_encoder
    )
    if drift_feature_extractor is not None:
        print(
            "Drift feature extractor: "
            f"{drift_feature_extractor.__class__.__name__} "
            f"(trainable={train_feature_encoder})"
        )

    ema = EMA(model, decay=ema_decay)
    optimizer_params = list(model.parameters())
    if train_feature_encoder:
        optimizer_params.extend(drift_feature_extractor.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
    )
    scheduler = WarmupLRScheduler(optimizer, warmup_steps=warmup_steps, base_lr=lr)

    def checkpoint_payload():
        payload = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "config": {
                "img_size": img_size,
                "hidden_size": hidden_size,
                "depth": depth,
                "num_heads": num_heads,
                "in_channels": in_channels,
                "dataset": name,
                "model_type": model_type,
                "model_preset": model_preset,
                "unet_base_channels": unet_base_channels,
                "unet_depth": unet_depth,
                "fog_preset": fog_preset,
                "fog_type": fog_type,
                "depth_mode": depth_mode,
                "depth_dir": depth_dir,
                "beta_range": fog_config["beta_range"],
                "a_range": fog_config["a_range"],
                "blur_sigma_range": fog_config["blur_sigma_range"],
                "noise_mode": noise_mode,
                "prediction_mode": effective_prediction_mode,
                "requested_prediction_mode": prediction_mode,
                "loss_mode": loss_mode,
                "drift_space": effective_drift_space,
                "drift_positive_mode": drift_positive_mode,
                "feature_encoder": effective_feature_encoder,
                "freeze_feature_encoder": effective_freeze_feature_encoder,
                "lambda_l1": lambda_l1,
                "lambda_l2": lambda_l2,
                "save_fog_debug": save_fog_debug,
                "fog_debug_dir": fog_debug_dir,
                "fog_debug_max_images": fog_debug_max_images,
            },
        }
        if has_external_feature_encoder and drift_feature_extractor is not None:
            payload["feature_encoder_state"] = drift_feature_extractor.state_dict()
        return payload

    def save_random_triple_grid(path: Path, n: int = 16) -> Dict[str, float]:
        indices = torch.randint(len(vis_set), (n,))
        samples = [vis_set[int(idx)] for idx in indices]
        vis_clean = torch.stack([sample[0] for sample in samples], dim=0).to(device)
        vis_hazy = torch.stack([sample[1] for sample in samples], dim=0).to(device)
        return save_triple_grid(
            ema.shadow,
            vis_clean,
            vis_hazy,
            path,
            noise_mode=noise_mode,
            prediction_mode=effective_prediction_mode,
            run_config=run_config,
            feature_extractor=drift_feature_extractor,
        )

    global_step = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        running = {"total_loss": 0.0, "drift_loss": 0.0, "l1_loss": 0.0, "l2_loss": 0.0, "n": 0}

        for batch_idx, (x_clean, x_hazy, _label) in enumerate(train_loader):
            x_clean = x_clean.to(device, non_blocking=True)
            x_hazy = x_hazy.to(device, non_blocking=True)
            expected_shape = (in_channels, img_size, img_size)
            if tuple(x_clean.shape[1:]) != expected_shape or tuple(x_hazy.shape[1:]) != expected_shape:
                raise RuntimeError(
                    f"Bad batch shape clean={tuple(x_clean.shape)}, hazy={tuple(x_hazy.shape)}, "
                    f"expected (*, {in_channels}, {img_size}, {img_size})"
                )

            z = make_noise(x_clean, noise_mode)
            if z.shape != x_clean.shape:
                raise RuntimeError(f"Noise shape {tuple(z.shape)} != clean {tuple(x_clean.shape)}")
            raw_output = model(z, x_hazy)
            if raw_output.shape != x_clean.shape:
                raise RuntimeError(f"Model output shape {tuple(raw_output.shape)} != clean {tuple(x_clean.shape)}")
            x_hat = apply_prediction_mode(raw_output, x_hazy, effective_prediction_mode)
            if x_hat.shape != x_clean.shape:
                raise RuntimeError(f"Dehazed output shape {tuple(x_hat.shape)} != clean {tuple(x_clean.shape)}")

            l1_loss = F.l1_loss(x_hat, x_clean)
            l2_loss = F.mse_loss(x_hat, x_clean)
            if loss_mode == "supervised":
                drift_loss = torch.zeros((), device=device)
                loss = lambda_l1 * l1_loss + lambda_l2 * l2_loss
            elif loss_mode in ("drift", "mixed"):
                drift_loss = compute_drift_loss(
                    x_hat,
                    x_clean,
                    drift_space=effective_drift_space,
                    positive_mode=drift_positive_mode,
                    feature_extractor=drift_feature_extractor,
                )
                loss = drift_loss + lambda_l1 * l1_loss + lambda_l2 * l2_loss
            else:
                raise ValueError(f"Unknown loss mode: {loss_mode}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            running["total_loss"] += loss.item()
            running["drift_loss"] += drift_loss.item()
            running["l1_loss"] += l1_loss.item()
            running["l2_loss"] += l2_loss.item()
            running["n"] += 1
            global_step += 1

            if global_step % effective_log_interval == 0:
                avg = {k: v / max(running["n"], 1) for k, v in running.items() if k != "n"}
                print(
                    f"Epoch {epoch + 1}/{epochs} | step {global_step} | "
                    f"total {avg['total_loss']:.4f} | drift {avg['drift_loss']:.4f} | "
                    f"l1 {avg['l1_loss']:.4f} | l2 {avg['l2_loss']:.4f} | grad {grad_norm:.2f} | "
                    f"lr {scheduler.get_lr():.6f}"
                )

            if sample_interval > 0 and global_step % sample_interval == 0:
                save_random_triple_grid(samples_dir / f"step_{global_step:06d}.png")

            if smoke_test and global_step >= 20:
                print("[smoke_test] early stop")
                # Final smoke-test image to confirm the visualisation path works.
                metrics = save_random_triple_grid(samples_dir / "smoke.png")
                if name in ("cifar", "cifar10"):
                    save_random_triple_grid(out / "cifar_dehaze_samples.png")
                print(format_metrics(metrics))
                torch.save(checkpoint_payload(), checkpoints_dir / "latest.pt")
                return

            if max_steps is not None and global_step >= max_steps:
                print(f"[max_steps] reached {max_steps}, stopping")
                metrics = save_random_triple_grid(samples_dir / "latest.png")
                if name in ("cifar", "cifar10"):
                    save_random_triple_grid(out / "cifar_dehaze_samples.png")
                print(format_metrics(metrics))
                torch.save(checkpoint_payload(), checkpoints_dir / "latest.pt")
                return

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1} done in {elapsed:.1f}s | "
            f"avg total {running['total_loss'] / max(running['n'], 1):.4f} | "
            f"avg drift {running['drift_loss'] / max(running['n'], 1):.4f}"
        )
        epoch_num = epoch + 1
        should_save_epoch = (
            epoch_save_interval > 0
            and (epoch_num % epoch_save_interval == 0 or epoch_num == epochs)
        )
        if should_save_epoch:
            epoch_name = f"epoch_{epoch_num:03d}"
            epoch_sample_path = samples_dir / f"{epoch_name}.png"
            metrics = save_random_triple_grid(epoch_sample_path)
            shutil.copyfile(epoch_sample_path, samples_dir / "latest.png")
            shutil.copyfile(epoch_sample_path.with_suffix(".txt"), samples_dir / "latest.txt")
            if name in ("cifar", "cifar10"):
                shutil.copyfile(epoch_sample_path, out / "cifar_dehaze_samples.png")
                shutil.copyfile(epoch_sample_path.with_suffix(".txt"), out / "cifar_dehaze_samples.txt")

            ckpt_payload = checkpoint_payload()
            torch.save(ckpt_payload, checkpoints_dir / f"{epoch_name}.pt")
            torch.save(ckpt_payload, checkpoints_dir / "latest.pt")
            print(f"  Saved samples/{epoch_name}.png and checkpoints/{epoch_name}.pt | {format_metrics(metrics)}")

    torch.save(checkpoint_payload(), checkpoints_dir / "final.pt")
    print(f"Saved final checkpoint to {checkpoints_dir / 'final.pt'}")


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def _tensor_summary(tensor: torch.Tensor, include_values: bool = False) -> Dict[str, Any]:
    flat = tensor.detach().float().cpu().reshape(-1)
    summary: Dict[str, Any] = {
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
    }
    if include_values:
        summary["values"] = [float(v) for v in flat.tolist()]
    return summary


def _save_debug_grid(
    tensor: torch.Tensor,
    path: Path,
    nrow: int,
    value_range: Tuple[float, float],
) -> None:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    save_image_grid(
        tensor.detach().cpu(),
        str(path),
        nrow=nrow,
        normalize=True,
        value_range=value_range,
    )


@torch.no_grad()
def save_fog_debug_outputs(
    dataset: Dataset,
    device: torch.device,
    fog_type: str,
    depth_mode: str,
    fog_preset: str,
    fog_config: Dict[str, Tuple[float, float]],
    output_dir: Path,
    fog_debug_dir: Optional[str],
    max_images: int,
    seed: int,
) -> None:
    if max_images <= 0:
        print("[fog_debug] skipped because --fog_debug_max_images <= 0")
        return
    n = min(max_images, len(dataset))
    debug_dir = Path(fog_debug_dir) if fog_debug_dir else output_dir / "fog_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    samples = [dataset[idx] for idx in range(n)]
    x_clean = torch.stack([sample[0] for sample in samples], dim=0).to(device)
    depth = None
    if depth_mode == "external":
        depth_getter = getattr(dataset, "fog_depth_for_index", None)
        if depth_getter is None:
            raise ValueError("depth_mode=external requires a dataset that can provide fog depth maps.")
        depth_items = [depth_getter(idx) for idx in range(n)]
        if any(item is None for item in depth_items):
            raise ValueError("depth_mode=external did not provide depth maps for fog debug.")
        depth = torch.stack(depth_items, dim=0).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    x_hazy, meta = apply_selected_fog(
        x_clean,
        fog_type=fog_type,
        beta_range=fog_config["beta_range"],
        a_range=fog_config["a_range"],
        blur_sigma_range=fog_config["blur_sigma_range"],
        depth_mode=depth_mode,
        depth=depth,
        generator=generator,
        return_meta=True,
    )

    nrow = max(1, int(math.ceil(math.sqrt(n))))
    files: Dict[str, str] = {}
    grids = [
        ("clean", x_clean, (-1.0, 1.0)),
        ("hazy", x_hazy, (-1.0, 1.0)),
        ("depth", meta["depth"], (0.0, 1.0)),
        ("transmission", meta["transmission"], (0.0, 1.0)),
    ]
    density_key = "beta_tilde" if "beta_tilde" in meta else "density" if "density" in meta else None
    if density_key is not None:
        grids.append(("density", meta[density_key], (0.0, 1.0)))

    for name, tensor, value_range in grids:
        path = debug_dir / f"{name}_grid.png"
        _save_debug_grid(tensor, path, nrow=nrow, value_range=value_range)
        files[name] = str(path)

    scalar_keys = ["beta", "atmospheric_light", "alpha"]
    map_keys = ["depth", "transmission"]
    if density_key is not None:
        map_keys.append(density_key)
    summary = {
        "fog_type": fog_type,
        "depth_mode": depth_mode,
        "fog_preset": fog_preset,
        "fog_config": _json_ready(meta.get("fog_config", fog_config)),
        "num_images": n,
        "files": files,
        "scalars": {
            key: _tensor_summary(meta[key], include_values=True)
            for key in scalar_keys
            if key in meta and isinstance(meta[key], torch.Tensor)
        },
        "maps": {
            key: _tensor_summary(meta[key])
            for key in map_keys
            if key in meta and isinstance(meta[key], torch.Tensor)
        },
    }
    summary_path = debug_dir / "metadata_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[fog_debug] saved fog debug grids and metadata to {debug_dir}")


@torch.no_grad()
def save_triple_grid(
    model: nn.Module,
    x_clean: torch.Tensor,
    x_hazy: torch.Tensor,
    path: Path,
    noise_mode: str,
    prediction_mode: str,
    run_config: Dict[str, Any],
    feature_extractor: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Save a grid with rows of [clean | hazy | dehazed]."""
    was_training = model.training
    feature_was_training = feature_extractor.training if feature_extractor is not None else False
    model.eval()
    if feature_extractor is not None:
        feature_extractor.eval()
    if x_clean.shape != x_hazy.shape:
        raise RuntimeError(f"Clean shape {tuple(x_clean.shape)} != hazy {tuple(x_hazy.shape)}")
    z = make_noise(x_clean, noise_mode)
    if z.shape != x_clean.shape:
        raise RuntimeError(f"Noise shape {tuple(z.shape)} != clean {tuple(x_clean.shape)}")
    raw_output = model(z, x_hazy)
    if raw_output.shape != x_clean.shape:
        raise RuntimeError(f"Model output shape {tuple(raw_output.shape)} != clean {tuple(x_clean.shape)}")
    x_hat = apply_prediction_mode(raw_output, x_hazy, prediction_mode)
    if x_hat.shape != x_clean.shape:
        raise RuntimeError(f"Dehazed output shape {tuple(x_hat.shape)} != clean {tuple(x_clean.shape)}")
    metrics = paired_dehaze_metrics(
        x_hat,
        x_hazy,
        x_clean,
        run_config,
        feature_extractor=feature_extractor,
    )

    # Interleave: clean_i, hazy_i, dehazed_i, clean_{i+1}, hazy_{i+1}, ...
    n = x_clean.shape[0]
    stacked = torch.stack([x_clean, x_hazy, x_hat], dim=1)  # (N, 3, C, H, W)
    grid = stacked.reshape(n * 3, *x_clean.shape[1:])
    save_image_grid(grid, str(path), nrow=3)
    save_metrics(path.with_suffix(".txt"), metrics, run_config)
    if was_training:
        model.train()
    if feature_extractor is not None and feature_was_training:
        feature_extractor.train()
    return metrics


def _mse_psnr(x: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    mse = F.mse_loss(x, target).item()
    psnr = 20.0 * math.log10(2.0) - 10.0 * math.log10(max(mse, 1e-12))
    return mse, psnr


def paired_dehaze_metrics(
    x_hat: torch.Tensor,
    x_hazy: torch.Tensor,
    x_clean: torch.Tensor,
    run_config: Dict[str, Any],
    feature_extractor: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Simple paired synthetic metrics on [-1, 1] tensors."""
    mse_hazy, psnr_hazy = _mse_psnr(x_hazy, x_clean)
    mse_dehazed, psnr_dehazed = _mse_psnr(x_hat, x_clean)
    l1_loss = F.l1_loss(x_hat, x_clean).item()
    l2_loss = F.mse_loss(x_hat, x_clean).item()
    loss_mode = str(run_config.get("loss_mode", "drift"))
    if loss_mode == "supervised":
        drift_loss = 0.0
        total_loss = float(run_config["lambda_l1"]) * l1_loss + float(run_config["lambda_l2"]) * l2_loss
    elif loss_mode in ("drift", "mixed"):
        drift_loss = compute_drift_loss(
            x_hat,
            x_clean,
            drift_space=str(run_config.get("drift_space", "pixel")),
            positive_mode=str(run_config.get("drift_positive_mode", "batch")),
            feature_extractor=feature_extractor,
        ).item()
        total_loss = (
            drift_loss
            + float(run_config["lambda_l1"]) * l1_loss
            + float(run_config["lambda_l2"]) * l2_loss
        )
    else:
        raise ValueError(f"Unknown loss mode: {loss_mode}")
    return {
        "mse_hazy": mse_hazy,
        "psnr_hazy": psnr_hazy,
        "mse_dehazed": mse_dehazed,
        "psnr_dehazed": psnr_dehazed,
        "drift_loss": drift_loss,
        "l1_loss": l1_loss,
        "l2_loss": l2_loss,
        "total_loss": total_loss,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"hazy MSE {metrics['mse_hazy']:.6f} | hazy PSNR {metrics['psnr_hazy']:.2f} dB | "
        f"dehazed MSE {metrics['mse_dehazed']:.6f} | dehazed PSNR {metrics['psnr_dehazed']:.2f} dB"
    )


def save_metrics(path: Path, metrics: Dict[str, float], run_config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_keys = [
        "dataset",
        "fog_preset",
        "fog_type",
        "depth_mode",
        "depth_dir",
        "beta_range",
        "blur_sigma_range",
        "noise_mode",
        "prediction_mode",
        "requested_prediction_mode",
        "loss_mode",
        "drift_space",
        "drift_positive_mode",
        "feature_encoder",
        "freeze_feature_encoder",
        "lambda_l1",
        "lambda_l2",
        "model_type",
        "model_preset",
        "hidden_size",
        "depth",
        "num_heads",
        "unet_base_channels",
        "unet_depth",
        "batch_size",
        "epochs",
    ]
    lines = [f"{key}: {run_config[key]}" for key in ordered_keys]
    lines.extend([
        f"mse_hazy: {metrics['mse_hazy']:.8f}",
        f"psnr_hazy: {metrics['psnr_hazy']:.4f}",
        f"mse_dehazed: {metrics['mse_dehazed']:.8f}",
        f"psnr_dehazed: {metrics['psnr_dehazed']:.4f}",
        f"drift_loss: {metrics['drift_loss']:.8f}",
        f"l1_loss: {metrics['l1_loss']:.8f}",
        f"l2_loss: {metrics['l2_loss']:.8f}",
        f"total_loss: {metrics['total_loss']:.8f}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Synthetic MNIST/CIFAR-10 dehazing toy example.")
    p.add_argument("--dataset", choices=["mnist", "cifar10", "folder", "fridge_pairs"], default="mnist")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--model_preset", choices=["small", "medium", "large", "custom"], default="small")
    p.add_argument("--model_type", choices=["dit", "unet", "residual_unet"], default="dit")
    p.add_argument("--hidden_size", type=int, default=192)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--unet_base_channels", type=int, default=32)
    p.add_argument("--unet_depth", type=int, default=3)
    p.add_argument("--fog_preset", choices=["mild", "medium", "heavy"], default=None,
                   help="Default: mild for CIFAR-10, heavy/original for MNIST.")
    p.add_argument("--fog_type", choices=["asm", "mcbm"], default="asm")
    p.add_argument("--depth_mode", choices=["synthetic", "flat", "external"], default="synthetic")
    p.add_argument("--depth_dir", type=str, default=None,
                   help="Folder of external depth maps. Required when --depth_mode external.")
    p.add_argument("--noise_mode", choices=["random", "zero"], default="random")
    p.add_argument("--prediction_mode", choices=["direct", "residual"], default="direct")
    p.add_argument("--loss_mode", choices=["drift", "supervised", "mixed"], default="drift")
    p.add_argument(
        "--drift_space",
        "--drift_loss",
        dest="drift_space",
        choices=["pixel", "feature"],
        default="pixel",
        help="Drift space used for drift computation. --drift_loss is a deprecated alias.",
    )
    p.add_argument("--drift_positive_mode", choices=["batch", "paired"], default="batch")
    p.add_argument("--feature_encoder", choices=["multiscale", "none"], default="none")
    p.add_argument(
        "--freeze_feature_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the feature encoder used by feature-space drift.",
    )
    p.add_argument("--lambda_l1", type=float, default=0.0)
    p.add_argument("--lambda_l2", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--save_dir", "--output_dir", dest="output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--recursive", action="store_true",
                   help="Recursively scan image files when --dataset folder or fridge_pairs.")
    p.add_argument("--save_fog_debug", action="store_true",
                   help="Save one small fog-generation debug batch before training.")
    p.add_argument("--fog_debug_dir", type=str, default=None,
                   help="Directory for fog debug grids. Default: <save_dir>/fog_debug.")
    p.add_argument("--fog_debug_max_images", type=int, default=8,
                   help="Maximum images to include in fog debug grids.")
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--sample_interval", type=int, default=400,
                   help="Optional step interval for extra samples (0 = only save per epoch).")
    p.add_argument("--epoch_save_interval", type=int, default=20,
                   help="Save epoch samples/checkpoints every N epochs (0 = disable).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke_test", action="store_true",
                   help="Run only ~20 steps and save one visualization for a quick sanity check.")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Optional hard cap on number of training steps.")
    args = p.parse_args()
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = "./data/mnist" if args.dataset == "mnist" else "./data"
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "./outputs/dehaze" if args.dataset == "mnist" else "./outputs/cifar_dehaze"
    img_size = args.img_size
    if img_size is None:
        img_size = 32

    train(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=img_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        model_preset=args.model_preset,
        model_type=args.model_type,
        unet_base_channels=args.unet_base_channels,
        unet_depth=args.unet_depth,
        fog_preset=args.fog_preset,
        fog_type=args.fog_type,
        depth_mode=args.depth_mode,
        depth_dir=args.depth_dir,
        noise_mode=args.noise_mode,
        prediction_mode=args.prediction_mode,
        loss_mode=args.loss_mode,
        drift_space=args.drift_space,
        drift_positive_mode=args.drift_positive_mode,
        feature_encoder=args.feature_encoder,
        freeze_feature_encoder=args.freeze_feature_encoder,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        data_dir=data_dir,
        output_dir=output_dir,
        device_name=args.device,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        sample_interval=args.sample_interval,
        epoch_save_interval=args.epoch_save_interval,
        seed=args.seed,
        smoke_test=args.smoke_test,
        max_steps=args.max_steps,
        recursive=args.recursive,
        save_fog_debug=args.save_fog_debug,
        fog_debug_dir=args.fog_debug_dir,
        fog_debug_max_images=args.fog_debug_max_images,
    )


if __name__ == "__main__":
    main()
