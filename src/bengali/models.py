"""Models: a fine-tuned CNN with three linear heads, and frozen encoders for linear probes."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import COMPONENTS, N_CLASSES


class MultiHead(nn.Module):
    """Three independent linear heads: root (168), vowel (11), consonant (7)."""

    def __init__(self, in_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.heads = nn.ModuleDict({c: nn.Linear(in_features, N_CLASSES[c]) for c in COMPONENTS})

    def forward(self, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.drop(feats)
        return {c: h(feats) for c, h in self.heads.items()}


class FinetuneNet(nn.Module):
    """timm backbone (ImageNet weights) + global pooling + MultiHead. Grayscale input."""

    # grayscale stats of ImageNet normalisation: timm sums RGB conv1 weights for in_chans=1
    MEAN, STD = 0.449, 0.226

    def __init__(self, backbone: str, pretrained: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=1)
        self.head = MultiHead(self.backbone.num_features, dropout)

    def forward(self, x_uint8: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (x_uint8.float() / 255.0 - self.MEAN) / self.STD
        return self.head(self.backbone(x.unsqueeze(1)))


# ---------------------------------------------------------------- frozen encoders


class FrozenEncoder(nn.Module):
    """Wraps a pretrained encoder; input: uint8 [B, H, W] preprocessed crops (ink bright)."""

    name: str = ""
    dim: int = 0

    def __init__(self, input_size: int, invert: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        # foundation models were pretrained on natural/document images => feed dark ink on light paper
        self.invert = invert

    def to_rgb(self, x_uint8: torch.Tensor, mean, std) -> torch.Tensor:
        x = x_uint8.float() / 255.0
        if self.invert:
            x = 1.0 - x
        x = x.unsqueeze(1)
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False, antialias=True)
        x = x.expand(-1, 3, -1, -1)
        mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std


class PixelEncoder(FrozenEncoder):
    """'No representation learning' control: raw downsampled pixels."""

    def __init__(self, input_size: int = 32, invert: bool = False) -> None:
        super().__init__(input_size, invert)
        self.name = f"pixels{input_size}"
        self.dim = input_size * input_size

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x = x_uint8.float().unsqueeze(1) / 255.0
        x = F.adaptive_avg_pool2d(x, self.input_size)
        return x.flatten(1)


class DinoV3Encoder(FrozenEncoder):
    """DINOv3 ViT (timm). Features = concat(CLS, mean of patch tokens) — the DINO linear-probe recipe."""

    def __init__(self, model_name: str = "vit_small_patch16_dinov3.lvd1689m", input_size: int = 224, invert: bool = True) -> None:
        super().__init__(input_size, invert)
        import timm

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
        cfg = self.model.pretrained_cfg
        self.mean, self.std = cfg["mean"], cfg["std"]
        self.name = model_name
        self.dim = 2 * self.model.num_features

    @torch.no_grad()
    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        t = self.model.forward_features(self.to_rgb(x_uint8, self.mean, self.std))
        cls = t[:, 0]
        patches = t[:, self.model.num_prefix_tokens :].mean(1)
        return torch.cat([cls, patches], dim=1)


class MobileClip2Encoder(FrozenEncoder):
    """Apple MobileCLIP2 image tower (open_clip). Features = image embedding (pre-normalisation)."""

    def __init__(self, model_name: str = "MobileCLIP2-S2", pretrained: str = "dfndr2b", input_size: int = 256, invert: bool = True) -> None:
        super().__init__(input_size, invert)
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.eval()
        norm = [t for t in preprocess.transforms if t.__class__.__name__ == "Normalize"][0]
        self.mean, self.std = tuple(norm.mean), tuple(norm.std)
        self.name = f"{model_name}.{pretrained}"
        self.n_params = count_params(self.model.visual)  # image tower only; the text tower is never used
        with torch.no_grad():  # infer the embedding size instead of hard-coding it
            self.dim = int(self.model.encode_image(torch.zeros(1, 3, input_size, input_size)).shape[1])

    @torch.no_grad()
    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(self.to_rgb(x_uint8, self.mean, self.std), normalize=False)


def build_encoder(cfg: dict) -> FrozenEncoder:
    kind = cfg["encoder"]
    kw = {k: v for k, v in cfg.items() if k in {"input_size", "invert", "model_name", "pretrained"}}
    if kind == "pixels":
        return PixelEncoder(**kw)
    if kind == "dinov3":
        return DinoV3Encoder(**kw)
    if kind == "mobileclip2":
        return MobileClip2Encoder(**kw)
    raise ValueError(kind)


def count_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def standardize(train: np.ndarray, *others: np.ndarray, eps: float = 1e-6):
    """Feature standardisation fitted on TRAIN rows only (no leakage)."""
    mu = train.mean(0, keepdims=True)
    sd = train.std(0, keepdims=True) + eps
    return [(a - mu) / sd for a in (train, *others)]
