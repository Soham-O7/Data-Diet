"""Image embedders. One interface: fit_transform(images) / transform(images) -> (N, d) float32, L2-normalised.

* PixelPCAEmbedder  - dependency-free fallback (flatten pixels -> global scale -> PCA). Works everywhere, but it is
                      NOT a semantic embedding: SemDeDup / difficulty / coverage quality will be lower than with DINOv2.
* DinoV2Embedder    - the plan's default (needs torch + network/cache for the torch.hub weights).
                      NOT exercised in the sandbox this was built in (torch unavailable) - test it on your machine.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def _l2norm(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (a / n).astype(np.float32)


class PixelPCAEmbedder:
    name = "pixel_pca"

    def __init__(self, n_components: int = 64, max_pixels: int = 4096, seed: int = 0):
        self.n_components = n_components
        self.max_pixels = max_pixels
        self.seed = seed

    def _flatten(self, images: np.ndarray) -> np.ndarray:
        from PIL import Image

        a = np.asarray(images)
        if a.ndim not in (3, 4):
            raise ValueError(f"expected images of shape (N,H,W) or (N,H,W,C), got {a.shape}")
        h, w = a.shape[1], a.shape[2]
        c = a.shape[3] if a.ndim == 4 else 1
        if h * w * c > self.max_pixels:
            # resize to 32x32 (keeps channels)
            u8 = a if a.dtype == np.uint8 else np.clip(a, 0, 255).astype(np.uint8)
            out = np.empty((len(a), 32, 32) + ((c,) if a.ndim == 4 else ()), dtype=np.float32)
            for i in range(len(a)):
                out[i] = np.asarray(Image.fromarray(u8[i]).resize((32, 32), Image.BILINEAR), dtype=np.float32)
            a = out
        return a.reshape(len(a), -1).astype(np.float32)

    def fit(self, images: np.ndarray) -> "PixelPCAEmbedder":
        # One GLOBAL scale (not per-pixel standardisation): per-pixel scaling blows up tiny noise on
        # near-constant background pixels and wrecks near-duplicate detection.
        F = self._flatten(images)
        self.scale_ = float(np.abs(F).max()) or 1.0
        k = min(self.n_components, F.shape[0] - 1, F.shape[1])
        self.pca_ = PCA(n_components=k, random_state=self.seed).fit(F / self.scale_)
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        F = self._flatten(images) / self.scale_
        return _l2norm(self.pca_.transform(F))

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        return self.fit(images).transform(images)


class DinoV2Embedder:
    """DINOv2 ViT-S/14 CLS embedding via torch.hub. Untested in the build sandbox (no torch)."""

    name = "dinov2_vits14"

    def __init__(self, batch_size: int = 128, device: str | None = None, model_name: str = "dinov2_vits14"):
        self.batch_size = batch_size
        self.device = device
        self.model_name = model_name
        self.model_ = None

    def _load(self):
        import torch  # lazy: only needed if this embedder is used

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_ = torch
        self.dev_ = dev
        self.model_ = torch.hub.load("facebookresearch/dinov2", self.model_name).to(dev).eval()
        self.mean_ = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
        self.std_ = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def fit(self, images):  # frozen model: nothing to fit
        if self.model_ is None:
            self._load()
        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            self._load()
        torch = self.torch_
        a = np.asarray(images)
        if a.ndim == 3:
            a = np.repeat(a[..., None], 3, axis=3)
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        outs = []
        with torch.no_grad():
            for s in range(0, len(a), self.batch_size):
                x = torch.from_numpy(a[s:s + self.batch_size]).to(self.dev_).permute(0, 3, 1, 2).float() / 255.0
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
                x = (x - self.mean_) / self.std_
                outs.append(self.model_(x).float().cpu().numpy())
        return _l2norm(np.concatenate(outs, axis=0))

    def fit_transform(self, images):
        return self.fit(images).transform(images)


def get_embedder(name: str = "pixel", **kw):
    if name in ("pixel", "pixel_pca"):
        return PixelPCAEmbedder(**kw)
    if name in ("dinov2", "dino"):
        return DinoV2Embedder(**kw)
    raise ValueError(f"unknown embedder {name!r}; use 'pixel' or 'dinov2'")
