"""CLIP encoders. Images and text land in the same embedding space, so one dot product compares them."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

try:  # iPhone HEIC photos, when the optional plugin is installed
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


class Encoder(Protocol):
    dim: int

    def encode_images(self, paths: list[Path]) -> np.ndarray: ...
    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class ClipEncoder:
    """open_clip wrapper. ViT-B-32 trained on LAION-2B: ~600 MB, fast on CPU and MPS."""

    def __init__(
        self,
        model: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        batch_size: int = 32,
    ):
        import open_clip
        import torch

        self.torch = torch
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        self.batch_size = batch_size
        self.dim = self.model.text_projection.shape[1]

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image, ImageOps

        out = []
        for i in range(0, len(paths), self.batch_size):
            batch = []
            for p in paths[i : i + self.batch_size]:
                with Image.open(p) as img:
                    batch.append(self.preprocess(ImageOps.exif_transpose(img).convert("RGB")))
            with self.torch.no_grad():
                feats = self.model.encode_image(self.torch.stack(batch).to(self.device))
            out.append(feats.float().cpu().numpy())
        return _normalize(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            feats = self.model.encode_text(self.tokenizer(texts).to(self.device))
        return _normalize(feats.float().cpu().numpy())
