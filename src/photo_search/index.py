"""Incremental photo index: only new or changed files are embedded on each run."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .encoder import Encoder

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic"}

# Prompt ensembling (from the CLIP paper): averaging a few phrasings beats a bare label.
TEMPLATES = ["a photo of {}.", "a picture of {}.", "an image showing {}.", "{}"]

DEFAULT_TAGS = [
    "a person", "a group of people", "a selfie", "a dog", "a cat", "food", "a beach",
    "mountains", "a city street", "a building", "a car", "a document", "a screenshot",
    "a receipt", "a whiteboard", "a sunset", "a forest", "a party", "a temple", "a wedding",
    "a classroom", "a laptop", "a chart or graph", "a flower", "snow", "rain", "night sky",
]  # fmt: skip


@dataclass
class Result:
    path: str
    score: float


def _fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def scan(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )


class PhotoIndex:
    def __init__(self, directory: Path, encoder: Encoder):
        self.dir, self.encoder = directory, encoder
        self.paths: list[str] = []
        self.prints: list[str] = []
        self.vectors = np.zeros((0, encoder.dim), np.float32)
        if (directory / "meta.json").exists():
            meta = json.loads((directory / "meta.json").read_text())
            self.paths, self.prints = meta["paths"], meta["prints"]
            self.vectors = np.load(directory / "vectors.npy")

    def __len__(self) -> int:
        return len(self.paths)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "vectors.npy", self.vectors)
        (self.dir / "meta.json").write_text(
            json.dumps({"paths": self.paths, "prints": self.prints})
        )

    def update(
        self, root: Path, batch: int = 64, progress: Callable[[int, int], None] | None = None
    ) -> dict:
        """Sync the index with a folder: embed new/changed files, drop deleted ones."""
        found = {str(p.resolve()): p for p in scan(root)}
        root_key = str(root.resolve())
        known = {p: (i, fp) for i, (p, fp) in enumerate(zip(self.paths, self.prints, strict=True))}

        keep = [
            i
            for p, (i, fp) in known.items()
            if not p.startswith(root_key) or (p in found and _fingerprint(found[p]) == fp)
        ]
        removed = len(self.paths) - len(keep)
        self.paths = [self.paths[i] for i in keep]
        self.prints = [self.prints[i] for i in keep]
        self.vectors = self.vectors[keep]

        indexed = set(self.paths)
        todo = [p for key, p in found.items() if key not in indexed]
        failed = 0
        for start in range(0, len(todo), batch):
            chunk = todo[start : start + batch]
            ok_paths, vecs = self._encode_safely(chunk)
            failed += len(chunk) - len(ok_paths)
            if ok_paths:
                self.paths += [str(p.resolve()) for p in ok_paths]
                self.prints += [_fingerprint(p) for p in ok_paths]
                self.vectors = np.vstack([self.vectors, vecs])
            if progress:
                progress(min(start + batch, len(todo)), len(todo))
            self.save()
        self.save()
        return {
            "added": len(todo) - failed,
            "removed": removed,
            "failed": failed,
            "total": len(self),
        }

    def _encode_safely(self, paths: list[Path]) -> tuple[list[Path], np.ndarray]:
        try:
            return paths, self.encoder.encode_images(paths)
        except Exception:
            ok, vecs = [], []
            for p in paths:  # isolate the corrupt file instead of losing the whole batch
                try:
                    vecs.append(self.encoder.encode_images([p])[0])
                    ok.append(p)
                except Exception:
                    pass
            return ok, np.array(vecs, np.float32).reshape(len(ok), self.encoder.dim)

    def _top(self, query: np.ndarray, k: int, exclude: str | None = None) -> list[Result]:
        if not len(self):
            return []
        sims = self.vectors @ query
        order = np.argsort(-sims)
        results = [Result(self.paths[i], float(sims[i])) for i in order if self.paths[i] != exclude]
        return results[:k]

    def search_text(self, query: str, k: int = 24) -> list[Result]:
        vecs = self.encoder.encode_texts([t.format(query) for t in TEMPLATES])
        q = vecs.mean(axis=0)
        return self._top(q / np.linalg.norm(q), k)

    def search_image(self, image: Path, k: int = 24) -> list[Result]:
        resolved = str(image.resolve())
        if resolved in self.paths:
            q = self.vectors[self.paths.index(resolved)]
        else:
            q = self.encoder.encode_images([image])[0]
        return self._top(q, k, exclude=resolved)

    def duplicates(self, threshold: float = 0.95) -> list[list[str]]:
        """Group near-duplicates (bursts, re-saves, resized copies) with union-find."""
        n = len(self)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for start in range(0, n, 1024):  # blockwise to keep memory flat on big libraries
            sims = self.vectors[start : start + 1024] @ self.vectors.T
            rows, cols = np.nonzero(sims >= threshold)
            for r, c in zip(rows + start, cols, strict=True):
                if r < c:
                    parent[find(r)] = find(c)

        groups: dict[int, list[str]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(self.paths[i])
        return sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)

    def tag(self, labels: list[str] | None = None, top: int = 3, temperature: float = 100.0):
        """Zero-shot tagging: softmax over label prompts, the same way CLIP does classification."""
        labels = labels or DEFAULT_TAGS
        protos = []
        for label in labels:
            v = self.encoder.encode_texts([t.format(label) for t in TEMPLATES]).mean(axis=0)
            protos.append(v / np.linalg.norm(v))
        logits = temperature * self.vectors @ np.stack(protos).T
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        tags = {}
        for i, row in enumerate(probs):
            best = np.argsort(-row)[:top]
            tags[self.paths[i]] = [(labels[j], float(row[j])) for j in best]
        return tags
