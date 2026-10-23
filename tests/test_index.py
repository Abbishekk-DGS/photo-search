"""Index logic tested with a fake encoder: images are solid colours, 'text' names a colour."""

import time

import numpy as np
from PIL import Image

from photo_search.index import PhotoIndex

COLOURS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}


class ColourEncoder:
    dim = 3

    def encode_images(self, paths):
        out = []
        for p in paths:
            with Image.open(p) as img:
                if img.size == (1, 1):
                    raise OSError("corrupt")
                out.append(np.asarray(img.convert("RGB"), np.float32).mean(axis=(0, 1)) + 1)
        v = np.array(out)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def encode_texts(self, texts):
        out = []
        for t in texts:
            name = next((c for c in COLOURS if c in t), "red")
            out.append(np.array(COLOURS[name], np.float32) + 1)
        v = np.array(out)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def make(path, colour, size=(8, 8)):
    Image.new("RGB", size, COLOURS[colour]).save(path)


def build(tmp_path):
    photos = tmp_path / "photos"
    (photos / "trip").mkdir(parents=True)
    make(photos / "a.png", "red")
    make(photos / "trip" / "b.png", "green")
    make(photos / "trip" / "c.png", "blue")
    make(photos / "a_copy.png", "red")
    return photos, PhotoIndex(tmp_path / "idx", ColourEncoder())


def test_text_search_ranks_by_meaning(tmp_path):
    photos, index = build(tmp_path)
    assert index.update(photos)["added"] == 4
    assert index.search_text("blue", k=1)[0].path.endswith("c.png")


def test_incremental_update(tmp_path):
    photos, index = build(tmp_path)
    index.update(photos)
    assert index.update(photos) == {"added": 0, "removed": 0, "failed": 0, "total": 4}

    (photos / "trip" / "b.png").unlink()
    time.sleep(1.1)  # fingerprint uses whole-second mtime
    make(photos / "a.png", "blue", size=(9, 9))
    stats = index.update(photos)
    assert stats == {"added": 1, "removed": 2, "failed": 0, "total": 3}

    reloaded = PhotoIndex(tmp_path / "idx", ColourEncoder())
    assert len(reloaded) == 3


def test_corrupt_file_is_isolated(tmp_path):
    photos, index = build(tmp_path)
    make(photos / "broken.png", "green", size=(1, 1))
    stats = index.update(photos)
    assert stats["added"] == 4 and stats["failed"] == 1


def test_similar_excludes_query_and_dupes_group(tmp_path):
    photos, index = build(tmp_path)
    index.update(photos)
    similar = index.search_image(photos / "a.png", k=3)
    assert all(not r.path.endswith("/a.png") for r in similar)
    assert similar[0].path.endswith("a_copy.png")
    groups = index.duplicates(threshold=0.99)
    assert len(groups) == 1 and len(groups[0]) == 2


def test_zero_shot_tags(tmp_path):
    photos, index = build(tmp_path)
    index.update(photos)
    tags = index.tag(["red", "green", "blue"], top=1)
    best = {p.rsplit("/", 1)[-1]: t[0][0] for p, t in tags.items()}
    assert best == {"a.png": "red", "a_copy.png": "red", "b.png": "green", "c.png": "blue"}
