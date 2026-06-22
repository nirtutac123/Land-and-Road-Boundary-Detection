from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .config import DATA_ROOT, LIST_ROOT, TEST_SPLIT_ROOT, resolve_data_path


@dataclass(frozen=True)
class SplitEntry:
    image_rel: str
    mask_rel: str | None = None
    labels: tuple[int, ...] | None = None


def read_split_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_list_entries(split_name: str) -> list[SplitEntry]:
    path = LIST_ROOT / split_name
    entries: list[SplitEntry] = []
    for line in read_split_file(path):
        parts = line.split()
        image_rel = parts[0]
        mask_rel = parts[1] if len(parts) > 1 and parts[1].startswith("/") else None
        # CULane's plain train/val/test lists contain only image paths.  The
        # segmentation labels live in a parallel directory, so discover them
        # here instead of silently treating those samples as unlabelled.
        if mask_rel is None:
            mask_rel = infer_mask_relative_path(image_rel)
        labels = tuple(int(value) for value in parts[2:]) if len(parts) > 2 else None
        entries.append(SplitEntry(image_rel=image_rel, mask_rel=mask_rel, labels=labels))
    return entries


def load_test_category_entries(file_stem: str) -> list[str]:
    return read_split_file(TEST_SPLIT_ROOT / f"{file_stem}.txt")


def resolve_image_path(image_rel: str) -> Path:
    return resolve_data_path(image_rel)


def resolve_mask_path(mask_rel: str | None) -> Path | None:
    if mask_rel is None:
        return None
    return resolve_data_path(mask_rel)


def infer_mask_relative_path(image_rel: str) -> str | None:
    """Return the available segmentation-mask path for an image, if any."""
    relative = Path(image_rel.lstrip("/")).with_suffix(".png")
    for mask_root in ("laneseg_label_w16", "laneseg_label_w16_test"):
        candidate = DATA_ROOT / mask_root / relative
        if candidate.is_file():
            return "/" + str(Path(mask_root) / relative)
    return None


def parse_lane_annotation(path: Path) -> list[np.ndarray]:
    lanes: list[np.ndarray] = []
    for raw_line in path.read_text().splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        values = [float(value) for value in raw_line.split()]
        points = np.array(values, dtype=np.float32).reshape(-1, 2)
        lanes.append(points)
    return lanes


def annotation_path_from_image(image_path: Path) -> Path:
    """Resolve a CULane line annotation in either supported dataset layout."""
    try:
        relative = image_path.relative_to(DATA_ROOT).with_suffix(".lines.txt")
        curated_annotation = DATA_ROOT / "annotations_new" / relative
        if curated_annotation.is_file():
            return curated_annotation
    except ValueError:
        pass
    return image_path.with_suffix(".lines.txt")


def image_from_relative(image_rel: str) -> Image.Image:
    return Image.open(resolve_image_path(image_rel)).convert("RGB")


def mask_from_relative(mask_rel: str) -> Image.Image:
    path = resolve_mask_path(mask_rel)
    if path is None:
        raise ValueError("A segmentation-mask path is required.")
    return Image.open(path).convert("L")


def available_entries(entries: Iterable[SplitEntry], require_mask: bool = False) -> list[SplitEntry]:
    """Keep entries whose local files are present (datasets are often partial)."""
    available: list[SplitEntry] = []
    for entry in entries:
        if not resolve_image_path(entry.image_rel).is_file():
            continue
        if require_mask:
            mask_path = resolve_mask_path(entry.mask_rel)
            if mask_path is None or not mask_path.is_file():
                continue
        available.append(entry)
    return available


def extract_driver_name(image_rel: str) -> str:
    parts = Path(image_rel).parts
    for part in parts:
        if part.startswith("driver_"):
            return part
    return "unknown_driver"


def extract_test_category(image_rel: str) -> str:
    for category in ["normal", "crowd", "hlight", "shadow", "noline", "arrow", "curve", "cross", "night"]:
        if category in image_rel:
            return category
    return "unknown"


def unique_drivers(entries: Iterable[SplitEntry]) -> list[str]:
    return sorted({extract_driver_name(entry.image_rel) for entry in entries})
