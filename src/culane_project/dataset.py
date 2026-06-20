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
    return image_path.with_suffix(".lines.txt")


def image_from_relative(image_rel: str) -> Image.Image:
    return Image.open(resolve_image_path(image_rel)).convert("RGB")


def mask_from_relative(mask_rel: str) -> Image.Image:
    return Image.open(resolve_mask_path(mask_rel)).convert("L")


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
