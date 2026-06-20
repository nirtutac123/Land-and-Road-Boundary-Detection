from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import (
    SplitEntry,
    annotation_path_from_image,
    extract_driver_name,
    load_test_category_entries,
    load_list_entries,
    parse_lane_annotation,
    resolve_image_path,
    unique_drivers,
)


@dataclass(frozen=True)
class SplitSummary:
    split_name: str
    total_entries: int
    unique_drivers: int
    unique_frames: int
    annotated_frames: int
    total_lanes: int


def summarize_split(split_name: str) -> SplitSummary:
    entries = load_list_entries(split_name)
    drivers = unique_drivers(entries)
    total_lanes = 0
    annotated_frames = 0
    for entry in entries:
        image_path = resolve_image_path(entry.image_rel)
        annotation_path = annotation_path_from_image(image_path)
        if annotation_path.exists():
            annotated_frames += 1
            total_lanes += len(parse_lane_annotation(annotation_path))
    return SplitSummary(
        split_name=split_name,
        total_entries=len(entries),
        unique_drivers=len(drivers),
        unique_frames=len({entry.image_rel for entry in entries}),
        annotated_frames=annotated_frames,
        total_lanes=total_lanes,
    )


def summarize_test_categories() -> dict[str, int]:
    category_counts: dict[str, int] = {}
    for index, category_name in enumerate(
        ["normal", "crowd", "hlight", "shadow", "noline", "arrow", "curve", "cross", "night"]
    ):
        category_counts[category_name] = len(load_test_category_entries(f"test{index}_{category_name}"))
    return category_counts


def collect_annotation_stats(entries: list[SplitEntry], max_items: int | None = None) -> dict[str, float]:
    annotation_counts = []
    for entry in entries[: max_items or len(entries)]:
        path = annotation_path_from_image(resolve_image_path(entry.image_rel))
        if path.exists():
            annotation_counts.append(len(parse_lane_annotation(path)))
    if not annotation_counts:
        return {"mean_lanes": 0.0, "max_lanes": 0.0, "min_lanes": 0.0}
    values = np.asarray(annotation_counts, dtype=np.float32)
    return {
        "mean_lanes": float(values.mean()),
        "max_lanes": float(values.max()),
        "min_lanes": float(values.min()),
    }


def summarize_dataset() -> dict[str, object]:
    train = summarize_split("train.txt")
    val = summarize_split("val.txt")
    test = summarize_split("test.txt")
    return {
        "train": train,
        "val": val,
        "test": test,
        "test_categories": summarize_test_categories(),
    }
