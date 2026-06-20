from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "CULane"
LIST_ROOT = DATA_ROOT / "list"
TEST_SPLIT_ROOT = LIST_ROOT / "test_split"

IMAGE_SIZE = (1640, 590)


def resolve_data_path(relative_path: str) -> Path:
    relative_path = relative_path.lstrip("/")
    return DATA_ROOT / relative_path
