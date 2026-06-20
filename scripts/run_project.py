from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from culane_project.analysis import summarize_dataset
from culane_project.dataset import load_list_entries
from culane_project.modeling import (
    aggregate_metrics,
    evaluate_model_on_entries,
    plot_training_curves,
    train_pixel_classifier,
)
from culane_project.visualize import overlay_annotation, overlay_prediction, save_overlay_gallery, build_html_gallery


OUTPUT_ROOT = PROJECT_ROOT / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)


def sample_entries(split_name: str, count: int) -> list:
    entries = load_list_entries(split_name)
    if count >= len(entries):
        return entries
    step = max(len(entries) // count, 1)
    sampled = entries[::step][:count]
    return sampled


def save_dataset_summary() -> dict:
    summary = summarize_dataset()
    serializable = {
        key: value.__dict__ if hasattr(value, "__dict__") else value
        for key, value in summary.items()
    }
    (OUTPUT_ROOT / "dataset_summary.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    rows = [
        {"split": key, **value.__dict__}
        for key, value in summary.items()
        if key in {"train", "val", "test"}
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "dataset_summary.csv", index=False)
    return summary


def save_visual_reports() -> None:
    for split_name, count in [("train.txt", 3), ("val.txt", 3), ("test.txt", 3)]:
        samples = sample_entries(split_name, count)
        gallery_path = OUTPUT_ROOT / f"{split_name.replace('.txt', '')}_annotations.html"
        save_overlay_gallery(samples, gallery_path, title=f"CULane {split_name} annotation overlays")


def train_and_evaluate() -> dict:
    train_entries = load_list_entries("train.txt")
    val_entries = load_list_entries("val.txt")
    test_entries = load_list_entries("test.txt")

    bundle = train_pixel_classifier(train_entries, max_images=24, max_pixels_per_image=3500, seed=42)
    plot_training_curves(bundle, OUTPUT_ROOT / "pixel_classifier_training_curve.png")

    metrics = {}
    for name, predictor in [
        ("hough", "hough"),
        ("color", "color"),
        ("pixel_classifier", bundle),
    ]:
        val_records = evaluate_model_on_entries(val_entries, predictor, max_items=10)
        test_records = evaluate_model_on_entries(test_entries, predictor, max_items=10)
        metrics[name] = {
            "val": aggregate_metrics(val_records),
            "test": aggregate_metrics(test_records),
        }
        pd.DataFrame(val_records).to_csv(OUTPUT_ROOT / f"{name}_val_details.csv", index=False)
        pd.DataFrame(test_records).to_csv(OUTPUT_ROOT / f"{name}_test_details.csv", index=False)

    (OUTPUT_ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    comparison_rows = []
    for model_name, split_map in metrics.items():
        for split_name, split_metrics in split_map.items():
            comparison_rows.append({"model": model_name, "split": split_name, **split_metrics})
    pd.DataFrame(comparison_rows).to_csv(OUTPUT_ROOT / "metrics_summary.csv", index=False)

    return {"bundle": bundle, "metrics": metrics}


def save_prediction_gallery(bundle) -> None:
    val_entries = load_list_entries("val.txt")[:3]
    items = []
    from culane_project.dataset import image_from_relative, mask_from_relative
    from culane_project.modeling import hough_lane_baseline, color_lane_baseline, predict_pixel_mask

    for entry in val_entries:
        image = image_from_relative(entry.image_rel)
        gt_mask = mask_from_relative(entry.mask_rel)
        hough_mask = hough_lane_baseline(image)
        color_mask = color_lane_baseline(image)
        learned_mask = predict_pixel_mask(bundle, image)
        items.extend(
            [
                {"image": overlay_annotation(entry.image_rel), "caption": f"GT {Path(entry.image_rel).name}", "description": "Ground truth overlay"},
                {"image": overlay_prediction(entry.image_rel, hough_mask), "caption": f"Hough {Path(entry.image_rel).name}", "description": "Hough baseline prediction"},
                {"image": overlay_prediction(entry.image_rel, color_mask), "caption": f"Color {Path(entry.image_rel).name}", "description": "Color-threshold baseline prediction"},
                {"image": overlay_prediction(entry.image_rel, learned_mask), "caption": f"Model {Path(entry.image_rel).name}", "description": "Trained pixel-classifier prediction"},
            ]
        )
    build_html_gallery(items, title="CULane prediction comparison gallery", output_path=OUTPUT_ROOT / "prediction_comparison.html")


def main() -> None:
    save_dataset_summary()
    save_visual_reports()
    trained = train_and_evaluate()
    save_prediction_gallery(trained["bundle"])
    print("Project artifacts written to", OUTPUT_ROOT)
    print(json.dumps(trained["metrics"], indent=2))


if __name__ == "__main__":
    main()
