from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

from .dataset import SplitEntry, image_from_relative, mask_from_relative, resolve_mask_path


@dataclass
class PixelClassifierBundle:
    model: HistGradientBoostingClassifier
    feature_names: tuple[str, ...] = ("r", "g", "b", "x", "y")


def _image_features(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    x_norm = xx.astype(np.float32) / max(width - 1, 1)
    y_norm = yy.astype(np.float32) / max(height - 1, 1)
    return np.dstack([rgb, x_norm[..., None], y_norm[..., None]]).reshape(-1, 5)


def build_pixel_training_set(
    entries: list[SplitEntry],
    max_images: int = 80,
    max_pixels_per_image: int = 6000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for entry in entries[:max_images]:
        if entry.mask_rel is None:
            continue
        image = image_from_relative(entry.image_rel)
        mask = np.asarray(mask_from_relative(entry.mask_rel), dtype=np.uint8)
        y = (mask > 0).astype(np.uint8).reshape(-1)
        x = _image_features(image)
        positive_idx = np.flatnonzero(y == 1)
        negative_idx = np.flatnonzero(y == 0)
        if len(positive_idx) == 0 or len(negative_idx) == 0:
            continue
        positive_take = min(len(positive_idx), max_pixels_per_image // 2)
        negative_take = min(len(negative_idx), max_pixels_per_image - positive_take)
        chosen_idx = np.concatenate(
            [
                rng.choice(positive_idx, size=positive_take, replace=False),
                rng.choice(negative_idx, size=negative_take, replace=False),
            ]
        )
        rng.shuffle(chosen_idx)
        features.append(x[chosen_idx])
        labels.append(y[chosen_idx])
    if not features:
        raise ValueError("No training data found for the pixel classifier.")
    return np.vstack(features), np.concatenate(labels)


def train_pixel_classifier(
    entries: list[SplitEntry],
    max_images: int = 80,
    max_pixels_per_image: int = 6000,
    seed: int = 42,
) -> PixelClassifierBundle:
    x_train, y_train = build_pixel_training_set(entries, max_images=max_images, max_pixels_per_image=max_pixels_per_image, seed=seed)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_depth=6,
        max_iter=80,
        min_samples_leaf=25,
        early_stopping=True,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return PixelClassifierBundle(model=model)


def predict_pixel_mask(bundle: PixelClassifierBundle, image: Image.Image, threshold: float = 0.5) -> np.ndarray:
    features = _image_features(image)
    probabilities = bundle.model.predict_proba(features)[:, 1]
    height, width = image.size[1], image.size[0]
    mask = (probabilities.reshape(height, width) >= threshold).astype(np.uint8) * 255
    return mask


def hough_lane_baseline(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=50)
    mask = np.zeros_like(gray, dtype=np.uint8)
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = line
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
    return mask


def color_lane_baseline(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    white_lane = ((value > 180) & (saturation < 100)).astype(np.uint8) * 255
    yellow_lane = (((hsv[:, :, 0] > 15) & (hsv[:, :, 0] < 40)) & (value > 120)).astype(np.uint8) * 255
    mask = np.maximum(white_lane, yellow_lane)
    mask = cv2.medianBlur(mask, 5)
    return mask


def segmentation_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    pred = (pred_mask > 0).reshape(-1)
    true = (true_mask > 0).reshape(-1)
    return {
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "f1": float(f1_score(true, pred, zero_division=0)),
        "iou": float(jaccard_score(true, pred, zero_division=0)),
    }


def evaluate_model_on_entries(
    entries: list[SplitEntry],
    predictor,
    max_items: int = 12,
    threshold: float = 0.5,
) -> list[dict[str, float | str]]:
    results: list[dict[str, float | str]] = []
    for entry in entries[:max_items]:
        if entry.mask_rel is None:
            continue
        image = image_from_relative(entry.image_rel)
        true_mask = np.asarray(mask_from_relative(entry.mask_rel), dtype=np.uint8)
        if predictor == "hough":
            pred_mask = hough_lane_baseline(image)
        elif predictor == "color":
            pred_mask = color_lane_baseline(image)
        else:
            pred_mask = predict_pixel_mask(predictor, image, threshold=threshold)
        metrics = segmentation_metrics(pred_mask, true_mask)
        metrics["image_rel"] = entry.image_rel
        results.append(metrics)
    return results


def aggregate_metrics(records: list[dict[str, float | str]]) -> dict[str, float]:
    keys = ["precision", "recall", "f1", "iou"]
    if not records:
        return {key: 0.0 for key in keys}
    return {key: float(np.mean([float(record[key]) for record in records])) for key in keys}


def plot_training_curves(bundle: PixelClassifierBundle, output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    model = bundle.model
    train_scores = getattr(model, "train_score_", [])
    validation_scores = getattr(model, "validation_score_", [])
    plt.figure(figsize=(8, 4.5))
    if len(train_scores):
        plt.plot(train_scores, label="train score")
    if len(validation_scores):
        plt.plot(validation_scores, label="validation score")
    plt.title("Pixel classifier progress")
    plt.xlabel("Iteration")
    plt.ylabel("Loss / score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path
