from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .dataset import SplitEntry, annotation_path_from_image, image_from_relative, parse_lane_annotation, resolve_image_path


LANE_COLORS = [
    (255, 64, 64),
    (64, 255, 96),
    (64, 128, 255),
    (255, 192, 64),
    (192, 64, 255),
    (64, 255, 224),
]


def _draw_lanes(draw: ImageDraw.ImageDraw, lanes: list[np.ndarray], width: int = 4) -> None:
    for index, lane in enumerate(lanes):
        color = LANE_COLORS[index % len(LANE_COLORS)]
        points = [(float(x), float(y)) for x, y in lane if x >= 0 and y >= 0]
        if len(points) < 2:
            continue
        draw.line(points, fill=color, width=width)
        for x, y in points:
            radius = 3
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def overlay_annotation(image_rel: str, annotation_path: Path | None = None) -> Image.Image:
    image = image_from_relative(image_rel).convert("RGBA")
    if annotation_path is None:
        annotation_path = annotation_path_from_image(resolve_image_path(image_rel))
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    lanes = parse_lane_annotation(annotation_path) if annotation_path.exists() else []
    _draw_lanes(draw, lanes)
    return Image.alpha_composite(image, overlay).convert("RGB")


def overlay_prediction(image_rel: str, prediction_mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    image = image_from_relative(image_rel).convert("RGBA")
    mask = (prediction_mask > 0).astype(np.uint8) * 255
    colored = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    colored[mask > 0] = [255, 0, 0, int(255 * alpha)]
    mask_image = Image.fromarray(colored, mode="RGBA")
    return Image.alpha_composite(image, mask_image).convert("RGB")


def image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_html_gallery(items: Iterable[dict[str, object]], title: str, output_path: Path) -> Path:
    cards = []
    for item in items:
        image = item["image"]
        if not isinstance(image, Image.Image):
            raise TypeError("Each item must provide a PIL Image under key 'image'.")
        caption = str(item.get("caption", ""))
        description = str(item.get("description", ""))
        image_uri = image_to_data_uri(image)
        cards.append(
            f"""
            <figure class='card'>
              <img src='{image_uri}' alt='{caption}' />
              <figcaption>
                <h3>{caption}</h3>
                <p>{description}</p>
              </figcaption>
            </figure>
            """
        )
    html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; font-family: Inter, system-ui, -apple-system, sans-serif; background: #0b1220; color: #e5eefb; }}
    header {{ padding: 24px 28px; border-bottom: 1px solid rgba(255,255,255,.08); background: linear-gradient(120deg, #0f172a, #111827); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p {{ margin: 0; opacity: .8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; padding: 20px; }}
    .card {{ margin: 0; background: #111827; border: 1px solid rgba(255,255,255,.08); border-radius: 18px; overflow: hidden; box-shadow: 0 18px 40px rgba(0,0,0,.25); }}
    .card img {{ width: 100%; display: block; background: #020617; }}
    figcaption {{ padding: 14px 16px 18px; }}
    h3 {{ margin: 0 0 6px; font-size: 16px; }}
    figcaption p {{ margin: 0; line-height: 1.45; color: #b7c4d9; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>Generated from CULane annotations and model outputs.</p>
  </header>
  <main class='grid'>
    {''.join(cards)}
  </main>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def save_overlay_gallery(sample_entries: list[SplitEntry], output_path: Path, title: str) -> Path:
    items = []
    for entry in sample_entries:
        items.append(
            {
                "image": overlay_annotation(entry.image_rel),
                "caption": Path(entry.image_rel).name,
                "description": f"{entry.image_rel}",
            }
        )
    return build_html_gallery(items, title=title, output_path=output_path)
