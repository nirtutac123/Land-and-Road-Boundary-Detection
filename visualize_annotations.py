#!/usr/bin/env python3
"""
Interactive HTML annotation visualizer for CULane dataset.
Generates overlays of lane annotations on images.
"""

import os
import json
from pathlib import Path
import base64
import numpy as np
import cv2
from collections import defaultdict

DATA_ROOT = Path("data/CULane")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_SPLIT_FILE = DATA_ROOT / "list" / "train_gt.txt"
VAL_SPLIT_FILE = DATA_ROOT / "list" / "val.txt"
TEST_SPLITS_DIR = DATA_ROOT / "list" / "test_split"

IMAGE_HEIGHT, IMAGE_WIDTH = 590, 1640


def load_lane_annotation(ann_path):
    """Parse lane annotation file."""
    lanes = []
    try:
        with open(ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                coords = [float(x) for x in line.split()]
                if len(coords) % 2 == 0 and len(coords) > 0:
                    points = [(int(coords[i]), int(coords[i+1])) for i in range(0, len(coords), 2)]
                    if len(points) > 1:
                        lanes.append(points)
    except:
        pass
    return lanes


def get_annotation_path(image_path):
    """Convert image path to annotation path."""
    # image_path: "driver_23_30frame/05151649_0422.MP4/00000.jpg"
    # Expected:   annotations_new/driver_23_30frame/05151649_0422.MP4/00000.lines.txt
    ann_rel_path = str(image_path).replace('.jpg', '.lines.txt')
    ann_path = DATA_ROOT / "annotations_new" / ann_rel_path
    return ann_path if ann_path.exists() else None


def draw_lanes_on_image(image_path, lanes, output_path):
    """Draw lane annotations on image and save."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Draw lanes with different colors
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    for idx, lane_points in enumerate(lanes):
        color = colors[idx % len(colors)]
        for i in range(len(lane_points) - 1):
            p1 = lane_points[i]
            p2 = lane_points[i + 1]
            cv2.line(img, p1, p2, color, 3)
        # Mark lane points
        for point in lane_points:
            cv2.circle(img, point, 4, color, -1)
    
    cv2.imwrite(str(output_path), img)
    return output_path


def image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_visualization(split_name, split_file, max_samples=20):
    """Generate HTML visualization for a split."""
    print(f"\n📊 Generating {split_name} visualization...")
    
    samples = []
    if split_file.exists():
        with open(split_file) as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                parts = line.strip().split()
                if parts:
                    img_path = parts[0]
                    # Remove leading slash if present
                    if img_path.startswith('/'):
                        img_path = img_path[1:]
                    samples.append(img_path)
    
    viz_dir = OUTPUT_DIR / f"{split_name}_viz"
    viz_dir.mkdir(exist_ok=True)
    
    html = f"""
    <html>
    <head>
        <title>CULane {split_name.upper()} Annotations</title>
        <style>
            body {{ font-family: Arial; margin: 20px; background: #1a1a1a; color: #fff; }}
            h1, h2 {{ color: #4CAF50; }}
            .sample {{ margin: 20px 0; padding: 15px; background: #2a2a2a; border-radius: 8px; border-left: 4px solid #4CAF50; }}
            .image-container {{ max-width: 800px; margin: 10px 0; }}
            img {{ max-width: 100%; height: auto; border-radius: 4px; }}
            .info {{ font-size: 12px; color: #aaa; margin: 5px 0; }}
            .stats {{ background: #333; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>🛣️ CULane Dataset - {split_name.upper()} Set Annotations</h1>
        <div class="stats">
            <strong>Total samples visualized: {len(samples)}</strong><br>
            <strong>Lane colors:</strong> Green (Lane 1), Blue (Lane 2), Red (Lane 3), Yellow (Lane 4)
        </div>
    """
    
    for idx, img_path in enumerate(samples):
        full_path = DATA_ROOT / img_path
        if not full_path.exists():
            continue
        
        ann_path = get_annotation_path(img_path)
        if not ann_path or not ann_path.exists():
            continue
        
        lanes = load_lane_annotation(ann_path)
        if not lanes:
            continue
        
        # Draw annotations
        output_img = viz_dir / f"sample_{idx:03d}.jpg"
        draw_lanes_on_image(full_path, lanes, output_img)
        
        # Embed in HTML
        try:
            img_b64 = image_to_base64(output_img)
            html += f"""
            <div class="sample">
                <h3>Sample {idx + 1}</h3>
                <div class="info">
                    <strong>Image:</strong> {img_path}<br>
                    <strong>Lanes detected:</strong> {len(lanes)}<br>
                    <strong>Total lane points:</strong> {sum(len(lane) for lane in lanes)}
                </div>
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{img_b64}" alt="Sample {idx}">
                </div>
            </div>
            """
        except Exception as e:
            print(f"    ⚠️ Error embedding sample {idx}: {e}")
    
    html += """
    </body>
    </html>
    """
    
    output_file = OUTPUT_DIR / f"{split_name}_annotations.html"
    with open(output_file, 'w') as f:
        f.write(html)
    print(f"  ✓ Saved to {output_file}")


def main():
    print("=" * 80)
    print("🎨 CULANE ANNOTATION VISUALIZER")
    print("=" * 80)
    
    # Visualize each split
    generate_visualization("train", TRAIN_SPLIT_FILE, max_samples=15)
    generate_visualization("val", VAL_SPLIT_FILE, max_samples=10)
    
    # Visualize test splits by category
    if TEST_SPLITS_DIR.exists():
        for cat_file in sorted(TEST_SPLITS_DIR.glob("*.txt"))[:3]:  # First 3 categories
            category = cat_file.stem
            print(f"\n📊 Generating {category} visualization...")
            
            viz_dir = OUTPUT_DIR / f"test_{category}_viz"
            viz_dir.mkdir(exist_ok=True)
            
            samples = []
            with open(cat_file) as f:
                for i, line in enumerate(f):
                    if i >= 8:
                        break
                    line = line.strip()
                    if line:
                        # Remove leading slash if present
                        if line.startswith('/'):
                            line = line[1:]
                        samples.append(line)
            
            html = f"""
            <html>
            <head>
                <title>CULane Test - {category.upper()}</title>
                <style>
                    body {{ font-family: Arial; margin: 20px; background: #1a1a1a; color: #fff; }}
                    h1, h2 {{ color: #FF9800; }}
                    .sample {{ margin: 20px 0; padding: 15px; background: #2a2a2a; border-radius: 8px; border-left: 4px solid #FF9800; }}
                    .image-container {{ max-width: 800px; margin: 10px 0; }}
                    img {{ max-width: 100%; height: auto; border-radius: 4px; }}
                    .info {{ font-size: 12px; color: #aaa; margin: 5px 0; }}
                </style>
            </head>
            <body>
                <h1>🚗 CULane TEST SET - {category.upper()}</h1>
            """
            
            for idx, img_path in enumerate(samples):
                full_path = DATA_ROOT / img_path
                if not full_path.exists():
                    continue
                
                ann_path = get_annotation_path(img_path)
                if ann_path and ann_path.exists():
                    lanes = load_lane_annotation(ann_path)
                else:
                    lanes = []
                
                # Draw annotations if available, otherwise show raw image
                output_img = viz_dir / f"sample_{idx:03d}.jpg"
                if lanes:
                    draw_lanes_on_image(full_path, lanes, output_img)
                else:
                    import shutil
                    shutil.copy(str(full_path), str(output_img))
                
                try:
                    img_b64 = image_to_base64(output_img)
                    lane_info = f"Lanes: {len(lanes)} | Points: {sum(len(lane) for lane in lanes)}" if lanes else "No annotations available (test set)"
                    html += f"""
                    <div class="sample">
                        <h3>Sample {idx + 1} - {category}</h3>
                        <div class="info">
                            <strong>Image:</strong> {img_path}<br>
                            <strong>{lane_info}</strong>
                        </div>
                        <div class="image-container">
                            <img src="data:image/jpeg;base64,{img_b64}" alt="Sample {idx}">
                        </div>
                    </div>
                    """
                except Exception as e:
                    print(f"    ⚠️ Error embedding sample {idx}: {e}")
            
            html += """
            </body>
            </html>
            """
            
            output_file = OUTPUT_DIR / f"test_{category}_annotations.html"
            with open(output_file, 'w') as f:
                f.write(html)
            print(f"  ✓ Saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output files saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
