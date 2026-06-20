#!/usr/bin/env python3
"""Quick test of image loading and base64 encoding."""

from pathlib import Path
import cv2
import base64

DATA_ROOT = Path("data/CULane")
train_file = DATA_ROOT / "list" / "train_gt.txt"

print("Testing visualization process...")
with open(train_file) as f:
    for i, line in enumerate(f):
        if i >= 2:
            break
        parts = line.strip().split()
        if parts:
            img_path = parts[0]
            if img_path.startswith('/'):
                img_path = img_path[1:]
            
            full_path = DATA_ROOT / img_path
            print(f'Sample {i}:')
            print(f'  Path: {img_path}')
            print(f'  Exists: {full_path.exists()}')
            
            if full_path.exists():
                img = cv2.imread(str(full_path))
                print(f'  Can read: {img is not None}')
                if img is not None:
                    print(f'  Shape: {img.shape}')
                    _, buffer = cv2.imencode('.jpg', img)
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    print(f'  Base64 length: {len(b64)}')
