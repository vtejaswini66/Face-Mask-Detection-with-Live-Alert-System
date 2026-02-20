"""
generate_dataset.py
-------------------
Generates a synthetic face-mask dataset for demonstration / offline testing.
Replace this step with a real Kaggle dataset (e.g. "Face Mask Detection" by
Prajna Bhandary) for production use.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

IMG_SIZE = 128
N_SAMPLES = 500          # samples per class
OUT_ROOT  = "dataset"

CLASSES = {
    "with_mask":    OUT_ROOT + "/with_mask",
    "without_mask": OUT_ROOT + "/without_mask",
}

SKIN_TONES = [
    (255, 224, 189), (240, 195, 145), (198, 134, 66),
    (141, 85,  36),  (90,  60,  20),
]

def _make_face(draw, cx, cy, r, skin):
    """Draw a simple oval face."""
    draw.ellipse([cx - r, cy - int(r * 1.2),
                  cx + r, cy + int(r * 1.2)], fill=skin)

def _make_eyes(draw, cx, cy, r):
    ew = max(int(r * 0.18), 4)
    for ex in [cx - int(r * 0.3), cx + int(r * 0.3)]:
        ey = cy - int(r * 0.1)
        draw.ellipse([ex - ew, ey - ew, ex + ew, ey + ew], fill=(50, 30, 20))

def _make_mask(draw, cx, cy, r, color):
    """Draw a surgical-mask rectangle over the lower face."""
    mx1 = cx - int(r * 0.85)
    mx2 = cx + int(r * 0.85)
    my1 = cy + int(r * 0.1)
    my2 = cy + int(r * 1.0)
    draw.rectangle([mx1, my1, mx2, my2], fill=color, outline=(180,180,200), width=2)
    # pleats
    for py in range(my1 + 6, my2 - 4, 8):
        draw.line([mx1 + 4, py, mx2 - 4, py], fill=(200, 200, 210), width=1)

def _make_mouth(draw, cx, cy, r):
    draw.arc([cx - int(r * 0.3), cy + int(r * 0.4),
              cx + int(r * 0.3), cy + int(r * 0.7)],
             start=0, end=180, fill=(180, 80, 80), width=3)

def generate_sample(with_mask: bool) -> Image.Image:
    w = h = IMG_SIZE
    bg_color = tuple(random.randint(180, 255) for _ in range(3))
    img = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)

    skin   = random.choice(SKIN_TONES)
    cx, cy = w // 2 + random.randint(-8, 8), h // 2 + random.randint(-8, 8)
    r      = random.randint(30, 40)

    _make_face(draw, cx, cy, r, skin)
    _make_eyes(draw, cx, cy, r)

    if with_mask:
        mask_colors = [(0, 120, 200), (200, 230, 255), (255, 255, 255),
                       (0, 160, 100), (200, 200, 200)]
        _make_mask(draw, cx, cy, r, random.choice(mask_colors))
    else:
        _make_mouth(draw, cx, cy, r)

    # subtle noise
    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, 8, arr.shape)
    arr  = np.clip(arr, 0, 255).astype(np.uint8)
    img  = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def main():
    for folder in CLASSES.values():
        os.makedirs(folder, exist_ok=True)

    for label, folder in CLASSES.items():
        is_mask = label == "with_mask"
        print(f"Generating {N_SAMPLES} '{label}' images …")
        for i in range(N_SAMPLES):
            img = generate_sample(is_mask)
            img.save(os.path.join(folder, f"{label}_{i:04d}.png"))

    print("✅  Dataset generated successfully in ./dataset/")


if __name__ == "__main__":
    main()
