"""Generate synthetic sample DENTEX-like dataset for testing and immediate reproducibility.

Creates panoramic-style grayscale images and a COCO-format annotation JSON
covering all four required diagnostic classes:
  - caries
  - deep_caries
  - periapical_lesion
  - impacted_tooth
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import random
import numpy as np
from PIL import Image, ImageDraw

CLASSES = [
    {"id": 1, "name": "caries"},
    {"id": 2, "name": "deep_caries"},
    {"id": 3, "name": "periapical_lesion"},
    {"id": 4, "name": "impacted_tooth"}
]

def generate_panoramic_sample(output_dir: str | Path, num_images: int = 12, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    out = Path(output_dir).resolve()
    images_dir = out / "images"
    anns_dir = out / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    anns_dir.mkdir(parents=True, exist_ok=True)

    images_list = []
    annotations_list = []
    ann_id = 1

    # Typical panoramic dimensions: ~1600x800 or ~2000x1000 (aspect ratio ~2:1)
    resolutions = [
        (1600, 800),
        (1800, 900),
        (2048, 1024),
        (1500, 750)
    ]

    for img_idx in range(1, num_images + 1):
        w, h = random.choice(resolutions)
        file_name = f"sample_xray_{img_idx:03d}.png"
        img_path = images_dir / file_name

        # Create panoramic-like synthetic image
        # Base background with horizontal and subtle curved gradients
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        arch = np.exp(-((yy + 0.3 * (xx**2) - 0.1)**2) / 0.15)
        base = (arch * 130 + np.random.normal(40, 10, (h, w))).clip(0, 255).astype(np.uint8)

        img = Image.fromarray(base)
        draw = ImageDraw.Draw(img)

        # Draw a dental arch curve with tooth-like blocks
        num_teeth = 14
        arch_x = np.linspace(w * 0.15, w * 0.85, num_teeth)
        tooth_boxes = []

        for tx in arch_x:
            # parabolic arch y position
            norm_x = (tx - w * 0.5) / (w * 0.35)
            ty = h * 0.5 + norm_x**2 * (h * 0.12)
            tw = random.randint(int(w * 0.03), int(w * 0.05))
            th = random.randint(int(h * 0.10), int(h * 0.18))
            x1 = int(tx - tw / 2)
            y1 = int(ty - th / 2)
            x2 = x1 + tw
            y2 = y1 + th
            # draw tooth structure
            draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=random.randint(180, 230), outline=250)
            tooth_boxes.append((x1, y1, tw, th))

        # Add 1 to 4 abnormalities per image
        num_abnormalities = random.randint(1, 4)
        chosen_teeth = random.sample(tooth_boxes, min(num_abnormalities, len(tooth_boxes)))

        for (tx1, ty1, tw, th) in chosen_teeth:
            category = random.choice(CLASSES)
            # Create a localized lesion bounding box inside/around the tooth
            if category["name"] in ("caries", "deep_caries"):
                bw = int(tw * random.uniform(0.35, 0.65))
                bh = int(th * random.uniform(0.25, 0.45))
                bx = int(tx1 + random.uniform(0, tw - bw))
                by = int(ty1 + random.uniform(0, th * 0.5))
            elif category["name"] == "periapical_lesion":
                # At the root tip (apex)
                bw = int(tw * random.uniform(0.40, 0.80))
                bh = int(th * random.uniform(0.30, 0.60))
                bx = int(tx1 + random.uniform(0, max(1, tw - bw)))
                by = int(ty1 + th - bh * 0.5)
            else: # impacted tooth
                # Angled / full tooth box
                bw = int(tw * random.uniform(0.9, 1.2))
                bh = int(th * random.uniform(0.8, 1.1))
                bx = tx1
                by = ty1

            # Clamp coordinates
            bx = max(0, min(w - 10, bx))
            by = max(0, min(h - 10, by))
            bw = max(10, min(w - bx, bw))
            bh = max(10, min(h - by, bh))

            # Render visual defect onto image
            defect_fill = 60 if "caries" in category["name"] else (40 if category["name"] == "periapical_lesion" else 210)
            draw.ellipse([bx, by, bx + bw, by + bh], fill=defect_fill, outline=20)

            annotations_list.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": category["id"],
                "category_name": category["name"],
                "bbox": [bx, by, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0
            })
            ann_id += 1

        img.save(img_path)
        images_list.append({
            "id": img_idx,
            "file_name": file_name,
            "width": w,
            "height": h
        })

    coco_dict = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": CLASSES
    }

    json_path = anns_dir / "sample_abnormal_teeth.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2)

    print(f"Generated {num_images} synthetic panoramic X-rays in: {images_dir}")
    print(f"Generated {len(annotations_list)} annotations across 4 classes in: {json_path}")
    return {"images_dir": str(images_dir), "annotations_path": str(json_path), "num_images": num_images, "num_annotations": len(annotations_list)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic sample DENTEX data for testing.")
    parser.add_argument("--output-dir", default="data/DENTEX_sample")
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_panoramic_sample(args.output_dir, args.num_images, args.seed)
