"""Extract context-padded crops for CNN classification from DENTEX dataset.

Uses strict image-level split manifest (results/split_manifest.csv)
to ensure zero patient/image leakage across train, val, and test splits.
Optimized: Loads each panoramic image once to extract all associated crops.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import cv2
import pandas as pd
from collections import defaultdict

# Canonical mapping
CANONICAL_CLASSES = ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]
DENTEX_TO_CANONICAL = {
    0: "impacted_tooth",
    1: "caries",
    2: "periapical_lesion",
    3: "deep_caries"
}

def extract_dentex_crops(
    json_path: Path = Path("data/archive/training_data/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"),
    xrays_dir: Path = Path("data/archive/training_data/training_data/quadrant-enumeration-disease/xrays"),
    manifest_csv: Path = Path("results/split_manifest.csv"),
    output_dir: Path = Path("artifacts/crops"),
    padding: float = 0.10
) -> dict:
    print(f"Loading annotations from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loading split manifest from: {manifest_csv}")
    df_splits = pd.read_csv(manifest_csv)
    img_split_map = dict(zip(df_splits["image_id"], df_splits["split"]))
    img_filename_map = {img["id"]: img["file_name"] for img in data["images"]}

    output_dir = Path(output_dir).resolve()
    for s in ("train", "val", "test"):
        for c in CANONICAL_CLASSES:
            (output_dir / s / c).mkdir(parents=True, exist_ok=True)

    # Group annotations by image_id
    anns_by_img = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_img[ann.get("image_id")].append(ann)

    rows = []
    skipped = 0
    total_imgs = len(data["images"])
    print(f"Processing {total_imgs} panoramic images and {len(data['annotations'])} annotations...")

    processed_imgs = 0
    for img_id, img_anns in anns_by_img.items():
        split_name = img_split_map.get(img_id)
        fname = img_filename_map.get(img_id)
        if not split_name or not fname:
            skipped += len(img_anns)
            continue

        img_path = xrays_dir / fname
        if not img_path.exists():
            skipped += len(img_anns)
            continue

        # Read image ONCE for all its annotations
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += len(img_anns)
            continue

        ih, iw = img.shape[:2]

        for ann in img_anns:
            c3 = ann.get("category_id_3")
            if c3 not in DENTEX_TO_CANONICAL:
                skipped += 1
                continue

            cname = DENTEX_TO_CANONICAL[c3]
            bbox = ann.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                skipped += 1
                continue

            bx, by, bw, bh = map(float, bbox)
            if bw <= 0 or bh <= 0:
                skipped += 1
                continue

            pad_w = bw * padding
            pad_h = bh * padding

            x1 = max(0, int(bx - pad_w))
            y1 = max(0, int(by - pad_h))
            x2 = min(iw, int(bx + bw + pad_w))
            y2 = min(ih, int(by + bh + pad_h))

            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue

            crop = img[y1:y2, x1:x2]
            ann_id = ann.get("id")
            crop_filename = f"img_{img_id}_ann_{ann_id}_{cname}.jpg"
            crop_path = output_dir / split_name / cname / crop_filename

            cv2.imwrite(str(crop_path), crop)

            rows.append({
                "path": str(crop_path).replace("\\", "/"),
                "label": cname,
                "image_id": img_id,
                "annotation_id": ann_id,
                "split": split_name,
                "bbox_x": int(bx),
                "bbox_y": int(by),
                "bbox_w": int(bw),
                "bbox_h": int(bh),
                "crop_w": int(x2 - x1),
                "crop_h": int(y2 - y1)
            })

        processed_imgs += 1
        if processed_imgs % 100 == 0 or processed_imgs == total_imgs:
            print(f"Processed {processed_imgs}/{total_imgs} images... (Extracted {len(rows)} crops)")

    df_out = pd.DataFrame(rows)
    out_manifest = output_dir / "split_manifest.csv"
    df_out.to_csv(out_manifest, index=False)
    print(f"\nExtracted {len(rows)} crops total (Skipped: {skipped})")
    print(f"Crop manifest saved to: {out_manifest}")

    summary = df_out.groupby(["split", "label"]).size().unstack(fill_value=0)
    print("\n--- Crop Distribution by Split and Class ---")
    print(summary.to_string())

    return {"crop_count": len(rows), "manifest": str(out_manifest), "summary": summary.to_dict()}

def main():
    parser = argparse.ArgumentParser(description="Extract context-padded abnormal tooth crops from DENTEX.")
    parser.add_argument("--output-dir", default="artifacts/crops")
    parser.add_argument("--padding", type=float, default=0.10)
    args = parser.parse_args()

    extract_dentex_crops(output_dir=Path(args.output_dir), padding=args.padding)

if __name__ == "__main__":
    main()
