"""DENTEX Dataset Organizer Utility.

Standardizes raw downloads from Grand Challenge (https://dentex.grand-challenge.org/data/)
into the canonical workspace layout:
  data/DENTEX/images/
  data/DENTEX/annotations/abnormal_teeth.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil
from dentistry.config import load_config
from dentistry.dataset import SUPPORTED_IMAGE_EXTS, parse_coco_annotations, validate_coco

def organize_dentex(source_dir: str | Path, target_dir: str | Path) -> dict:
    source = Path(source_dir).resolve()
    target = Path(target_dir).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")

    target_images = target / "images"
    target_anns = target / "annotations"
    target_images.mkdir(parents=True, exist_ok=True)
    target_anns.mkdir(parents=True, exist_ok=True)

    # 1. Discover annotation files
    candidate_anns = list(source.rglob("*.json"))
    if not candidate_anns:
        raise FileNotFoundError(f"No JSON annotation files found under {source}")

    print(f"Discovered {len(candidate_anns)} JSON file(s). Looking for DENTEX abnormal tooth annotations...")
    chosen_ann = None
    best_match_count = -1
    for ann_path in candidate_anns:
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "annotations" in data and "images" in data:
                ann_count = len(data["annotations"])
                if ann_count > best_match_count:
                    best_match_count = ann_count
                    chosen_ann = ann_path
        except Exception:
            continue

    if not chosen_ann:
        raise ValueError(f"Could not identify a valid COCO annotation JSON in {source}")

    print(f"Selected annotation file: {chosen_ann} ({best_match_count} annotations)")

    # 2. Discover and organize images
    image_files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
    print(f"Found {len(image_files)} image files. Linking/copying to {target_images}...")
    copied_images = 0
    for img in image_files:
        dest = target_images / img.name
        if not dest.exists():
            try:
                shutil.copy2(img, dest)
            except Exception as e:
                print(f"Warning: could not copy {img}: {e}")
        copied_images += 1

    # 3. Copy annotation file
    dest_ann = target_anns / "abnormal_teeth.json"
    shutil.copy2(chosen_ann, dest_ann)
    print(f"Saved canonical annotation to {dest_ann}")

    # 4. Validate
    parsed = parse_coco_annotations(dest_ann, target_images)
    validation = validate_coco(parsed)
    summary = {
        "status": "success",
        "images_copied": copied_images,
        "annotation_path": str(dest_ann),
        "validation": validation
    }
    print("Organization complete!")
    print(f"Total images registered: {validation['image_count']}")
    print(f"Total annotations: {validation['annotation_count']}")
    print(f"Class breakdown: {validation['class_counts']}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Organize raw DENTEX download into canonical folder layout.")
    parser.add_argument("--dentex-dir", required=True, help="Directory containing unzipped DENTEX download")
    parser.add_argument("--target-dir", default=None, help="Target directory (default: from configs/default.yaml)")
    args = parser.parse_args()

    cfg = load_config()
    target = args.target_dir or cfg["paths"]["raw_data_dir"]
    organize_dentex(args.dentex_dir, target)

if __name__ == "__main__":
    main()
