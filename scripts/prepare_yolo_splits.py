"""Strict image-level train/val/test split and Ultralytics YOLO format conversion.

Crucial medical ML rule: Splitting is done strictly at the IMAGE level
(by image_id / patient X-ray) so that no teeth or crops from the same image
leak between train, val, and test splits.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil
import random
import yaml
import pandas as pd
from dentistry.config import load_config
from dentistry.dataset import parse_coco_annotations, validate_coco, DEFAULT_CLASSES, _normalise_category_name

def split_images_stratified(images: list[dict], annotations: list[dict], target_classes: list[str],
                            train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15,
                            seed: int = 42) -> dict[str, list[dict]]:
    """Partition images deterministically into train/val/test preserving class representation."""
    random.seed(seed)
    
    # Map image_id -> set of classes present
    img_classes: dict[int, set[str]] = {img["id"]: set() for img in images}
    for ann in annotations:
        if not ann.get("bbox_valid", True):
            continue
        cname = _normalise_category_name(ann.get("category_name", ""))
        if cname in target_classes and ann["image_id"] in img_classes:
            img_classes[ann["image_id"]].add(cname)

    # Group images by primary/rarest class or multi-label signature
    img_lookup = {img["id"]: img for img in images}
    shuffled_ids = list(img_lookup.keys())
    random.shuffle(shuffled_ids)

    # Sort images so images with rare classes are distributed first
    class_rarity = {c: sum(c in img_classes[iid] for iid in shuffled_ids) for c in target_classes}
    def rarity_score(iid):
        present = img_classes[iid]
        if not present:
            return 9999
        return min(class_rarity.get(c, 999) for c in present)

    shuffled_ids.sort(key=rarity_score)

    train_ids, val_ids, test_ids = [], [], []
    total_imgs = len(shuffled_ids)
    target_train = max(1, int(round(total_imgs * train_ratio)))
    target_val = max(1, int(round(total_imgs * val_ratio)))
    
    for iid in shuffled_ids:
        # Distribute into train, val, test while respecting capacities
        if len(train_ids) < target_train:
            train_ids.append(iid)
        elif len(val_ids) < target_val:
            val_ids.append(iid)
        else:
            test_ids.append(iid)

    # Fallback safety if test set ended up empty with very small N
    if not test_ids and len(train_ids) > 2:
        test_ids.append(train_ids.pop())

    return {
        "train": [img_lookup[iid] for iid in train_ids],
        "val": [img_lookup[iid] for iid in val_ids],
        "test": [img_lookup[iid] for iid in test_ids]
    }

def build_yolo_dataset(annotation_json: str | Path, image_root: str | Path,
                       output_dir: str | Path, manifest_out: str | Path,
                       cfg: dict) -> dict:
    ann_path = Path(annotation_json).resolve()
    img_root = Path(image_root).resolve()
    out_dir = Path(output_dir).resolve()
    manifest_path = Path(manifest_out).resolve()
    
    target_classes = cfg.get("classes", DEFAULT_CLASSES)
    class_to_idx = {name: idx for idx, name in enumerate(target_classes)}

    parsed = parse_coco_annotations(ann_path, img_root)
    validation = validate_coco(parsed)
    print(f"Loaded {validation['image_count']} images and {validation['annotation_count']} annotations.")

    splits_cfg = cfg.get("split", {})
    splits = split_images_stratified(
        images=parsed["images"],
        annotations=parsed["annotations"],
        target_classes=target_classes,
        train_ratio=splits_cfg.get("train_ratio", 0.70),
        val_ratio=splits_cfg.get("val_ratio", 0.15),
        test_ratio=splits_cfg.get("test_ratio", 0.15),
        seed=splits_cfg.get("seed", 42)
    )

    # Create YOLO directory structure
    for split_name in ("train", "val", "test"):
        (out_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    anns_by_image: dict[int, list[dict]] = {}
    for ann in parsed["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    stats_per_split = {"train": {c: 0 for c in target_classes},
                       "val": {c: 0 for c in target_classes},
                       "test": {c: 0 for c in target_classes}}

    for split_name, img_list in splits.items():
        for img_info in img_list:
            src_img = img_info.get("path")
            if not src_img or not Path(src_img).exists():
                # Try locating in image_root directly
                cand = img_root / img_info["file_name"]
                if cand.exists():
                    src_img = str(cand)
                else:
                    print(f"Warning: image file not found for {img_info['file_name']}, skipping.")
                    continue

            src_img_path = Path(src_img)
            dst_img_path = out_dir / "images" / split_name / src_img_path.name
            shutil.copy2(src_img_path, dst_img_path)

            # Generate YOLO label file: <class_id> <cx> <cy> <nw> <nh>
            w, h = float(img_info["width"]), float(img_info["height"])
            label_lines = []
            img_anns = anns_by_image.get(img_info["id"], [])
            classes_in_img = set()

            for ann in img_anns:
                if not ann.get("bbox_valid", True) or not ann.get("bbox"):
                    continue
                cname = _normalise_category_name(ann.get("category_name", ""))
                if cname not in class_to_idx:
                    continue

                cid = class_to_idx[cname]
                bx, by, bw, bh = map(float, ann["bbox"])
                x1 = max(0.0, min(w, bx))
                y1 = max(0.0, min(h, by))
                x2 = max(0.0, min(w, bx + bw))
                y2 = max(0.0, min(h, by + bh))
                if x2 <= x1 or y2 <= y1:
                    continue

                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                nw = (x2 - x1) / w
                nh = (y2 - y1) / h

                label_lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                stats_per_split[split_name][cname] += 1
                classes_in_img.add(cname)

            lbl_dst = out_dir / "labels" / split_name / f"{src_img_path.stem}.txt"
            lbl_dst.write_text("\n".join(label_lines), encoding="utf-8")

            manifest_rows.append({
                "image_id": img_info["id"],
                "file_name": src_img_path.name,
                "split": split_name,
                "width": int(w),
                "height": int(h),
                "annotation_count": len(label_lines),
                "classes_present": ";".join(sorted(classes_in_img))
            })

    # Save split manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(manifest_path, index=False)
    print(f"Saved strict image-level split manifest to {manifest_path}")

    # Generate dataset.yaml for YOLO
    yolo_yaml = {
        "path": str(out_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(target_classes)}
    }
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yolo_yaml, f, sort_keys=False)
    print(f"Generated YOLO dataset configuration: {yaml_path}")

    summary = {
        "dataset_yaml": str(yaml_path),
        "manifest_path": str(manifest_path),
        "total_images": len(manifest_rows),
        "split_counts": {s: len(splits[s]) for s in ("train", "val", "test")},
        "annotations_per_split": stats_per_split
    }
    print("\n--- YOLO Split Summary ---")
    print(json.dumps(summary, indent=2))
    return summary

def main():
    parser = argparse.ArgumentParser(description="Prepare strict image-level train/val/test splits in Ultralytics YOLO format.")
    parser.add_argument("--annotations", required=True, help="Path to COCO JSON annotations")
    parser.add_argument("--image-root", required=True, help="Path to raw image folder")
    parser.add_argument("--output-dir", default="data/processed/yolo", help="Output YOLO directory")
    parser.add_argument("--manifest-out", default="artifacts/manifests/image_splits.csv", help="Split manifest CSV path")
    args = parser.parse_args()

    cfg = load_config()
    build_yolo_dataset(args.annotations, args.image_root, args.output_dir, args.manifest_out, cfg)

if __name__ == "__main__":
    main()
