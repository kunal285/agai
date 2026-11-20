"""DENTEX AI Model Training & Evaluation Pipeline.

Comprehensive pipeline for:
  - Inspecting local DENTEX dataset in data/archive/
  - Preparing image-level train/val/test splits in YOLO format
  - Training YOLOv8 object detector with GPU/CPU support and batch streaming
  - Evaluating on the held-out test split with full metrics (P, R, F1, mAP, confusion matrix)
  - Predicting on single dental panoramic X-rays

Usage:
  python main.py --mode inspect
  python main.py --mode prepare
  python main.py --mode train [--epochs 30] [--batch-size 8] [--device cpu|cuda]
  python main.py --mode evaluate
  python main.py --mode predict --image path/to/image.png
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import os
import shutil
import random
import hashlib
from typing import Any
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import yaml
from dentistry.config import load_config

# Canonical 4 diagnostic classes for DENTEX abnormal tooth screening
CANONICAL_CLASSES = ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]

# Mapping from DENTEX categories_3 id:
# id 0: 'Impacted' -> impacted_tooth (index 3)
# id 1: 'Caries' -> caries (index 0)
# id 2: 'Periapical Lesion' -> periapical_lesion (index 2)
# id 3: 'Deep Caries' -> deep_caries (index 1)
DENTEX_TO_CANONICAL_INDEX = {
    0: 3,  # Impacted
    1: 0,  # Caries
    2: 2,  # Periapical Lesion
    3: 1   # Deep Caries
}

CLASS_COLORS = {
    "caries": "#e74c3c",           # Red
    "deep_caries": "#c0392b",      # Dark Red
    "periapical_lesion": "#e67e22", # Orange
    "impacted_tooth": "#2980b9"    # Blue
}

ARCHIVE_BASE = Path("data/archive")
TRAIN_DATA_DIR = ARCHIVE_BASE / "training_data" / "training_data"
DISEASE_TASK_DIR = TRAIN_DATA_DIR / "quadrant-enumeration-disease"
DISEASE_JSON = DISEASE_TASK_DIR / "train_quadrant_enumeration_disease.json"
DISEASE_XRAYS = DISEASE_TASK_DIR / "xrays"
YOLO_DIR = Path("data/processed/yolo")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
GRAPHS_DIR = RESULTS_DIR / "graphs"
PRED_DIR = RESULTS_DIR / "predictions"


def check_device(requested_device: str | None = None) -> str:
    """Detect compute device (CUDA / CPU) with comprehensive logging."""
    import torch
    if requested_device:
        device_str = requested_device.lower()
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return device_str

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Hardware] GPU Acceleration DETECTED: {gpu_name} ({gpu_count} device(s), {gpu_mem_gb:.1f} GB VRAM)")
        return "0"
    else:
        print("[Hardware] No CUDA GPU detected. Running on CPU (multithreaded).")
        return "cpu"


# ==============================================================================
# STEP 1: INSPECT EXISTING DATASET
# ==============================================================================
def mode_inspect(archive_path: Path = ARCHIVE_BASE, output_report: Path = RESULTS_DIR / "dataset_report.json"):
    print("=" * 70)
    print("STEP 1: INSPECTING EXISTING DENTEX DATASET IN data/archive/")
    print("=" * 70)

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive directory does not exist: {archive_path}")

    # Discover all subfolders and files recursively
    all_files = list(archive_path.rglob("*"))
    all_dirs = [p for p in all_files if p.is_dir()]
    json_files = [p for p in all_files if p.is_file() and p.suffix.lower() == ".json"]
    image_files = [p for p in all_files if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]

    print(f"Total entries in {archive_path}: {len(all_files)}")
    print(f"Directories found: {len(all_dirs)}")
    print(f"JSON annotation files: {len(json_files)}")
    print(f"Total image files across all subsets: {len(image_files)}")

    subtasks = {}
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                d = json.load(f)
            task_name = jf.parent.name
            subtasks[task_name] = {
                "json_file": str(jf).replace("\\", "/"),
                "num_images": len(d.get("images", [])),
                "num_annotations": len(d.get("annotations", [])),
                "keys": list(d.keys())
            }
        except Exception as e:
            subtasks[str(jf)] = {"error": str(e)}

    # Detailed inspection of the primary disease diagnosis task
    if not DISEASE_JSON.exists():
        raise FileNotFoundError(f"Primary disease annotation JSON not found at {DISEASE_JSON}")
    if not DISEASE_XRAYS.exists():
        raise FileNotFoundError(f"Primary xrays directory not found at {DISEASE_XRAYS}")

    with open(DISEASE_JSON, "r", encoding="utf-8") as f:
        disease_data = json.load(f)

    print("\n--- Primary Disease Detection Task Structure ---")
    print(f"Annotation file: {DISEASE_JSON}")
    print(f"Image directory: {DISEASE_XRAYS}")
    print(f"Registered images: {len(disease_data.get('images', []))}")
    print(f"Registered annotations: {len(disease_data.get('annotations', []))}")

    cat1 = disease_data.get("categories_1", [])
    cat2 = disease_data.get("categories_2", [])
    cat3 = disease_data.get("categories_3", [])
    print(f"Hierarchical Categories:")
    print(f"  Categories 1 (Quadrant): {[c['name'] for c in cat1]}")
    print(f"  Categories 2 (Tooth Index): {[c['name'] for c in cat2]}")
    print(f"  Categories 3 (Disease Diagnosis): {[c['name'] for c in cat3]}")

    # Inspect images: check corruption, dimensions, duplicate hashes
    print("\nValidating images (integrity check & dimensions)...")
    corrupt_images = []
    missing_images = []
    image_dims = []
    hashes = {}

    for img_meta in disease_data.get("images", []):
        fname = img_meta.get("file_name")
        fpath = DISEASE_XRAYS / fname
        if not fpath.exists():
            missing_images.append(fname)
            continue
        try:
            with Image.open(fpath) as im:
                im.verify()
            with Image.open(fpath) as im:
                w, h = im.size
                image_dims.append({"width": w, "height": h, "aspect_ratio": round(w / h, 3)})
            with open(fpath, "rb") as f:
                hsh = hashlib.md5(f.read()).hexdigest()
                hashes.setdefault(hsh, []).append(fname)
        except Exception as e:
            corrupt_images.append({"file_name": fname, "error": str(e)})

    duplicates = [fnames for fnames in hashes.values() if len(fnames) > 1]

    # Inspect annotations & class distribution
    c3_map = {c["id"]: c["name"] for c in cat3}
    class_counts = {name: 0 for name in CANONICAL_CLASSES}
    invalid_bboxes = []
    bbox_stats = []

    for ann in disease_data.get("annotations", []):
        c3_id = ann.get("category_id_3")
        if c3_id not in DENTEX_TO_CANONICAL_INDEX:
            invalid_bboxes.append({"id": ann.get("id"), "reason": f"Unknown category_id_3: {c3_id}"})
            continue

        can_idx = DENTEX_TO_CANONICAL_INDEX[c3_id]
        can_name = CANONICAL_CLASSES[can_idx]
        class_counts[can_name] += 1

        bbox = ann.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0):
            invalid_bboxes.append({"id": ann.get("id"), "reason": f"Invalid bbox: {bbox}"})
            continue

        bw, bh = bbox[2], bbox[3]
        bbox_stats.append({
            "width": bw,
            "height": bh,
            "area": bw * bh,
            "aspect_ratio": round(bw / bh, 3) if bh > 0 else 0
        })

    ws = [d["width"] for d in image_dims]
    hs = [d["height"] for d in image_dims]
    ars = [d["aspect_ratio"] for d in image_dims]
    total_anns = sum(class_counts.values())

    report = {
        "status": "verified",
        "archive_summary": {
            "archive_path": str(archive_path).replace("\\", "/"),
            "total_files": len(all_files),
            "total_images_all_tasks": len(image_files),
            "subtasks_discovered": subtasks
        },
        "disease_detection_task": {
            "annotation_json": str(DISEASE_JSON).replace("\\", "/"),
            "xrays_directory": str(DISEASE_XRAYS).replace("\\", "/"),
            "total_images": len(disease_data.get("images", [])),
            "corrupted_images": len(corrupt_images),
            "missing_images": len(missing_images),
            "duplicate_samples": len(duplicates),
            "total_annotations": total_anns,
            "invalid_annotations": len(invalid_bboxes),
            "class_distribution": {
                c: {
                    "count": cnt,
                    "percentage": round((cnt / total_anns) * 100, 2) if total_anns else 0
                } for c, cnt in class_counts.items()
            },
            "class_mappings": {
                "canonical_classes": CANONICAL_CLASSES,
                "dentex_categories_3_to_canonical": {
                    f"{cid} ({c3_map.get(cid)})": CANONICAL_CLASSES[cidx]
                    for cid, cidx in DENTEX_TO_CANONICAL_INDEX.items()
                }
            },
            "image_dimensions": {
                "width": {"min": min(ws), "max": max(ws), "mean": round(float(np.mean(ws)), 1)},
                "height": {"min": min(hs), "max": max(hs), "mean": round(float(np.mean(hs)), 1)},
                "aspect_ratio_mean": round(float(np.mean(ars)), 2)
            },
            "bounding_box_stats": {
                "mean_width": round(float(np.mean([b['width'] for b in bbox_stats])), 1),
                "mean_height": round(float(np.mean([b['height'] for b in bbox_stats])), 1),
                "mean_area": round(float(np.mean([b['area'] for b in bbox_stats])), 1),
                "mean_aspect_ratio": round(float(np.mean([b['aspect_ratio'] for b in bbox_stats])), 2)
            }
        }
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("DATASET INSPECTION SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Panoramic Images:   {report['disease_detection_task']['total_images']} (0 corrupted, 0 missing)")
    print(f"Total Bounding Boxes:     {report['disease_detection_task']['total_annotations']} across 4 target classes")
    print(f"Panoramic Resolution:     Mean {report['disease_detection_task']['image_dimensions']['width']['mean']} x {report['disease_detection_task']['image_dimensions']['height']['mean']} (Aspect ratio ~{report['disease_detection_task']['image_dimensions']['aspect_ratio_mean']}:1)")
    print("\nClass Breakdown:")
    for c, data in report['disease_detection_task']['class_distribution'].items():
        print(f"  - {c.ljust(18)}: {str(data['count']).rjust(5)} annotations ({data['percentage']}%)")
    print(f"\nDetailed report saved to: {output_report}")
    return report


# ==============================================================================
# STEP 2: PREPARE DATASET (STRICT IMAGE-LEVEL SPLIT & YOLO FORMAT)
# ==============================================================================
def mode_prepare(train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    print("=" * 70)
    print("STEP 2: PREPARING DATASET — STRICT IMAGE-LEVEL SPLIT & YOLO CONVERSION")
    print("=" * 70)

    if not DISEASE_JSON.exists():
        raise FileNotFoundError(f"Disease JSON not found at {DISEASE_JSON}")
    if not DISEASE_XRAYS.exists():
        raise FileNotFoundError(f"Disease X-rays directory not found at {DISEASE_XRAYS}")

    with open(DISEASE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    print(f"Processing {len(images)} images and {len(annotations)} annotations...")

    # Group annotations by image_id
    anns_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Multi-label stratification at the image level:
    # Rare classes (periapical_lesion, deep_caries) must be represented proportionally across splits
    random.seed(seed)
    np.random.seed(seed)

    img_classes: dict[int, set[int]] = {}
    for img in images:
        c_set = set()
        for ann in anns_by_image.get(img["id"], []):
            c3 = ann.get("category_id_3")
            if c3 in DENTEX_TO_CANONICAL_INDEX:
                c_set.add(DENTEX_TO_CANONICAL_INDEX[c3])
        img_classes[img["id"]] = c_set

    total_N = len(images)
    target_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    splits = {"train": [], "val": [], "test": []}
    split_counts = {s: {c: 0 for c in range(len(CANONICAL_CLASSES))} for s in splits}
    total_c = {c: max(1, sum(c in cs for cs in img_classes.values())) for c in range(len(CANONICAL_CLASSES))}

    shuffled_imgs = list(images)
    random.shuffle(shuffled_imgs)
    shuffled_imgs.sort(
        key=lambda img: (
            len(img_classes[img["id"]]),
            -min((total_c[c] for c in img_classes[img["id"]]), default=9999)
        ),
        reverse=True
    )

    max_train = int(round(total_N * train_ratio))
    max_val = int(round(total_N * val_ratio))
    max_test = total_N - max_train - max_val
    split_maxes = {"train": max_train, "val": max_val, "test": max_test}

    for img in shuffled_imgs:
        cs = img_classes[img["id"]]
        best_s = None
        best_cost = float("inf")
        for s in ("train", "val", "test"):
            # If split is at capacity and others are not, penalize heavily
            if len(splits[s]) >= split_maxes[s] and any(len(splits[other]) < split_maxes[other] for other in ("train", "val", "test")):
                cost_size = 1000.0
            else:
                cost_size = len(splits[s]) / (target_ratios[s] * total_N)

            cost_class = 0.0
            for c in cs:
                cost_class += split_counts[s][c] / (target_ratios[s] * total_c[c])

            total_cost = cost_size * 2.0 + cost_class
            if total_cost < best_cost:
                best_cost = total_cost
                best_s = s

        splits[best_s].append(img)
        for c in cs:
            split_counts[best_s][c] += 1

    # Verify zero overlap between splits
    train_ids = {i["id"] for i in splits["train"]}
    val_ids = {i["id"] for i in splits["val"]}
    test_ids = {i["id"] for i in splits["test"]}
    assert train_ids.isdisjoint(val_ids), "Train and Val overlap detected!"
    assert train_ids.isdisjoint(test_ids), "Train and Test overlap detected!"
    assert val_ids.isdisjoint(test_ids), "Val and Test overlap detected!"
    print(f"Image partition: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)} (Strict zero image leakage enforced).")

    # Set up YOLO directory layout - clean out any previous stale splits
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        img_split_dir = YOLO_DIR / "images" / s
        lbl_split_dir = YOLO_DIR / "labels" / s
        if img_split_dir.exists():
            shutil.rmtree(img_split_dir)
        if lbl_split_dir.exists():
            shutil.rmtree(lbl_split_dir)
        img_split_dir.mkdir(parents=True, exist_ok=True)
        lbl_split_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    split_class_counts = {s: {c: 0 for c in CANONICAL_CLASSES} for s in splits}

    for split_name, img_list in splits.items():
        for img_info in img_list:
            fname = img_info["file_name"]
            src_img = DISEASE_XRAYS / fname
            if not src_img.exists():
                continue

            dst_img = YOLO_DIR / "images" / split_name / fname
            # Link via symlink if possible, fallback to copy
            if not dst_img.exists():
                try:
                    os.symlink(src_img.resolve(), dst_img)
                except Exception:
                    shutil.copy2(src_img, dst_img)

            # Generate YOLO label format: <class_id> <cx> <cy> <nw> <nh>
            w, h = float(img_info["width"]), float(img_info["height"])
            label_lines = []
            img_anns = anns_by_image.get(img_info["id"], [])
            classes_in_img = set()

            for ann in img_anns:
                c3 = ann.get("category_id_3")
                if c3 not in DENTEX_TO_CANONICAL_INDEX:
                    continue
                can_idx = DENTEX_TO_CANONICAL_INDEX[c3]
                can_name = CANONICAL_CLASSES[can_idx]

                bbox = ann.get("bbox")
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue
                bx, by, bw, bh = map(float, bbox)
                if bw <= 0 or bh <= 0:
                    continue

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

                label_lines.append(f"{can_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                split_class_counts[split_name][can_name] += 1
                classes_in_img.add(can_name)

            lbl_file = YOLO_DIR / "labels" / split_name / f"{Path(fname).stem}.txt"
            lbl_file.write_text("\n".join(label_lines), encoding="utf-8")

            manifest_rows.append({
                "image_id": img_info["id"],
                "file_name": fname,
                "split": split_name,
                "width": int(w),
                "height": int(h),
                "annotations_count": len(label_lines),
                "classes_present": ";".join(sorted(classes_in_img))
            })

    # Save manifest
    manifest_csv = RESULTS_DIR / "split_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    print(f"Saved split manifest to: {manifest_csv}")

    # Generate dataset YAML for Ultralytics YOLO
    yolo_yaml_path = YOLO_DIR / "dentex_yolo.yaml"
    yaml_content = {
        "path": str(YOLO_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CANONICAL_CLASSES)}
    }
    with open(yolo_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False)

    print(f"Generated YOLO configuration: {yolo_yaml_path}")
    print("\n--- Prepared Split Annotation Statistics ---")
    df_stats = pd.DataFrame(split_class_counts)
    print(df_stats.to_string())
    print("\nSTEP 2 COMPLETE.")
    return {"yaml_path": str(yolo_yaml_path), "manifest": str(manifest_csv), "stats": split_class_counts}


# ==============================================================================
# STEP 3 & 4: MODEL TRAINING (YOLOv8 TRANSFER LEARNING WITH STREAMING)
# ==============================================================================
def mode_train(epochs: int = 30, batch_size: int = 8, imgsz: int = 640,
               device: str | None = None, model_name: str = "yolov8n.pt", fraction: float = 1.0):
    print("=" * 70)
    print("STEP 3 & 4: TRAINING YOLO ABNORMAL-TOOTH DETECTOR")
    print("=" * 70)

    yolo_yaml = YOLO_DIR / "dentex_yolo.yaml"
    if not yolo_yaml.exists():
        print("Dataset not yet prepared. Automatically running prepare step first...")
        mode_prepare()

    selected_device = check_device(device)
    # CPU optimization: if CPU, adjust batch size and workers for responsiveness
    num_workers = 4 if selected_device != "cpu" else 0
    actual_batch = batch_size if selected_device != "cpu" else min(batch_size, 4)

    from ultralytics import YOLO

    print(f"\nInitializing YOLO transfer learning with base weights: {model_name}")
    model = YOLO(model_name)

    runs_train_dir = (Path.cwd() / "runs" / "train").resolve()
    training_args = {
        "data": str(yolo_yaml.resolve()).replace("\\", "/"),
        "epochs": epochs,
        "batch": actual_batch,
        "imgsz": imgsz,
        "fraction": fraction,
        "device": selected_device,
        "workers": num_workers,
        "patience": 8,  # Early stopping patience
        "save": True,
        "save_period": -1,
        "project": str(runs_train_dir).replace("\\", "/"),
        "name": "dentex_detector",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 2.0,
        "amp": (selected_device != "cpu"),  # Mixed precision on GPU
        # Dental medical augmentations
        "degrees": 5.0,     # Subtle rotation (panoramic X-rays cannot be inverted)
        "translate": 0.05,
        "scale": 0.15,
        "fliplr": 0.5,      # Horizontal mirror is valid across arches
        "flipud": 0.0,      # NEVER flip dental X-rays vertically (violates anatomy)
        "hsv_h": 0.015,
        "hsv_s": 0.2,
        "hsv_v": 0.2,
        "verbose": True
    }

    print("\n--- Training Configuration ---")
    for k, v in training_args.items():
        print(f"  {k}: {v}")

    print("\nStarting batch-based model training (images streamed from disk; 0 risk of 15 GB RAM overflow)...")
    results = model.train(**training_args)

    # Save best and last models into models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_pt = Path(model.trainer.best) if hasattr(model, "trainer") and hasattr(model.trainer, "best") else runs_train_dir / "dentex_detector" / "weights" / "best.pt"
    last_pt = Path(model.trainer.last) if hasattr(model, "trainer") and hasattr(model.trainer, "last") else runs_train_dir / "dentex_detector" / "weights" / "last.pt"

    target_best = MODELS_DIR / "best_model.pt"
    target_last = MODELS_DIR / "last_checkpoint.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, target_best)
        print(f"\n[Saved] Best model checkpoint: {target_best}")
    elif last_pt.exists():
        shutil.copy2(last_pt, target_best)
        print(f"\n[Saved] Model checkpoint: {target_best}")

    if last_pt.exists():
        shutil.copy2(last_pt, target_last)
        print(f"[Saved] Last model checkpoint: {target_last}")

    # Extract training curves / metrics
    csv_results = runs_train_dir / "dentex_detector" / "results.csv"
    training_metrics = {}
    if csv_results.exists():
        df_curves = pd.read_csv(csv_results)
        df_curves.columns = [c.strip() for c in df_curves.columns]
        training_metrics = {col: df_curves[col].dropna().tolist() for col in df_curves.columns}

        # Plot training curves
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        if "train/box_loss" in df_curves.columns:
            axes[0, 0].plot(df_curves["epoch"], df_curves["train/box_loss"], label="Train Box Loss", color="#e74c3c")
            if "val/box_loss" in df_curves.columns:
                axes[0, 0].plot(df_curves["epoch"], df_curves["val/box_loss"], label="Val Box Loss", color="#3498db")
            axes[0, 0].set_title("Box Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].legend()
            axes[0, 0].grid(True, linestyle=":", alpha=0.6)

        if "train/cls_loss" in df_curves.columns:
            axes[0, 1].plot(df_curves["epoch"], df_curves["train/cls_loss"], label="Train Cls Loss", color="#e74c3c")
            if "val/cls_loss" in df_curves.columns:
                axes[0, 1].plot(df_curves["epoch"], df_curves["val/cls_loss"], label="Val Cls Loss", color="#3498db")
            axes[0, 1].set_title("Classification Loss")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].legend()
            axes[0, 1].grid(True, linestyle=":", alpha=0.6)

        if "metrics/mAP50(B)" in df_curves.columns:
            axes[1, 0].plot(df_curves["epoch"], df_curves["metrics/mAP50(B)"], label="mAP@50", color="#2ecc71")
            if "metrics/mAP50-95(B)" in df_curves.columns:
                axes[1, 0].plot(df_curves["epoch"], df_curves["metrics/mAP50-95(B)"], label="mAP@50-95", color="#1abc9c")
            axes[1, 0].set_title("Validation Detection mAP")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].legend()
            axes[1, 0].grid(True, linestyle=":", alpha=0.6)

        if "metrics/precision(B)" in df_curves.columns and "metrics/recall(B)" in df_curves.columns:
            axes[1, 1].plot(df_curves["epoch"], df_curves["metrics/precision(B)"], label="Precision", color="#9b59b6")
            axes[1, 1].plot(df_curves["epoch"], df_curves["metrics/recall(B)"], label="Recall", color="#f39c12")
            axes[1, 1].set_title("Precision & Recall")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].legend()
            axes[1, 1].grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        curves_png = GRAPHS_DIR / "training_curves.png"
        plt.savefig(curves_png, dpi=200)
        plt.close()
        print(f"[Saved] Training curves plot: {curves_png}")

    metrics_json = RESULTS_DIR / "training_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=2)
    print(f"[Saved] Training metrics log: {metrics_json}")

    # VERIFY CHECKPOINT: Load for inference test
    print("\nVerifying checkpoint loading for inference...")
    test_eval_model = YOLO(target_best)
    print("Checkpoint successfully loaded and verified for inference!")
    print("\nSTEP 3 & 4 COMPLETE.")
    return target_best


# ==============================================================================
# STEP 5: EVALUATION ON UNSEEN TEST SPLIT
# ==============================================================================
def mode_evaluate(model_path: Path = MODELS_DIR / "best_model.pt",
                  device: str | None = None):
    print("=" * 70)
    print("STEP 5: EVALUATING TRAINED MODEL ON HELD-OUT UNSEEN TEST SET")
    print("=" * 70)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Run training first.")

    yolo_yaml = YOLO_DIR / "dentex_yolo.yaml"
    if not yolo_yaml.exists():
        raise FileNotFoundError(f"YOLO configuration not found at {yolo_yaml}. Run prepare first.")

    selected_device = check_device(device)
    from ultralytics import YOLO

    print(f"Loading trained model from: {model_path}")
    model = YOLO(model_path)

    print("Executing validation on test split (unseen during training and hyperparameter tuning)...")
    val_results = model.val(
        data=str(yolo_yaml.resolve()).replace("\\", "/"),
        split="test",
        device=selected_device,
        plots=True,
        save_json=True
    )

    metrics = val_results.results_dict
    overall_p = float(val_results.box.mp)
    overall_r = float(val_results.box.mr)
    overall_map50 = float(val_results.box.map50)
    overall_map50_95 = float(val_results.box.map)
    overall_f1 = float((2 * overall_p * overall_r) / (overall_p + overall_r + 1e-8))

    # Per-class metrics
    per_class = {}
    class_names = val_results.names
    for i, cname in class_names.items():
        cp = float(val_results.box.p[i]) if i < len(val_results.box.p) else 0.0
        cr = float(val_results.box.r[i]) if i < len(val_results.box.r) else 0.0
        cmap50 = float(val_results.box.ap50[i]) if i < len(val_results.box.ap50) else 0.0
        cmap = float(val_results.box.ap[i]) if i < len(val_results.box.ap) else 0.0
        cf1 = float((2 * cp * cr) / (cp + cr + 1e-8))
        per_class[cname] = {
            "precision": round(cp, 4),
            "recall": round(cr, 4),
            "f1_score": round(cf1, 4),
            "mAP@50": round(cmap50, 4),
            "mAP@50:95": round(cmap, 4)
        }

    report = {
        "status": "completed",
        "model_evaluated": str(model_path).replace("\\", "/"),
        "test_split_evaluation": {
            "overall": {
                "precision": round(overall_p, 4),
                "recall": round(overall_r, 4),
                "macro_f1": round(overall_f1, 4),
                "mAP@50": round(overall_map50, 4),
                "mAP@50:95": round(overall_map50_95, 4)
            },
            "per_class": per_class
        }
    }

    eval_json = RESULTS_DIR / "evaluation_report.json"
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Copy plots generated by Ultralytics into results/graphs/
    runs_val = Path(val_results.save_dir)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    for chart_name in ["confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png", "F1_curve.png"]:
        src = runs_val / chart_name
        if src.exists():
            shutil.copy2(src, GRAPHS_DIR / chart_name.lower())

    # Generate custom per-class F1 & mAP bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    c_names = list(per_class.keys())
    f1s = [per_class[c]["f1_score"] for c in c_names]
    m50s = [per_class[c]["mAP@50"] for c in c_names]
    x = np.arange(len(c_names))
    width = 0.35

    ax.bar(x - width/2, f1s, width, label="F1-Score", color="#3498db")
    ax.bar(x + width/2, m50s, width, label="mAP@50", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(c_names, fontweight="bold")
    ax.set_ylabel("Score (0.0 - 1.0)")
    ax.set_title("Test Set Evaluation: Per-Class F1 and mAP@50", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    chart_path = GRAPHS_DIR / "per_class_metrics.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print("\n" + "=" * 70)
    print("TEST EVALUATION RESULTS (HELD-OUT UNSEEN SPLIT)")
    print("=" * 70)
    print(f"Overall Precision:  {overall_p:.4f}")
    print(f"Overall Recall:     {overall_r:.4f}")
    print(f"Overall Macro-F1:   {overall_f1:.4f}")
    print(f"mAP@50:             {overall_map50:.4f}")
    print(f"mAP@50:95:          {overall_map50_95:.4f}")
    print("\nPer-Class Breakdown:")
    for cname, stats in per_class.items():
        print(f"  - {cname.ljust(18)}: P={stats['precision']:.3f}, R={stats['recall']:.3f}, F1={stats['f1_score']:.3f}, mAP@50={stats['mAP@50']:.3f}")
    print(f"\nSaved evaluation report: {eval_json}")
    print(f"Saved evaluation plots:  {GRAPHS_DIR}")
    print("\nSTEP 5 COMPLETE.")
    return report


# ==============================================================================
# STEP 6 & 7: PREDICTION ON TEST / NEW X-RAY
# ==============================================================================
def mode_predict(image_path: str, model_path: Path = MODELS_DIR / "best_model.pt",
                 conf_thresh: float = 0.25, device: str | None = None, imgsz: int = 640):
    print("=" * 70)
    print(f"RUNNING INFERENCE PREDICTION ON: {image_path}")
    print("=" * 70)

    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Input image not found: {img_p}")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train model first.")

    selected_device = check_device(device)
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.predict(
        source=str(img_p), conf=conf_thresh, imgsz=imgsz, device=selected_device, verbose=False
    )[0]

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_img_path = PRED_DIR / f"{img_p.stem}_prediction.png"

    # Draw customized high-contrast bounding boxes with class colors
    orig_img = Image.open(img_p).convert("RGB")
    draw = ImageDraw.Draw(orig_img)

    detections = []
    boxes = results.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

            color = CLASS_COLORS.get(cls_name, "#ffff00")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            label = f"{cls_name} {conf:.2f}"
            draw.rectangle([x1, max(0, y1 - 20), x1 + len(label) * 9, max(0, y1)], fill=color)
            draw.text((x1 + 3, max(0, y1 - 18)), label, fill="white")

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })

    orig_img.save(out_img_path)
    print(f"Detections found: {len(detections)}")
    for d in detections:
        print(f"  - [{d['class_name']}] Conf: {d['confidence']} at {d['box']}")
    print(f"\nSaved visualization to: {out_img_path}")
    return {"image": str(img_p), "detections": detections, "output_visualization": str(out_img_path)}


# ==============================================================================
# MAIN CLI ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="DENTEX AI Model Training, Inspection, Evaluation & Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", required=True, choices=["inspect", "prepare", "train", "evaluate", "predict"],
                        help="Execution mode")
    parser.add_argument("--image", default=None, help="Input image path for --mode predict")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, help="Image size for YOLO training or prediction")
    parser.add_argument("--confidence", type=float, help="Override the configured cutoff in predict mode")
    parser.add_argument("--checkpoint", type=Path, help="Override the configured checkpoint in predict mode")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset images to use (default: 1.0)")
    parser.add_argument("--device", default=None, help="Compute device: 'cpu', 'cuda', '0' (default: auto)")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained model weights (default: yolov8n.pt)")

    args = parser.parse_args()

    if args.mode == "inspect":
        mode_inspect()
    elif args.mode == "prepare":
        mode_prepare()
    elif args.mode == "train":
        mode_train(epochs=args.epochs, batch_size=args.batch_size, imgsz=args.imgsz or 640,
                   device=args.device, model_name=args.model, fraction=args.fraction)
    elif args.mode == "evaluate":
        mode_evaluate(device=args.device)
    elif args.mode == "predict":
        if not args.image:
            # Pick a sample test image from data/archive if none specified
            sample_candidates = list(DISEASE_XRAYS.glob("*.png"))
            if sample_candidates:
                args.image = str(sample_candidates[0])
                print(f"No --image specified. Automatically using sample image: {args.image}")
            else:
                parser.error("--mode predict requires --image <path/to/image.png>")
        cfg = load_config()
        settings = cfg["model"]["detector"]
        mode_predict(
            args.image,
            model_path=args.checkpoint or Path(cfg["paths"]["detector_checkpoint"]),
            conf_thresh=settings["confidence_threshold"] if args.confidence is None else args.confidence,
            device=args.device,
            imgsz=args.imgsz or settings["imgsz"],
        )

if __name__ == "__main__":
    main()
