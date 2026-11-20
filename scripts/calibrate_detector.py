"""Select a detector operating point on validation and compare it on test."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dentistry.config import load_config
from dentistry.metrics import greedy_match_boxes, save_metrics
from dentistry.models import YoloDetector
from dentistry.quality import read_image


DENTEX_CLASSES = dict(enumerate(("impacted_tooth", "caries", "periapical_lesion", "deep_caries")))


def load_splits(manifest, annotations):
    records = json.loads(annotations.read_text(encoding="utf-8"))
    names = {image["id"]: image["file_name"] for image in records["images"]}
    boxes = {image_id: [] for image_id in names}
    for annotation in records["annotations"]:
        label = DENTEX_CLASSES.get(annotation.get("category_id_3"))
        if label is not None:
            x, y, width, height = annotation["bbox"]
            boxes[annotation["image_id"]].append(
                {"bbox": [x, y, x + width, y + height], "label": label}
            )
    splits = {split: [] for split in ("train", "val", "test")}
    seen = set()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            image_id = int(row["image_id"])
            name = row["file_name"]
            if names[image_id] != name or name in seen:
                raise ValueError(f"Image appears more than once or disagrees with annotations: {name}")
            seen.add(name)
            splits[row["split"]].append((name, boxes[image_id]))
    if not splits["val"] or not splits["test"]:
        raise ValueError("Calibration requires nonempty validation and test splits")
    return splits


def evaluate(detector, images, image_dir, thresholds, classes):
    counts = {threshold: {label: [0, 0, 0] for label in classes} for threshold in thresholds}
    for name, truth in images:
        predictions = detector.predict(read_image(image_dir / name))
        for label in classes:
            expected = [box for box in truth if box["label"] == label]
            predicted = [
                {"bbox": box["bbox"], "confidence": box["detector_confidence"]}
                for box in predictions if box["detector_class"] == label
            ]
            for threshold in thresholds:
                matched = greedy_match_boxes(
                    expected, [box for box in predicted if box["confidence"] > threshold]
                )
                total = counts[threshold][label]
                total[0] += matched["true_positives"]
                total[1] += matched["false_positives"]
                total[2] += matched["false_negatives"]

    results = {}
    for threshold, totals in counts.items():
        per_class = {}
        for label, (tp, fp, fn) in totals.items():
            per_class[label] = {
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
                "support": tp + fn,
                "tp": tp, "fp": fp, "fn": fn,
            }
        results[threshold] = {
            "macro_f1": sum(row["f1"] for row in per_class.values()) / len(classes),
            "per_class": per_class,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/split_manifest.csv"))
    parser.add_argument("--dataset", type=Path, default=Path(
        "data/archive/training_data/training_data/quadrant-enumeration-disease"
    ))
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 960])
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--baseline-imgsz", type=int)
    parser.add_argument("--baseline-confidence", type=float)
    parser.add_argument("--output", type=Path, default=Path("results/detector_calibration.json"))
    args = parser.parse_args()
    cfg = load_config()
    settings = cfg["model"]["detector"]
    baseline = (
        (args.baseline_checkpoint or Path(cfg["paths"]["detector_checkpoint"])).resolve(),
        args.baseline_imgsz if args.baseline_imgsz is not None else settings["imgsz"],
        args.baseline_confidence if args.baseline_confidence is not None else settings["confidence_threshold"],
    )
    if min(baseline[1], *args.imgsz) < 32 or not 0 < baseline[2] < 1:
        parser.error("Image sizes must be at least 32 and baseline confidence must be between 0 and 1")
    splits = load_splits(args.manifest, args.dataset / "train_quadrant_enumeration_disease.json")
    image_dir = args.dataset / "xrays"
    thresholds = sorted({0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.35, baseline[2]})
    candidates = dict.fromkeys([
        baseline[:2],
        *((checkpoint.resolve(), size)
          for checkpoint in [baseline[0], *args.checkpoint] for size in args.imgsz),
    ])

    def score(checkpoint, size, split, confidence_levels):
        print(f"Scoring {checkpoint.name} at {size} on {split} ({len(splits[split])} images)", flush=True)
        detector = YoloDetector(checkpoint, min(confidence_levels), size)
        if set(detector.model.names.values()) != set(cfg["classes"]):
            raise ValueError(f"Checkpoint classes do not match the configured DENTEX classes: {checkpoint}")
        return evaluate(detector, splits[split], image_dir, confidence_levels, cfg["classes"])

    validation = []
    for checkpoint, size in candidates:
        scores = score(checkpoint, size, "val", thresholds)
        for confidence, metrics in scores.items():
            validation.append({
                "checkpoint": str(checkpoint), "imgsz": size, "confidence": confidence, **metrics,
            })
    selected = max(validation, key=lambda row: (
        row["macro_f1"], row["checkpoint"] == str(baseline[0]), -row["imgsz"], row["confidence"]
    ))
    chosen = (Path(selected["checkpoint"]), selected["imgsz"], selected["confidence"])
    test = {}
    for name, setting in (("baseline", baseline), ("selected", chosen)):
        checkpoint, size, confidence = setting
        if name == "selected" and setting == baseline:
            test[name] = test["baseline"]
        else:
            test[name] = {
                "checkpoint": str(checkpoint), "imgsz": size, "confidence": confidence,
                **score(checkpoint, size, "test", [confidence])[confidence],
            }
    save_metrics({
        "metric": "Class-aware macro-F1 at IoU 0.50",
        "image_counts": {name: len(images) for name, images in splits.items()},
        "validation_candidates": validation,
        "validation_selected": selected,
        "test": test,
    }, args.output)
    print(f"Test macro-F1: {test['baseline']['macro_f1']:.4f} -> {test['selected']['macro_f1']:.4f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
