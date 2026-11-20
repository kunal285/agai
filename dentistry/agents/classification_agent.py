"""Agent 4: Classification Agent.

Executes the compact EfficientNet-B0 CNN on context-padded crops for diagnosis confirmation.
Compares classifier output with detector proposals, identifying agreements and disagreements.
Never treats model agreement as proof of diagnostic correctness.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from dentistry.models import crop_bbox

class DentalClassificationAgent:
    name = "classification"

    def __init__(self, classifier=None):
        self.classifier = classifier

    def run(self, image: np.ndarray, regions: list[dict[str, Any]]) -> dict[str, Any]:
        if self.classifier is None:
            return {
                "agent": self.name,
                "status": "skipped",
                "message": "Classifier not configured or checkpoint unavailable",
                "classified_regions": regions,
            }

        classified = []
        for r in regions:
            r_copy = dict(r)
            bbox = r.get("bbox")
            if not bbox:
                r_copy["errors"] = r_copy.get("errors", []) + ["Missing bounding box for classification"]
                classified.append(r_copy)
                continue

            try:
                crop, coords = crop_bbox(image, bbox, padding=0.10)
                r_copy["crop_bbox_xyxy"] = list(coords)
                pred = self.classifier.predict(crop)
                r_copy["classifier"] = pred

                det_class = r_copy.get("detector_class")
                cls_class = pred.get("class_name")
                r_copy["model_agreement"] = (det_class == cls_class) if (det_class and cls_class) else None

            except Exception as e:
                r_copy["errors"] = r_copy.get("errors", []) + [f"Classification failed: {e}"]
                r_copy["classifier"] = None
                r_copy["model_agreement"] = None

            classified.append(r_copy)

        return {
            "agent": self.name,
            "status": "success",
            "classified_count": len(classified),
            "regions": classified,
        }
