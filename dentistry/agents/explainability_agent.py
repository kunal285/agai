"""Agent 5: Explainability Agent.

Integrates Grad-CAM on the classifier's final convolutional layer.
Generates heatmaps and links them to model predictions.
States model focus and limitations; never describes Grad-CAM as proof of disease.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from dentistry.models import crop_bbox
from dentistry.gradcam import generate_gradcam

class ExplainabilityAgent:
    name = "explainability"

    def __init__(self, classifier=None, gradcam_dir: str | Path = "artifacts/gradcam"):
        self.classifier = classifier
        self.gradcam_dir = Path(gradcam_dir)
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)

    def run(self, image: np.ndarray, regions: list[dict[str, Any]], run_id: str = "run") -> dict[str, Any]:
        if self.classifier is None:
            return {
                "agent": self.name,
                "status": "skipped",
                "message": "Classifier unavailable; Grad-CAM cannot be computed",
                "regions": regions,
            }

        model = getattr(self.classifier, "model", None)
        if model is None:
            return {
                "agent": self.name,
                "status": "skipped",
                "message": "Model instance not accessible on classifier",
                "regions": regions,
            }

        target_layer = None
        if hasattr(model, "features"):
            target_layer = model.features[-1]
        elif hasattr(model, "layer4"):
            target_layer = model.layer4[-1]

        if target_layer is None:
            return {
                "agent": self.name,
                "status": "skipped",
                "message": "Target convolutional feature layer not found on model",
                "regions": regions,
            }

        updated_regions = []
        for idx, r in enumerate(regions):
            r_copy = dict(r)
            bbox = r.get("bbox")
            if not bbox:
                updated_regions.append(r_copy)
                continue

            try:
                crop, _ = crop_bbox(image, bbox, padding=0.10)
                tensor = self.classifier.preprocess(crop)
                out_path = self.gradcam_dir / f"{run_id}_region_{idx+1}_gradcam.jpg"
                cam_res = generate_gradcam(model, tensor, target_layer, crop, out_path)
                r_copy["gradcam_path"] = cam_res.get("output_path")
                r_copy["gradcam_note"] = (
                    "Grad-CAM visualizes network sensitivity; it does NOT constitute biological or pathological proof."
                )
            except Exception as e:
                r_copy["errors"] = r_copy.get("errors", []) + [f"Grad-CAM generation failed: {e}"]
                r_copy["gradcam_path"] = None

            updated_regions.append(r_copy)

        return {
            "agent": self.name,
            "status": "success",
            "gradcam_count": sum(1 for r in updated_regions if r.get("gradcam_path")),
            "regions": updated_regions,
        }
