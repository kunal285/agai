"""Grad-CAM Explainability Service for visual feature sensitivity analysis."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from dentistry.gradcam import generate_gradcam

class GradCAMService:
    """Service wrapper for computing and managing Grad-CAM heatmaps."""

    def __init__(self, classifier_model=None, target_layer=None, output_dir: str | Path = "artifacts/gradcam"):
        self.classifier_model = classifier_model
        self.target_layer = target_layer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_target_layer(self, model=None):
        m = model or self.classifier_model
        if hasattr(m, "features"):
            return m.features[-1]
        elif hasattr(m, "layer4"):
            return m.layer4[-1]
        return None

    def explain_crop(
        self,
        model,
        input_tensor,
        crop_bgr: np.ndarray,
        output_filename: str = "gradcam_output.jpg",
    ) -> dict[str, Any]:
        """Generates a Grad-CAM overlay for a given candidate crop."""
        layer = self.target_layer or self.resolve_target_layer(model)
        if layer is None:
            raise RuntimeError("Could not automatically locate the target convolutional layer for Grad-CAM.")

        out_path = self.output_dir / output_filename
        result = generate_gradcam(model, input_tensor, layer, crop_bgr, out_path)
        return {
            "target_class_id": result.get("target_class_id"),
            "heatmap": result.get("heatmap"),
            "overlay_bgr": result.get("overlay_bgr"),
            "output_path": str(out_path),
            "disclaimer": "Grad-CAM highlights neural feature sensitivity; it is not clinical proof of pathology.",
        }
