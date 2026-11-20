"""Agent 1: Image Quality Agent.

Checks image readability, blur (Laplacian variance), contrast (std dev), dimensions,
and detects corrupted or unreadable images. Rejects compromised scans at the gate.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from dentistry.quality import assess_image

class ImageQualityAgent:
    name = "image_quality"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def run(self, image: np.ndarray, config: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg = config or self.config
        assessment = assess_image(image, cfg)

        acceptable = bool(assessment.get("acceptable", False))
        metrics = assessment.get("metrics", {})
        warnings = assessment.get("warnings", [])

        # PS-01 required outputs + backward-compatible keys
        return {
            "agent": self.name,
            "quality_status": "acceptable" if acceptable else "poor",
            "blur_score": float(metrics.get("laplacian_variance", 0.0)),
            "contrast_score": float(metrics.get("contrast_std", 0.0)),
            "image_dimensions": {
                "width": int(metrics.get("width", 0)),
                "height": int(metrics.get("height", 0)),
            },
            "rejection_reason": warnings if not acceptable else None,
            "review_required": not acceptable,
            # Backward-compatibility fields
            "acceptable": acceptable,
            "status": "acceptable" if acceptable else "poor",
            "warnings": warnings,
            "metrics": metrics,
            "quality_gate_passed": acceptable,
        }
