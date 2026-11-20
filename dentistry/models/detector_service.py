"""YOLOv8n Object Detector Service for abnormal dental region proposal."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from dentistry.models import YoloDetector, ModelUnavailable

class YoloDetectorService(YoloDetector):
    """Production service wrapper for YOLOv8n dental abnormality detection."""

    def __init__(self, checkpoint: str | Path = "models/best_model.pt", confidence: float = 0.075, image_size: int = 640):
        super().__init__(checkpoint=checkpoint, confidence=confidence, image_size=image_size)

    def detect_abnormalities(self, image: np.ndarray, confidence: float | None = None) -> list[dict[str, Any]]:
        """Executes calibrated YOLOv8n inference on full panoramic X-ray."""
        if confidence is not None:
            old_conf = self.confidence
            self.confidence = confidence
            try:
                return self.predict(image)
            finally:
                self.confidence = old_conf
        return self.predict(image)
