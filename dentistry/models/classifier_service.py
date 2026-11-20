"""EfficientNet-B0 Classifier Service for fine-grained differential diagnosis."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from dentistry.models import TorchvisionCropClassifier, crop_bbox, ModelUnavailable

class EfficientNetClassifierService(TorchvisionCropClassifier):
    """Production service wrapper for EfficientNet-B0 tooth crop differential classification."""

    def __init__(
        self,
        checkpoint: str | Path = "models/classifier/best.pt",
        classes: list[str] | None = None,
        architecture: str = "efficientnet_b0",
        image_size: int = 224,
        device: str | None = None,
    ):
        target_classes = classes or ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]
        super().__init__(
            checkpoint=checkpoint,
            classes=target_classes,
            architecture=architecture,
            image_size=image_size,
            device=device,
        )

    def classify_crop(self, crop: np.ndarray) -> dict[str, Any]:
        """Runs differential classification on a 224x224 RGB tooth crop."""
        return self.predict(crop)

    def extract_and_classify(self, image: np.ndarray, bbox: list[float], padding: float = 0.10) -> dict[str, Any]:
        """Crops an anatomical region with padding and runs classification."""
        crop, coords = crop_bbox(image, bbox, padding=padding)
        pred = self.classify_crop(crop)
        return {
            "crop": crop,
            "crop_coords": list(coords),
            "prediction": pred,
        }
