"""Agent 2: Detection and Diagnosis Agent.

Coordinates:
1. YOLOv8n object detection (abnormal tooth localization).
2. Context-padded crop extraction (10% anatomical padding).
3. EfficientNet-B0 differential classification across 4 DENTEX categories.
4. Grad-CAM visual explainability generation on convolutional feature layer.
5. Model disagreement identification between detector and classifier.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np

from dentistry.models import crop_bbox
from dentistry.gradcam import generate_gradcam

class DetectionDiagnosisAgent:
    name = "detection_diagnosis"

    def __init__(
        self,
        detector,
        classifier=None,
        gradcam_dir: str | Path = "artifacts/gradcam",
        confidence_threshold: float = 0.075,
    ):
        self.detector = detector
        self.classifier = classifier
        self.gradcam_dir = Path(gradcam_dir)
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold

    def run(
        self,
        image: np.ndarray,
        image_id: str = "upload",
        confidence_threshold: float | None = None,
    ) -> dict[str, Any]:
        conf = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        if hasattr(self.detector, "confidence"):
            old_conf = self.detector.confidence
            self.detector.confidence = conf
            try:
                detections = self.detector.predict(image)
            finally:
                self.detector.confidence = old_conf
        else:
            detections = self.detector.predict(image)

        regions = []
        bboxes = []
        cls_results = []
        cls_probs = []
        gradcam_refs = []
        any_disagreement = False

        for i, d in enumerate(detections):
            reg_id = f"region_{i+1}"
            raw_box = [float(x) for x in d["bbox"]]
            bboxes.append(raw_box)
            det_class = d.get("detector_class") or d.get("class_name", "unknown")
            det_conf = float(d.get("detector_confidence") or d.get("confidence", 0.0))

            region = {
                "region_id": reg_id,
                "bbox": raw_box,
                "crop_bbox_xyxy": None,
                "detector_class": det_class,
                "detector_confidence": det_conf,
                "classifier": None,
                "gradcam_path": None,
                "model_agreement": None,
                "errors": [],
            }

            # 1. Padded crop extraction & 2. Classification
            try:
                crop, coords = crop_bbox(image, raw_box, padding=0.10)
                region["crop_bbox_xyxy"] = list(coords)

                if self.classifier:
                    pred = self.classifier.predict(crop)
                    region["classifier"] = pred
                    cls_results.append(pred)
                    cls_probs.append(pred.get("probabilities", {}))

                    cls_class = pred.get("class_name")
                    agreement = (det_class == cls_class) if (det_class and cls_class) else None
                    region["model_agreement"] = agreement
                    if agreement is False:
                        any_disagreement = True

                    # 3. Grad-CAM generation
                    try:
                        model = getattr(self.classifier, "model", None)
                        target_layer = None
                        if hasattr(model, "features"):
                            target_layer = model.features[-1]
                        elif hasattr(model, "layer4"):
                            target_layer = model.layer4[-1]

                        if target_layer is not None:
                            tensor = self.classifier.preprocess(crop)
                            cam_path = self.gradcam_dir / f"{image_id}_{i+1}.jpg"
                            cam = generate_gradcam(model, tensor, target_layer, crop, cam_path)
                            region["gradcam_path"] = cam.get("output_path")
                            gradcam_refs.append(cam.get("output_path"))
                    except Exception as e:
                        region["errors"].append(f"Grad-CAM unavailable: {e}")
                        gradcam_refs.append(None)
                else:
                    cls_results.append(None)
                    cls_probs.append({})
                    gradcam_refs.append(None)

            except Exception as e:
                region["errors"].append(f"Region crop processing failed: {e}")
                cls_results.append(None)
                cls_probs.append({})
                gradcam_refs.append(None)

            regions.append(region)

        # PS-01 required outputs + backward-compatible keys
        return {
            "agent": self.name,
            "status": "success",
            "detection_count": len(regions),
            "detection_results": regions,
            "bounding_boxes": bboxes,
            "classification_results": cls_results,
            "class_probabilities": cls_probs,
            "gradcam_references": gradcam_refs,
            "model_disagreement": any_disagreement,
            # Backward compatibility
            "regions": regions,
        }
