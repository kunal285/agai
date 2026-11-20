"""Agent 3: Clinical Validation Agent.

Validates detection and classification outputs against safety thresholds:
- Detector confidence threshold (<0.25)
- Classifier confidence threshold (<0.65)
- Model disagreement detection (YOLO vs EfficientNet-B0)
- Simplifies technical messages into plain clinical language
- Groups warnings by tooth and removes repeated warnings
- Routes cases into three statuses:
  1. No Additional Review Flag (Needs Clinical Confirmation)
  2. Dentist Review Needed
  3. Review Pending
"""
from __future__ import annotations
from typing import Any
from dentistry.safety.clinical_guardrails import ClinicalGuardrails, DISCLAIMER_TEXT

class ValidationAgent:
    name = "validation"

    def __init__(self, detector_threshold: float = 0.25, classifier_threshold: float = 0.65):
        self.detector_threshold = detector_threshold
        self.classifier_threshold = classifier_threshold
        self.guardrails = ClinicalGuardrails(
            detector_review_threshold=detector_threshold,
            classifier_review_threshold=classifier_threshold,
        )

    def run(self, detection_result: dict[str, Any], quality_result: dict[str, Any] | None = None) -> dict[str, Any]:
        q_res = quality_result or {"acceptable": True, "status": "acceptable"}
        val_output = self.guardrails.validate_screening_output(detection_result, q_res)

        regions = detection_result.get("regions", [])
        flagged_region_ids = []
        uncertainty_flags = []

        # Enrich each region with grouped, simplified clinical validation data
        for r, v in zip(regions, val_output.get("validated_regions", [])):
            r["tooth_label"] = v["tooth_label"]
            r["predicted_finding"] = v["predicted_finding"]
            r["confidence_label"] = v["confidence_label"]
            r["confidence_value"] = v["confidence_value"]
            r["review_status"] = v["review_status"]
            r["status_color"] = v["status_color"]
            r["clinical_message"] = v["clinical_message"]
            r["simplified_reasons"] = v["simplified_reasons"]
            r["technical_reasons"] = v["technical_reasons"]
            r["review_required"] = v["review_required"]
            r["review_reasons"] = v["review_reasons"]
            r["model_agreement"] = v.get("model_agreement")
            r["is_disagreement"] = v.get("is_disagreement", False)
            r["is_low_confidence"] = v.get("is_low_confidence", False)

            if v["review_required"]:
                flagged_region_ids.append(r.get("region_id", "unknown_region"))
                uncertainty_flags.extend(v["review_reasons"])

        if not regions and q_res.get("acceptable", False):
            flagged_region_ids.append("no_detections_disclaimer")
            uncertainty_flags.append("No abnormal regions detected. Mandatory disclaimer: Absence of detection is NOT proof of health.")

        review_req = val_output.get("human_review_required", True)
        all_reasons = val_output.get("review_reasons", [])

        # PS-01 required outputs + backward-compatible keys
        return {
            "agent": self.name,
            "validation_status": "needs_review" if review_req else "validated",
            "review_status": val_output.get("review_status", "Dentist Review Needed" if review_req else "No Additional Review Flag"),
            "review_summary": val_output.get("review_summary", {}),
            "simplified_reasons": val_output.get("simplified_reasons", []),
            "technical_reasons": val_output.get("technical_reasons", []),
            "uncertainty_flags": list(dict.fromkeys(uncertainty_flags)),
            "human_review_required": bool(review_req),
            "validation_reasons": all_reasons,
            # Backward-compatibility fields
            "review_required": bool(review_req),
            "review_reasons": all_reasons,
            "flagged_regions": flagged_region_ids,
            "regions": regions,
            "validated_regions": val_output.get("validated_regions", []),
            "disclaimer": DISCLAIMER_TEXT,
        }

# Alias for compatibility
ClinicalValidationAgent = ValidationAgent
