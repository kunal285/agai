"""Validation rules and clinical guardrails for dental X-ray screening."""
from __future__ import annotations
from typing import Any
from dentistry.safety.clinical_guardrails import ClinicalGuardrails, DISCLAIMER_TEXT

class ValidationRules(ClinicalGuardrails):
    """Encapsulates clinical threshold validation, model disagreement checking, and human review routing."""

    def __init__(
        self,
        detector_review_threshold: float = 0.25,
        classifier_review_threshold: float = 0.65,
        supported_classes: list[str] | None = None,
    ):
        super().__init__(
            detector_review_threshold=detector_review_threshold,
            classifier_review_threshold=classifier_review_threshold,
            supported_classes=supported_classes,
        )

    def evaluate_triage(
        self,
        detection_result: dict[str, Any],
        quality_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluates whether the scan requires human dental review or can proceed as standard screening record."""
        return self.validate_screening_output(
            detection_result=detection_result,
            quality_result=quality_result or {"acceptable": True, "status": "acceptable"},
        )
