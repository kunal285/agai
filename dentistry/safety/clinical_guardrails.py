"""Clinical safety guardrails and medical AI validation rules."""
from __future__ import annotations
from typing import Any

DISCLAIMER_TEXT = "Educational research prototype • Not a clinical diagnostic or treatment system"

# Human-friendly class name mappings
FRIENDLY_CLASS_NAMES = {
    "caries": "Tooth Decay (Caries)",
    "deep_caries": "Deep Tooth Decay",
    "periapical_lesion": "Root-Tip Infection (Periapical Lesion)",
    "impacted_tooth": "Impacted Tooth",
}

def format_class_name(raw_name: str | None) -> str:
    if not raw_name:
        return "Uncertain Finding"
    return FRIENDLY_CLASS_NAMES.get(raw_name, raw_name.replace("_", " ").title())

def format_confidence_label(conf: float) -> str:
    pct = int(round(conf * 100))
    if conf < 0.35:
        return f"Low ({pct}%)"
    elif conf < 0.70:
        return f"Moderate ({pct}%)"
    else:
        return f"High ({pct}%)"

def get_disagreement_message(det_class: str, cls_class: str) -> str:
    pair = {det_class, cls_class}
    if pair == {"caries", "deep_caries"}:
        return "AI models disagree about the severity of tooth decay."
    elif "periapical_lesion" in pair and ("caries" in pair or "deep_caries" in pair):
        return "AI models disagree between tooth decay and a root-tip infection."
    elif "impacted_tooth" in pair and ("caries" in pair or "deep_caries" in pair):
        return "AI models disagree between tooth decay and an impacted tooth."
    else:
        return "The AI models gave different results. A dentist should check this tooth."

class ClinicalGuardrails:
    """Enforces non-diagnostic research boundaries, simplified language, and review triage."""

    def __init__(
        self,
        detector_review_threshold: float = 0.25,
        classifier_review_threshold: float = 0.65,
        supported_classes: list[str] | None = None,
    ):
        self.detector_review_threshold = detector_review_threshold
        self.classifier_review_threshold = classifier_review_threshold
        self.supported_classes = supported_classes or [
            "caries", "deep_caries", "periapical_lesion", "impacted_tooth"
        ]

    def validate_region(self, region: dict[str, Any], tooth_index: int = 1) -> dict[str, Any]:
        """Validates a single candidate region for clinical confidence, model agreement, and triage."""
        technical_reasons = []
        simplified_reasons = []

        detector_class = region.get("detector_class")
        detector_conf = float(region.get("detector_confidence") or 0.0)

        classifier_data = region.get("classifier") or {}
        classifier_class = classifier_data.get("class_name")
        classifier_conf = float(classifier_data.get("confidence") or 0.0)

        is_low_det = False
        is_low_cls = False
        is_disagreement = False

        # Check supported classes
        if detector_class and detector_class not in self.supported_classes:
            technical_reasons.append(f"Detector class '{detector_class}' is unsupported or unverified")
            simplified_reasons.append("AI could not verify this finding type.")

        # Confidence checks
        if detector_conf < self.detector_review_threshold:
            is_low_det = True
            technical_reasons.append(
                f"Low detector confidence ({detector_conf:.2f} < {self.detector_review_threshold:.2f})"
            )
            simplified_reasons.append("AI could not clearly identify this area.")

        if not classifier_data:
            technical_reasons.append("Secondary CNN classification confirmation unavailable")
            simplified_reasons.append("AI confirmation is not available for this area.")
        else:
            if classifier_conf < self.classifier_review_threshold:
                is_low_cls = True
                technical_reasons.append(
                    f"Low classifier confidence ({classifier_conf:.2f} < {self.classifier_review_threshold:.2f})"
                )
                simplified_reasons.append("AI is not fully sure about this result.")

            if classifier_class and classifier_class not in self.supported_classes:
                technical_reasons.append(f"Classifier class '{classifier_class}' is unsupported")
                simplified_reasons.append("AI classified an unsupported condition.")

        # Disagreement check between detector and classifier
        agreement = None
        if detector_class and classifier_class:
            agreement = (detector_class == classifier_class)
            if not agreement:
                is_disagreement = True
                technical_reasons.append(
                    f"Model disagreement: Detector proposed '{detector_class}' but CNN classified as '{classifier_class}'"
                )
                disagree_msg = get_disagreement_message(detector_class, classifier_class)
                simplified_reasons.append(disagree_msg)

        # Region errors
        if region.get("errors"):
            technical_reasons.extend(region["errors"])
            simplified_reasons.append("Processing note: region required manual verification.")

        # Determine Review Status and Message
        review_required = bool(technical_reasons)
        unique_simplified = list(dict.fromkeys(simplified_reasons))

        if not review_required and agreement is True:
            review_status = "No Additional Review Flag"
            status_color = "green"
            clinical_message = "AI Result – Needs Clinical Confirmation"
            primary_reason = "Models agree. Clinical confirmation still required."
        elif is_disagreement:
            review_status = "Dentist Review Needed"
            status_color = "red"
            clinical_message = "The AI models gave different results. A dentist should check this tooth."
            primary_reason = get_disagreement_message(detector_class or "", classifier_class or "")
        elif is_low_det or is_low_cls:
            review_status = "Dentist Review Needed"
            status_color = "yellow"
            clinical_message = "The AI is not fully sure about this finding. Please ask a dentist to review it."
            primary_reason = "The AI is not fully sure about this result."
        else:
            review_status = "Dentist Review Needed" if review_required else "No Additional Review Flag"
            status_color = "yellow" if review_required else "green"
            clinical_message = "Dentist review needed." if review_required else "AI Result – Needs Clinical Confirmation"
            primary_reason = unique_simplified[0] if unique_simplified else "Standard verification."

        # Best predicted finding for human presentation
        best_finding_key = classifier_class if (classifier_conf >= detector_conf and classifier_class) else (detector_class or classifier_class)
        friendly_finding = format_class_name(best_finding_key)
        best_conf = max(detector_conf, classifier_conf)

        # Provide backward-compatible review_reasons containing both technical and simplified strings
        combined_review_reasons = technical_reasons + [s for s in unique_simplified if s not in technical_reasons]

        return {
            "region_id": region.get("region_id", f"region_{tooth_index}"),
            "tooth_label": f"Tooth #{tooth_index}",
            "tooth_index": tooth_index,
            "predicted_finding": friendly_finding,
            "finding_key": best_finding_key,
            "confidence_label": format_confidence_label(best_conf),
            "confidence_value": best_conf,
            "detector_class": detector_class,
            "detector_confidence": detector_conf,
            "classifier_class": classifier_class,
            "classifier_confidence": classifier_conf if classifier_data else None,
            "model_agreement": agreement,
            "is_disagreement": is_disagreement,
            "is_low_confidence": (is_low_det or is_low_cls),
            "review_required": review_required,
            "review_status": review_status,
            "status_color": status_color,
            "clinical_message": clinical_message,
            "primary_reason": primary_reason,
            "simplified_reasons": unique_simplified,
            "technical_reasons": technical_reasons,
            "review_reasons": combined_review_reasons,
            "clinical_disclaimer": DISCLAIMER_TEXT,
        }

    def validate_screening_output(self, detection_result: dict[str, Any], quality_result: dict[str, Any]) -> dict[str, Any]:
        """Performs global clinical validation across all regions and quality metrics."""
        regions = detection_result.get("regions", [])
        validated_regions = []
        global_technical_reasons = []
        global_simplified_reasons = []

        # Check image quality
        if not quality_result.get("acceptable", False):
            msg = "Image quality was poor or blurred; radiograph rejected by quality gate."
            global_technical_reasons.append("Image quality requirements not satisfied; radiograph rejected by quality gate")
            global_simplified_reasons.append(msg)

        # Check empty detections: in a screening system, 0 detections does NOT equal confirmed healthy
        if not regions and quality_result.get("acceptable", False):
            msg = "No abnormalities detected by AI. Note: Absence of detection is not proof of health."
            global_technical_reasons.append("No abnormal regions detected. Mandatory disclaimer: Absence of detection is NOT proof of health.")
            global_simplified_reasons.append(msg)

        total_review_needed = 0
        total_low_confidence = 0
        total_disagreements = 0

        for idx, r in enumerate(regions, 1):
            v_reg = self.validate_region(r, tooth_index=idx)
            validated_regions.append(v_reg)
            if v_reg["review_required"]:
                total_review_needed += 1
                global_technical_reasons.extend(v_reg["technical_reasons"])
                global_simplified_reasons.extend(v_reg["simplified_reasons"])
            if v_reg["is_low_confidence"]:
                total_low_confidence += 1
            if v_reg["is_disagreement"]:
                total_disagreements += 1

        unique_technical = list(dict.fromkeys(global_technical_reasons))
        unique_simplified = list(dict.fromkeys(global_simplified_reasons))
        human_review_required = total_review_needed > 0 or not quality_result.get("acceptable", False) or len(unique_technical) > 0

        review_summary = {
            "total_teeth_requiring_review": total_review_needed,
            "low_confidence_findings": total_low_confidence,
            "model_disagreements": total_disagreements,
            "pending_dentist_reviews": total_review_needed,
            "review_status_label": "Dentist's Review Needed" if human_review_required else "No Additional Review Flag",
        }

        # Provide both technical & simplified for backward compatibility
        combined_reasons = unique_technical + [s for s in unique_simplified if s not in unique_technical]

        return {
            "human_review_required": human_review_required,
            "review_status": "Dentist Review Needed" if human_review_required else "No Additional Review Flag",
            "review_summary": review_summary,
            "simplified_reasons": unique_simplified,
            "technical_reasons": unique_technical,
            "review_reasons": combined_reasons,
            "validated_regions": validated_regions,
            "never_treat_agreement_as_proof": True,
            "never_declare_healthy_automatically": True,
            "disclaimer": DISCLAIMER_TEXT,
        }
