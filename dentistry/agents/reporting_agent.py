"""Agent 4: Reporting Agent.

Collects validated agent outputs from Agent 3 (Clinical Validation Agent),
generates tooth-wise screening summaries, creates structured JSON reports,
and includes clinical review statuses, simplified reasons, and model versions.
"""
from __future__ import annotations
from typing import Any
from datetime import datetime, timezone
from dentistry.safety.clinical_guardrails import DISCLAIMER_TEXT

class ReportingAgent:
    name = "reporting"

    def run(
        self,
        image_name: str,
        quality: dict[str, Any],
        detection: dict[str, Any],
        validation: dict[str, Any],
        workflow_status: str = "completed",
        analysis_id: str | None = None,
        model_versions: dict[str, str] | None = None,
        execution_trace: dict[str, Any] | None = None,
        token_usage: dict[str, Any] | None = None,
        ai_synthesis: str | None = None,
    ) -> dict[str, Any]:
        regions = []
        visual_evidence = {"image_name": image_name, "gradcam_overlays": [], "bounding_boxes": []}
        confidence_scores = []
        simplified_findings = []
        affected_teeth = []
        findings_requiring_review = []
        uncertain_findings = []

        validated_regions = validation.get("validated_regions") or validation.get("regions", [])

        for idx, r in enumerate(validated_regions, 1):
            pred = r.get("classifier") or {}
            box = r.get("bbox") or r.get("bbox_xyxy")
            t_label = r.get("tooth_label", f"Tooth #{idx}")
            finding = r.get("predicted_finding") or pred.get("class_name") or r.get("detector_class") or "Uncertain Finding"
            conf_label = r.get("confidence_label") or f"{int(round(float(r.get('detector_confidence') or 0.0)*100))}%"
            rev_stat = r.get("review_status", "Dentist Review Needed" if r.get("review_required") else "No Additional Review Flag")

            reg_entry = {
                "region_id": r.get("region_id", f"region_{idx}"),
                "tooth_label": t_label,
                "bbox_xyxy": box,
                "crop_bbox_xyxy": r.get("crop_bbox_xyxy"),
                "detector_class": r.get("detector_class"),
                "detector_confidence": r.get("detector_confidence"),
                "predicted_class": pred.get("class_name"),
                "classifier_confidence": pred.get("confidence"),
                "class_scores": pred.get("probabilities"),
                "model_agreement": r.get("model_agreement"),
                "predicted_finding": finding,
                "confidence_label": conf_label,
                "review_status": rev_stat,
                "status_color": r.get("status_color", "red" if r.get("review_required") else "green"),
                "clinical_message": r.get("clinical_message", "AI Result – Needs Clinical Confirmation"),
                "primary_reason": r.get("primary_reason", ""),
                "simplified_reasons": r.get("simplified_reasons", []),
                "technical_reasons": r.get("technical_reasons", []),
                "review_required": bool(r.get("review_required", True)),
                "review_reasons": r.get("review_reasons", []),
                "gradcam_path": r.get("gradcam_path"),
                "warnings": r.get("errors", []),
            }
            regions.append(reg_entry)

            affected_teeth.append(t_label)

            simplified_findings.append({
                "tooth_number": t_label,
                "predicted_finding": finding,
                "confidence": conf_label,
                "review_status": rev_stat,
                "reason_for_review": r.get("primary_reason") or (r.get("simplified_reasons", ["Standard verification"])[0] if r.get("simplified_reasons") else "Standard verification"),
                "clinical_message": r.get("clinical_message"),
            })

            if r.get("review_required"):
                findings_requiring_review.append(t_label)
            if r.get("is_low_confidence") or r.get("is_disagreement"):
                uncertain_findings.append(t_label)

            if box:
                visual_evidence["bounding_boxes"].append({"region_id": r.get("region_id", f"region_{idx}"), "tooth_label": t_label, "bbox": box})
            if r.get("gradcam_path"):
                visual_evidence["gradcam_overlays"].append(r["gradcam_path"])

            confidence_scores.append({
                "tooth_label": t_label,
                "region_id": r.get("region_id", f"region_{idx}"),
                "confidence_label": conf_label,
                "detector_confidence": r.get("detector_confidence"),
                "classifier_confidence": pred.get("confidence"),
            })

        agent_statuses = (execution_trace or {}).get("agent_statuses", {
            "image_quality": "completed",
            "detection_diagnosis": detection.get("status", "unknown"),
            "validation": "completed",
            "reporting": "completed",
        })

        warnings = list(quality.get("warnings", []))
        for r in regions:
            warnings.extend(r.get("warnings", []))
        if not quality.get("acceptable", True):
            warnings.append("Image failed quality gate; automated findings may be unreliable")

        aid = analysis_id or f"analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        versions = model_versions or {
            "detector": "YOLOv8n (DENTEX trained)",
            "classifier": "EfficientNet-B0 (DENTEX transfer learning)",
            "gradcam": "PyTorch Grad-CAM (Target: features[-1])",
            "orchestrator": "Four-Agent System (OpenAI Responses + Python Safe Dispatcher)",
        }

        # Build clean review summary
        rev_summary = validation.get("review_summary") or {
            "total_teeth_requiring_review": len(findings_requiring_review),
            "low_confidence_findings": len(uncertain_findings),
            "model_disagreements": sum(1 for r in regions if r.get("model_agreement") is False),
            "pending_dentist_reviews": len(findings_requiring_review),
            "review_status_label": "Dentist's Review Needed" if len(findings_requiring_review) > 0 else "No Additional Review Flag",
        }

        if not ai_synthesis:
            try:
                from dentistry.llm.openai_client import OpenAIOrchestrationClient
                client = OpenAIOrchestrationClient()
                ai_synthesis = client._build_dynamic_synthesis(aid, quality, regions, validation)
            except Exception:
                ai_synthesis = (
                    f"**Dental Screening AI Synthesis (Analysis ID: {aid})**\n\n"
                    f"Screening workflow completed with {len(regions)} suspicious regions evaluated. "
                    f"Human dentist review status: {rev_summary.get('review_status_label', 'Dentist Review Needed')}. "
                    "All findings require clinical confirmation."
                )

        return {
            "schema_version": "1.0",
            "analysis_id": aid,
            "image_quality": quality,
            "screening_findings": regions,
            "confidence_scores": confidence_scores,
            "visual_evidence": visual_evidence,
            "validation_status": validation.get("validation_status", "needs_review" if validation.get("review_required") else "validated"),
            "clinical_review_status": validation.get("review_status", "Dentist Review Needed" if validation.get("review_required") else "No Additional Review Flag"),
            "dentist_review_summary": rev_summary,
            "simplified_findings": simplified_findings,
            "affected_tooth_numbers": affected_teeth,
            "findings_requiring_dentist_review": findings_requiring_review,
            "uncertain_findings": uncertain_findings,
            "simplified_reasons": validation.get("simplified_reasons", []),
            "human_review_status": {
                "required": bool(validation.get("review_required", True)),
                "reasons": validation.get("validation_reasons", validation.get("review_reasons", [])),
                "review_status_label": rev_summary.get("review_status_label", "Dentist's Review Needed"),
            },
            "processing_time": {
                "agent_latencies_ms": (execution_trace or {}).get("agent_latencies_ms", {}),
                "total_latency_ms": (execution_trace or {}).get("total_latency_ms", 0.0),
            },
            "model_versions": versions,
            "research_disclaimer": DISCLAIMER_TEXT,
            # Backward-compatibility fields
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "image_name": image_name,
            "workflow_status": workflow_status,
            "educational_research_only": True,
            "not_for_clinical_decisions": True,
            "detection_count": len(regions),
            "human_review_required": bool(validation.get("review_required", True)),
            "review_reasons": validation.get("review_reasons", []),
            "regions": regions,
            "warnings": list(dict.fromkeys(warnings)),
            "agent_status": agent_statuses,
            "execution_trace": execution_trace or {},
            "token_usage": token_usage or {},
            "ai_synthesis": ai_synthesis,
            "disclaimer": DISCLAIMER_TEXT,
        }
