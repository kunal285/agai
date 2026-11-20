"""Tool registry and dispatcher for Agentic AI dental screening."""
from __future__ import annotations
import json
import logging
from typing import Any, Callable
from pathlib import Path
import numpy as np

from dentistry.llm.schemas import (
    CheckImageQualityInput,
    ExecuteYoloDetectionInput,
    ExtractToothCropsInput,
    ClassifyToothRegionsInput,
    GenerateGradcamHeatmapInput,
    ValidatePredictionsInput,
    RouteForHumanReviewInput,
    CreateScreeningReportInput,
    ValidateImageInput,
    QualityCheckInput,
    YoloDetectionInput,
    EfficientNetClassificationInput,
    GradcamInput,
    ComparePredictionsInput,
    ValidateDetectionResultsInput,
    RequestDentistReviewInput,
    GenerateStructuredReportInput,
)

log = logging.getLogger(__name__)

class ToolRegistry:
    """Manages tool registration, JSON schema generation, and safe execution dispatching."""

    def __init__(self):
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._input_models: dict[str, type] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        input_model: type,
        handler: Callable[..., dict[str, Any]],
    ) -> None:
        """Registers a Python function as an agent-callable tool with an OpenAPI-compatible JSON schema."""
        schema_props = input_model.model_json_schema()
        props = schema_props.get("properties", {})
        req = schema_props.get("required", [])

        self._tools[name] = handler
        self._input_models[name] = input_model
        self._schemas[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": req,
                },
            },
        }

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Returns the list of tool definitions formatted for the OpenAI API."""
        return list(self._schemas.values())

    def execute_tool(self, tool_name: str, arguments: dict[str, Any] | str) -> dict[str, Any]:
        """Safely validates tool inputs and dispatches execution to the registered handler."""
        if tool_name not in self._tools:
            return {"status": "error", "error": f"Tool '{tool_name}' is not registered."}

        try:
            if isinstance(arguments, str):
                args_dict = json.loads(arguments) if arguments.strip() else {}
            else:
                args_dict = arguments or {}

            model = self._input_models[tool_name]
            validated_args = model(**args_dict).model_dump()
            handler = self._tools[tool_name]
            result = handler(**validated_args)
            return {"status": "success", "tool": tool_name, "result": result}
        except Exception as e:
            log.exception(f"Tool execution failed for '{tool_name}'")
            return {"status": "error", "tool": tool_name, "error": str(e)}

def build_default_tool_registry(
    context_data: dict[str, Any],
    quality_fn: Callable[..., dict[str, Any]],
    detection_fn: Callable[..., dict[str, Any]],
    classification_fn: Callable[..., dict[str, Any]],
    gradcam_fn: Callable[..., dict[str, Any]],
    validation_fn: Callable[..., dict[str, Any]],
    reporting_fn: Callable[..., dict[str, Any]],
) -> ToolRegistry:
    """Builds and wires up the default tool registry bound to the active session context."""
    registry = ToolRegistry()

    # Tool 1: check_image_quality
    def _run_quality(image_ref: str = "input_xray") -> dict[str, Any]:
        return quality_fn()

    registry.register_tool(
        name="check_image_quality",
        description="Assesses radiograph image quality (contrast, brightness, Laplacian blur variance) to enforce quality gate.",
        input_model=CheckImageQualityInput,
        handler=_run_quality,
    )
    registry.register_tool(
        name="run_quality_check",
        description="Assesses radiograph image quality (contrast, brightness, Laplacian blur variance) to enforce quality gate.",
        input_model=QualityCheckInput,
        handler=_run_quality,
    )

    # Tool 2: execute_yolo_detection
    def _run_detection(image_ref: str = "input_xray", confidence_threshold: float = 0.075) -> dict[str, Any]:
        return detection_fn(confidence_threshold)

    registry.register_tool(
        name="execute_yolo_detection",
        description="Executes YOLOv8n object detection to propose candidate abnormal tooth bounding boxes on the full radiograph.",
        input_model=ExecuteYoloDetectionInput,
        handler=_run_detection,
    )
    registry.register_tool(
        name="run_yolo_detection",
        description="Executes YOLOv8n object detection to propose candidate abnormal tooth bounding boxes on the full radiograph.",
        input_model=YoloDetectionInput,
        handler=_run_detection,
    )

    # Tool 3: extract_tooth_crops
    def _extract_crops(region_ids: list[str] | None = None, padding: float = 0.10) -> dict[str, Any]:
        regs = context_data.get("detections", {}).get("regions", [])
        return {"extracted_count": len(regs), "padding": padding, "region_ids": [r["region_id"] for r in regs]}

    registry.register_tool(
        name="extract_tooth_crops",
        description="Extracts candidate bounding box crops with anatomical context padding (default 10%).",
        input_model=ExtractToothCropsInput,
        handler=_extract_crops,
    )

    # Tool 4: classify_tooth_regions
    def _run_classification(region_id: str, crop_bbox_xyxy: list[float]) -> dict[str, Any]:
        return classification_fn(region_id, crop_bbox_xyxy)

    registry.register_tool(
        name="classify_tooth_regions",
        description="Runs EfficientNet-B0 transfer learning differential classification on candidate tooth crops.",
        input_model=ClassifyToothRegionsInput,
        handler=_run_classification,
    )
    registry.register_tool(
        name="run_efficientnet_classification",
        description="Runs EfficientNet-B0 transfer learning differential classification on candidate tooth crops.",
        input_model=EfficientNetClassificationInput,
        handler=_run_classification,
    )

    # Tool 5: generate_gradcam_heatmap
    def _run_gradcam(region_id: str) -> dict[str, Any]:
        return gradcam_fn(region_id)

    registry.register_tool(
        name="generate_gradcam_heatmap",
        description="Generates a Grad-CAM sensitivity heatmap on the classifier's final convolutional layer for visual explainability.",
        input_model=GenerateGradcamHeatmapInput,
        handler=_run_gradcam,
    )
    registry.register_tool(
        name="generate_gradcam",
        description="Generates a Grad-CAM sensitivity heatmap on the classifier's final convolutional layer for visual explainability.",
        input_model=GradcamInput,
        handler=_run_gradcam,
    )

    # Tool 6: validate_predictions
    def _validate_results(findings: list[dict[str, Any]]) -> dict[str, Any]:
        return validation_fn(findings)

    registry.register_tool(
        name="validate_predictions",
        description="Performs clinical validation checks on confidence levels, errors, and triggers human review flags.",
        input_model=ValidatePredictionsInput,
        handler=_validate_results,
    )
    registry.register_tool(
        name="validate_detection_results",
        description="Performs clinical validation checks on confidence levels, errors, and triggers human review flags.",
        input_model=ValidateDetectionResultsInput,
        handler=_validate_results,
    )

    # Tool 7: route_for_human_review
    def _request_review(reason: str, flagged_regions: list[str] | None = None) -> dict[str, Any]:
        return {
            "human_review_requested": True,
            "reason": reason,
            "flagged_regions": flagged_regions or [],
            "triage_priority": "high" if "quality" in reason.lower() else "standard",
        }

    registry.register_tool(
        name="route_for_human_review",
        description="Routes ambiguous, low-confidence, or poor-quality cases to a qualified dental practitioner.",
        input_model=RouteForHumanReviewInput,
        handler=_request_review,
    )
    registry.register_tool(
        name="request_dentist_review",
        description="Routes ambiguous, low-confidence, or poor-quality cases to a qualified dental practitioner.",
        input_model=RequestDentistReviewInput,
        handler=_request_review,
    )

    # Tool 8: create_screening_report
    def _generate_report(analysis_id: str, synthesis_notes: str | None = None) -> dict[str, Any]:
        return reporting_fn(analysis_id, synthesis_notes)

    registry.register_tool(
        name="create_screening_report",
        description="Compiles the final structured JSON screening report with metadata, findings, and disclaimers.",
        input_model=CreateScreeningReportInput,
        handler=_generate_report,
    )
    registry.register_tool(
        name="generate_structured_report",
        description="Compiles the final structured JSON screening report with metadata, findings, and disclaimers.",
        input_model=GenerateStructuredReportInput,
        handler=_generate_report,
    )

    # Tool 9: validate_xray_image (pre-flight check)
    def _validate_image(image_ref: str) -> dict[str, Any]:
        image = context_data.get("image")
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {"valid": False, "reason": "No valid radiograph array found in context"}
        h, w = image.shape[:2]
        return {"valid": True, "dimensions": {"width": w, "height": h}, "channels": 3 if image.ndim == 3 else 1}

    registry.register_tool(
        name="validate_xray_image",
        description="Validates that the input radiograph is properly decoded, non-empty, and has valid dental dimensions.",
        input_model=ValidateImageInput,
        handler=_validate_image,
    )

    # Tool 10: compare_model_predictions
    def _compare_preds(
        region_id: str,
        detector_class: str,
        detector_confidence: float,
        classifier_class: str,
        classifier_confidence: float,
    ) -> dict[str, Any]:
        agreement = (detector_class == classifier_class)
        return {
            "region_id": region_id,
            "detector_class": detector_class,
            "detector_confidence": detector_confidence,
            "classifier_class": classifier_class,
            "classifier_confidence": classifier_confidence,
            "agreement": agreement,
            "note": "Agreement noted; not proof of disease" if agreement else "Model disagreement flagged for human review",
        }

    registry.register_tool(
        name="compare_model_predictions",
        description="Compares YOLO proposal vs. EfficientNet prediction to identify agreements or diagnostic discrepancies.",
        input_model=ComparePredictionsInput,
        handler=_compare_preds,
    )

    return registry
