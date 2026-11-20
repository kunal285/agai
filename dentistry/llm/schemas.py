"""Pydantic schemas for Agentic AI tool calling and structured reports."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Tool Calling Input Schemas (PS-01 Named Tools)
# ---------------------------------------------------------

class CheckImageQualityInput(BaseModel):
    image_ref: str = Field(default="input_xray", description="Image identifier to evaluate for blur, contrast, and resolution")

class ExecuteYoloDetectionInput(BaseModel):
    image_ref: str = Field(default="input_xray", description="Image identifier to run YOLOv8 object detection on")
    confidence_threshold: float = Field(0.075, ge=0.01, le=1.0, description="Minimum confidence score threshold")

class ExtractToothCropsInput(BaseModel):
    region_ids: list[str] = Field(default_factory=list, description="IDs of candidate regions to extract")
    padding: float = Field(0.10, ge=0.0, le=0.5, description="Context padding fraction")

class ClassifyToothRegionsInput(BaseModel):
    region_id: str = Field(..., description="Region ID to crop and classify")
    crop_bbox_xyxy: list[float] = Field(..., description="[x1, y1, x2, y2] bounding box coordinates")

class GenerateGradcamHeatmapInput(BaseModel):
    region_id: str = Field(..., description="Region ID for which to generate Grad-CAM interpretability overlay")

class ValidatePredictionsInput(BaseModel):
    findings: list[dict[str, Any]] = Field(..., description="List of detected and classified dental regions")

class RouteForHumanReviewInput(BaseModel):
    reason: str = Field(..., description="Clinical reason why human review is required")
    flagged_regions: list[str] = Field(default_factory=list, description="IDs of ambiguous or low-confidence regions")

class CreateScreeningReportInput(BaseModel):
    analysis_id: str = Field(..., description="Unique screening analysis identifier")
    synthesis_notes: str | None = Field(default=None, description="Objective AI synthesis of the screening process")

# ---------------------------------------------------------
# Backward Compatibility Schemas
# ---------------------------------------------------------

class ValidateImageInput(BaseModel):
    image_ref: str = Field(..., description="Image identifier or file path to inspect")

class QualityCheckInput(CheckImageQualityInput):
    pass

class YoloDetectionInput(ExecuteYoloDetectionInput):
    pass

class EfficientNetClassificationInput(ClassifyToothRegionsInput):
    pass

class GradcamInput(GenerateGradcamHeatmapInput):
    pass

class ComparePredictionsInput(BaseModel):
    region_id: str = Field(..., description="Region ID being compared")
    detector_class: str = Field(..., description="Predicted class from YOLOv8 detector")
    detector_confidence: float = Field(..., description="Confidence score from detector")
    classifier_class: str = Field(..., description="Predicted class from EfficientNet classifier")
    classifier_confidence: float = Field(..., description="Confidence score from classifier")

class ValidateDetectionResultsInput(ValidatePredictionsInput):
    pass

class RequestDentistReviewInput(RouteForHumanReviewInput):
    pass

class GenerateStructuredReportInput(CreateScreeningReportInput):
    pass

# ---------------------------------------------------------
# Structured AI Explanation Output
# ---------------------------------------------------------

class RegionSummary(BaseModel):
    region_id: str
    anatomical_location: str
    detector_finding: str
    detector_confidence: float
    classifier_finding: str | None = None
    classifier_confidence: float | None = None
    agreement: bool | None = None
    gradcam_visual_evidence: str | None = None
    triage_recommendation: Literal["dentist_review", "standard_record"]

class AIScreeningSynthesis(BaseModel):
    analysis_id: str
    quality_summary: str
    detected_abnormalities_count: int
    findings_summary: list[RegionSummary]
    disagreements_noted: list[str] = Field(default_factory=list)
    human_review_required: bool
    mandatory_disclaimer: str = "Educational research prototype • Not a clinical diagnostic or treatment system"
    limitations: list[str] = Field(default_factory=list)
