"""Comprehensive automated tests for Advanced Agentic AI multi-agent dental screening."""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from dentistry.agents.quality_agent import ImageQualityAgent
from dentistry.agents.detection_agent import DentalDetectionAgent
from dentistry.agents.classification_agent import DentalClassificationAgent
from dentistry.agents.explainability_agent import ExplainabilityAgent
from dentistry.agents.validation_agent import ClinicalValidationAgent
from dentistry.agents.reporting_agent import ReportingAgent
from dentistry.agents.orchestrator_agent import MasterOrchestratorAgent
from dentistry.llm.openai_client import OpenAIOrchestrationClient
from dentistry.llm.tool_registry import ToolRegistry, build_default_tool_registry
from dentistry.llm.schemas import (
    ValidateImageInput,
    QualityCheckInput,
    YoloDetectionInput,
    EfficientNetClassificationInput,
    ComparePredictionsInput,
    RequestDentistReviewInput,
    GenerateStructuredReportInput,
)
from dentistry.safety.privacy import sanitize_metadata, check_image_transmission_allowed
from dentistry.safety.clinical_guardrails import ClinicalGuardrails, DISCLAIMER_TEXT
from dentistry.monitoring.usage_tracker import UsageTracker
from dentistry.monitoring.agent_logger import AgentExecutionLogger

# ---------------------------------------------------------------------------
# Fixtures and Dummies
# ---------------------------------------------------------------------------

class MockDetector:
    def __init__(self, detections=None):
        self.detections = detections if detections is not None else [
            {"bbox": [10.0, 10.0, 60.0, 60.0], "detector_class": "caries", "detector_confidence": 0.85}
        ]
        self.confidence = 0.075

    def predict(self, image):
        return self.detections

class MockClassifier:
    def __init__(self, pred_class="caries", confidence=0.88):
        self.pred_class = pred_class
        self.confidence = confidence
        self.classes = ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]

    def predict(self, crop):
        return {
            "class_name": self.pred_class,
            "confidence": self.confidence,
            "probabilities": {c: 0.1 for c in self.classes},
        }

    def preprocess(self, crop):
        import torch
        return torch.zeros((1, 3, 224, 224), dtype=torch.float32)

@pytest.fixture
def sample_xray():
    """Generates a synthetic high-texture image that passes the quality gate."""
    return np.random.default_rng(42).integers(30, 220, (300, 600, 3), dtype=np.uint8)

@pytest.fixture
def blurred_xray():
    """Generates an image that fails the quality gate due to low variance."""
    return np.full((300, 600, 3), 120, dtype=np.uint8)

# ---------------------------------------------------------------------------
# Test 1: Individual Agent Execution
# ---------------------------------------------------------------------------

def test_quality_agent_accepts_good_image(sample_xray):
    agent = ImageQualityAgent()
    res = agent.run(sample_xray)
    assert res["agent"] == "image_quality"
    assert res["acceptable"] is True
    assert res["quality_gate_passed"] is True

def test_quality_agent_rejects_blurred_image(blurred_xray):
    agent = ImageQualityAgent()
    res = agent.run(blurred_xray)
    assert res["acceptable"] is False
    assert res["quality_gate_passed"] is False
    assert any("blur" in w.lower() or "contrast" in w.lower() for w in res["warnings"])

def test_detection_agent_runs_calibrated_yolo(sample_xray):
    det = MockDetector([
        {"bbox": [20, 30, 80, 90], "detector_class": "impacted_tooth", "detector_confidence": 0.72}
    ])
    agent = DentalDetectionAgent(det, confidence_threshold=0.075)
    res = agent.run(sample_xray)
    assert res["status"] == "success"
    assert res["detection_count"] == 1
    assert res["regions"][0]["detector_class"] == "impacted_tooth"
    assert res["regions"][0]["detector_confidence"] == 0.72

def test_classification_agent_compares_and_flags_agreement(sample_xray):
    cls_model = MockClassifier(pred_class="caries", confidence=0.91)
    agent = DentalClassificationAgent(cls_model)
    regions = [
        {"region_id": "r1", "bbox": [10, 10, 50, 50], "detector_class": "caries", "detector_confidence": 0.8}
    ]
    res = agent.run(sample_xray, regions)
    assert res["status"] == "success"
    assert res["regions"][0]["classifier"]["class_name"] == "caries"
    assert res["regions"][0]["model_agreement"] is True

def test_classification_agent_flags_disagreement(sample_xray):
    cls_model = MockClassifier(pred_class="deep_caries", confidence=0.75)
    agent = DentalClassificationAgent(cls_model)
    regions = [
        {"region_id": "r1", "bbox": [10, 10, 50, 50], "detector_class": "caries", "detector_confidence": 0.8}
    ]
    res = agent.run(sample_xray, regions)
    assert res["regions"][0]["classifier"]["class_name"] == "deep_caries"
    assert res["regions"][0]["model_agreement"] is False

def test_clinical_validation_agent_routes_disagreement_and_low_confidence():
    agent = ClinicalValidationAgent(detector_threshold=0.25, classifier_threshold=0.65)
    regions = [
        {
            "region_id": "r1",
            "detector_class": "caries",
            "detector_confidence": 0.15,  # Low detector confidence
            "classifier": {"class_name": "deep_caries", "confidence": 0.50},  # Disagreement & low conf
            "errors": [],
        }
    ]
    res = agent.run({"regions": regions})
    assert res["review_required"] is True
    reasons = " ".join(res["review_reasons"]).lower()
    assert "low detector confidence" in reasons
    assert "model disagreement" in reasons
    assert "low classifier confidence" in reasons

def test_reporting_agent_schema_and_disclaimer():
    agent = ReportingAgent()
    rep = agent.run(
        image_name="test_opg.png",
        quality={"acceptable": True, "status": "acceptable", "warnings": []},
        detection={"status": "success", "regions": []},
        validation={"review_required": True, "review_reasons": ["Disclaimers"]},
        analysis_id="test_001",
    )
    assert rep["schema_version"] == "1.0"
    assert rep["analysis_id"] == "test_001"
    assert rep["educational_research_only"] is True
    assert rep["not_for_clinical_decisions"] is True
    assert DISCLAIMER_TEXT in rep["disclaimer"]

# ---------------------------------------------------------------------------
# Test 2: Tool Registry & Safe Dispatching
# ---------------------------------------------------------------------------

def test_tool_registry_registration_and_execution(sample_xray):
    registry = ToolRegistry()

    def mock_val(image_ref: str):
        return {"valid": True, "ref": image_ref}

    registry.register_tool(
        name="validate_xray_image",
        description="Validate image",
        input_model=ValidateImageInput,
        handler=mock_val,
    )

    tools = registry.get_openai_tools()
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "validate_xray_image"

    # Test safe execution
    res = registry.execute_tool("validate_xray_image", {"image_ref": "test.png"})
    assert res["status"] == "success"
    assert res["result"]["valid"] is True

    # Test unknown tool
    res_unknown = registry.execute_tool("nonexistent_tool", {})
    assert res_unknown["status"] == "error"

def test_tool_registry_argument_validation():
    registry = ToolRegistry()
    registry.register_tool(
        name="compare_model_predictions",
        description="Compare predictions",
        input_model=ComparePredictionsInput,
        handler=lambda **kwargs: kwargs,
    )
    # Missing required argument should fail gracefully
    res = registry.execute_tool("compare_model_predictions", {"region_id": "r1"})
    assert res["status"] == "error"

# ---------------------------------------------------------------------------
# Test 3: Privacy, Safety & Guardrails
# ---------------------------------------------------------------------------

def test_privacy_scrubbing_redacts_phi():
    raw_metadata = {
        "patient_name": "John Doe",
        "ssn": "123-45-6789",
        "notes": "Patient ID: 987654 visited on 12/04/2023 with email test@clinic.org",
        "detector_confidence": 0.88,
    }
    clean = sanitize_metadata(raw_metadata)
    assert "[REDACTED_PHI]" in clean["ssn"]
    assert "[REDACTED_PHI]" in clean["notes"]
    assert clean["detector_confidence"] == 0.88

def test_never_upload_raw_image_to_llm():
    assert check_image_transmission_allowed() is False

def test_absence_of_detection_does_not_mean_healthy():
    guardrails = ClinicalGuardrails()
    res = guardrails.validate_screening_output(
        detection_result={"regions": []},
        quality_result={"acceptable": True, "status": "acceptable"},
    )
    assert res["human_review_required"] is True
    assert any("absence of detection is not proof of health" in r.lower() for r in res["review_reasons"])

# ---------------------------------------------------------------------------
# Test 4: Offline Inference Fallback & Error Recovery
# ---------------------------------------------------------------------------

def test_offline_mode_when_api_disabled(sample_xray, tmp_path):
    client = OpenAIOrchestrationClient(enabled=False)
    assert client.is_available() is False

    engine = MasterOrchestratorAgent(
        detector=MockDetector(),
        classifier=MockClassifier(),
        reports_dir=tmp_path / "reports",
        gradcam_dir=tmp_path / "cam",
        llm_client=client,
    )

    report = engine.run(sample_xray, image_name="offline_test.png", use_llm_if_available=False)
    assert report["workflow_status"] in ("completed", "needs_review")
    assert report["detection_count"] == 1
    assert "token_usage" in report
    assert report["token_usage"]["api_calls"] == 0

def test_quality_gate_stops_pipeline(blurred_xray, tmp_path):
    client = OpenAIOrchestrationClient(enabled=False)
    engine = MasterOrchestratorAgent(
        detector=MockDetector(),
        classifier=MockClassifier(),
        reports_dir=tmp_path / "reports",
        llm_client=client,
    )
    report = engine.run(blurred_xray, image_name="blurred.png")
    assert report["image_quality"]["acceptable"] is False
    assert report["human_review_required"] is True
    assert report["execution_order"] == ["image_quality", "validation", "reporting"]
    # Check that detection was not run on rejected image
    assert report["detection_count"] == 0

# ---------------------------------------------------------------------------
# Test 5: Usage & Latency Tracking
# ---------------------------------------------------------------------------

def test_usage_tracker_token_and_cost_calculation():
    tracker = UsageTracker()
    rec = tracker.record_usage(
        model="gpt-6.1-sol",
        prompt_tokens=1000,
        completion_tokens=500,
        latency_ms=120.0,
        agent_name="orchestrator",
    )
    assert rec["total_tokens"] == 1500
    assert rec["estimated_cost_usd"] > 0.0

    summary = tracker.get_summary("gpt-6.1-sol")
    assert summary["api_calls"] == 1
    assert summary["prompt_tokens"] == 1000
    assert summary["completion_tokens"] == 500
    assert summary["total_estimated_cost_usd"] == rec["estimated_cost_usd"]

def test_agent_logger_timings():
    logger = AgentExecutionLogger()
    logger.start_agent("quality_agent")
    dur = logger.complete_agent("quality_agent", "success")
    assert dur >= 0.0
    summary = logger.get_summary()
    assert "quality_agent" in summary["agent_statuses"]
    assert summary["agent_statuses"]["quality_agent"] == "success"

# ---------------------------------------------------------------------------
# Test 6: PS-01 Four-Agent Architecture & Tool Suite
# ---------------------------------------------------------------------------

def test_detection_diagnosis_agent_ps01(sample_xray, tmp_path):
    from dentistry.agents.detection_diagnosis_agent import DetectionDiagnosisAgent
    det = MockDetector([
        {"bbox": [20, 20, 70, 70], "detector_class": "caries", "detector_confidence": 0.88}
    ])
    cls_model = MockClassifier(pred_class="deep_caries", confidence=0.74)
    agent = DetectionDiagnosisAgent(det, cls_model, gradcam_dir=tmp_path / "cam")
    res = agent.run(sample_xray, image_id="test_run")
    assert res["status"] == "success"
    assert res["detection_count"] == 1
    assert "bounding_boxes" in res
    assert "classification_results" in res
    assert "class_probabilities" in res
    assert "gradcam_references" in res
    assert res["model_disagreement"] is True  # caries != deep_caries

def test_four_agent_orchestrator_ps01(sample_xray, tmp_path):
    from dentistry.agents.orchestrator import FourAgentOrchestrator
    orchestrator = FourAgentOrchestrator(
        detector=MockDetector(),
        classifier=MockClassifier(),
        reports_dir=tmp_path / "reports",
        gradcam_dir=tmp_path / "cam",
        llm_client=OpenAIOrchestrationClient(enabled=False),
    )
    report = orchestrator.run(sample_xray, image_name="test_4agent.png", save=True)
    assert report["workflow_status"] in ("completed", "needs_review")
    assert report["image_quality"]["quality_status"] == "acceptable"
    assert report["execution_order"] == ["image_quality", "detection_diagnosis", "validation", "reporting"]
    assert "model_versions" in report
    assert "processing_time" in report

def test_ps01_eight_tool_registry_names():
    from dentistry.llm.tool_registry import build_default_tool_registry
    registry = build_default_tool_registry(
        context_data={},
        quality_fn=lambda: {"acceptable": True},
        detection_fn=lambda conf: {"status": "success", "regions": []},
        classification_fn=lambda rid, box: {"prediction": {}},
        gradcam_fn=lambda rid: {"gradcam_path": None},
        validation_fn=lambda findings: {"human_review_required": False},
        reporting_fn=lambda aid, notes=None: {"analysis_id": aid},
    )
    tools = registry.get_openai_tools()
    tool_names = [t["function"]["name"] for t in tools]
    required_ps01_tools = [
        "check_image_quality",
        "execute_yolo_detection",
        "extract_tooth_crops",
        "classify_tooth_regions",
        "generate_gradcam_heatmap",
        "validate_predictions",
        "route_for_human_review",
        "create_screening_report",
    ]
    for req in required_ps01_tools:
        assert req in tool_names, f"PS-01 tool '{req}' missing from ToolRegistry"

def test_services_and_rules():
    from dentistry.models.detector_service import YoloDetectorService
    from dentistry.models.classifier_service import EfficientNetClassifierService
    from dentistry.explainability.gradcam_service import GradCAMService
    from dentistry.safety.validation_rules import ValidationRules

    rules = ValidationRules(0.25, 0.65)
    triage = rules.evaluate_triage({"regions": []}, {"acceptable": True})
    assert triage["human_review_required"] is True

    gcam = GradCAMService()
    assert gcam.output_dir.exists()

