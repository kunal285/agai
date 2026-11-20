"""Agentic AI Orchestrator coordinating the 4 functional agents.

Workflow:
1. Image Quality Agent -> Checks blur, contrast, dimensions. Rejects if poor.
2. Detection & Diagnosis Agent -> Runs YOLOv8, extracts crops, runs EfficientNet-B0, generates Grad-CAM.
3. Validation Agent -> Validates confidence, detects model disagreements, routes uncertain cases.
4. Reporting Agent -> Compiles tooth-wise structured report and AI synthesis.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from uuid import uuid4
from typing import Any
import numpy as np

from dentistry.quality import read_image
from dentistry.agents.quality_agent import ImageQualityAgent
from dentistry.agents.detection_diagnosis_agent import DetectionDiagnosisAgent
from dentistry.agents.validation_agent import ValidationAgent
from dentistry.agents.reporting_agent import ReportingAgent
from dentistry.monitoring.agent_logger import AgentExecutionLogger
from dentistry.monitoring.usage_tracker import UsageTracker
from dentistry.llm.openai_client import OpenAIOrchestrationClient
from dentistry.llm.tool_registry import build_default_tool_registry
from dentistry.llm.prompts import ORCHESTRATOR_SYSTEM_PROMPT

log = logging.getLogger(__name__)

class FourAgentOrchestrator:
    name = "four_agent_orchestrator"

    def __init__(
        self,
        detector,
        classifier=None,
        config: dict[str, Any] | None = None,
        reports_dir: str | Path = "artifacts/reports",
        gradcam_dir: str | Path = "artifacts/gradcam",
        llm_client: OpenAIOrchestrationClient | None = None,
    ):
        self.config = config or {}
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.gradcam_dir = Path(gradcam_dir)
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)

        self.detector = detector
        self.classifier = classifier

        m = self.config.get("model", {})
        det_conf = float(m.get("detector", {}).get("confidence_threshold", 0.075))
        det_review = float(m.get("detector", {}).get("review_confidence_threshold", 0.25))
        cls_review = float(m.get("classifier", {}).get("confidence_threshold", 0.65))

        # Four Primary Functional Agents
        self.quality_agent = ImageQualityAgent(self.config)
        self.detection_diagnosis_agent = DetectionDiagnosisAgent(
            detector=detector,
            classifier=classifier,
            gradcam_dir=self.gradcam_dir,
            confidence_threshold=det_conf,
        )
        self.validation_agent = ValidationAgent(
            detector_threshold=det_review,
            classifier_threshold=cls_review,
        )
        self.reporting_agent = ReportingAgent()

        # Monitoring and centralized LLM client
        self.logger = AgentExecutionLogger()
        self.usage_tracker = UsageTracker()
        self.llm_client = llm_client or OpenAIOrchestrationClient(
            usage_tracker=self.usage_tracker,
            agent_logger=self.logger,
        )

    def run(
        self,
        source: str | Path | bytes | np.ndarray,
        image_name: str | None = None,
        save: bool = True,
        use_llm_if_available: bool = True,
    ) -> dict[str, Any]:
        """Runs the four-agent screening workflow with OpenAI tool-calling or deterministic fallback."""
        run_id = uuid4().hex[:12]
        image = read_image(source)
        name = image_name or (Path(source).name if isinstance(source, (str, Path)) else f"upload_{run_id}.png")

        self.logger.start_agent(self.name)

        state: dict[str, Any] = {
            "run_id": run_id,
            "image_name": name,
            "image": image,
            "quality": None,
            "detections": {"regions": [], "detection_count": 0, "status": "pending"},
            "validation": None,
            "report": None,
            "ai_synthesis": None,
        }

        # Attempt OpenAI Agentic Tool-calling if enabled and active
        if use_llm_if_available and self.llm_client.is_available():
            report = self._run_agentic_llm_workflow(state, run_id, name)
        else:
            report = self._run_deterministic_workflow(state, run_id, name)

        self.logger.complete_agent(self.name, status=report.get("workflow_status", "completed"))

        report["execution_trace"] = self.logger.get_summary()
        report["token_usage"] = self.usage_tracker.get_summary(self.llm_client.model)
        report["run_id"] = run_id
        if not report.get("image_quality", {}).get("acceptable", True):
            report["execution_order"] = ["image_quality", "validation", "reporting"]
        else:
            report["execution_order"] = ["image_quality", "detection_diagnosis", "validation", "reporting"]

        if save:
            path = self.reports_dir / f"{run_id}.json"
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            report["report_path"] = str(path)

        return report

    def _run_deterministic_workflow(self, state: dict[str, Any], run_id: str, name: str) -> dict[str, Any]:
        """Deterministic 4-agent state machine."""
        image = state["image"]

        # Step 1: Agent 1 - Image Quality Agent
        self.logger.start_agent(self.quality_agent.name)
        try:
            quality = self.quality_agent.run(image, self.config)
            self.logger.complete_agent(self.quality_agent.name, "success")
        except Exception as e:
            self.logger.fail_agent(self.quality_agent.name, str(e))
            quality = {"acceptable": False, "status": "poor", "warnings": [f"Quality agent failure: {e}"]}
        state["quality"] = quality

        # Check Image Quality Gate
        if not quality.get("acceptable", False):
            # Short-circuit: Quality failed. Stop deep learning inference.
            self.logger.start_agent(self.validation_agent.name)
            validation = self.validation_agent.run({"regions": []}, quality)
            self.logger.complete_agent(self.validation_agent.name, "needs_review")

            self.logger.start_agent(self.reporting_agent.name)
            report = self.reporting_agent.run(
                image_name=name,
                quality=quality,
                detection={"status": "skipped_quality_gate", "regions": []},
                validation=validation,
                workflow_status="needs_review",
                analysis_id=run_id,
                ai_synthesis="Automated screening stopped at Image Quality Gate. Scan was rejected due to quality heuristics.",
            )
            self.logger.complete_agent(self.reporting_agent.name, "completed")
            return report

        # Step 2: Agent 2 - Detection and Diagnosis Agent
        self.logger.start_agent(self.detection_diagnosis_agent.name)
        try:
            detection = self.detection_diagnosis_agent.run(image, image_id=run_id)
            self.logger.complete_agent(self.detection_diagnosis_agent.name, "success")
        except Exception as e:
            self.logger.fail_agent(self.detection_diagnosis_agent.name, str(e))
            detection = {"status": "error", "error": str(e), "regions": []}
        state["detections"] = detection

        regions = detection.get("regions", [])

        # Step 3: Agent 3 - Validation Agent
        self.logger.start_agent(self.validation_agent.name)
        try:
            validation = self.validation_agent.run(detection, quality)
            self.logger.complete_agent(self.validation_agent.name, "success")
        except Exception as e:
            self.logger.fail_agent(self.validation_agent.name, str(e))
            validation = {"review_required": True, "review_reasons": [f"Validation error: {e}"], "regions": regions}
        state["validation"] = validation

        # Step 4: AI Synthesis Generation
        ai_synthesis = self.llm_client.generate_clinical_synthesis(
            analysis_id=run_id,
            quality_data=quality,
            findings_data=regions,
            validation_data=validation,
        )

        # Step 5: Agent 4 - Reporting Agent
        self.logger.start_agent(self.reporting_agent.name)
        wf_status = "completed" if not validation.get("review_required") else "needs_review"
        report = self.reporting_agent.run(
            image_name=name,
            quality=quality,
            detection=detection,
            validation=validation,
            workflow_status=wf_status,
            analysis_id=run_id,
            ai_synthesis=ai_synthesis,
        )
        self.logger.complete_agent(self.reporting_agent.name, "completed")
        return report

    def _run_agentic_llm_workflow(self, state: dict[str, Any], run_id: str, name: str) -> dict[str, Any]:
        """Multi-turn tool-calling orchestration with OpenAI API."""
        image = state["image"]

        def q_fn():
            res = self.quality_agent.run(image, self.config)
            state["quality"] = res
            return res

        def d_fn(conf):
            res = self.detection_diagnosis_agent.run(image, image_id=run_id, confidence_threshold=conf)
            state["detections"] = res
            return res

        def c_fn(reg_id, bbox):
            if self.classifier:
                crop, coords = crop_bbox(image, bbox, padding=0.10)
                pred = self.classifier.predict(crop)
                return {"region_id": reg_id, "prediction": pred, "crop_coords": list(coords)}
            return {"error": "Classifier unavailable"}

        def g_fn(reg_id):
            regs = [r for r in state["detections"].get("regions", []) if r.get("region_id") == reg_id]
            if regs:
                return {"region_id": reg_id, "gradcam_path": regs[0].get("gradcam_path")}
            return {"error": f"Region {reg_id} not found"}

        def v_fn(findings):
            res = self.validation_agent.run({"regions": findings}, state.get("quality"))
            state["validation"] = res
            return res

        def r_fn(analysis_id, notes=None):
            return {"status": "ready"}

        registry = build_default_tool_registry(
            context_data=state,
            quality_fn=q_fn,
            detection_fn=d_fn,
            classification_fn=c_fn,
            gradcam_fn=g_fn,
            validation_fn=v_fn,
            reporting_fn=r_fn,
        )

        tools = registry.get_openai_tools()
        messages = [{
            "role": "user",
            "content": f"Begin screening analysis for radiograph '{name}'. Analysis ID: '{run_id}'. "
                       "Start by checking image quality.",
        }]

        max_turns = 12
        for _ in range(max_turns):
            response = self.llm_client.run_agent_turn(
                agent_name="four_agent_orchestrator",
                system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
            )

            if response["status"] != "success" or not response.get("tool_calls"):
                break

            for tc in response["tool_calls"]:
                tool_name = tc["name"]
                args = tc["arguments"]
                self.logger.start_agent(f"tool_{tool_name}")
                exec_result = registry.execute_tool(tool_name, args)
                self.logger.complete_agent(f"tool_{tool_name}", exec_result.get("status", "success"))

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tool_name, "arguments": args if isinstance(args, str) else json.dumps(args)},
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tool_name,
                    "content": json.dumps(exec_result),
                })

            if state.get("quality") and not state["quality"].get("acceptable"):
                break

        return self._run_deterministic_workflow(state, run_id, name)

# Aliases for backward compatibility
ScreeningOrchestrator = FourAgentOrchestrator
MasterOrchestratorAgent = FourAgentOrchestrator
