"""Deterministic four-agent orchestration; no LLM-generated diagnoses."""
from __future__ import annotations
import json, logging
from pathlib import Path
from uuid import uuid4
from dentistry.quality import read_image
from dentistry.agents.quality_agent import ImageQualityAgent
from dentistry.agents.detection_agent import DetectionDiagnosisAgent
from dentistry.agents.validation_agent import ValidationAgent
from dentistry.agents.reporting_agent import ReportingAgent

log=logging.getLogger(__name__)
class ScreeningOrchestrator:
    def __init__(self, detector, classifier=None, config=None, reports_dir="artifacts/reports", gradcam_dir="artifacts/gradcam"):
        self.config=config or {}; self.reports_dir=Path(reports_dir); self.reports_dir.mkdir(parents=True,exist_ok=True)
        self.quality_agent=ImageQualityAgent(); self.detection_agent=DetectionDiagnosisAgent(detector,classifier,gradcam_dir)
        m=self.config.get("model",{})
        self.validation_agent=ValidationAgent(float(m.get("detector_confidence",0.25)),float(m.get("classifier_confidence_threshold",0.65)))
        self.reporting_agent=ReportingAgent()
    def run(self, source, image_name=None, save=True):
        run_id=uuid4().hex[:12]; image=read_image(source)
        name=image_name or (Path(source).name if isinstance(source,(str,Path)) else f"upload_{run_id}.png")
        statuses=[]
        quality=self.quality_agent.run(image,self.config); statuses.append("image_quality")
        if not quality["acceptable"]:
            report=self.reporting_agent.run(name,quality,{"status":"skipped_quality_gate","regions":[]},{"review_required":True,"regions":[]},"needs_review")
            report["agent_status"]["detection_diagnosis"]="skipped_due_to_quality"; report["execution_order"]=statuses+["validation","reporting"]
        else:
            try: detection=self.detection_agent.run(image,run_id); statuses.append("detection_diagnosis")
            except Exception as e:
                log.exception("Detection/diagnosis agent failed")
                detection={"status":"error","error":str(e),"regions":[]}
            validation=self.validation_agent.run(detection); statuses.append("validation")
            report=self.reporting_agent.run(name,quality,detection,validation,"completed" if detection.get("status")=="success" else "needs_review")
            if detection.get("error"): report["warnings"].append(f"Detection agent error: {detection['error']}")
            report["execution_order"]=statuses+["reporting"]
        report["run_id"]=run_id
        if save:
            path=self.reports_dir/f"{run_id}.json"; path.write_text(json.dumps(report,indent=2),encoding="utf-8"); report["report_path"]=str(path)
        return report
