from datetime import datetime, timezone
class ReportingAgent:
    name="reporting"
    def run(self, image_name, quality, detection, validation, workflow_status="completed"):
        regions=[]
        for r in validation.get("regions",[]):
            pred=r.get("classifier") or {}
            regions.append({"region_id":r.get("region_id"),"bbox_xyxy":r.get("bbox"),"crop_bbox_xyxy":r.get("crop_bbox_xyxy"),
                "detector_class":r.get("detector_class"),"detector_confidence":r.get("detector_confidence"),
                "predicted_class":pred.get("class_name"),"classifier_confidence":pred.get("confidence"),
                "class_scores":pred.get("probabilities"),"review_required":bool(r.get("review_required",True)),
                "review_reasons":r.get("review_reasons",[]),"gradcam_path":r.get("gradcam_path"),"warnings":r.get("errors",[])})
        return {"schema_version":"1.0","created_at_utc":datetime.now(timezone.utc).isoformat(),"image_name":image_name,
            "workflow_status":workflow_status,"educational_research_only":True,"not_for_clinical_decisions":True,
            "image_quality":quality,"detection_count":len(regions),"human_review_required":bool(validation.get("review_required",True)),
            "regions":regions,"warnings":list(quality.get("warnings",[])),"agent_status":{"image_quality":"completed","detection_diagnosis":detection.get("status","unknown"),"validation":"completed","reporting":"completed"},
            "disclaimer":"Research prototype only. Outputs are not a clinical diagnosis and must not guide treatment."}
