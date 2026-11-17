class ValidationAgent:
    name="validation"
    def __init__(self, detector_threshold: float=0.25, classifier_threshold: float=0.65):
        self.detector_threshold=detector_threshold; self.classifier_threshold=classifier_threshold
    def run(self, detection_result):
        review=[]
        for r in detection_result.get("regions",[]):
            reasons=[]
            if float(r.get("detector_confidence",0)) < self.detector_threshold: reasons.append("detector confidence below review threshold")
            pred=r.get("classifier")
            if pred is None: reasons.append("classifier unavailable; diagnosis confirmation missing")
            elif float(pred.get("confidence",0)) < self.classifier_threshold: reasons.append("classifier confidence below review threshold")
            if r.get("errors"): reasons.extend(r["errors"])
            r["review_required"]=bool(reasons); r["review_reasons"]=reasons
            if reasons: review.append(r["region_id"])
        if not detection_result.get("regions"): review.append("no_detections")
        return {"agent":self.name,"review_required":bool(review),"flagged_regions":review,"regions":detection_result.get("regions",[])}
