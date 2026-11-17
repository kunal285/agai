from __future__ import annotations
from pathlib import Path
import cv2
from dentistry.models import crop_bbox, ModelUnavailable
from dentistry.gradcam import generate_gradcam

class DetectionDiagnosisAgent:
    name="detection_diagnosis"
    def __init__(self, detector, classifier=None, gradcam_dir: str|Path="artifacts/gradcam"):
        self.detector=detector; self.classifier=classifier; self.gradcam_dir=Path(gradcam_dir)
    def run(self,image, image_id="upload"):
        detections=self.detector.predict(image); regions=[]
        for i,d in enumerate(detections):
            region={"region_id":f"region_{i+1}",**d,"classifier":None,"gradcam_path":None,"errors":[]}
            try:
                crop,coords=crop_bbox(image,d["bbox"]); region["crop_bbox_xyxy"]=list(coords)
                if self.classifier:
                    pred=self.classifier.predict(crop); region["classifier"]=pred
                    try:
                        model=self.classifier.model
                        target_layer=None
                        if hasattr(model,"features"): target_layer=model.features[-1]
                        elif hasattr(model,"layer4"): target_layer=model.layer4[-1]
                        if target_layer is not None:
                            tensor=self.classifier.preprocess(crop)
                            path=self.gradcam_dir/f"{image_id}_{i+1}.jpg"
                            cam=generate_gradcam(model,tensor,target_layer,crop,path)
                            region["gradcam_path"]=cam["output_path"]
                    except Exception as e: region["errors"].append(f"Grad-CAM unavailable: {e}")
            except Exception as e: region["errors"].append(f"Region processing failed: {e}")
            regions.append(region)
        return {"agent":self.name,"status":"success","detection_count":len(detections),"regions":regions}
