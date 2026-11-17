"""Optional model adapters. Missing checkpoints never trigger fake predictions."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import cv2, numpy as np
from PIL import Image

class ModelUnavailable(RuntimeError): pass

class YoloDetector:
    def __init__(self, checkpoint: str | Path, confidence: float=0.25, image_size: int=640):
        self.checkpoint=Path(checkpoint); self.confidence=confidence; self.image_size=image_size
        if not self.checkpoint.exists(): raise ModelUnavailable(f"Detector checkpoint not found: {self.checkpoint}. Train the detector first.")
        try:
            from ultralytics import YOLO
        except ImportError as e: raise ModelUnavailable("Ultralytics is not installed. Install with: pip install -e '.[models]'") from e
        self.model=YOLO(str(self.checkpoint))
    def predict(self, image: np.ndarray) -> list[dict[str,Any]]:
        result=self.model.predict(source=image,conf=self.confidence,imgsz=self.image_size,verbose=False)[0]
        output=[]
        if result.boxes is None: return output
        names=result.names
        for b in result.boxes:
            xyxy=b.xyxy[0].detach().cpu().numpy().tolist()
            cls=int(b.cls[0].item()); conf=float(b.conf[0].item())
            output.append({"bbox":[float(x) for x in xyxy],"detector_class_id":cls,"detector_class":str(names.get(cls,cls)),"detector_confidence":conf})
        return output

class TorchvisionCropClassifier:
    def __init__(self, checkpoint: str | Path, classes: list[str], architecture: str="efficientnet_b0", image_size: int=224, device: str|None=None):
        self.checkpoint=Path(checkpoint); self.classes=classes; self.architecture=architecture; self.image_size=image_size
        if not self.checkpoint.exists(): raise ModelUnavailable(f"Classifier checkpoint not found: {self.checkpoint}. Train the classifier first.")
        try:
            import torch
            from torchvision import models
        except Exception as e: raise ModelUnavailable(f"PyTorch/torchvision unavailable: {e}") from e
        self.torch=torch; self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        ctor=getattr(models,architecture,None)
        if ctor is None: raise ValueError(f"Unsupported torchvision architecture: {architecture}")
        self.model=ctor(weights=None)
        if hasattr(self.model,"classifier"):
            layer=self.model.classifier[-1]
            self.model.classifier[-1]=torch.nn.Linear(layer.in_features,len(classes))
        elif hasattr(self.model,"fc"):
            self.model.fc=torch.nn.Linear(self.model.fc.in_features,len(classes))
        else: raise ValueError("Could not identify classifier head for architecture")
        checkpoint_data=torch.load(self.checkpoint,map_location=self.device,weights_only=False)
        if isinstance(checkpoint_data,dict) and "classes" in checkpoint_data:
            saved_classes=list(checkpoint_data["classes"])
            if list(classes) != saved_classes:
                raise ValueError(f"Classifier class mapping mismatch. Checkpoint classes={saved_classes}; configured classes={list(classes)}. Use the checkpoint's saved order.")
        state=checkpoint_data.get("model_state_dict",checkpoint_data) if isinstance(checkpoint_data,dict) else checkpoint_data
        self.model.load_state_dict(state); self.model.to(self.device).eval()
        self.mean=np.array([0.485,0.456,0.406],dtype=np.float32); self.std=np.array([0.229,0.224,0.225],dtype=np.float32)
    def preprocess(self, crop: np.ndarray):
        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB); rgb=cv2.resize(rgb,(self.image_size,self.image_size))
        x=rgb.astype(np.float32)/255.0; x=(x-self.mean)/self.std
        return self.torch.from_numpy(x.transpose(2,0,1).copy()).float().unsqueeze(0).to(self.device)
    def predict(self, crop: np.ndarray) -> dict[str,Any]:
        x=self.preprocess(crop)
        with self.torch.inference_mode(): probs=self.torch.softmax(self.model(x),dim=1)[0].cpu().numpy()
        idx=int(np.argmax(probs))
        return {"class_name":self.classes[idx],"class_id":idx,"confidence":float(probs[idx]),"probabilities":{c:float(p) for c,p in zip(self.classes,probs)}}

def crop_bbox(image: np.ndarray, bbox: list[float], padding: float=0.04) -> tuple[np.ndarray,tuple[int,int,int,int]]:
    h,w=image.shape[:2]; x1,y1,x2,y2=map(float,bbox)
    if not np.all(np.isfinite([x1,y1,x2,y2])): raise ValueError("Bounding box contains non-finite coordinates")
    if x2<=x1 or y2<=y1: raise ValueError("Bounding box has non-positive area")
    px=(x2-x1)*padding; py=(y2-y1)*padding
    a=max(0,int(x1-px)); b=max(0,int(y1-py)); c=min(w,int(x2+px)); d=min(h,int(y2+py))
    if c<=a or d<=b: raise ValueError("Bounding box lies outside the image")
    return image[b:d,a:c].copy(),(a,b,c,d)
