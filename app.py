"""Streamlit dashboard. Real inference is enabled only when checkpoints and dependencies exist."""
import json
from pathlib import Path
import cv2, numpy as np
import streamlit as st
from PIL import Image
from dentistry.config import load_config
from dentistry.models import YoloDetector, TorchvisionCropClassifier, ModelUnavailable
from dentistry.orchestrator import ScreeningOrchestrator
from dentistry.visualize import draw_detections

st.set_page_config(page_title="AI Dentistry Screening",page_icon="🦷",layout="wide")
st.title("🦷 AI-Assisted Dental X-ray Screening")
st.caption("Educational research prototype • Not a clinical diagnostic or treatment system")
cfg=load_config(); upload=st.file_uploader("Upload a panoramic dental X-ray",type=["png","jpg","jpeg","tif","tiff","bmp"])
with st.sidebar:
    st.header("Model status")
    detector_path=Path(cfg["paths"]["detector_checkpoint"]); classifier_path=Path(cfg["paths"]["classifier_checkpoint"])
    st.write("YOLO checkpoint:","available" if detector_path.exists() else "missing — training required")
    st.write("CNN checkpoint:","available" if classifier_path.exists() else "missing — training required")
    st.warning("Do not use this prototype for clinical decisions.")
if upload:
    try:
        raw=upload.getvalue(); arr=np.frombuffer(raw,np.uint8); bgr=cv2.imdecode(arr,cv2.IMREAD_COLOR)
        if bgr is None: st.error("This file could not be decoded as an image."); st.stop()
        st.image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),caption="Uploaded X-ray",use_container_width=True)
        if st.button("Run screening workflow",type="primary"):
            try:
                detector=YoloDetector(detector_path,cfg["model"]["detector_confidence"],cfg["model"]["detector_imgsz"])
                import torch
                checkpoint_meta=torch.load(classifier_path,map_location="cpu",weights_only=False)
                classes=checkpoint_meta.get("classes",cfg.get("classes",[])) if isinstance(checkpoint_meta,dict) else cfg.get("classes",[])
                architecture=checkpoint_meta.get("architecture",cfg["model"]["classifier_architecture"]) if isinstance(checkpoint_meta,dict) else cfg["model"]["classifier_architecture"]
                image_size=int(checkpoint_meta.get("image_size",cfg["model"]["classifier_image_size"])) if isinstance(checkpoint_meta,dict) else cfg["model"]["classifier_image_size"]
                classifier=TorchvisionCropClassifier(classifier_path,classes,architecture,image_size)
                engine=ScreeningOrchestrator(detector,classifier,cfg,cfg["paths"]["reports_dir"])
                report=engine.run(raw,image_name=upload.name,save=True)
                st.session_state["report"]=report; st.session_state["regions"]=report.get("regions",[])
            except ModelUnavailable as e: st.error(str(e))
            except Exception as e: st.exception(e)
    except Exception as e: st.error(f"Could not process upload: {e}")
if st.session_state.get("report"):
    report=st.session_state["report"]
    c1,c2,c3=st.columns(3); c1.metric("Regions detected",report.get("detection_count",0)); c2.metric("Image quality",report.get("image_quality",{}).get("status","unknown")); c3.metric("Human review", "Required" if report.get("human_review_required") else "Flag not raised")
    st.subheader("Agent workflow"); st.write(" → ".join(report.get("execution_order",[])))
    st.subheader("Structured report"); st.json(report)
    st.download_button("Download JSON report",json.dumps(report,indent=2),file_name="dental_screening_report.json",mime="application/json")
    for region in report.get("regions",[]):
        with st.expander(f"{region['region_id']} — {region.get('predicted_class') or 'unconfirmed'}"):
            st.write(region)
            path=region.get("gradcam_path")
            if path and Path(path).exists(): st.image(path,caption="Grad-CAM sensitivity visualization; not clinical proof")
else:
    st.info("Upload an image and run screening after both trained checkpoints are installed. This interface does not generate sample or fabricated predictions.")
