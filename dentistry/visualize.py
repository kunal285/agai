import cv2, numpy as np

def draw_detections(image: np.ndarray, regions: list[dict]) -> np.ndarray:
    out=image.copy()
    for r in regions:
        box=r.get("bbox")
        if not box or len(box)!=4: continue
        x1,y1,x2,y2=map(int,box); label=r.get("classifier",{}).get("class_name") or r.get("detector_class","region")
        conf=r.get("classifier",{}).get("confidence",r.get("detector_confidence",0))
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,220,80),2)
        cv2.putText(out,f"{label} {float(conf):.2f}",(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,80),2,cv2.LINE_AA)
    return out
