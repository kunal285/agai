"""Image quality measurements; heuristic flags are not clinical judgements."""
from pathlib import Path
from typing import Any
import cv2
import numpy as np

def assess_image(image: np.ndarray, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg=(config or {}).get("quality", {})
    if image is None or not isinstance(image,np.ndarray) or image.size == 0:
        return {"status":"poor","acceptable":False,"warnings":["Image is empty or unreadable"],"metrics":{}}
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    h,w=gray.shape[:2]
    metrics={"width":int(w),"height":int(h),"brightness_mean":float(np.mean(gray)),"contrast_std":float(np.std(gray)),"laplacian_variance":float(cv2.Laplacian(gray,cv2.CV_64F).var())}
    warnings=[]
    if w<int(cfg.get("min_width",256)) or h<int(cfg.get("min_height",128)): warnings.append("Image dimensions are below configured minimum")
    if metrics["contrast_std"]<float(cfg.get("min_contrast_std",8)): warnings.append("Low contrast detected")
    if metrics["laplacian_variance"]<float(cfg.get("min_laplacian_variance",10)): warnings.append("Possible blur detected")
    if not float(cfg.get("min_brightness",5))<=metrics["brightness_mean"]<=float(cfg.get("max_brightness",250)): warnings.append("Brightness is outside configured range")
    return {"status":"poor" if warnings else "acceptable","acceptable":not warnings,"warnings":warnings,"metrics":metrics}

def read_image(source: str | Path | bytes | np.ndarray) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, bytes):
        arr=np.frombuffer(source,dtype=np.uint8); image=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    else: image=cv2.imread(str(source),cv2.IMREAD_COLOR)
    if image is None: raise ValueError("Unsupported, corrupted, or unreadable image")
    return image
