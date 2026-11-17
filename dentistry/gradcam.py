"""Grad-CAM for torchvision-style classifiers."""
from pathlib import Path
import cv2, numpy as np

def generate_gradcam(model, input_tensor, target_layer, original_bgr: np.ndarray | None=None, output_path: str|Path|None=None):
    import torch
    activations=[]; gradients=[]
    def forward_hook(_m,_i,out): activations.append(out)
    def backward_hook(_m,_gi,go): gradients.append(go[0])
    h1=target_layer.register_forward_hook(forward_hook); h2=target_layer.register_full_backward_hook(backward_hook)
    try:
        model.eval(); model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.detach().requires_grad_(True)
        logits=model(input_tensor); target=int(logits.argmax(dim=1).item()); logits[0,target].backward()
        if not activations or not gradients: raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
        weights=gradients[0].mean(dim=(2,3),keepdim=True)
        cam=torch.relu((weights*activations[0]).sum(dim=1,keepdim=True))
        cam=torch.nn.functional.interpolate(cam,size=input_tensor.shape[-2:],mode="bilinear",align_corners=False)[0,0]
        cam=cam.detach().cpu().numpy(); cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
        heat=np.uint8(255*cam); heat=cv2.applyColorMap(heat,cv2.COLORMAP_JET)
        if original_bgr is not None:
            heat=cv2.resize(heat,(original_bgr.shape[1],original_bgr.shape[0])); overlay=cv2.addWeighted(original_bgr,0.58,heat,0.42,0)
        else: overlay=heat
        if output_path:
            p=Path(output_path); p.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(p),overlay): raise OSError(f"Could not write Grad-CAM image: {p}")
        return {"target_class_id":target,"heatmap":cam,"overlay_bgr":overlay,"output_path":str(output_path) if output_path else None}
    finally:
        h1.remove(); h2.remove()
