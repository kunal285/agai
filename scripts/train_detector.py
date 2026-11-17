import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
import argparse, json
from dentistry.training import train_yolo
p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--output",default="models/detector"); p.add_argument("--base-model",default="yolov8n.pt"); p.add_argument("--epochs",type=int,default=50); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--batch",type=int,default=8); p.add_argument("--device",default=None); p.add_argument("--resume",action="store_true"); a=p.parse_args()
print(json.dumps(train_yolo(a.data,a.output,a.base_model,a.epochs,a.imgsz,a.batch,a.device,a.resume),indent=2))
