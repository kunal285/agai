import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
import argparse, json
from dentistry.training import train_classifier
p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--output",default="models/classifier"); p.add_argument("--architecture",default="efficientnet_b0"); p.add_argument("--image-size",type=int,default=224); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch",type=int,default=16); p.add_argument("--lr",type=float,default=3e-4); a=p.parse_args()
print(json.dumps(train_classifier(a.manifest,a.output,a.architecture,a.image_size,a.epochs,a.batch,a.lr),indent=2))
