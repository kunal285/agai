import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
import argparse, yaml
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--dataset-root",required=True); p.add_argument("--classes",nargs="+",required=True); p.add_argument("--output",default="data/processed/dentex.yaml"); a=p.parse_args()
root=Path(a.dataset_root).resolve()
# Expected YOLO layout: images/{train,val,test}, labels/{train,val,test}; create only config, not fake files.
data={"path":str(root),"train":"images/train","val":"images/val","test":"images/test","names":{i:n for i,n in enumerate(a.classes)}}
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(yaml.safe_dump(data,sort_keys=False),encoding="utf-8"); print(f"Wrote {out}; verify the referenced split directories and labels exist.")
