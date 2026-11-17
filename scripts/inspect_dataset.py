import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
import argparse, json
from pathlib import Path
from dentistry.config import load_config
from dentistry.dataset import inventory_dataset, find_annotation_files, parse_coco_annotations, validate_coco
p=argparse.ArgumentParser(); p.add_argument("--root",default=None); p.add_argument("--output",default="artifacts/dataset_inventory.json"); args=p.parse_args()
cfg=load_config(); root=Path(args.root or cfg["paths"]["dataset_root"]); result=inventory_dataset(root); result["annotation_summaries"]=[]
for ann in find_annotation_files(root):
    try: result["annotation_summaries"].append({"file":str(ann),**validate_coco(parse_coco_annotations(ann,root))})
    except Exception as e: result["annotation_summaries"].append({"file":str(ann),"error":str(e)})
out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps({k:v for k,v in result.items() if k not in ("images",)},indent=2))
