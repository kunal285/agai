import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
import argparse, json
from pathlib import Path
from dentistry.dataset import parse_coco_annotations, validate_coco, coco_to_yolo, extract_crops, DEFAULT_CLASSES
p=argparse.ArgumentParser(); p.add_argument("annotation_json"); p.add_argument("--image-root",default=None); p.add_argument("--yolo-out",default="data/processed/yolo/labels"); p.add_argument("--crops-out",default="artifacts/crops"); p.add_argument("--report",default="artifacts/dataset_validation.json"); args=p.parse_args()
parsed=parse_coco_annotations(args.annotation_json,args.image_root); validation=validate_coco(parsed)
# Use actual annotation category names for detector labels; diagnostic crop labels are limited to verified names.
yolo=coco_to_yolo(parsed,args.yolo_out,[c["name"] for c in parsed["categories"]]); crops=extract_crops(parsed,args.crops_out,DEFAULT_CLASSES)
report={"validation":validation,"yolo_conversion":yolo,"crop_extraction":crops}; out=Path(args.report); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
