"""Dataset discovery and flexible COCO-style DENTEX annotation utilities."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_CLASSES = ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]

def inventory_dataset(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    if not root.exists():
        return {"root": str(root), "exists": False, "image_count": 0, "annotation_files": [], "images": [], "errors": ["Dataset root does not exist"]}
    images = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS)
    anns = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"})
    bad, dimensions, hashes = [], [], {}
    for p in images:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                dimensions.append({"path": str(p), "width": im.width, "height": im.height})
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            hashes.setdefault(digest, []).append(str(p))
        except Exception as e:
            bad.append({"path": str(p), "error": str(e)})
    duplicates = [v for v in hashes.values() if len(v) > 1]
    return {"root": str(root), "exists": True, "image_count": len(images), "valid_image_count": len(dimensions),
            "annotation_files": [str(p) for p in anns], "images": dimensions, "corrupt_images": bad,
            "duplicate_image_groups": duplicates, "errors": []}

def _normalise_category_name(value: Any) -> str:
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"periapicallesion":"periapical_lesion", "impactedtooth":"impacted_tooth", "deepcaries":"deep_caries"}
    return aliases.get(s.replace("_", ""), aliases.get(s, s))

def parse_coco_annotations(path: str | Path, image_root: str | Path | None = None) -> dict[str, Any]:
    """Parse COCO JSON and retain unknown category names instead of guessing mappings."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("images", []), list) or not isinstance(data.get("annotations", []), list):
        raise ValueError(f"Unsupported annotation JSON schema: {path}")
    cats = {c["id"]: str(c.get("name", c["id"])) for c in data.get("categories", []) if "id" in c}
    image_map = {}
    root = Path(image_root) if image_root else path.parent
    for im in data.get("images", []):
        file_name = str(im.get("file_name", ""))
        candidates = [root / file_name, path.parent / file_name]
        found = next((p.resolve() for p in candidates if p.exists()), None)
        image_map[im.get("id")] = {"id": im.get("id"), "file_name": file_name, "path": str(found) if found else None,
                                     "width": im.get("width"), "height": im.get("height"), "metadata": im}
    annotations = []
    for a in data.get("annotations", []):
        bbox = a.get("bbox")
        valid = isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(x, (int,float)) and np.isfinite(x) for x in bbox) and bbox[2] > 0 and bbox[3] > 0
        annotations.append({"id": a.get("id"), "image_id": a.get("image_id"), "category_id": a.get("category_id"),
                            "category_name": cats.get(a.get("category_id"), "unknown"), "bbox": list(bbox) if valid else None,
                            "bbox_valid": bool(valid), "iscrowd": a.get("iscrowd", 0), "raw": a})
    return {"images": list(image_map.values()), "annotations": annotations, "categories": [{"id": k, "name": v} for k,v in cats.items()], "source": str(path)}

def find_annotation_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*.json") if p.is_file() and any(k in p.name.lower() for k in ("annot", "train", "val", "test", "dentex", "coco"))) if root.exists() else []

def validate_coco(parsed: dict[str, Any]) -> dict[str, Any]:
    image_ids = {i["id"] for i in parsed["images"]}
    missing_image = [a["id"] for a in parsed["annotations"] if a["image_id"] not in image_ids]
    missing_files = [i["file_name"] for i in parsed["images"] if not i["path"]]
    invalid_boxes = [a["id"] for a in parsed["annotations"] if not a["bbox_valid"]]
    class_counts: dict[str,int] = {}
    for a in parsed["annotations"]:
        name = a["category_name"]
        class_counts[name] = class_counts.get(name,0)+1
    return {"image_count": len(parsed["images"]), "annotation_count": len(parsed["annotations"]), "categories": parsed["categories"],
            "class_counts": class_counts, "missing_image_references": missing_image, "missing_image_files": missing_files,
            "invalid_bboxes": invalid_boxes, "supported_diagnostic_categories_present": sorted(set(DEFAULT_CLASSES) & set(class_counts))}

def coco_to_yolo(parsed: dict[str, Any], output_dir: str | Path, class_names: list[str] | None = None) -> dict[str, Any]:
    """Convert COCO pixel xywh boxes to YOLO normalized center xywh; unknown classes are skipped explicitly."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    names = class_names or [c["name"] for c in parsed["categories"]]
    class_to_idx = {n:i for i,n in enumerate(names)}
    cat_name = {c["id"]: c["name"] for c in parsed["categories"]}
    by_image: dict[Any,list[str]] = {}
    skipped = []
    dims = {i["id"]:(i.get("width"),i.get("height")) for i in parsed["images"]}
    for a in parsed["annotations"]:
        name = cat_name.get(a["category_id"], a["category_name"])
        if not a["bbox_valid"] or name not in class_to_idx or a["image_id"] not in dims:
            skipped.append(a["id"]); continue
        w,h = dims[a["image_id"]]
        if not w or not h:
            skipped.append(a["id"]); continue
        x,y,bw,bh = a["bbox"]
        x1=max(0.0,min(float(w),float(x))); y1=max(0.0,min(float(h),float(y)))
        x2=max(0.0,min(float(w),float(x+bw))); y2=max(0.0,min(float(h),float(y+bh)))
        if x2<=x1 or y2<=y1: skipped.append(a["id"]); continue
        cx=((x1+x2)/2)/w; cy=((y1+y2)/2)/h; nw=(x2-x1)/w; nh=(y2-y1)/h
        by_image.setdefault(a["image_id"], []).append(f"{class_to_idx[name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    image_names = {i["id"]: Path(i["file_name"]).stem for i in parsed["images"]}
    for image_id, stem in image_names.items():
        (output_dir / f"{stem}.txt").write_text("\n".join(by_image.get(image_id, [])), encoding="utf-8")
    return {"output_dir": str(output_dir), "label_files": len(image_names), "labels_written": sum(map(len,by_image.values())), "skipped_annotation_ids": skipped, "class_to_idx": class_to_idx}

def extract_crops(parsed: dict[str, Any], output_dir: str | Path, class_names: list[str] | None = None, padding: float = 0.08) -> dict[str, Any]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    names = class_names or DEFAULT_CLASSES
    cat_name = {c["id"]: _normalise_category_name(c["name"]) for c in parsed["categories"]}
    img_map = {i["id"]: i for i in parsed["images"]}; rows=[]; failures=[]
    for a in parsed["annotations"]:
        name = cat_name.get(a["category_id"], _normalise_category_name(a["category_name"]))
        if name not in names or not a["bbox_valid"]: continue
        info = img_map.get(a["image_id"])
        if not info or not info.get("path"): failures.append({"annotation_id":a["id"],"reason":"missing image file"}); continue
        image = cv2.imread(info["path"])
        if image is None: failures.append({"annotation_id":a["id"],"reason":"unreadable image"}); continue
        h,w=image.shape[:2]; x,y,bw,bh=map(float,a["bbox"])
        px,py=bw*padding,bh*padding
        x1=max(0,int(x-px)); y1=max(0,int(y-py)); x2=min(w,int(x+bw+px)); y2=min(h,int(y+bh+py))
        if x2<=x1 or y2<=y1: failures.append({"annotation_id":a["id"],"reason":"empty crop"}); continue
        cls_dir=output_dir/name; cls_dir.mkdir(parents=True,exist_ok=True)
        out=cls_dir/f"image_{a['image_id']}_ann_{a['id']}.jpg"
        if not cv2.imwrite(str(out),image[y1:y2,x1:x2]): failures.append({"annotation_id":a["id"],"reason":"crop write failed"}); continue
        rows.append({"path":str(out),"image_id":a["image_id"],"annotation_id":a["id"],"label":name,"x1":x1,"y1":y1,"x2":x2,"y2":y2})
    import csv
    if rows:
        with (output_dir/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return {"crop_count":len(rows),"failures":failures,"class_counts":{n:sum(r['label']==n for r in rows) for n in names},"manifest":str(output_dir/"manifest.csv")}
