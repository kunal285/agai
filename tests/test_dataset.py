import json
from pathlib import Path
from dentistry.dataset import parse_coco_annotations, validate_coco, coco_to_yolo, extract_crops
from scripts.prepare_yolo_splits import split_images_stratified

def test_coco_parse_and_yolo_conversion(tmp_path):
    from PIL import Image
    import numpy as np
    img = tmp_path / 'x.png'
    Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(str(img))
    ann = tmp_path / 'annotations.json'
    ann.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "x.png", "width": 200, "height": 100}],
        "categories": [{"id": 4, "name": "caries"}],
        "annotations": [{"id": 9, "image_id": 1, "category_id": 4, "bbox": [20, 10, 40, 20]}]
    }))
    parsed = parse_coco_annotations(ann, tmp_path)
    assert validate_coco(parsed)["invalid_bboxes"] == []
    result = coco_to_yolo(parsed, tmp_path / 'labels', ['caries'])
    line = (tmp_path / 'labels' / 'x.txt').read_text().strip().split()
    assert result['labels_written'] == 1 and line[0] == '0'
    assert line[1:] == ['0.200000', '0.200000', '0.200000', '0.200000']

def test_invalid_box_is_flagged(tmp_path):
    p = tmp_path / 'a.json'
    p.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "missing.png"}],
        "categories": [{"id": 1, "name": "caries"}],
        "annotations": [{"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 0, 3]}]
    }))
    d = parse_coco_annotations(p)
    v = validate_coco(d)
    assert v['invalid_bboxes'] == [2] and v['missing_image_files'] == ['missing.png']

def test_strict_image_level_splitting_no_leakage():
    """Verify zero overlap between train, val, and test image sets."""
    images = [{"id": i, "file_name": f"img_{i}.png", "width": 1600, "height": 800} for i in range(1, 21)]
    annotations = [
        {"id": 100 + i, "image_id": (i % 20) + 1, "category_name": "caries", "bbox": [10, 10, 50, 50], "bbox_valid": True}
        for i in range(40)
    ]
    splits = split_images_stratified(images, annotations, ["caries"], train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    
    train_ids = {img["id"] for img in splits["train"]}
    val_ids = {img["id"] for img in splits["val"]}
    test_ids = {img["id"] for img in splits["test"]}
    
    # 1. Zero intersection between any splits
    assert train_ids.isdisjoint(val_ids), "Train and Val images must not overlap!"
    assert train_ids.isdisjoint(test_ids), "Train and Test images must not overlap!"
    assert val_ids.isdisjoint(test_ids), "Val and Test images must not overlap!"
    
    # 2. Complete partition
    all_assigned = train_ids | val_ids | test_ids
    assert all_assigned == {img["id"] for img in images}
