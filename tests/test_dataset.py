import json
from pathlib import Path
from dentistry.dataset import parse_coco_annotations, validate_coco, coco_to_yolo, extract_crops

def test_coco_parse_and_yolo_conversion(tmp_path):
    import cv2, numpy as np
    img=tmp_path/'x.png'; cv2.imwrite(str(img),np.zeros((100,200,3),dtype=np.uint8))
    ann=tmp_path/'annotations.json'; ann.write_text(json.dumps({"images":[{"id":1,"file_name":"x.png","width":200,"height":100}],"categories":[{"id":4,"name":"caries"}],"annotations":[{"id":9,"image_id":1,"category_id":4,"bbox":[20,10,40,20]}]}))
    parsed=parse_coco_annotations(ann,tmp_path); assert validate_coco(parsed)["invalid_bboxes"]==[]
    result=coco_to_yolo(parsed,tmp_path/'labels',['caries']); line=(tmp_path/'labels'/'x.txt').read_text().strip().split()
    assert result['labels_written']==1 and line[0]=='0'; assert line[1:]==['0.200000','0.200000','0.200000','0.200000']

def test_invalid_box_is_flagged(tmp_path):
    p=tmp_path/'a.json'; p.write_text(json.dumps({"images":[{"id":1,"file_name":"missing.png"}],"categories":[{"id":1,"name":"caries"}],"annotations":[{"id":2,"image_id":1,"category_id":1,"bbox":[0,0,0,3]}]}))
    d=parse_coco_annotations(p); v=validate_coco(d); assert v['invalid_bboxes']==[2] and v['missing_image_files']==['missing.png']
