import numpy as np, pytest
from dentistry.metrics import box_iou, greedy_match_boxes, classification_metrics

def test_iou_and_greedy_matching():
    assert box_iou([0,0,10,10],[0,0,10,10])==1.0
    r=greedy_match_boxes([{'bbox':[0,0,10,10]}],[{'bbox':[0,0,10,10],'confidence':.9}],.5)
    assert r['true_positives']==1 and r['false_negatives']==0 and r['precision']==1

def test_classification_metrics():
    m=classification_metrics(['a','b','a'],['a','b','b'],['a','b']); assert 0<=m['macro_f1']<=1 and len(m['confusion_matrix'])==2

def test_gradcam_generates_shape():
    torch=pytest.importorskip('torch')
    from dentistry.gradcam import generate_gradcam
    model=torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(4,2))
    x=torch.rand(1,3,32,32); out=generate_gradcam(model,x,model[0]); assert out['heatmap'].shape==(32,32) and out['overlay_bgr'].shape==(32,32,3)
