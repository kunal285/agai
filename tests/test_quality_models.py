import numpy as np, pytest
from dentistry.quality import assess_image, read_image
from dentistry.models import crop_bbox, YoloDetector, TorchvisionCropClassifier, ModelUnavailable

def test_quality_rejects_empty(): assert assess_image(None)['acceptable'] is False

def test_quality_metrics():
    image=np.random.default_rng(1).integers(0,255,(300,500,3),dtype=np.uint8)
    result=assess_image(image); assert result['metrics']['width']==500 and 'laplacian_variance' in result['metrics']

def test_crop_bbox_preserves_clipped_coordinates():
    im=np.zeros((100,200,3),dtype=np.uint8); crop,box=crop_bbox(im,[-5,10,50,60]); assert box==(0,8,52,62) and crop.shape[0]>0

def test_bad_bbox_rejected():
    with pytest.raises(ValueError): crop_bbox(np.zeros((10,10,3),dtype=np.uint8),[5,5,2,2])

def test_missing_checkpoint_fails_clearly(tmp_path):
    with pytest.raises(ModelUnavailable): YoloDetector(tmp_path/'absent.pt')
    with pytest.raises(ModelUnavailable): TorchvisionCropClassifier(tmp_path/'absent.pt',['caries'])

def test_corrupt_upload_rejected():
    with pytest.raises(ValueError): read_image(b"not an image")
