import json
import numpy as np
from dentistry.orchestrator import ScreeningOrchestrator
from dentistry.agents.validation_agent import ValidationAgent
from dentistry.agents.reporting_agent import ReportingAgent

class DummyDetector:
    def __init__(self, rows=None): self.rows=rows or []
    def predict(self,image): return self.rows

def test_low_confidence_routed_for_review():
    result={"regions":[{"region_id":"r1","detector_confidence":0.1,"classifier":{"confidence":0.4},"errors":[]}]}
    out=ValidationAgent(.25,.65).run(result); assert out['review_required'] and out['regions'][0]['review_required']

def test_orchestrator_order_and_json_report(tmp_path):
    # High texture image should pass simple quality heuristics.
    image=np.random.default_rng(7).integers(0,255,(300,500,3),dtype=np.uint8)
    import cv2
    p=tmp_path/'x.png'; cv2.imwrite(str(p),image)
    engine=ScreeningOrchestrator(DummyDetector([]),None,{},tmp_path/'reports',tmp_path/'cam')
    report=engine.run(str(p))
    assert report['execution_order']==['image_quality','detection_diagnosis','validation','reporting']
    assert report['human_review_required'] is True and report['detection_count']==0
    saved=json.loads((tmp_path/'reports'/f"{report['run_id']}.json").read_text())
    assert saved['schema_version']=='1.0'

def test_reporting_does_not_invent_diagnosis():
    r=ReportingAgent().run('x.png',{'status':'acceptable','warnings':[]},{'status':'success','regions':[]},{'review_required':True,'regions':[]})
    assert r['regions']==[] and r['detection_count']==0
