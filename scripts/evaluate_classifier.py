"""Evaluate a trained classifier checkpoint on an explicit held-out split manifest."""
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
import argparse, json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from dentistry.metrics import classification_metrics

p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--checkpoint',default='models/classifier/best.pt'); p.add_argument('--output',default='artifacts/metrics/classifier_test.json'); p.add_argument('--batch',type=int,default=16); a=p.parse_args()
df=pd.read_csv(a.manifest)
if 'split' not in df or 'label' not in df or 'path' not in df: raise SystemExit('manifest must contain split,label,path columns')
test=df[df.split=='test'].copy()
if test.empty: raise SystemExit('No test rows. Do not substitute validation rows for final test evaluation.')
ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False); classes=ckpt['classes']; arch=ckpt['architecture']; size=int(ckpt['image_size']); ctor=getattr(models,arch)
model=ctor(weights=None)
from torch import nn
if hasattr(model,'classifier'): model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,len(classes))
else: model.fc=nn.Linear(model.fc.in_features,len(classes))
model.load_state_dict(ckpt['model_state_dict']); device='cuda' if torch.cuda.is_available() else 'cpu'; model.to(device).eval()
tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
class DS(Dataset):
 def __len__(self): return len(test)
 def __getitem__(self,i):
  row=test.iloc[i]
  with Image.open(row.path) as im: x=tf(im.convert('RGB'))
  return x,classes.index(row.label),row.path
loader=DataLoader(DS(),batch_size=a.batch,shuffle=False,num_workers=0); yt=[]; yp=[]; rows=[]
with torch.inference_mode():
 for x,y,paths in loader:
  probs=torch.softmax(model(x.to(device)),1).cpu(); pred=probs.argmax(1)
  yt.extend(y.tolist()); yp.extend(pred.tolist())
  for path,truth,p,prob in zip(paths,y.tolist(),pred.tolist(),probs.tolist()): rows.append({'path':path,'true_class':classes[truth],'predicted_class':classes[p],'confidence':max(prob),'correct':truth==p,'probabilities':dict(zip(classes,prob))})
metrics=classification_metrics([classes[i] for i in yt],[classes[i] for i in yp],classes); metrics.update({'split':'test','sample_count':len(test),'checkpoint':str(Path(a.checkpoint).resolve()),'classes':classes,'device':device,'predictions':rows})
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(metrics,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in metrics.items() if k not in ('classification_report','confusion_matrix','per_class','predictions')},indent=2)); print(f'Wrote {out}')
