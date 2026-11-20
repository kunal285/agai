"""Training entry points; model dependencies are imported only when training starts."""
from pathlib import Path
import random, json
import numpy as np

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError: pass

def train_yolo(data_yaml, output_dir="models/detector", model_name="yolov8n.pt", epochs=50, imgsz=640, batch=8, device=None, resume=False):
    try: from ultralytics import YOLO
    except ImportError as e: raise RuntimeError("Install detector dependencies: pip install -e '.[models]'") from e
    if not Path(data_yaml).exists(): raise FileNotFoundError(f"YOLO dataset YAML not found: {data_yaml}")
    model=YOLO(str(output_dir)+"/last.pt") if resume and (Path(output_dir)/"last.pt").exists() else YOLO(model_name)
    result=model.train(data=str(data_yaml),epochs=int(epochs),imgsz=int(imgsz),batch=int(batch),project=str(Path(output_dir).parent),name=Path(output_dir).name,device=device,resume=resume,seed=42,exist_ok=True)
    best=Path(result.save_dir)/"weights"/"best.pt"
    if not best.exists(): raise RuntimeError(f"Training finished but best checkpoint was not found at {best}")
    return {"best_checkpoint":str(best),"save_dir":str(result.save_dir)}

def train_classifier(manifest_csv, output_dir="models/classifier", architecture="efficientnet_b0", image_size=224, epochs=20, batch_size=16, learning_rate=3e-4, seed=42, fraction=1.0):
    import pandas as pd, torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image
    df=pd.read_csv(manifest_csv)
    required={"path","label","split"}
    if not required.issubset(df.columns): raise ValueError(f"Manifest must include columns {sorted(required)}; got {list(df.columns)}")
    train_df=df[df.split=="train"].copy(); val_df=df[df.split=="val"].copy()
    if fraction < 1.0:
        sampled_train = []
        for _, grp in train_df.groupby("label"):
            n_samp = max(2, int(round(len(grp) * fraction)))
            sampled_train.append(grp.sample(n=min(len(grp), n_samp), random_state=seed))
        train_df = pd.concat(sampled_train, ignore_index=True)

        sampled_val = []
        for _, grp in val_df.groupby("label"):
            n_samp = max(2, int(round(len(grp) * fraction)))
            sampled_val.append(grp.sample(n=min(len(grp), n_samp), random_state=seed))
        val_df = pd.concat(sampled_val, ignore_index=True)
    if train_df.empty or val_df.empty: raise ValueError("Manifest requires non-empty train and val splits")
    classes=sorted(train_df["label"].unique().tolist()); class_to_idx={c:i for i,c in enumerate(classes)}
    if set(val_df["label"])-set(classes): raise ValueError("Validation labels contain classes missing from training split")
    seed_everything(seed); device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {architecture} on {device} (Train crops: {len(train_df)}, Val crops: {len(val_df)})...")
    tf=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    class Crops(Dataset):
        def __init__(self,frame): self.frame=frame.reset_index(drop=True)
        def __len__(self): return len(self.frame)
        def __getitem__(self,i):
            row=self.frame.iloc[i]
            with Image.open(row.path) as im: x=tf(im.convert("RGB"))
            return x,class_to_idx[row.label]
    train_loader=DataLoader(Crops(train_df),batch_size=batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(Crops(val_df),batch_size=batch_size,shuffle=False,num_workers=0)
    ctor=getattr(models,architecture,None)
    if ctor is None: raise ValueError(f"Unknown torchvision architecture {architecture}")
    try: model=ctor(weights="DEFAULT")
    except Exception: model=ctor(weights=None)
    if hasattr(model,"classifier"): model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,len(classes))
    elif hasattr(model,"fc"): model.fc=nn.Linear(model.fc.in_features,len(classes))
    else: raise ValueError("Unsupported classifier architecture head")
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    # Compute inverse-frequency class weights to combat extreme class imbalance
    class_counts = [sum(train_df.label == c) for c in classes]
    total_train = len(train_df)
    class_weights = torch.tensor([total_train / (len(classes) * max(1, cnt)) for cnt in class_counts], dtype=torch.float32).to(device)
    criterion=nn.CrossEntropyLoss(weight=class_weights)
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True); best=-1; history=[]
    from sklearn.metrics import f1_score
    for epoch in range(int(epochs)):
        model.train(); tr_loss=0.0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True); loss=criterion(model(x),y); loss.backward(); opt.step(); tr_loss+=loss.item()*len(y)
        model.eval(); correct=total=0; ys=[]; ps=[]
        with torch.inference_mode():
            for x,y in val_loader:
                logits=model(x.to(device)); pred=logits.argmax(1).cpu(); ys.extend(y.tolist()); ps.extend(pred.tolist()); correct+=int((pred==y).sum()); total+=len(y)
        val_acc=correct/max(total,1)
        val_f1=float(f1_score(ys, ps, average="macro", zero_division=0))
        row={"epoch":epoch+1,"train_loss":round(tr_loss/len(train_df),4),"val_accuracy":round(val_acc,4),"val_macro_f1":round(val_f1,4)}; history.append(row)
        if val_f1>best or (val_f1==best and val_acc>best):
            best=val_f1; torch.save({"model_state_dict":model.state_dict(),"classes":classes,"architecture":architecture,"image_size":image_size,"class_to_idx":class_to_idx,"epoch":epoch+1,"val_accuracy":val_acc,"val_macro_f1":val_f1},output/"best.pt")
        print(json.dumps(row))
    (output/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    (output/"metadata.json").write_text(json.dumps({"classes":classes,"architecture":architecture,"best_val_accuracy":best,"epochs_completed":len(history),"device":device},indent=2),encoding="utf-8")
    return {"best_checkpoint":str(output/"best.pt"),"best_val_accuracy":best,"history":str(output/"history.json"),"device":device}
