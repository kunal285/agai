"""Evaluation helpers with explicit, reproducible matching rules."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

def classification_metrics(y_true, y_pred, labels):
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,labels=labels,average=None,zero_division=0)
    return {"accuracy":float(accuracy_score(y_true,y_pred)),"macro_f1":float(precision_recall_fscore_support(y_true,y_pred,labels=labels,average="macro",zero_division=0)[2]),
            "per_class":{str(c):{"precision":float(p[i]),"recall":float(r[i]),"f1":float(f[i])} for i,c in enumerate(labels)},
            "confusion_matrix":confusion_matrix(y_true,y_pred,labels=labels).tolist(),"classification_report":classification_report(y_true,y_pred,labels=labels,zero_division=0,output_dict=True)}

def box_iou(a,b):
    ax1,ay1,ax2,ay2=map(float,a); bx1,by1,bx2,by2=map(float,b)
    ix=max(0,min(ax2,bx2)-max(ax1,bx1)); iy=max(0,min(ay2,by2)-max(ay1,by1)); inter=ix*iy
    union=max(0,ax2-ax1)*max(0,ay2-ay1)+max(0,bx2-bx1)*max(0,by2-by1)-inter
    return inter/union if union>0 else 0.0

def greedy_match_boxes(ground_truth, predictions, iou_threshold=0.5):
    """Class-agnostic greedy one-to-one IoU matching, descending prediction score."""
    gt=list(ground_truth); pred=sorted(predictions,key=lambda x:float(x.get("confidence",0)),reverse=True); used=set(); tp=fp=0
    for p in pred:
        best_i,best_v=None,0.0
        for i,g in enumerate(gt):
            if i not in used:
                v=box_iou(p["bbox"],g["bbox"])
                if v>best_v: best_i,best_v=i,v
        if best_i is not None and best_v>=iou_threshold: tp+=1; used.add(best_i)
        else: fp+=1
    return {"true_positives":tp,"false_positives":fp,"false_negatives":len(gt)-tp,"iou_threshold":iou_threshold,
            "precision":tp/(tp+fp) if tp+fp else 0.0,"recall":tp/len(gt) if gt else (1.0 if not pred else 0.0)}

def save_metrics(metrics, path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(metrics,indent=2),encoding="utf-8"); return str(p)
