"""Create deterministic group-aware splits from crop manifest; grouping by source image prevents crop leakage."""
import argparse
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("manifest"); p.add_argument("--output",default="artifacts/crops/split_manifest.csv"); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
df=pd.read_csv(a.manifest)
if not {"path","label","image_id"}.issubset(df.columns): raise SystemExit("manifest requires path,label,image_id")
if df.image_id.nunique()<3: raise SystemExit("Need at least three unique source images to create train/val/test splits without group leakage")
groups=df.image_id.astype(str); outer=GroupShuffleSplit(n_splits=1,test_size=.15,random_state=a.seed); trainval_idx,test_idx=next(outer.split(df,df.label,groups))
tv=df.iloc[trainval_idx].copy(); test=df.iloc[test_idx].copy(); inner=GroupShuffleSplit(n_splits=1,test_size=.1764705882,random_state=a.seed+1); tr_idx,val_idx=next(inner.split(tv,tv.label,tv.image_id.astype(str)))
tv.loc[tv.index[tr_idx],"split"]="train"; tv.loc[tv.index[val_idx],"split"]="val"; test["split"]="test"; out=pd.concat([tv,test]).sort_index(); dest=Path(a.output); dest.parent.mkdir(parents=True,exist_ok=True); out.to_csv(dest,index=False)
print(out.groupby(["split","label"]).size().to_string()); print(f"Saved {dest}")
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
