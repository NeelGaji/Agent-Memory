#!/usr/bin/env python3
import argparse, gzip, json
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise SystemExit("Install: python3 -m pip install huggingface_hub numpy")

try:
    import numpy as np
except ImportError:
    np = None

REPO_ID="yali30/findingdory-habitat"
FILES=[
"findingdory/train/episodes.json.gz",
"findingdory/train/transformations.npy",
"findingdory/train/viewpoints.npy",
"findingdory/val/episodes.json.gz",
"findingdory/val/transformations.npy",
"findingdory/val/viewpoints.npy",
]

def preview(v,n=1200):
    try: s=json.dumps(v,ensure_ascii=False)
    except Exception: s=repr(v)
    return s if len(s)<=n else s[:n]+"...<truncated>"

def compact(v,depth=0):
    if depth>=3:
        if isinstance(v,dict): return {"type":"dict","keys":list(v)[:50]}
        if isinstance(v,list): return {"type":"list","length":len(v)}
        return {"type":type(v).__name__,"example":v}
    if isinstance(v,dict):
        return {"type":"dict","keys":list(v)[:80],
                "children":{k:compact(x,depth+1) for k,x in list(v.items())[:20]}}
    if isinstance(v,list):
        return {"type":"list","length":len(v),
                "first":compact(v[0],depth+1) if v else None}
    return {"type":type(v).__name__,"example":v}

def walk(v,path="$",out=None):
    if out is None: out=[]
    if len(out)>=200: return out
    toks=("object","receptacle","transform","position","rotation","goal","target",
          "interaction","pick","place","state","room","task","rigid","scene","episode")
    if isinstance(v,dict):
        for k,x in v.items():
            if any(t in str(k).lower() for t in toks):
                out.append({"path":f"{path}.{k}","type":type(x).__name__,"preview":preview(x,700)})
                if len(out)>=200: return out
            walk(x,f"{path}.{k}",out)
    elif isinstance(v,list):
        for i,x in enumerate(v[:5]): walk(x,f"{path}[{i}]",out)
    return out

def inspect_gz(path):
    with gzip.open(path,"rt",encoding="utf-8") as f: d=json.load(f)
    r={"top_level_summary":compact(d),"interesting_fields":walk(d)}
    if isinstance(d,dict) and isinstance(d.get("episodes"),list):
        eps=d["episodes"]; r["episode_count"]=len(eps)
        r["episode_examples"]=[{"keys":list(e.keys()),"preview":preview(e,6000)}
                               for e in eps[:3] if isinstance(e,dict)]
    return r

def inspect_npy(path):
    if np is None: return {"error":"numpy not installed"}
    a=np.load(path,allow_pickle=True)
    flat=a.reshape(-1) if a.size else a
    samples=[]
    for x in flat[:3]:
        try:
            if hasattr(x,"tolist"): x=x.tolist()
        except Exception: pass
        samples.append(preview(x,2500))
    return {"shape":list(a.shape),"dtype":str(a.dtype),"ndim":int(a.ndim),"samples":samples}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir",default="findingdory_probe")
    a=ap.parse_args()
    out=Path(a.out_dir); meta=out/"metadata"; meta.mkdir(parents=True,exist_ok=True)
    downloaded={}
    print(f"Downloading metadata only from {REPO_ID} ...")
    for i,rel in enumerate(FILES,1):
        print(f"[{i}/{len(FILES)}] {rel}")
        try:
            p=hf_hub_download(repo_id=REPO_ID,filename=rel,repo_type="dataset",local_dir=str(meta))
            downloaded[rel]=Path(p)
        except Exception as e:
            print("  ERROR:",e)
    report={"repo_id":REPO_ID,"splits":{},"notes":[
        "Metadata-only probe; no HSSD scenes, RGB/video, robot assets, or policy checkpoints.",
        "Goal: determine whether released episode metadata exposes real object/receptacle transformations and interaction structure usable by StateMem."
    ]}
    for split in ("train","val"):
        sr={}
        ep=f"findingdory/{split}/episodes.json.gz"
        tr=f"findingdory/{split}/transformations.npy"
        vp=f"findingdory/{split}/viewpoints.npy"
        if ep in downloaded: sr["episodes"]=inspect_gz(downloaded[ep])
        if tr in downloaded: sr["transformations"]=inspect_npy(downloaded[tr])
        if vp in downloaded: sr["viewpoints"]=inspect_npy(downloaded[vp])
        report["splits"][split]=sr
    outp=out/"findingdory_schema_report.json"
    outp.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("\n"+"="*80)
    print("FINDINGDORY METADATA PROBE COMPLETE")
    for split,sr in report["splits"].items():
        print(f"{split}: episodes={sr.get('episodes',{}).get('episode_count')} "
              f"transformations={sr.get('transformations',{}).get('shape')} "
              f"viewpoints={sr.get('viewpoints',{}).get('shape')}")
    print("Report:",outp)

if __name__=="__main__":
    main()
