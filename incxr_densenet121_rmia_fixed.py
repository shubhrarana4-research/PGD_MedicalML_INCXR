#!/usr/bin/env python3
"""IN-CXR + DenseNet-121 + offline RMIA.

Reuses:
  /home/shubhra/RMIA_IN_CXR/exp/incxr_16/{candidates.csv,labels.npy,keep.npy,target_keep.npy}
  /home/shubhra/RMIA_IN_CXR/results/incxr_densenet121/target/target_densenet121.pt

Does NOT retrain the target.

Stages:
  preflight   validate artifacts/checkpoint
  train-refs  train 16 DenseNet-121 references (resumable)
  infer       18-query inference for target + refs
  attack      released-code-style offline RMIA scoring
  infer-attack
  all

DenseNet target-compatible preprocessing:
  train: resize -> reflect pad -> random crop -> hflip -> ToTensor; no normalization
  infer: resize -> reflect pad -> fixed 3x3 crop x 2 flip states; no normalization

RMIA settings retained from the existing IN-CXR scorer:
  refs=16, 8 OUT/sample, queries=18, T=2, m=0.6, n=4 (range(2,n)),
  offline a=0.3, trim=0.2, gamma=2, LR>gamma, >9/18 majority,
  target-OUT population with self exclusion.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import trim_mean
from sklearn.metrics import auc, roc_auc_score, roc_curve

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

NUM_REFS = 16
NUM_QUERIES = 18
NUM_CLASSES = 2
OFFLINE_A = 0.3
GAMMA = 2.0
TEMPERATURE = 2.0
TRIM_PROPORTION = 0.2
TAYLOR_M = 0.6
TAYLOR_N = 4
REF_SEED_OFFSET = 1000


def log(s):
    print(f"[DENSENET-RMIA] {s}", flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_):
    s = torch.initial_seed() % (2**32)
    np.random.seed(s)
    random.seed(s)


def load_torch(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def dump_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def device_of(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else
                        "cpu" if name == "auto" else name)


def P(args):
    root = Path(args.project).expanduser().resolve()
    exp = root / "exp" / "incxr_16"
    dense = root / "results" / "incxr_densenet121"
    rmia = dense / "rmia"
    return {
        "root": root,
        "data": root / "data" / "IN-CXR",
        "exp": exp,
        "candidates": exp / "candidates.csv",
        "labels": exp / "labels.npy",
        "keep": exp / "keep.npy",
        "target_keep": exp / "target_keep.npy",
        "target_ckpt": dense / "target" / "target_densenet121.pt",
        "models": rmia / "models",
        "logs": rmia / "training_logs",
        "inf": rmia / "inference",
        "attack": rmia / "attack",
    }


def ref_ckpt(p, r):
    return p["models"] / f"ref_{r:02d}.pt"


def ref_log(p, r):
    return p["logs"] / f"ref_{r:02d}.csv"


def ref_logits(p, r):
    return p["inf"] / f"ref_{r:02d}_logits.npy"


def target_logits(p):
    return p["inf"] / "target_logits.npy"


def load_candidates(p):
    with p["candidates"].open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "relative_path" not in reader.fieldnames:
            raise RuntimeError("candidates.csv must contain relative_path")
        return list(reader)


def load_artifacts(p):
    labels = np.load(p["labels"]).astype(np.int64).reshape(-1)
    keep = np.load(p["keep"]).astype(bool)
    tkeep = np.load(p["target_keep"]).astype(bool).reshape(-1)
    candidates = load_candidates(p)
    n = len(labels)
    if len(candidates) != n:
        raise RuntimeError(f"candidates={len(candidates)} labels={n}")
    if keep.shape != (NUM_REFS, n):
        raise RuntimeError(f"keep.npy shape {keep.shape}; expected {(NUM_REFS,n)}")
    if tkeep.shape != (n,):
        raise RuntimeError(f"target_keep shape {tkeep.shape}; expected {(n,)}")
    if not np.all(keep.sum(axis=0) == NUM_REFS // 2):
        raise RuntimeError("Each candidate must be IN in exactly 8/16 references")
    if not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise RuntimeError(f"Expected binary labels 0/1, got {np.unique(labels)}")
    return candidates, labels, keep, tkeep


def image_path(p, row):
    x = p["data"] / row["relative_path"]
    if not x.is_file():
        raise FileNotFoundError(x)
    return x


def make_model():
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
    return m


def target_config(p):
    if not p["target_ckpt"].exists():
        raise FileNotFoundError(p["target_ckpt"])
    ck = load_torch(p["target_ckpt"])
    if ck.get("architecture") not in ("densenet121", "DenseNet121", "DenseNet-121"):
        raise RuntimeError(f"Unexpected target architecture: {ck.get('architecture')}")
    if ck.get("pretrained", False):
        raise RuntimeError("Expected scratch DenseNet target")
    if "model" not in ck:
        raise RuntimeError("Target checkpoint missing 'model' key")
    cfg = {
        "image_size": int(ck.get("image_size", 224)),
        "pad": int(ck.get("pad", 28)),
        "epochs": int(ck.get("epochs", 100)),
        "batch_size": int(ck.get("batch_size", 64)),
        "grad_accum": int(ck.get("grad_accum", 1)),
        "lr": float(ck.get("lr", 0.1)),
        "momentum": float(ck.get("momentum", 0.9)),
        "weight_decay": float(ck.get("weight_decay", 5e-4)),
        "seed": int(ck.get("seed", 314159265)),
        "num_parameters": int(ck.get("num_parameters", 0)),
    }
    return ck, cfg


def load_target(p, dev):
    ck, cfg = target_config(p)
    m = make_model()
    m.load_state_dict(ck["model"], strict=True)
    return m.to(dev).eval(), cfg


def load_ref(p, r, dev):
    ck = load_torch(ref_ckpt(p, r))
    if not ck.get("complete", False):
        raise RuntimeError(f"Incomplete checkpoint {ref_ckpt(p,r)}")
    m = make_model()
    m.load_state_dict(ck["model_state"], strict=True)
    return m.to(dev).eval(), ck


class TrainDS(Dataset):
    def __init__(self, p, candidates, labels, size, pad):
        self.p, self.c, self.y = p, candidates, labels
        self.tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
            transforms.Pad(pad, padding_mode="reflect"),
            transforms.RandomCrop((size, size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        with Image.open(image_path(self.p, self.c[i])) as im:
            x = self.tf(im.convert("RGB"))
        return x, int(self.y[i]), i


class QueryDS(Dataset):
    def __init__(self, p, candidates, labels, q, size, pad):
        self.p, self.c, self.y = p, candidates, labels
        self.q, self.size, self.pad = q, size, pad
        self.offsets = [0, pad, 2 * pad]
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        with Image.open(image_path(self.p, self.c[i])) as im:
            im = im.convert("RGB")
            im = TF.resize(im, [self.size, self.size], interpolation=InterpolationMode.BILINEAR, antialias=True)
            if self.pad:
                im = TF.pad(im, [self.pad]*4, padding_mode="reflect")
            flip = self.q >= 9
            k = self.q % 9
            top, left = self.offsets[k // 3], self.offsets[k % 3]
            if flip:
                im = TF.hflip(im)
            im = TF.crop(im, top, left, self.size, self.size)
            x = TF.to_tensor(im)
        return x, int(self.y[i]), i


def optimizer_for(m, cfg):
    decay, no_decay = [], []
    for q in m.parameters():
        if not q.requires_grad: continue
        (decay if q.ndim > 1 else no_decay).append(q)
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], momentum=cfg["momentum"], nesterov=False)


def train_ref(args, p, candidates, labels, keep, cfg, r, dev):
    seed = cfg["seed"] + REF_SEED_OFFSET + r
    seed_all(seed)
    path, logpath = ref_ckpt(p, r), ref_log(p, r)
    if args.restart_ref and path.exists():
        path.unlink()
        if logpath.exists(): logpath.unlink()

    m = make_model().to(dev)
    opt = optimizer_for(m, cfg)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    amp = dev.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)
    start = 0

    if path.exists():
        ck = load_torch(path)
        expected = {"architecture":"densenet121", "image_size":cfg["image_size"],
                    "pad":cfg["pad"], "epochs":cfg["epochs"], "batch_size":cfg["batch_size"],
                    "grad_accum":cfg["grad_accum"], "lr":cfg["lr"],
                    "momentum":cfg["momentum"], "weight_decay":cfg["weight_decay"], "seed":seed}
        for k,v in expected.items():
            if k in ck and ck[k] != v:
                raise RuntimeError(f"ref_{r:02d}: checkpoint {k}={ck[k]} != {v}; use --restart-ref")
        if ck.get("complete", False):
            log(f"ref_{r:02d}: complete; skip")
            return
        m.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        sch.load_state_dict(ck["scheduler_state"])
        if ck.get("scaler_state"): scaler.load_state_dict(ck["scaler_state"])
        start = int(ck["epoch"]) + 1
        log(f"ref_{r:02d}: resume epoch {start+1}")

    idx = np.flatnonzero(keep[r])
    ds = TrainDS(p, candidates, labels, cfg["image_size"], cfg["pad"])
    logpath.parent.mkdir(parents=True, exist_ok=True)
    if not logpath.exists():
        with logpath.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch","loss","accuracy","lr","seconds"])
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*72}\nTRAIN ref_{r:02d}: n={len(idx)} seed={seed} device={dev}\n{'='*72}")
    for epoch in range(start, cfg["epochs"]):
        g = torch.Generator(); g.manual_seed(seed + epoch)
        loader = DataLoader(Subset(ds, idx.tolist()), batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=args.workers, pin_memory=(dev.type=="cuda"),
                            persistent_workers=False, worker_init_fn=seed_worker, generator=g)
        m.train(); opt.zero_grad(set_to_none=True)
        loss_sum = 0.0; correct = 0; seen = 0; t0 = time.time()
        for step,(x,y,_) in enumerate(loader):
            x,y = x.to(dev,non_blocking=True), y.to(dev,non_blocking=True)
            with torch.amp.autocast(device_type=dev.type, enabled=amp):
                z = m(x); raw = criterion(z,y); loss = raw / cfg["grad_accum"]
            scaler.scale(loss).backward()
            if (step+1) % cfg["grad_accum"] == 0 or step+1 == len(loader):
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            b = y.size(0); loss_sum += float(raw.detach())*b
            correct += int((z.argmax(1)==y).sum()); seen += b
        sch.step(); secs = time.time()-t0
        ep_loss, ep_acc = loss_sum/seen, correct/seen
        lr = opt.param_groups[0]["lr"]
        print(f"ref_{r:02d} epoch {epoch+1:03d}/{cfg['epochs']:03d} loss={ep_loss:.6f} acc={100*ep_acc:.3f}% lr={lr:.8g} time={secs:.1f}s", flush=True)
        with logpath.open("a",newline="") as f:
            csv.writer(f).writerow([epoch+1,ep_loss,ep_acc,lr,secs])
        atomic_save({
            "model_state":m.state_dict(), "optimizer_state":opt.state_dict(),
            "scheduler_state":sch.state_dict(), "scaler_state":scaler.state_dict(),
            "epoch":epoch, "complete":epoch+1==cfg["epochs"], "role":"reference", "ref_id":r,
            "architecture":"densenet121", "pretrained":False, "num_classes":2,
            "image_size":cfg["image_size"], "pad":cfg["pad"], "normalization":"none",
            "padding_mode":"reflect", "seed":seed, "train_size":int(len(idx)),
            "epochs":cfg["epochs"], "batch_size":cfg["batch_size"], "grad_accum":cfg["grad_accum"],
            "lr":cfg["lr"], "momentum":cfg["momentum"], "weight_decay":cfg["weight_decay"],
            "weight_decay_rule":"ndim>1 only", "scheduler":"CosineAnnealingLR",
            "target_recipe":str(p["target_ckpt"]),
        }, path)
    del m,opt,sch,scaler; gc.collect()
    if dev.type=="cuda": torch.cuda.empty_cache()


@torch.inference_mode()
def infer_model(args, p, candidates, labels, cfg, role, r, dev):
    if role == "target":
        name, out = "target", target_logits(p)
        m,_ = load_target(p,dev)
    else:
        name, out = f"ref_{r:02d}", ref_logits(p,r)
        m,_ = load_ref(p,r,dev)
    expected = (len(labels),NUM_QUERIES,NUM_CLASSES)
    if out.exists() and not args.overwrite_logits:
        a=np.load(out,mmap_mode="r")
        if a.shape==expected and np.isfinite(a).all():
            log(f"{name}: valid logits exist; skip"); del m; return
    out.parent.mkdir(parents=True,exist_ok=True)
    arr=np.lib.format.open_memmap(out,mode="w+",dtype=np.float32,shape=expected)
    amp=dev.type=="cuda" and not args.no_amp; m.eval()
    for q in range(NUM_QUERIES):
        ds=QueryDS(p,candidates,labels,q,cfg["image_size"],cfg["pad"])
        loader=DataLoader(ds,batch_size=args.infer_batch_size,shuffle=False,num_workers=args.workers,
                          pin_memory=(dev.type=="cuda"),persistent_workers=False,worker_init_fn=seed_worker)
        cursor=0
        for x,_,idx in loader:
            actual=idx.numpy(); expected_idx=np.arange(cursor,cursor+len(idx))
            if not np.array_equal(actual,expected_idx): raise RuntimeError("Candidate order changed")
            x=x.to(dev,non_blocking=True)
            with torch.amp.autocast(device_type=dev.type,enabled=amp): z=m(x)
            arr[actual,q,:]=z.float().cpu().numpy(); cursor+=len(idx)
        if cursor!=len(labels): raise RuntimeError(f"{name} query {q}: {cursor}/{len(labels)}")
        arr.flush(); print(f"{name}: query {q+1:02d}/18 complete",flush=True)
    if not np.isfinite(arr).all(): raise FloatingPointError(f"{name}: non-finite logits")
    arr.flush(); del arr,m; gc.collect()
    if dev.type=="cuda": torch.cuda.empty_cache()
    log(f"saved {out}")


def factorial(n):
    v=1
    for i in range(2,n+1): v*=i
    return v


def get_taylor(x,n):
    """Paper-faithful nth-order Taylor approximation of exp(x).

    Equation (7) of RMIA defines apx(a) = sum_{i=0}^n a^i / i!.
    Therefore n=4 MUST include the quartic term a^4 / 4!.
    """
    power=x
    t=x+1.0
    for i in range(2,n+1):
        power=power*x
        t=t+power/factorial(i)
    return t


def taylor_signal(logits,labels):
    z=logits/TEMPERATURE; t=get_taylor(z,TAYLOR_N); tsum=t.sum(1).reshape(-1,1)
    true_z=z.gather(1,labels.reshape(-1,1)); true_t=t.gather(1,labels.reshape(-1,1))
    tsum=tsum-true_t; soft_true=get_taylor(true_z-TAYLOR_M,TAYLOR_N); tsum=tsum+soft_true
    return torch.flatten(soft_true/tsum)


def logits_to_signal(logits,labels):
    # mmap_mode="r" returns a non-writable NumPy view. Copying here removes
    # PyTorch's non-writable-array warning and does not alter the logits.
    z=torch.from_numpy(np.array(logits, dtype=np.float64, copy=True))
    y=torch.as_tensor(labels,dtype=torch.long)
    if tuple(z.shape)!=(len(labels),NUM_QUERIES,NUM_CLASSES): raise RuntimeError(f"bad logits shape {tuple(z.shape)}")
    out=torch.empty((len(labels),NUM_QUERIES),dtype=torch.float64)
    for q in range(NUM_QUERIES): out[:,q]=taylor_signal(z[:,q,:],y)
    a=out.numpy()
    if not np.isfinite(a).all() or np.any(a<=0): raise FloatingPointError("invalid RMIA signal; no clipping applied")
    return a


def build_pr_out(keep,refsig):
    R,N,Q=refsig.shape; K=R//2
    if keep.shape!=(R,N) or not np.all((~keep).sum(0)==K): raise RuntimeError("bad reference membership bank")
    vals=np.empty((N,K,Q),dtype=np.float64)
    for j in range(N): vals[j]=refsig[~keep[:,j],j,:]
    out=np.asarray(trim_mean(vals,proportiontocut=TRIM_PROPORTION,axis=1),dtype=np.float64)
    if not np.isfinite(out).all() or np.any(out<=0): raise FloatingPointError("invalid Pr_OUT")
    return out


def ratio_from(target_sig,pr_out):
    pr=((1+OFFLINE_A)/2)*pr_out + ((1-OFFLINE_A)/2)
    r=target_sig/pr
    if not np.isfinite(r).all() or np.any(r<=0): raise FloatingPointError("invalid RMIA ratio")
    return r


def rmia_score(ratio,tkeep,dev,chunk):
    N,Q=ratio.shape; pop=np.flatnonzero(~tkeep); M=len(pop)
    rt=torch.as_tensor(ratio,dtype=torch.float64,device=dev); rz=rt[torch.as_tensor(pop,device=dev)]
    score=np.empty(N,dtype=np.float64)
    for start in range(0,N,chunk):
        end=min(start+chunk,N); wins=torch.zeros((end-start,M),dtype=torch.uint8,device=dev)
        for q in range(Q): wins += ((rt[start:end,q,None]/rz[None,:,q])>GAMMA).to(torch.uint8)
        score[start:end]=(wins>(Q//2)).to(torch.float64).mean(1).cpu().numpy()
        print(f"RMIA pairwise scoring: {start}:{end}/{N}",flush=True)
    score[pop] *= M/(M-1)
    return score,pop


def metrics(y,s):
    fpr,tpr,thr=roc_curve(y.astype(int),s,drop_intermediate=False)
    out={"auc":float(auc(fpr,tpr)),"best_balanced_accuracy":float(np.max(1-(fpr+(1-tpr))/2)),
         "tpr_at_fpr":{},"threshold_at_fpr":{},"actual_fpr_at_operating_point":{}}
    for target in (1e-2,1e-3,1e-4,1e-5,0.0):
        valid=np.where(fpr<=target)[0]; k=str(target)
        if len(valid)==0: chosen=None
        else: chosen=valid[np.argmax(tpr[valid])]
        out["tpr_at_fpr"][k]=0.0 if chosen is None else float(tpr[chosen])
        out["threshold_at_fpr"][k]=None if chosen is None else float(thr[chosen])
        out["actual_fpr_at_operating_point"][k]=0.0 if chosen is None else float(fpr[chosen])
    return out,fpr,tpr,thr


def diag(y,v):
    y=np.asarray(y,bool); v=np.asarray(v,float)
    return {"auc_member_higher":float(roc_auc_score(y.astype(int),v)),
            "auc_member_lower":float(roc_auc_score(y.astype(int),-v)),
            "member_mean":float(v[y].mean()),"nonmember_mean":float(v[~y].mean()),
            "mean_gap_member_minus_nonmember":float(v[y].mean()-v[~y].mean())}


def preflight(args,p):
    c,y,keep,tkeep=load_artifacts(p); _,cfg=target_config(p)
    print("\n"+"="*76+"\nIN-CXR DENSENET-121 RMIA PREFLIGHT\n"+"="*76)
    print("Candidates          :",len(y)); print("Members             :",int(tkeep.sum())); print("Nonmembers          :",int((~tkeep).sum()))
    print("Reference models    :",NUM_REFS); print("IN refs/candidate   :",int(keep.sum(0)[0])); print("OUT refs/candidate  :",int((~keep).sum(0)[0]))
    print("Queries             :",NUM_QUERIES); print("Target checkpoint   :",p["target_ckpt"])
    print("Architecture        : DenseNet-121 scratch"); print("Parameters          :",cfg["num_parameters"])
    print("Image size          :",cfg["image_size"]); print("Reflect pad         :",cfg["pad"]); print("Normalization       : NONE")
    print("Epochs              :",cfg["epochs"]); print("Batch               :",cfg["batch_size"]); print("Grad accum          :",cfg["grad_accum"])
    print("LR / momentum / WD  :",cfg["lr"],cfg["momentum"],cfg["weight_decay"])
    print("RMIA a/gamma/T      :",OFFLINE_A,GAMMA,TEMPERATURE); print("Taylor m/n          :",TAYLOR_M,TAYLOR_N); print("Trim                :",TRIM_PROPORTION)
    print("Min nonzero FPR     :",1/int((~tkeep).sum()),f"({100/int((~tkeep).sum()):.6f}%)")
    for i in [0,len(c)//2,len(c)-1]: image_path(p,c[i])
    complete=[]; incomplete=[]; missing=[]
    for r in range(NUM_REFS):
        q=ref_ckpt(p,r)
        if not q.exists(): missing.append(r)
        elif load_torch(q).get("complete",False): complete.append(r)
        else: incomplete.append(r)
    print("Complete refs       :",complete); print("Incomplete refs     :",incomplete); print("Missing refs        :",missing)
    print("NOTE: DenseNet preprocessing differs from the earlier ResNet-50 run; this is not a perfectly pure architecture-only ablation.")
    print("="*76)
    return c,y,keep,tkeep,cfg


def stage_train(args,p):
    c,y,keep,_,cfg=preflight(args,p); dev=device_of(args.device)
    start=max(0,args.ref_start); end=NUM_REFS if args.ref_end is None else min(NUM_REFS,args.ref_end)
    if not (0<=start<end<=NUM_REFS): raise ValueError(f"bad ref range {start}:{end}")
    for r in range(start,end): train_ref(args,p,c,y,keep,cfg,r,dev)


def stage_infer(args,p):
    c,y,_,_,cfg=preflight(args,p); dev=device_of(args.device)
    for r in range(NUM_REFS):
        if not ref_ckpt(p,r).exists() or not load_torch(ref_ckpt(p,r)).get("complete",False):
            raise RuntimeError(f"Reference {r:02d} not complete")
    if args.infer_target: infer_model(args,p,c,y,cfg,"target",None,dev)
    start=max(0,args.ref_start); end=NUM_REFS if args.ref_end is None else min(NUM_REFS,args.ref_end)
    for r in range(start,end): infer_model(args,p,c,y,cfg,"reference",r,dev)


def stage_attack(args,p):
    _,y,keep,tkeep,cfg=preflight(args,p); n=len(y); dev=device_of(args.attack_device)
    tp=target_logits(p)
    if not tp.exists(): raise FileNotFoundError(tp)
    tz=np.load(tp,mmap_mode="r"); target_sig=logits_to_signal(tz,y)
    refsig=np.empty((NUM_REFS,n,NUM_QUERIES),dtype=np.float64)
    for r in range(NUM_REFS):
        q=ref_logits(p,r)
        if not q.exists(): raise FileNotFoundError(q)
        refsig[r]=logits_to_signal(np.load(q,mmap_mode="r"),y); print(f"ref_{r:02d} signal OK",flush=True)
    pr_out=build_pr_out(keep,refsig); ratio=ratio_from(target_sig,pr_out); score,pop=rmia_score(ratio,tkeep,dev,args.x_chunk)
    met,fpr,tpr,thr=metrics(tkeep,score)
    diagnostics={"stage1_target_signal_mean":diag(tkeep,target_sig.mean(1)),
                 "stage2_normalized_ratio_mean":diag(tkeep,ratio.mean(1)),
                 "stage3_final_rmia_score":diag(tkeep,score)}
    summary={
        "paper":"Low-Cost High-Power Membership Inference Attacks (ICML 2024)",
        "experiment":"IN-CXR + DenseNet-121 scratch + offline RMIA adaptation",
        "num_candidates":n,"members":int(tkeep.sum()),"nonmembers":int((~tkeep).sum()),
        "reference_models":NUM_REFS,"in_refs_per_candidate":8,"out_refs_per_candidate":8,"queries":18,
        "query_scheme":"3x3 reflect-pad crops x 2 horizontal-flip states",
        "preprocessing":"no normalization; reflect padding",
        "architecture_ablation_warning":"Earlier ResNet-50 run used 0.5 normalization and zero padding; direct model comparison is not purely architecture-only.",
        "rmia":{"offline_a":OFFLINE_A,"gamma":GAMMA,"temperature":TEMPERATURE,"soft_margin_m":TAYLOR_M,"taylor_n":TAYLOR_N,
                "taylor_semantics":"released code range(2,n)","trim_proportion":TRIM_PROPORTION,"pairwise":"LR > gamma",
                "query_aggregation":">9 of 18","population":"target OUT","self_exclusion":True,"clipping":"none"},
        "empirical_fpr_resolution":1/int((~tkeep).sum()),"metrics":met,"diagnostics":diagnostics,
        "target_checkpoint":str(p["target_ckpt"])}
    p["attack"].mkdir(parents=True,exist_ok=True)
    np.save(p["attack"]/"target_signal.npy",target_sig); np.save(p["attack"]/"pr_out.npy",pr_out); np.save(p["attack"]/"ratio.npy",ratio)
    np.save(p["attack"]/"rmia_score.npy",score); np.save(p["attack"]/"population_indices.npy",pop)
    np.save(p["attack"]/"fpr.npy",fpr); np.save(p["attack"]/"tpr.npy",tpr); np.save(p["attack"]/"thresholds.npy",thr)
    dump_json(summary,p["attack"]/"rmia_metrics.json")
    print("\n"+"="*76+"\nFINAL DENSENET-121 RMIA RESULT\n"+"="*76)
    print(f"AUC                  : {met['auc']:.6f}"); print(f"Best bal. accuracy   : {met['best_balanced_accuracy']:.6f}")
    for label,key in [("1%","0.01"),("0.1%","0.001"),("0.01%","0.0001"),("0%","0.0")]:
        print(f"TPR @ FPR {label:<5}      : {100*met['tpr_at_fpr'][key]:.4f}%  (actual FPR={100*met['actual_fpr_at_operating_point'][key]:.6f}%)")
    print("Diagnostics:")
    for k,v in diagnostics.items(): print(f"  {k}: AUC={v['auc_member_higher']:.6f}, member_mean={v['member_mean']:.6g}, nonmember_mean={v['nonmember_mean']:.6g}")
    print("Saved:",p["attack"]/"rmia_metrics.json"); print("="*76)


def parse_args():
    a=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a.add_argument("--stage",choices=["preflight","train-refs","infer","attack","infer-attack","all"],default="preflight")
    a.add_argument("--project",default="/home/shubhra/RMIA_IN_CXR")
    a.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    a.add_argument("--attack-device",choices=["auto","cpu","cuda"],default="auto")
    a.add_argument("--workers",type=int,default=0)
    a.add_argument("--infer-batch-size",type=int,default=64)
    a.add_argument("--ref-start",type=int,default=0); a.add_argument("--ref-end",type=int,default=None)
    a.add_argument("--infer-target",action=argparse.BooleanOptionalAction,default=True)
    a.add_argument("--overwrite-logits",action="store_true"); a.add_argument("--restart-ref",action="store_true")
    a.add_argument("--no-amp",action="store_true"); a.add_argument("--x-chunk",type=int,default=128)
    return a.parse_args()


def main():
    args=parse_args(); p=P(args)
    if args.stage=="preflight": preflight(args,p)
    elif args.stage=="train-refs": stage_train(args,p)
    elif args.stage=="infer": stage_infer(args,p)
    elif args.stage=="attack": stage_attack(args,p)
    elif args.stage=="infer-attack": stage_infer(args,p); stage_attack(args,p)
    elif args.stage=="all": stage_train(args,p); stage_infer(args,p); stage_attack(args,p)


if __name__=="__main__": main()
