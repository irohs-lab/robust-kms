#!/usr/bin/env python3
"""SPSA attack on benign klr.KernelLogisticRegression checkpoints."""
import argparse, json, math, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import dataset

def arguments():
    a=argparse.ArgumentParser()
    a.add_argument("--checkpoint",required=True)
    a.add_argument("--output",default=None)
    a.add_argument("--norm",choices=["Linf","L2"],default="Linf")
    a.add_argument("--budgets",nargs="+",type=float,default=[0,1,2,4,8,12,16,20,24,32,40,55])
    a.add_argument("--device",default="cuda:0")
    a.add_argument("--seed",type=int,default=42)
    a.add_argument("--iters",type=int,default=100)
    a.add_argument("--sample_size",type=int,default=32)
    a.add_argument("--batch_size",type=int,default=16)
    a.add_argument("--query_batch_size",type=int,default=128)
    a.add_argument("--eta",type=float,default=None)
    a.add_argument("--lr",type=float,default=None)
    a.add_argument("--max_samples",type=int,default=None)
    a.add_argument("--stop_on_error",action="store_true")
    return a.parse_args()

def get_data(ckpt,device):
    cfg=ckpt["dataset_config"]
    name=cfg["dataset"]
    if name not in ("kmnist","fmnist","qmnist","emnist"):
        raise ValueError(f"Adapt dataset loader for {name}")
    # Stored classes=[0,1] is not applicable to this 10-class model.
    _,_,loader=dataset.load_fmnist(classes=None,which_mnist=name,multiclass=True)
    xs,ys=[],[]
    for x,y in loader:
        xs.append(x.flatten(1))
        ys.append(y.argmax(1) if y.ndim==2 else y)
    return torch.cat(xs).to(device=device,dtype=torch.float32),torch.cat(ys).to(device=device,dtype=torch.long)

class Predictor:
    def __init__(self,model,chunk):
        self.model,self.chunk=model,chunk
    @torch.no_grad()
    def __call__(self,x):
        outputs=[]
        for part in x.flatten(1).split(self.chunk):
            out=self.model.decision_function(part)
            if not torch.is_tensor(out):
                out=torch.as_tensor(out,device=part.device)
            if out.shape!=(len(part),self.model.num_classes):
                raise ValueError(f"Unexpected logits shape {out.shape}")
            outputs.append(out)
        return torch.cat(outputs)

@torch.no_grad()
def accuracy(predict,x,y,chunk):
    correct=0
    for i in range(0,len(x),chunk):
        correct+=(predict(x[i:i+chunk]).argmax(1)==y[i:i+chunk]).sum().item()
    return 100*correct/len(x)

@torch.no_grad()
def attack_batch(predict,x,y,eps,eta,lr,iters,S,norm,generator):
    original=x.detach()
    adv=original.clone()
    B,D=adv.shape
    for iteration in range(iters):
        v=torch.empty(S,B,D,device=x.device,dtype=x.dtype).bernoulli_(0.5,generator=generator).mul_(2).sub_(1)
        plus=(adv.unsqueeze(0)+eta*v).clamp(0,1)
        minus=(adv.unsqueeze(0)-eta*v).clamp(0,1)
        queries=torch.cat([plus,minus],0).reshape(2*S*B,D)
        labels=y.repeat(2*S)
        losses=[]
        for i in range(0,len(queries),predict.chunk):
            j=i+predict.chunk
            logits=predict(queries[i:j])
            losses.append(F.cross_entropy(logits,labels[i:j],reduction="none"))
        L=torch.cat(losses).reshape(2*S,B)
        grad=(((L[:S]-L[S:]).unsqueeze(-1)/(2*eta))*v).mean(0)
        if norm=="Linf":
            adv=adv+lr*grad.sign()
            delta=(adv-original).clamp(-eps,eps)
        else:
            adv=adv+lr*grad/grad.norm(p=2,dim=1,keepdim=True).clamp_min(1e-12)
            delta=adv-original
            delta=delta*(eps/delta.norm(p=2,dim=1,keepdim=True).clamp_min(1e-12)).clamp(max=1)
        adv=(original+delta).clamp(0,1)
        if iteration==0 or (iteration+1)%10==0 or iteration+1==iters:
            print(f"    SPSA iteration {iteration+1}/{iters}",flush=True)
    return adv.detach()

def main():
    a=arguments()
    torch.manual_seed(a.seed)
    device=torch.device(a.device)
    # Unpickling is unsafe for untrusted checkpoints.
    ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if not isinstance(ckpt,dict) or "model" not in ckpt:
        raise ValueError("Expected benign checkpoint containing a 'model' object")
    model=ckpt["model"]
    if not hasattr(model,"decision_function") or not hasattr(model,"X_train_"):
        raise ValueError("Checkpoint does not contain fitted klr model")
    model.X_train_=model.X_train_.to(device)
    model.alpha_=model.alpha_.to(device)
    x,y=get_data(ckpt,device)
    if a.max_samples is not None:
        x,y=x[:a.max_samples],y[:a.max_samples]
    if x.shape[1]!=model.X_train_.shape[1] or y.max()>=model.num_classes:
        raise ValueError("Data dimensions/classes do not match checkpoint")
    predict=Predictor(model,a.query_batch_size)
    print(f"Dataset={ckpt['dataset_config']['dataset']} classes={model.num_classes} samples={len(x)}")
    print(f"Centers={tuple(model.X_train_.shape)} alpha={tuple(model.alpha_.shape)}")
    clean=accuracy(predict,x,y,a.query_batch_size)
    print(f"Clean accuracy: {clean:.2f}%")
    saved=ckpt.get("metrics",{}).get("test_accuracy")
    if saved is not None:
        print(f"Checkpoint accuracy: {100*float(saved):.2f}%")
        if a.max_samples is None and abs(clean-100*float(saved))>1:
            print("WARNING: Check preprocessing/label mapping; clean accuracy differs from checkpoint")
    out=Path(a.output) if a.output else Path(a.checkpoint).with_name("benign_KLR_SPSA_"+a.norm+".json")
    out.parent.mkdir(parents=True,exist_ok=True)
    records,errors=[],[]
    for budget in a.budgets:
        eps=budget/255 if a.norm=="Linf" else budget
        eta=float(a.eta if a.eta is not None else np.clip(eps/2,1e-3,0.01))
        lr=float(a.lr if a.lr is not None else eps/10)
        print(f"\nBudget={budget} eps={eps:.6g} eta={eta:.6g} lr={lr:.6g}",flush=True)
        try:
            if eps==0:
                robust,max_delta=clean,0.0
            else:
                correct,max_delta=0,0.0
                for batch_idx,start in enumerate(range(0,len(x),a.batch_size)):
                    end=min(start+a.batch_size,len(x))
                    print(f"  batch {batch_idx+1}/{math.ceil(len(x)/a.batch_size)}",flush=True)
                    g=torch.Generator(device=device)
                    g.manual_seed(a.seed+batch_idx)
                    adv=attack_batch(predict,x[start:end],y[start:end],eps,eta,lr,a.iters,a.sample_size,a.norm,g)
                    diff=(adv-x[start:end]).flatten(1)
                    magnitude=diff.abs().max().item() if a.norm=="Linf" else diff.norm(p=2,dim=1).max().item()
                    if magnitude>eps+1e-5:
                        raise RuntimeError(f"Attack exceeded epsilon: {magnitude} > {eps}")
                    max_delta=max(max_delta,magnitude)
                    correct+=(predict(adv).argmax(1)==y[start:end]).sum().item()
                robust=100*correct/len(x)
            records.append({"eps_float":eps,"eps_budget":budget,"norm":a.norm,"attack":"spsa",
                "pgd_acc":robust,"clean_acc":clean,"max_perturbation":max_delta,
                "iterations":a.iters,"sample_size":a.sample_size,"batch_size":a.batch_size,
                "query_batch_size":a.query_batch_size,"eta":eta if eps else 0,"lr":lr if eps else 0})
            print(f"Robust accuracy: {robust:.2f}%",flush=True)
        except Exception as e:
            traceback.print_exc()
            errors.append({"budget":budget,"error":f"{type(e).__name__}: {e}"})
            if a.stop_on_error:
                raise
        finally:
            output={"results":[{"path":str(Path(a.checkpoint).resolve()),
                "kernel":ckpt.get("model_config",{}).get("kernel"),
                "classes":list(range(model.num_classes)),
                "bandwidth":ckpt.get("model_config",{}).get("length_scale"),
                "attack":"spsa","pgd":records}],"errors":errors}
            out.write_text(json.dumps(output,indent=2)+"\n")
    print(f"Saved {out}")

if __name__=="__main__":
    main()
