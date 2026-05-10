import os, json, time, csv
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser(description="Adversarial Training")
# -------------------------
# Model
# -------------------------
class FCNN2(nn.Module):
    """
    depth = number of Linear layers INCLUDING output layer.
    depth=2: in->w ->2
    depth=3: in->w ->w ->2
    ...
    Output logits shape: (B, num_classes)
    """
    def __init__(self, input_dim: int, width: int, depth: int, dropout: float = 0.0, num_classes: int = 2):
        super().__init__()
        assert depth >= 2

        layers = []
        layers += [nn.Linear(input_dim, width), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(width, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

# LeNet
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # after conv/pool, feature map size is 5x5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # First conv + pool
        if (x.ndim==3):
            x=x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        # Second conv + pool
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # logits

        return x

# -------------------------
# Helpers
# -------------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def onehot_to_index(y: torch.Tensor) -> torch.Tensor:
    # y: (B,2) -> (B,)
    return y.argmax(dim=1).long()

@torch.no_grad()
def batch_accuracy(logits: torch.Tensor, y_onehot: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    if y_onehot.ndim >= 2:                  # (B,2)
        y_idx = onehot_to_index(y_onehot)
    else:
        y_idx = y_onehot
        y_idx = y_idx.long()
    return (preds == y_idx).float().mean().item()

def infer_input_dim(train_loader, device: torch.device) -> int:
    x, _ = next(iter(train_loader))
    x = x.to(device)
    return x.view(x.size(0), -1).shape[1]


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, loader, optimizer, alpha, epsilon, num_iters, norm, device,model_type) -> Dict[str, float]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if model_type=="lenet":
            x = x.reshape(x.size(0), 1, 28, 28) # add channel dim for CNN

        if y.ndim >= 2:
            y_idx = onehot_to_index(y).long()
        else:
            y_idx = y.long()

        if norm == "Linf":
            x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-epsilon, epsilon)
            x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)

        elif norm == "L2":
            delta = torch.randn_like(x)

            # For flattened input.
            # delta = delta / delta.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)

            # For flattened and image input.
            delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1).view(-1, *([1] * (delta.ndim - 1)))
            delta =     delta / delta_norm.clamp(min=1e-12)
            # radius = torch.rand(x.size(0), 1, device=x.device) * epsilon

            radius = torch.rand(
                x.size(0),
                *([1] * (x.ndim - 1)),
                device=x.device,
            ) * epsilon

            x_adv = x.clone().detach() + delta * radius
            x_adv = torch.clamp(x_adv, 0, 1)

        for _ in range(num_iters):
            x_adv.requires_grad_(True)

            logits_adv = model(x_adv)
            loss_adv = F.cross_entropy(logits_adv, y_idx)

            grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]

            with torch.no_grad():
                if norm == "Linf":
                    x_adv = x_adv + alpha * grad.sign()
                    x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
                    x_adv = torch.clamp(x_adv, 0, 1)
                elif norm == "L2":
                    # x_adv = x_adv + alpha * grad/grad.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)

                    grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1).view(-1, *([1] * (grad.ndim - 1)))
                    x_adv = x_adv + alpha * grad / grad_norm.clamp(min=1e-12)

                    delta = x_adv - x

                    # For flattened inputs
                    # delta_norm = delta.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                    
                    # For cnn's.
                    delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1).view(-1, *([1] * (delta.ndim - 1)))

                    factor = torch.minimum(torch.ones_like(delta_norm), epsilon / delta_norm.clamp(min=1e-12))
                    x_adv = x + delta * factor
                    x_adv = torch.clamp(x_adv, 0, 1)

            x_adv = x_adv.detach()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y_idx)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += batch_accuracy(logits.detach(), y_idx) * bs
        n += bs

    return {"loss": total_loss / n, "acc": total_acc / n}

@torch.no_grad()
def evaluate(model, loader, device,model_type) -> Dict[str, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if model_type == "lenet":
            x = x.reshape(x.size(0), 1, 28, 28)

        logits = model(x)
        if y.ndim >= 2:                  # (B,2)
            y_idx = onehot_to_index(y)
        else:
            y_idx = y
            y_idx = y_idx.long()
        loss = F.cross_entropy(logits, y_idx)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += batch_accuracy(logits, y) * bs
        n += bs

    return {"loss": total_loss / n, "acc": total_acc / n}

@torch.enable_grad()
def evaluate_robust(model, loader, alpha, epsilon, num_iters, norm, device,model_type) -> Dict[str, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if model_type == "lenet":
            x = x.reshape(x.size(0), 1, 28, 28)

        if y.ndim >= 2:
            y_idx = onehot_to_index(y).long()
        else:
            y_idx = y.long()

        if norm == "Linf":
            x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-epsilon, epsilon)
            x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)

        elif norm == "L2":
            delta = torch.randn_like(x)

            # For flattened inputs.
            # delta = delta / delta.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)


            delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1).view(-1, *([1] * (delta.ndim - 1)))
            delta = delta / delta_norm.clamp(min=1e-12)
            
            # radius = torch.rand(x.size(0), 1, device=x.device) * epsilon

            radius = torch.rand(
                x.size(0),
                *([1] * (x.ndim - 1)),
                device=x.device,
            ) * epsilon

            x_adv = x.clone().detach() + delta * radius
            x_adv = torch.clamp(x_adv, 0, 1)

        for _ in range(num_iters):
            x_adv.requires_grad_(True)

            logits_adv = model(x_adv)
            loss_adv = F.cross_entropy(logits_adv, y_idx)

            grad = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]

            with torch.no_grad():
                if norm == "Linf":
                    x_adv = x_adv + alpha * grad.sign()
                    x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
                    x_adv = torch.clamp(x_adv, 0, 1)
                elif norm == "L2":
                    # For flattened
                    # x_adv = x_adv + alpha * grad / grad.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                    grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1).view(-1, *([1] * (grad.ndim - 1)))

                    x_adv = x_adv + alpha * grad / grad_norm.clamp(min=1e-12)

                    # For CNN's
                    delta = x_adv - x

                    # For flattened
                    # delta_norm = delta.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)

                    # For CNN.
                    delta_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1).view(-1, *([1] * (delta.ndim - 1)))

                    factor = torch.minimum(torch.ones_like(delta_norm), epsilon / delta_norm.clamp(min=1e-12))
                    x_adv = x + delta * factor
                    x_adv = torch.clamp(x_adv, 0, 1)

            x_adv = x_adv.detach()

        with torch.no_grad():
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y_idx)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += batch_accuracy(logits, y_idx) * bs
        n += bs

    return {"loss": total_loss / n, "acc": total_acc / n}

# -------------------------
# Hyperparams + Optimizer
# -------------------------
@dataclass
class HParams:
    depth: int
    width: int
    weight_decay: float
    lr: float
    epochs: int = 30
    momentum: float = 0.9
    nesterov: bool = True
    dropout: float = 0.0
    seed: int = 42

def make_sgd(model: nn.Module, lr: float, weight_decay: float, momentum: float, nesterov: bool):
    # For SGD, weight_decay is true L2 regularization
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=weight_decay
    )


# -------------------------
# Logging
# -------------------------
def append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")

def write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in keys})

import math
# -------------------------
# Sweep (SGD)
# -------------------------
def sweep_fcnn_sgd(
    train_loader,
    val_loader,
    num_classes,
    out_dir: str = "./fcnn_sweep_sgd_bin_w_",
    # depths=( 2,3,4,5,),
    # depths=( 2,4,),
    depths=(1,),
    # widths=(64,128,256),
    # widths=(128,256),
    widths=(1,),
    weight_decays=(1e-2,1e-3,1e-4),   # kept as you stated (duplicate allowed)
    # weight_decays=(1e-3,1e-4),   # kept as you stated (duplicate allowed)
    lrs=(0.1,0.03,0.01),                    # good default grid for SGD
    # lrs=(0.1,0.03),                    # good default grid for SGD
    epochs: int = 50,
    momentum: float = 0.9,
    nesterov: bool = True,
    dropout: float = 0.0,
    seed: int = 42,    
    alpha: float = 2/255,
    epsilon: float = 8/255,
    num_iters: int = 10,
    norm: str = "Linf",
    model_type: str = "fcnn"
):
    """
    Runs full grid search and logs:
      - history.jsonl: per-epoch metrics for every run
      - runs.jsonl: per-run best metrics + hyperparams
      - runs.csv: same per-run summary in CSV
      - best_model.pt: best overall checkpoint (by val_acc)
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = infer_input_dim(train_loader, device)

    history_path = os.path.join(out_dir, "history_3.jsonl")
    runs_path    = os.path.join(out_dir, "runs_3.jsonl")
    csv_path     = os.path.join(out_dir, "runs3.csv")
    best_path    = os.path.join(out_dir, "best_model.pt")


    run_summaries = []
    run_id = 0 # changed due to interruption in run.
    
    best_val_acc = -float("inf")
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    for d in depths:
        for w in widths:
            for wd in weight_decays:
                for lr in lrs:
                    run_id += 1
                    hp = HParams(
                        depth=int(d),
                        width=int(w),
                        weight_decay=float(wd),
                        lr=float(lr),
                        epochs=int(epochs),
                        momentum=float(momentum),
                        nesterov=bool(nesterov),
                        dropout=float(dropout),
                        seed=int(seed),
                    )

                    print("Running training again for class")

                    seed_all(hp.seed)

                    if model_type == "fcnn":
                        model = FCNN2(
                            input_dim=input_dim,
                            width=hp.width,
                            depth=hp.depth,
                            dropout=hp.dropout,
                            num_classes=num_classes,
                        ).to(device)

                    elif model_type == "lenet":
                        model = LeNet(num_classes=num_classes).to(device)

                    optimizer = make_sgd(model, hp.lr, hp.weight_decay, hp.momentum, hp.nesterov)

                    t0 = time.time()
                    run_best_val_acc = -float("inf")
                    run_best_val_loss = float("inf")
                    run_best_epoch = -1
                    run_best_train_acc = None
                    run_best_train_loss = None

                    for epoch in range(1, hp.epochs + 1):
                        tr = train_one_epoch(model, train_loader, optimizer, alpha, epsilon, num_iters, norm, device,model_type)
                        va_clean = evaluate(model, val_loader, device, model_type)
                        if epoch ==1 or epoch % 5 == 0 or epoch == hp.epochs:
                            va_robust = evaluate_robust(model, val_loader, alpha, epsilon, num_iters, norm, device, model_type)

                            # per-epoch log
                            append_jsonl(history_path, {
                                "run_id": run_id,
                                "epoch": epoch,
                                "hparams": asdict(hp),
                                "train_loss": tr["loss"],
                                "train_acc": tr["acc"],
                                "val_clean_loss": va_clean["loss"],
                                "val_clean_acc": va_clean["acc"],
                                "val_robust_loss": va_robust["loss"],
                                "val_robust_acc": va_robust["acc"],
                                "epsilon": epsilon,
                                "alpha": alpha,
                                "num_iters": num_iters,
                                "norm": norm,
                            })

                            print(
                                f"[{run_id:03d}] d={hp.depth} w={hp.width} wd={hp.weight_decay:g} lr={hp.lr:g} "
                                f"ep={epoch:02d}/{hp.epochs} "
                                f"tr_acc={tr['acc']*100:.2f} "
                                f"va_clean={va_clean['acc']*100:.2f} "
                                f"va_robust={va_robust['acc']*100:.2f}"
                            )

                            if not math.isnan(va_robust["acc"]) and va_robust["acc"] > run_best_val_acc:
                                run_best_val_acc = va_robust["acc"]
                                run_best_val_loss = va_robust["loss"]
                                run_best_epoch = epoch
                                run_best_train_acc = tr["acc"]
                                run_best_train_loss = tr["loss"]

                            if not math.isnan(va_robust["acc"]) and va_robust["acc"] > best_val_acc:
                                best_val_acc = va_robust["acc"]
                                best_val_loss = va_robust["loss"]
                                best_epoch = epoch
                                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                                torch.save({
                                    "state_dict": best_state,
                                    "input_dim": input_dim,
                                    "hparams": asdict(hp),
                                    "best_val_robust_acc": best_val_acc,
                                    "best_epoch": best_epoch,
                                    "epsilon": epsilon,
                                    "alpha": alpha,
                                    "num_iters": num_iters,
                                    "norm": norm,
                                }, best_path)
                            # if va["loss"] < best_val_loss and not math.isnan(va["loss"]):
                            #     best_val_acc = va["acc"]
                            #     best_val_loss = va["loss"]
                            #     best_epoch = epoch
                            #     best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                            #     best_train_acc = tr["acc"]
                            #     best_train_loss = tr["loss"]

                    dt = time.time() - t0

                    run_summary = {
                        "run_id": run_id,
                        "input_dim": input_dim,
                        "train_size": len(train_loader.dataset),
                        "val_size": len(val_loader.dataset),
                        "depth": hp.depth,
                        "width": hp.width,
                        "weight_decay": hp.weight_decay,
                        "lr": hp.lr,
                        "epochs": hp.epochs,
                        "momentum": hp.momentum,
                        "nesterov": hp.nesterov,
                        "dropout": hp.dropout,
                        "best_epoch": run_best_epoch,
                        "best_val_robust_acc": run_best_val_acc,
                        "best_val_robust_loss": run_best_val_loss,
                        "time_sec": dt,
                        "best_train_acc": run_best_train_acc,
                        "best_train_loss": run_best_train_loss,
                        "epsilon": epsilon,
                        "alpha": alpha,
                        "num_iters": num_iters,
                        "norm": norm,

                    }

                    append_jsonl(runs_path, run_summary)
                    run_summaries.append(run_summary)

    # write CSV summary
    write_csv(csv_path, run_summaries)

# -------------------------
# Usage
# -------------------------
# You already have:
if __name__ == "__main__":
    # trainloader, valloader = load_fmnist()

    parser.add_argument("--MC", action="store_true")
    parser.add_argument("--DATASET", type = str)
    parser.add_argument("--NORM", type = str)
    parser.add_argument("--MODEL", type=str, default="fcnn", choices=["fcnn", "lenet"])

    args = parser.parse_args()

    mc = args.MC
    Dataset = args.DATASET
    norm = args.NORM

    if mc == True or Dataset in ["bank"]:
        print("Running multiclass sweep...")
        classes1 = [0]
        classes2 = [1]

    if mc == False and Dataset != "bank":
        print("Running per-class pairwise sweep...")
        # 0,8,1,7
        classes1 = [0,1,2,3,4,5,6,7,8,9] # 4 is remaining for qmnist class-pair L-inf.
        classes2 = [1,2,3,4,5,6,7,8,9]

    global trainloader
    global valloader
    global testloader

    for cl1 in classes1:
        for cl2 in range(cl1+1, len(classes2)+1):
        # for cl2 in range(7,8):
            classes = (cl1,cl2)

            if Dataset == "svhn":

                def load_dataset_svhn(NUM_CLASSES=2):

                    return dataset.get_svhn(svhn_class=NUM_CLASSES)

                trainloader, valloader, testloader = load_dataset_svhn(10)

                def load_dataset_svhn(classes):

                    return dataset.get_svhn(classes = classes)

                trainloader, valloader, testloader = load_dataset_svhn(classes=(cl1,cl2))

                print("I am svhn")

            elif Dataset == "fmnist":

                #for two class fmnist
                # def ld_fmnist(class1,class2):
                # For 10-class fmnist
                def ld_fmnist():

                    return dataset.load_fmnist(classes=(cl1,cl2) , which_mnist='fmnist', multiclass = mc)

                trainloader, valloader, testloader = ld_fmnist()

                print("I am FMNIST")

                # print("Loaded FMNIST10 dataset")

            elif Dataset == "qmnist":
                def ld_fmnist():

                    return dataset.load_fmnist(classes=(cl1,cl2) , which_mnist='qmnist', multiclass = mc)

                trainloader, valloader, testloader = ld_fmnist()

                print("I am qmnist")

            elif Dataset == "kmnist":
                def ld_fmnist():

                    return dataset.load_fmnist(classes=(cl1,cl2) ,  which_mnist='kmnist', multiclass=mc)

                trainloader, valloader, testloader = ld_fmnist()

                print("I am KMNIST")
            elif Dataset == "emnist":
                def ld_fmnist():

                    return dataset.load_fmnist(classes=(cl1,cl2) ,  which_mnist='emnist', multiclass = mc)

                trainloader, valloader, testloader = ld_fmnist()

                print("I am EMNIST")
            elif Dataset == "bank":
                def ld_bank():

                    return dataset.uci_bank()

                trainloader, testloader = ld_bank()
                valloader = testloader

                print("I am Bank")
            for x, y in trainloader:
                print("Batch x shape:", x.shape)
                print("Batch y shape:", y.shape)
                break
            if mc == False:
                num_classes = 2
            else:
                num_classes = 10
            print("Number of classes:", num_classes)

            if args.MODEL == "fcnn":

                if norm == "Linf":
                    # EPSILONS = [1/255, 2/255, 4/255, 8/255]
                    EPSILONS = [6/255]
                    num_iters = 40

                    for eps in EPSILONS:
                        alpha = eps/4
                        numerator = int(round(eps*255))
                        print(f"Running with epsilon={eps} and alpha={alpha}...")
                        if mc == False:
                            best = sweep_fcnn_sgd(trainloader, valloader, num_classes=num_classes, out_dir= f"./FCNN_nn_{Dataset}_per_class/linf_eps_{numerator}/fcnn_sweep_sgd_{Dataset}_{classes}", epochs=50, alpha=alpha, epsilon=eps, num_iters=num_iters, norm=norm, model_type=args.MODEL)
                        else:
                            best = sweep_fcnn_sgd(trainloader, valloader, num_classes=num_classes, out_dir= f"./FCNN_nn_{Dataset}_mc/linf_eps_{numerator}/fcnn_sweep_sgd_{Dataset}_mc", epochs=50, alpha=alpha, epsilon=eps, num_iters=num_iters, norm=norm, model_type=args.MODEL)

                elif norm == "L2":
                    # EPSILONS = [0.5, 1.0]
                    EPSILONS = [1.0]
                    num_iters = 40

                    for eps in EPSILONS:
                        alpha = eps/4
                        print(f"Running with epsilon={eps} and alpha={alpha}...")
                        if mc == False:
                            best = sweep_fcnn_sgd(trainloader, valloader, num_classes=num_classes, out_dir= f"./FCNN_nn_{Dataset}_per_class/l2_eps_{eps}/fcnn_sweep_sgd_{Dataset}_{classes}", epochs=50, alpha=alpha, epsilon=eps, num_iters=num_iters, norm=norm, model_type=args.MODEL)
                        else:
                            best = sweep_fcnn_sgd(trainloader, valloader, num_classes=num_classes, out_dir= f"./FCNN_nn_{Dataset}_mc/l2_eps_{eps}/fcnn_sweep_sgd_{Dataset}_mc", epochs=50, alpha=alpha, epsilon=eps, num_iters=num_iters, norm=norm, model_type=args.MODEL)

            elif args.MODEL == "lenet":
                if norm == "Linf":
                    # EPSILONS = [1/255, 2/255, 4/255, 8/255]
                    EPSILONS = [6/255]
                    num_iters = 40

                    for eps in EPSILONS:
                        alpha = eps/4
                        numerator = int(round(eps*255))
                        print(f"Running with epsilon={eps} and alpha={alpha}...")
                        if mc == False:
                            best = sweep_fcnn_sgd(trainloader, valloader, num_classes=num_classes, out_dir= f"./FCNN_lenet_{Dataset}_per_class/linf_eps_{numerator}/fcnn_sweep_sgd_{Dataset}_{classes}", epochs=50, alpha=alpha, epsilon=eps, num_iters=num_iters, norm=norm, model_type=args.MODEL)
                        else:
                            best = sweep_fcnn_sgd(trainloader, valloader, num_classes=num_classes, out_dir= f"./FCNN_lenet_{Dataset}_mc/linf_eps_{numerator}/fcnn_sweep_sgd_{Dataset}_mc", epochs=50, alpha=alpha, epsilon=eps, num_iters=num_iters, norm=norm, model_type=args.MODEL)

                elif norm == "L2":
                    # EPSILONS = [0.5, 1.0]
                    EPSILONS = [1.0]
                    num_iters = 40

                    for eps in EPSILONS:
                        alpha = eps/4 
                        print(f"Running with epsilon={eps} and alpha={alpha}...")
                        if mc == False:
                            best = sweep_fcnn_sgd(trainloader, valloader, num_classes=num_classes, out_dir= f"./FCNN_lenet_{Dataset}_per_class/l2_eps_{eps}/fcnn_sweep_sgd_{Dataset}_{classes}", epochs=50, alpha=alpha, epsilon=eps, num_iters=num_iters, norm=norm, model_type=args.MODEL)
                        else:
                            best = sweep_fcnn_sgd(trainloader, valloader, num_classes=num_classes, out_dir= f"./FCNN_lenet_{Dataset}_mc/l2_eps_{eps}/fcnn_sweep_sgd_{Dataset}_mc", epochs=50, alpha=alpha, epsilon=eps, num_iters=num_iters, norm=norm, model_type=args.MODEL)
