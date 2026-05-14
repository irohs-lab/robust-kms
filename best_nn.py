import os, json, time, csv
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--MC', action='store_true', default=False)
parser.add_argument('--DATASET', type=str, default='qmnist')
parser.add_argument('--CNN', action='store_true', default=False)
args = parser.parse_args()

mc = args.MC
Dataset = args.DATASET
cnn = args.CNN

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

# Only for LeNet
def make_scheduler(
    optimizer,
    scheduler_name: str = "plateau",
    epochs: int = 200,
    factor: float = 0.5,
    patience: int = 3,
    min_lr: float = 1e-5,
):
    if scheduler_name is None or str(scheduler_name).lower() == "none":
        return None

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_lr,
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, epochs // 3),
            gamma=factor,
        )
    else:
        raise ValueError(f"Unknown scheduler_name: {scheduler_name}")

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
def train_one_epoch(model, loader, optimizer, device,cnn=False) -> Dict[str, float]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if cnn == False:
            logits = model(x) 
        else:
            x = x.reshape(x.size(0), 1, 28, 28) # add channel dim for CNN
            logits = model(x) # add channel dim for CNN    
        # print("Logits shape:", logits.shape)
        if y.ndim >= 2:                  # (B,2)
            y_idx = onehot_to_index(y)
        else:
            y_idx = y
            y_idx = y_idx.long()   # or y_idx = y.long()                       # (B,)
        
        
        loss = F.cross_entropy(logits, y_idx)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += batch_accuracy(logits.detach(), y) * bs
        n += bs

    return {"loss": total_loss / n, "acc": total_acc / n}

@torch.no_grad()
def evaluate(model, loader, device,cnn=False) -> Dict[str, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if cnn == False:
            logits = model(x) 
        else:
            x = x.reshape(x.size(0), 1, 28, 28) # add channel dim for CNN
            logits = model(x) # add channel dim for CNN    
        # print("Logits shape:", logits.shape)
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

def sweep_fcnn_sgd(
    train_loader,
    val_loader,
    num_classes,
    out_dir: str = "./fcnn_sweep_sgd_bin_w_",
    # depths=(2, 3, 4, 5),
    depths=(4,),
    # widths=(64, 128, 256),
    widths=(256,),
    # weight_decays=(1e-2, 1e-3, 1e-4),
    weight_decays=(1e-4,),
    # lrs=(0.1, 0.03, 0.01),
    lrs=(0.03,),
    epochs: int = 200,
    momentum: float = 0.9,
    nesterov: bool = True,
    dropout: float = 0.0,
    seed: int = 42,
    cnn: bool = False,
    scheduler_name: str = "plateau",
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 3,
    min_lr: float = 1e-5,
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = infer_input_dim(train_loader, device)

    history_path = os.path.join(out_dir, "history_3.jsonl")
    runs_path    = os.path.join(out_dir, "runs_3.jsonl")
    csv_path     = os.path.join(out_dir, "runs3.csv")
    best_path    = os.path.join(out_dir, "best_model.pt")

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    run_summaries = []
    run_id = 0

    best_val_acc = -float("inf")
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    best_hparams = None
    best_run_id = -1
    best_model_name = None

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

                    if not cnn:
                        model = FCNN2(
                            input_dim=input_dim,
                            width=hp.width,
                            depth=hp.depth,
                            dropout=hp.dropout,
                            num_classes=num_classes,
                        ).to(device)
                        model_name = "FCNN2"
                    else:
                        model = LeNet(num_classes=num_classes).to(device)
                        model_name = "LeNet"

                    optimizer = make_sgd(
                        model,
                        hp.lr,
                        hp.weight_decay,
                        hp.momentum,
                        hp.nesterov,
                    )

# No schheduler for NIPS project
                    # scheduler = make_scheduler(
                    #     optimizer,
                    #     scheduler_name=scheduler_name,
                    #     epochs=hp.epochs,
                    #     factor=scheduler_factor,
                    #     patience=scheduler_patience,
                    #     min_lr=min_lr,
                    # )

                    t0 = time.time()

                    run_best_val_acc = -float("inf")
                    run_best_val_loss = float("inf")
                    run_best_epoch = -1
                    run_best_train_acc = None
                    run_best_train_loss = None
                    run_best_state = None

                    for epoch in range(1, hp.epochs + 1):
                        tr = train_one_epoch(model, train_loader, optimizer, device, cnn=cnn)
                        va = evaluate(model, val_loader, device, cnn=cnn)

                        current_lr = optimizer.param_groups[0]["lr"]
                        current_lr = hp.lr

                        append_jsonl(history_path, {
                            "run_id": run_id,
                            "epoch": epoch,
                            "hparams": asdict(hp),
                            "model_name": model_name,
                            "cnn": cnn,
                            "train_loss": tr["loss"],
                            "train_acc": tr["acc"],
                            "val_loss": va["loss"],
                            "val_acc": va["acc"],
                            "current_lr": current_lr,
                            # "scheduler_name": scheduler_name,
                        })

                        print(
                            f"[{run_id:03d}] "
                            f"model={model_name} d={hp.depth} w={hp.width} "
                            f"wd={hp.weight_decay:g} lr={hp.lr:g} "
                            f"ep={epoch:02d}/{hp.epochs} "
                            f"tr_acc={tr['acc']*100:.2f} "
                            f"va_acc={va['acc']*100:.2f} "
                            f"curr_lr={current_lr:.3e}"
                        )

                        if not math.isnan(va["acc"]) and va["acc"] > run_best_val_acc:
                            run_best_val_acc = va["acc"]
                            run_best_val_loss = va["loss"]
                            run_best_epoch = epoch
                            run_best_train_acc = tr["acc"]
                            run_best_train_loss = tr["loss"]
                            run_best_state = {
                                k: v.detach().cpu()
                                for k, v in model.state_dict().items()
                            }

                        if not math.isnan(va["acc"]) and va["acc"] > best_val_acc:
                            best_val_acc = va["acc"]
                            best_val_loss = va["loss"]
                            best_epoch = epoch
                            best_run_id = run_id
                            best_hparams = asdict(hp)
                            best_model_name = model_name
                            best_state = {
                                k: v.detach().cpu()
                                for k, v in model.state_dict().items()
                            }

                            torch.save({
                                "state_dict": best_state,
                                "input_dim": input_dim,
                                "num_classes": num_classes,
                                "model_name": model_name,
                                "cnn": cnn,
                                "hparams": best_hparams,
                                "run_id": best_run_id,
                                "best_val_acc": best_val_acc,
                                "best_val_loss": best_val_loss,
                                "best_epoch": best_epoch,
                                # "scheduler_name": scheduler_name,
                                # "scheduler_factor": scheduler_factor,
                                # "scheduler_patience": scheduler_patience,
                                "min_lr": min_lr,
                            }, best_path)

                        # if scheduler is not None:
                        #     if scheduler_name.lower() == "plateau":
                        #         scheduler.step(va["loss"])
                        #     else:
                        #         scheduler.step()

                    dt = time.time() - t0

                    run_ckpt_path = os.path.join(
                        ckpt_dir,
                        f"run_{run_id:03d}_{model_name}_d{hp.depth}_w{hp.width}_wd{hp.weight_decay:g}_lr{hp.lr:g}.pt"
                    )

                    if run_best_state is not None:
                        torch.save({
                            "state_dict": run_best_state,
                            "input_dim": input_dim,
                            "num_classes": num_classes,
                            "model_name": model_name,
                            "cnn": cnn,
                            "hparams": asdict(hp),
                            "run_id": run_id,
                            "best_epoch": run_best_epoch,
                            "best_val_acc": run_best_val_acc,
                            "best_val_loss": run_best_val_loss,
                            "best_train_acc": run_best_train_acc,
                            "best_train_loss": run_best_train_loss,
                            # "scheduler_name": scheduler_name,
                            # "scheduler_factor": scheduler_factor,
                            # "scheduler_patience": scheduler_patience,
                            "min_lr": min_lr,
                        }, run_ckpt_path)

                    run_summary = {
                        "run_id": run_id,
                        "model_name": model_name,
                        "cnn": cnn,
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
                        # "scheduler_name": scheduler_name,
                        # "scheduler_factor": scheduler_factor,
                        # "scheduler_patience": scheduler_patience,
                        "min_lr": min_lr,
                        "best_epoch": run_best_epoch,
                        "best_val_acc": run_best_val_acc,
                        "best_val_loss": run_best_val_loss,
                        "time_sec": dt,
                        "best_train_acc": run_best_train_acc,
                        "best_train_loss": run_best_train_loss,
                        "checkpoint_path": run_ckpt_path,
                    }

                    append_jsonl(runs_path, run_summary)
                    run_summaries.append(run_summary)

    write_csv(csv_path, run_summaries)

    best_overall = {
        "run_id": best_run_id,
        "model_name": best_model_name,
        "cnn": cnn,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_path": best_path,
        "hparams": best_hparams,
    }

    print("\n=== SWEEP DONE (SGD) ===")
    print(f"Best val acc: {best_val_acc*100:.2f}%")
    print(f"Best run_id: {best_run_id}")
    print(f"Best model: {best_model_name}")
    print(f"Best hparams: {best_hparams}")
    print(f"Saved best checkpoint: {best_path}")
    print(f"Per-epoch history: {history_path}")
    print(f"Per-run summary: {runs_path}")
    print(f"CSV summary: {csv_path}")

    return best_overall

# -------------------------
# Usage
# -------------------------
# You already have:
if __name__ == "__main__":
    # trainloader, valloader = load_fmnist()

    if mc == True or Dataset in ["bank"]:
        print("Running multiclass sweep...")
        classes1 = [0]
        classes2 = [1]

    if mc == False and Dataset != "bank":
        print("Running per-class pairwise sweep...")
        classes1 = [0,1,2,3,4,5,6,7,8]
        classes2 = [1,2,3,4,5,6,7,8,9] 


    global trainloader
    global valloader
    global testloader

    for cl1 in classes1:
        # for cl2 in range(cl1+1, len(classes2)+1):
        for cl2 in range(9,10):
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

            if cnn == False:
                if mc == False:
                    best = sweep_fcnn_sgd(trainloader, valloader, num_classes=num_classes, out_dir= f"./FCNN_nn_{Dataset}_per_class/fcnn_sweep_sgd_{Dataset}_{classes}",epochs=200)
                else:
                    best = sweep_fcnn_sgd(trainloader, valloader, num_classes=num_classes, out_dir= f"./FCNN_nn_{Dataset}_mc/fcnn_sweep_sgd_{Dataset}_mc",epochs=200)
            else:
                if mc == False:
                    best = sweep_fcnn_sgd(trainloader, valloader, num_classes=num_classes, out_dir= f"./FCNN_lenet_{Dataset}_per_class/fcnn_lenet_sweep_sgd_{Dataset}_{classes}", depths=(1,),
    widths=(1,),
    weight_decays=(1e-2,1e-3,1e-4),
    lrs=(0.1,0.03,0.01),epochs=50,cnn = True)
                else:
                    best = sweep_fcnn_sgd(trainloader, valloader, num_classes=num_classes, out_dir= f"./FCNN_lenet_{Dataset}_mc/fcnn_lenet_sweep_sgd_{Dataset}_mc",    depths=(1,),
    widths=(1,),
    weight_decays=(1e-2,1e-3,1e-4),
    lrs=(0.1,0.03,0.01),epochs=50,cnn = True)
