import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchkernels.kernels.radial import laplacian, gaussian
from autoattack import AutoAttack

# import sys
# sys.path.append("/users/student/rs/rahulky/robust-rfms/nfa_src/src")
import dataset

from pathlib import Path

import os
import json
from typing import Dict, Any, List, Tuple

import argparse
parser = argparse.ArgumentParser(description="Adversarial Attack Evaluation")

parser.add_argument("--DATASET", type=str)
parser.add_argument("--AA_norm", type=str)
# parser.add_argument("--PGD", action = "store_true")
# parser.add_argument("--BB_attack", type=str)
parser.add_argument("--VERSION", type=str)
parser.add_argument("--MC", action="store_true")
parser.add_argument("--Kernel_Name", type=str)

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AutoAttack setup
AA_VERSION = "standard"
AA_ATTACKS = ["apgd-ce"]

# AA_NORM    = "L2"

# pgd = False

# bb_attack = "spsa"

# version = "rfm"

# Dataset = "qmnist"

# KERNEL_NAME = "laplacian"

# mc = False

Dataset = args.DATASET
AA_NORM = args.AA_norm
# pgd = args.PGD
# bb_attack = args.BB_attack
pgd=False
bb_attack = 'spsa'
version = args.VERSION
mc = args.MC
KERNEL_NAME = args.Kernel_Name

# DIR = "./svhn_class_pair_laplacian/"
DIR=f"./{Dataset}2/{KERNEL_NAME}_{version}/"

loss_fn = lambda xb, yb: F.cross_entropy(model_rfm(X_train, xb, Ms, sols, bandwidth, iteration_idx) , yb, reduction="none")                      # forward


print(f"Running {version} version with kernel {KERNEL_NAME} on dataset {Dataset} with {'multi-class' if mc else 'binary'} setting, using {'PGD' if pgd else bb_attack} attack under {AA_NORM} threat model.")
if not mc:
    INPUT_JSON = f"{DIR}{KERNEL_NAME}_{version}.json"   # <-- your 45 entries file
else:
    DIR = f"./{Dataset}2/{KERNEL_NAME}_{version}_mc/"
    INPUT_JSON = f"{DIR}{KERNEL_NAME}_{version}.json"

if not mc:
    OUT_JSON   = f"{DIR}PGD_result_{AA_NORM}_{'bb_' if bb_attack != 'False' else ''}{KERNEL_NAME}_{version}_{Dataset}_class_pair.json"   # <-- output
else:
    OUT_JSON   = f"{DIR}PGD_result_{AA_NORM}_{'bb_' if bb_attack != 'False' else ''}{KERNEL_NAME}_{version}_{Dataset}_mc.json"   # <-- output

# epsilons in pixel scale (as in your script)
if AA_NORM == "Linf":
    # EPSILONS = [0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 32.0, 40.0, 55.0]
    EPSILONS = [32.0, 40.0, 50.0]
else:
    EPSILONS = [0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

def get_data(loader):
    X = []
    y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)

global trainloader
global testloader

from updated_rfm import *

def normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x /x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)

# Our forward function for RFM model
def model_rfm(X_train, X_test, Ms, sols, bandwidth, i):

    dtype = X_train.dtype

    torch.autograd.set_detect_anomaly(True)

    # Safe conversions from NumPy -> Torch, on the right device/dtype
    M_t    = torch.as_tensor(Ms[i],  device = DEVICE, dtype=dtype)   # (1024, 1024)
    alpha  = torch.as_tensor(sols[i], device = DEVICE, dtype=dtype)   # (2, n_train)
    M_t = torch.nan_to_num(M_t, nan=0.0, posinf=1.0, neginf=-1.0)
    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=-1.0)

    # Build K_test with consistent dtype/device
    # Decide orientation. Here we compute K(X_train, X_test) -> (n_train, n_test)
    # K_test = laplace_kernel_M(X_train, X_test, bandwidth, M_t)
    # Lapl_k=lambda samples,centers: laplacian(normalize(samples),normalize(centers))
    if KERNEL_NAME == "laplacian":
        K_test = laplacian(normalize(X_train), normalize(X_test), bandwidth, M_t, False).to(DEVICE)  # alternative kernel
    elif KERNEL_NAME == "gaussian":
        K_test = gaussian(normalize(X_train), normalize(X_test), bandwidth, M_t, False).to(DEVICE)  # alternative kernel
    K_test = K_test.to(dtype=dtype)  # ensure same dtype, just in case
    K_test = torch.nan_to_num(K_test, nan=0.0, posinf=1.0, neginf=-1.0)
    preds = (alpha @ K_test).T

    return preds

def spsa_fast(loss_fn, x, y, epsilon=0.03, eta=0.01, lr=0.01,
              iterations=40, sample_size=128, norm="Linf"):

    x_adv = x.detach().clone()
    B, D = x_adv.shape
    eta = float(np.clip(epsilon / 2, 1e-3, 0.01))

    print("I'm in spsa fast")

    for _ in range(iterations):
        print("I am rademacher")

        # if norm == "Linf":
        v = torch.empty(sample_size, B, D, device = x.device).bernoulli_(0.5).mul_(2).sub_(1)
        # else:
        #     v = torch.randn(sample_size, B, D, device=x.device)
        #     v = v / v.norm(p=2, dim=2, keepdim=True).clamp_min(1e-12)

        x_plus  = (x_adv.unsqueeze(0) + eta * v).clamp(0, 1)
        x_minus = (x_adv.unsqueeze(0) - eta * v).clamp(0, 1)

        x_pm = torch.cat([x_plus, x_minus], dim=0)          # [2S, B, D]
        x_pm = x_pm.reshape(2 * sample_size * B, D)         # [2SB, D]

        y_rep = y.repeat(2 * sample_size)

        losses = loss_fn(x_pm, y_rep)                       # [2SB]
        losses = losses.view(2 * sample_size, B)

        loss_plus  = losses[:sample_size]                   # [S, B]
        loss_minus = losses[sample_size:]                   # [S, B]

        diff = (loss_plus - loss_minus).unsqueeze(2)         # [S, B, 1]

        grad_estimate = ((diff / (2 * eta)) * v).mean(dim=0) # [B, D]

        if norm == "Linf":
            x_adv = x_adv + lr * torch.sign(grad_estimate)
        else:
            g_norm = grad_estimate.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            x_adv = x_adv + lr * grad_estimate / g_norm

        delta = x_adv - x

        if norm == "Linf":
            delta = torch.clamp(delta, -epsilon, epsilon)
        else:
            d_norm = delta.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            delta = delta * (epsilon / d_norm).clamp(max=1.0)

        x_adv = torch.clamp(x + delta, 0, 1).detach()

    return x_adv

def get_adv_acc(epsilon=4/255, alpha=1/255, num_iters=15, iteration_idx=0):
    correct = 0
    total = 0

    X_test_dev = X_test.to(DEVICE)
    labels_all = torch.argmax(y_test.to(DEVICE), dim=-1)

    attack_batch_size = 128
    spsa_sample_size = 16   # start small; increase if memory allows

    for start in range(0, X_test_dev.size(0), attack_batch_size):
        end = start + attack_batch_size

        xb = X_test_dev[start:end]
        yb = labels_all[start:end]

        X_adv = spsa_fast(
            loss_fn,
            xb,
            yb,
            epsilon=epsilon,
            lr=alpha,
            iterations=num_iters,
            sample_size=spsa_sample_size,
            norm=AA_NORM,
        )

        with torch.no_grad():
            preds = torch.argmax(
                model_rfm(X_train, X_adv, Ms, sols, bandwidth, iteration_idx),
                dim=-1
            )

        correct += torch.sum(preds == yb).item()
        total += xb.size(0)

        del xb, yb, X_adv, preds
        torch.cuda.empty_cache()

    acc = 100.0 * correct / total
    print(f"Adversarial accuracy for epsilon={epsilon:.5f}: {acc:.2f}%")
    return acc

from pathlib import Path

with open(INPUT_JSON, "r") as f:
    summary = json.load(f)

results_out: List[Dict[str, Any]] = []
errors_out: List[Dict[str, str]] = []

for entry in summary.get("results", summary):  # supports either {"results":[...]} or bare list
    try:

        exp_path = entry["path"]
        if mc == False:
            classes = entry["classes"]  # expected like [c1, c2] or ["0","1"]

        # print(f"[INFO] Processing entry: classes={classes}, path={exp_path}")

        BASE_DIR = Path(__file__).resolve().parent
        # ARTIFACT_ROOT = BASE_DIR / f"{Dataset}2"
        ARTIFACT_ROOT = BASE_DIR / f"{DIR}"
        # ARTIFACT_ROOT = BASE_DIR + DIR
        exp_path = (ARTIFACT_ROOT / exp_path).resolve()

        # exp_path = "/janaki/common/adv_kernels/artifacts_gauss/" + exp_path

        print("Loading experiment from:", exp_path)

        # Parse classes only for two-class case
        if mc == False:
            c1, c2 = int(classes[0]), int(classes[1])
            clas = (c1, c2)

        # Load dataset for the pair
        # trainloader, testloader = load_fmnist(clas)
        # trainloader, testloader = load_fmnist()
        if Dataset == "svhn":
            if mc == False:
                trainloader, valloader, testloader = dataset.load_svhn(classes=(c1,c2))
            else:
                trainloader, valloader, testloader = dataset.load_svhn()
        elif Dataset == "fmnist":
            if mc == False:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=(c1,c2) , which_mnist='fmnist', multiclass = mc)
            else:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=(0,1) , which_mnist='fmnist', multiclass = mc)
        elif Dataset == "kmnist":
            if mc == False:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=(c1,c2) , which_mnist='kmnist', multiclass = mc)
            else:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=(0,1) , which_mnist='kmnist', multiclass = mc)
        elif Dataset == "qmnist":
            if mc == False:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=(c1,c2) , which_mnist='qmnist', multiclass = mc)
            else:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=(0,1) , which_mnist='qmnist', multiclass = mc)
        elif Dataset == "emnist":
            if mc == False:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=(c1,c2) , which_mnist='emnist', multiclass = mc)
                    
            else:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=(0,1) , which_mnist='emnist', multiclass = mc)
        # Load checkpoint (RFM gaussian results)
        if not os.path.isfile(exp_path):
            raise FileNotFoundError(f"Checkpoint not found: {exp_path}")

        checkpoint = torch.load(exp_path, weights_only=False, map_location=DEVICE)

        Ms = checkpoint["Ms"]
        mses = checkpoint["mses"]
        sols = checkpoint["sols"]

        # L is bandwidth
        bandwidth = checkpoint["bandwidth"]
        iteration = checkpoint["iteration"]
        X_train = checkpoint["X_train"]
        y_train = checkpoint["y_train"]
        best_round_accuracy = checkpoint["best_round_accuracy"]
        diff_rfm_kernel_acc = checkpoint["diff_rfm_kernel_acc"]
        # X_test = checkpoint["X_test"]
        # y_test = checkpoint["y_test"]
        # X_train, y_train = get_data(trainloader)
        X_test, y_test = get_data(testloader)

        print("M:",len(Ms))
        print("Sols:",len(sols))
        print("M[0]:",Ms[0].shape)
        print("sols[0]:",sols[0].shape)
        print("bandwidth:",bandwidth)
        print("X_train:",X_train.shape)
        print("y_train:",y_train.shape)
        print("X_test:",X_test.shape)
        print("y_test:",y_test.shape)

        # breakpoint()

        # choose iteration (you used 0th iter)
        if version == "kernel":
            iteration_idx = 0
        else:
            iteration_idx = iteration

        print(f"Using iteration index: {iteration_idx}")

        pair_adv = []
        for eps_pix in EPSILONS:
            if AA_NORM in ("Linf"):
                eps = eps_pix/255.0
            else:
                eps = eps_pix
            if bb_attack == "spsa":
                acc = get_adv_acc(epsilon=eps, alpha=(eps/10), num_iters=100, iteration_idx=iteration_idx)
                # if eps < 4.0:
                #     acc = get_adv_acc(epsilon=eps, alpha=(eps/4), num_iters=40, iteration_idx=iteration_idx)
                # else:
                #     acc = get_adv_acc(epsilon=eps, alpha=(eps/8) ,num_iters=50, iteration_idx=iteration_idx)
            else:
                acc = get_adv_acc(epsilon=eps, alpha=(eps/10) ,num_iters=100, iteration_idx=iteration_idx)

            pair_adv.append({
                    "eps_float": eps,
                    "norm": "L2" if AA_NORM == "L2" else "Linf",
                    "attack": "apgd-ce" if AA_NORM == "L2" and pgd==True else ("pgd" if AA_NORM == "Linf" and pgd==True else "spsa" if bb_attack=="spsa" else "square"),
                    "pgd_acc": float(acc),
                    })

        results_out.append({
            "path": str(exp_path),
            "kernel": KERNEL_NAME,
            "classes": [int(clas[0]), int(clas[1])] if not mc else None,
            "bandwidth": bandwidth,
            "iteration_idx": iteration_idx,
            "attack": "apgd-ce" if AA_NORM == "L2" and pgd==True else ("pgd" if AA_NORM == "Linf" and pgd==True else ("spsa" if bb_attack=="spsa" else ("square"))),
            "pgd": pair_adv,
        })

                    # print(f"Adversarial accuracy for eps={j} (i.e., {eps_float:.5f} in [0,1] scale): {acc:.2f}")
        if not mc:
            filename = f"{DIR}adv_acc_{AA_NORM}_{'bb_' if bb_attack != 'False' else ''}{KERNEL_NAME}_{version}.json"
            with open(filename, "w") as f:
                json.dump(results_out, f, indent=2)
        else:
            filename = f"{DIR}adv_acc_{AA_NORM}_{'bb_' if bb_attack != 'False' else ''}{KERNEL_NAME}_{version}_mc.json"
            with open(filename, "w") as f:
                json.dump(results_out, f, indent=2)

    except Exception as e:
        errors_out.append({
            "path": entry.get("path", "UNKNOWN"),
            # "classes": str(entry.get("classes", "UNKNOWN")),
            "error": f"{type(e).__name__}: {e}",
        })

out_obj = {"results": results_out, "errors": errors_out}

with open(OUT_JSON, "w") as f:
    json.dump(out_obj, f, indent=2)

print(f"[DONE] Saved results to: {OUT_JSON}")
if errors_out:
    print(f"[WARN] {len(errors_out)} entries failed. Check 'errors' in {OUT_JSON}.")
