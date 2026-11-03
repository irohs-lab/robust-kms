import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchkernels.kernels.radial import laplacian, gaussian
from autoattack import AutoAttack
from pathlib import Path
# Replace laplace_kernel_M with laplacian(X_train, X_test, bandwidth, Mt, False)

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
parser.add_argument("--PGD", action = "store_true")
parser.add_argument("--BB_attack", type=str, default="False")
parser.add_argument("--VERSION", type=str)
parser.add_argument("--MC", action="store_true")
parser.add_argument("--Kernel_Name", type=str)
parser.add_argument("--SEED", type=int, default=42)

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
pgd = args.PGD
bb_attack = args.BB_attack
version = args.VERSION
mc = args.MC
KERNEL_NAME = args.Kernel_Name
seed = args.SEED

# DIR = "./svhn_class_pair_laplacian/"
DIR=f"./{Dataset}2/{KERNEL_NAME}_{version}/"

# DIR=f"./"

print(f"Running {version} version with kernel {KERNEL_NAME} on dataset {Dataset} with {'multi-class' if mc else 'binary'} setting, using {'PGD' if pgd else bb_attack} attack under {AA_NORM} threat model.")
# if version == "rfm":
if not mc:
    INPUT_JSON = f"{DIR}{KERNEL_NAME}_{version}.json"   # <-- your 45 entries file
else:
    DIR = f"./{Dataset}2/{KERNEL_NAME}_{version}_mc/"
    INPUT_JSON = f"{DIR}{KERNEL_NAME}_{version}.json"
# elif version == "kernel":
#     if not mc:
#         INPUT_JSON = f"{DIR}{KERNEL_NAME}_{version}.json"
#     else:
#         INPUT_JSON = f"{DIR}{KERNEL_NAME}_{version}_mc.json"
if not mc:
    OUT_JSON   = f"{DIR}/seed_{seed}/PGD_result_{AA_NORM}_{'bb_' if bb_attack != 'False' else ''}{KERNEL_NAME}_{version}_{Dataset}_class_pair.json"   # <-- output
else:
    # OUT_JSON   = f"{DIR}/seed_{seed}/PGD_result_{AA_NORM}_{'bb_' if bb_attack != 'False' else ''}{KERNEL_NAME}_{version}_{Dataset}_mc.json"   # <-- output
    OUT_JSON   = f"{DIR}/seed_{seed}/PGD_result_{AA_NORM}_{'bb_' if bb_attack != 'False' else ''}{KERNEL_NAME}_{version}_{Dataset}_mc_over_iters1.json"   # <-- output

# epsilons in pixel scale (as in your script)
if AA_NORM == "Linf":
    EPSILONS = [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 32.0, 40.0, 55.0]
else:
    EPSILONS = [0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]


          # change to "Linf" if you want Linf threat model

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
    # return x /x.norm(p=2, dim=-1, keepdim=True)

# Ms, mses , sols, bandwidth, X_train, y_train, X_test, y_test  = rfm(trainloader, testloader, iters=4, loader=True, classif=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# X_train = X_train.to(device)
# y_train = y_train.to(device)
# X_test = X_test.to(device)
# y_test = y_test.to(device)

# Our forward function for RFM model
def model_rfm(X_train, X_test, Ms, sols, bandwidth, i):

    dtype = X_train.dtype

    torch.autograd.set_detect_anomaly(True)

    # Safe conversions from NumPy -> Torch, on the right device/dtype
    M_t    = torch.as_tensor(Ms[i],  device = device, dtype=dtype)   # (1024, 1024)
    alpha  = torch.as_tensor(sols[i], device = device, dtype=dtype)   # (2, n_train)
    M_t = torch.nan_to_num(M_t, nan=0.0, posinf=1.0, neginf=-1.0)
    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=-1.0)

    # Build K_test with consistent dtype/device
    # Decide orientation. Here we compute K(X_train, X_test) -> (n_train, n_test)
    # K_test = laplace_kernel_M(X_train, X_test, bandwidth, M_t)
    # Lapl_k=lambda samples,centers: laplacian(normalize(samples),normalize(centers))
    if KERNEL_NAME == "laplacian":
        K_test = laplacian(normalize(X_train), normalize(X_test), bandwidth, M_t, False).to(device)  # alternative kernel
    elif KERNEL_NAME == "gaussian":
        K_test = gaussian(normalize(X_train), normalize(X_test), bandwidth, M_t, False).to(device)  # alternative kernel
    K_test = K_test.to(dtype=dtype)  # ensure same dtype, just in case
    K_test = torch.nan_to_num(K_test, nan=0.0, posinf=1.0, neginf=-1.0)
    preds = (alpha @ K_test).T

    return preds

def spsa(loss_fn,x,y,epsilon=0.03, eta=0.01, lr=0.01, iterations=40, sample_size=128,norm="Linf"):
    x_adv = x.detach().clone()
    print("Shape of x_adv:", x_adv.shape)

    B, D = x_adv.shape

    eta = np.clip(epsilon / 2, 1e-3, 0.01)
    for i in range(iterations):
        grad_estimate = torch.zeros_like(x_adv)
        for _ in range(sample_size):
            # Gaussian for l-2, rademacher for l-inf
            # if norm == "Linf":
            print("I am Rademacher direction")
            v = torch.empty_like(x_adv).bernoulli_(0.5).mul_(2).sub_(1)
            # else:
            #     print("I am Gaussian direction")
            #     v = torch.randn_like(x_adv)
            #     if v.dim() == 1:
            #         v = v / v.norm(p=2).clamp_min(1e-12)
            #     else:
            #         v = v / v.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

            with torch.no_grad():
                loss_plus  = loss_fn((x_adv + eta * v).clamp(0, 1), y)
                loss_minus = loss_fn((x_adv - eta * v).clamp(0, 1), y)

            diff = (loss_plus - loss_minus)          # shape: (B,) or scalar
            diff = diff.view(-1, 1)            # for images

            grad_estimate += (diff / (2 * eta)) * v
        
        grad_estimate /= sample_size

        if norm=="Linf":
            # lr needs to be smaller for huge epsilons, for FCNN.
            x_adv = x_adv + lr * torch.sign(grad_estimate)
        else:
            x_adv = x_adv + lr * grad_estimate / torch.norm(grad_estimate, p=2, dim=1, keepdim=True).clamp_min(1e-12)        
        
        delta=x_adv-x
        
        if norm=="Linf":
            delta=torch.clamp(delta,min=-epsilon,max=epsilon)
        else:
            # delta=epsilon*delta/torch.norm(delta,p=2,dim=1, keepdim=True)
            delta_norm = torch.norm(delta, p=2, dim=1, keepdim=True).clamp_min(1e-12)
            scale = (epsilon / delta_norm).clamp(max=1.0)
            delta = delta * scale
        
        x_adv=x+delta

        if torch.isnan(delta).any():
            print("NaNs in delta")

        if torch.isnan(x_adv).any():
            print("NaNs in x_adv")
        x_adv=torch.clamp(x_adv,min=0,max=1)
    
    return x_adv

def spsa_fast(loss_fn, x, y, epsilon=0.03, eta=0.01, lr=0.01,
              iterations=40, sample_size=128, norm="Linf"):

    x_adv = x.detach().clone()
    B, D = x_adv.shape
    eta = float(np.clip(epsilon / 2, 1e-3, 0.01))

    for _ in range(iterations):

        # if norm == "Linf":
        v = torch.empty(sample_size, B, D, device=x.device).bernoulli_(0.5).mul_(2).sub_(1)
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
def pgd_attack_on_ker(
    X_train,
    Ms,
    sols,
    bandwidth,
    X_test,
    y_test,
    epsilon,
    alpha,
    num_iters,
    iteration_idx=0,
):
    if X_test.ndim > 2:
        X_nat = X_test.flatten(1).clone().detach().to(device)
    else:
        X_nat = X_test.clone().detach().to(device)

    y = y_test.clone().detach()

    if y.ndim >= 2:
        y = y.argmax(dim=1)

    y = y.to(device).long()

    print("X_nat shape:", X_nat.shape)
    print("y shape:", y.shape)
    print(
        "Attack:",
        "APGD-CE" if pgd else bb_attack,
        "epsilon:", epsilon,
        "alpha:", alpha,
        "iterations:", num_iters,
    )

    def forw(x):
        return model_rfm(
            X_train,
            x,
            Ms,
            sols,
            bandwidth,
            iteration_idx,
        )

    # Clean evaluation: do not run AutoAttack at epsilon=0.
    if epsilon == 0:
        return X_nat.detach()

    if pgd:
        adversary = AutoAttack(
            forw,
            norm=AA_NORM,
            eps=epsilon,
            version="standard",
            seed=seed,
        )

        adversary.attacks_to_run = ["apgd-ce"]

        print("AutoAttack input shape:", X_nat.shape)

        # Batch size intentionally remains the complete test set.
        X_adv = adversary.run_standard_evaluation(
            X_nat,
            y,
            bs=X_nat.shape[0],
        )

    elif bb_attack.lower() == "spsa":

        def loss_fn(x, labels):
            logits = model_rfm(
                X_train,
                x,
                Ms,
                sols,
                bandwidth,
                iteration_idx,
            )

            return F.cross_entropy(
                logits,
                labels,
                reduction="none",
            )

        X_adv = spsa_fast(
            loss_fn=loss_fn,
            x=X_nat,
            y=y,
            epsilon=epsilon,
            eta=min(max(epsilon / 2, 1e-3), 0.01),
            lr=alpha,
            iterations=num_iters,
            sample_size=128,
            norm=AA_NORM,
        )

    else:
        raise ValueError(
            "No attack selected. Use --PGD for APGD-CE "
            "or --BB_attack spsa for SPSA."
        )

    return X_adv.detach()

def prepare_labels(y):
    y = y.to(device)

    if y.ndim >= 2:
        y = y.argmax(dim=1)

    return y.long()


def select_stronger_adversarial(
    X_adv_previous,
    X_adv_current,
    labels,
    iteration_idx,
):
    """
    Select the stronger adversarial example independently for each sample.

    Priority:
    1. Prefer a misclassified point over a correctly classified point.
    2. If both have the same success status, prefer the point with
       the larger cross-entropy loss.
    """
    with torch.no_grad():
        logits_previous = model_rfm(
            X_train,
            X_adv_previous,
            Ms,
            sols,
            bandwidth,
            iteration_idx,
        )

        logits_current = model_rfm(
            X_train,
            X_adv_current,
            Ms,
            sols,
            bandwidth,
            iteration_idx,
        )

        loss_previous = F.cross_entropy(
            logits_previous,
            labels,
            reduction="none",
        )

        loss_current = F.cross_entropy(
            logits_current,
            labels,
            reduction="none",
        )

        success_previous = (
            logits_previous.argmax(dim=1) != labels
        )

        success_current = (
            logits_current.argmax(dim=1) != labels
        )

        use_current = (
            (~success_previous & success_current)
            |
            (
                (success_previous == success_current)
                & (loss_current > loss_previous)
            )
        )

    # Inputs are flattened, so use_current[:, None] broadcasts over features.
    X_adv_best = torch.where(
        use_current[:, None],
        X_adv_current,
        X_adv_previous,
    )

    return X_adv_best.detach()


def calculate_accuracy(
    X_adv,
    labels,
    iteration_idx,
):
    with torch.no_grad():
        logits = model_rfm(
            X_train,
            X_adv,
            Ms,
            sols,
            bandwidth,
            iteration_idx,
        )

        predictions = logits.argmax(dim=1)

        accuracy = (
            100.0
            * predictions.eq(labels).float().mean().item()
        )

    return accuracy

def get_adv_acc(
    epsilon=4 / 255,
    alpha=1 / 255,
    num_iters=15,
    iteration_idx=0,
):
    """
    Generate adversarial examples and calculate their accuracy.

    Returns
    -------
    acc : float
        Accuracy on the newly generated adversarial examples.

    X_adv : torch.Tensor
        Newly generated adversarial examples.
    """
    X_adv = pgd_attack_on_ker(
        X_train=X_train,
        Ms=Ms,
        sols=sols,
        bandwidth=bandwidth,
        X_test=X_test,
        y_test=y_test,
        epsilon=epsilon,
        alpha=alpha,
        num_iters=num_iters,
        iteration_idx=iteration_idx,
    )

    labels = prepare_labels(y_test)

    acc = calculate_accuracy(
        X_adv,
        labels,
        iteration_idx,
    )

    print(
        f"New attack accuracy for epsilon={epsilon:.5f}: "
        f"{acc:.2f}%"
    )

    return acc, X_adv

# epsilons = [0, 0.1, 1, 2, 4, 6, 8, 12, 20]

# for j in epsilons:
#     print(f"Adversarial accuracies for eps{j}:", get_adv_acc(epsilon=j/255, alpha= (j/255)/4, num_iters=30))

# Epsilons in pixel scale (your current choice)
# epsilons = [0, 0.1, 1, 2, 4, 6, 8, 12, 20]



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
            # if c1 == 8 and c2 == 9 and KERNEL_NAME == "laplacian":
            #     c1 = 9
            #     c2 = 8
            #     clas = (c1, c2)
            # else:
            #     clas = (c1, c2)
            clas = (c1, c2)

        # Load dataset for the pair
        # trainloader, testloader = load_fmnist(clas)
        # trainloader, testloader = load_fmnist()
        if Dataset == "svhn":
            if mc == False:
                trainloader, valloader, testloader = dataset.get_svhn(classes=(c1,c2))
            else:
                trainloader, valloader, testloader = dataset.get_svhn(multiclass=mc)
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
        elif Dataset == "cifar10":
            if mc ==False:
                trainloader, valloader, testloader = dataset.get_cifar(classes=(c1,c2), multiclass = mc)
            else:
                trainloader, valloader, testloader = dataset.get_cifar(classes=(0,1), multiclass = mc)

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
        # if version == "kernle":
        #     pass
        else:
            # for i in range(0,len(Ms)):
            iteration_idx = iteration
                # iteration_idx = i

        print(f"Using iteration index: {iteration_idx}")

        pair_adv = []

        # Previous strongest adversarial examples.
        best_X_adv = None

        if X_test.ndim > 2:
            X_clean = X_test.flatten(1).clone().detach().to(device)
        else:
            X_clean = X_test.clone().detach().to(device)

        labels = prepare_labels(y_test)

        for eps_pix in EPSILONS:

            if AA_NORM == "Linf":
                eps = eps_pix / 255.0
            else:
                eps = float(eps_pix)

            if eps == 0:
                # Direct clean evaluation.
                X_adv_current = X_clean.clone()

                current_acc = calculate_accuracy(
                    X_adv_current,
                    labels,
                    iteration_idx,
                )

            elif bb_attack.lower() == "spsa":
                current_acc, X_adv_current = get_adv_acc(
                    epsilon=eps,
                    alpha=eps / 4,
                    num_iters=40,
                    iteration_idx=iteration_idx,
                )

            else:
                current_acc, X_adv_current = get_adv_acc(
                    epsilon=eps,
                    alpha=eps / 10,
                    num_iters=100,
                    iteration_idx=iteration_idx,
                )

            # Retain the strongest point from all radii evaluated so far.
            if best_X_adv is None:
                best_X_adv = X_adv_current.detach()
            else:
                best_X_adv = select_stronger_adversarial(
                    X_adv_previous=best_X_adv,
                    X_adv_current=X_adv_current,
                    labels=labels,
                    iteration_idx=iteration_idx,
                )

            cumulative_acc = calculate_accuracy(
                best_X_adv,
                labels,
                iteration_idx,
            )

            if AA_NORM == "L2":
                perturbation_norms = (
                    best_X_adv - X_clean
                ).norm(p=2, dim=1)
            else:
                perturbation_norms = (
                    best_X_adv - X_clean
                ).abs().amax(dim=1)

            print(
                f"Epsilon={eps:.6f} | "
                f"Current attack accuracy={current_acc:.2f}% | "
                f"Cumulative attack accuracy={cumulative_acc:.2f}% | "
                f"Maximum perturbation={perturbation_norms.max().item():.6f}"
            )

            # Because EPSILONS is increasing, every previously retained
            # adversarial example must be feasible at the current radius.
            if perturbation_norms.max().item() > eps + 1e-4:
                raise RuntimeError(
                    "An adversarial example violates the current "
                    f"{AA_NORM} radius: "
                    f"maximum norm={perturbation_norms.max().item():.6f}, "
                    f"epsilon={eps:.6f}"
                )

            if pgd:
                attack_name = "apgd-ce"
            elif bb_attack.lower() == "spsa":
                attack_name = "spsa"
            else:
                attack_name = "unknown"

            pair_adv.append({
                "eps_float": float(eps),
                "norm": AA_NORM,
                "attack": attack_name,

                # This is the accuracy after retaining attacks found
                # at smaller radii.
                "pgd_acc": float(cumulative_acc),

                # Useful for diagnosing independent APGD failures.
                "current_attack_acc": float(current_acc),

                "max_perturbation_norm": float(
                    perturbation_norms.max().item()
                ),
            })

        results_out.append({
            "path": str(exp_path),
            "kernel": KERNEL_NAME,
            "classes": (
                [int(clas[0]), int(clas[1])]
                if not mc
                else None
            ),
            "bandwidth": bandwidth,
            "iteration_idx": iteration_idx,
            "attack": attack_name,
            "pgd": pair_adv,
        })

                            # print(f"Adversarial accuracy for eps={j} (i.e., {eps_float:.5f} in [0,1] scale): {acc:.2f}")
                # if not mc:
                #     filename = f"{DIR}/seed_{seed}/adv_acc_{AA_NORM}_{'bb_' if bb_attack != 'False' else ''}{KERNEL_NAME}_{version}.json"
                #     with open(filename, "w") as f:
                #         json.dump(results_out, f, indent=2)
                # else:
                #     filename = f"{DIR}/seed_{seed}/adv_acc_{AA_NORM}_{'bb_' if bb_attack != 'False' else ''}{KERNEL_NAME}_{version}_mc.json"
                #     with open(filename, "w") as f:
                #         json.dump(results_out, f, indent=2)

        if not mc:
            filename = (
                f"{DIR}/seed_{seed}/"
                f"adv_acc_{AA_NORM}_{'bb_' if bb_attack != 'False' else ''}"
                f"{KERNEL_NAME}_{version}.json"
            )
        else:
            filename = (
                f"{DIR}/seed_{seed}/"
                f"adv_acc_{AA_NORM}_{'bb_' if bb_attack != 'False' else ''}"
                f"{KERNEL_NAME}_{version}_mc.json"
            )

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w") as f:
            json.dump(results_out, f, indent=2)

    # except Exception as e:
    #     errors_out.append({
    #         "path": entry.get("path", "UNKNOWN"),
    #         # "classes": str(entry.get("classes", "UNKNOWN")),
    #         "error": f"{type(e).__name__}: {e}",
    #     })
    except Exception as e:
        import traceback

        print("\n[ERROR] Experiment failed:")
        traceback.print_exc()

        errors_out.append({
            "path": entry.get("path", "UNKNOWN"),
            "error": f"{type(e).__name__}: {e}",
        })

out_obj = {"results": results_out, "errors": errors_out}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

with open(OUT_JSON, "w") as f:
    json.dump(out_obj, f, indent=2)

print(f"[DONE] Saved results to: {OUT_JSON}")
if errors_out:
    print(f"[WARN] {len(errors_out)} entries failed. Check 'errors' in {OUT_JSON}.")


# python -u svhn.py \
#   --DATASET fmnist \
#   --AA_norm L2 \
#   --PGD \
#   --BB_attack False \
#   --VERSION kernel \
#   --Kernel_Name gaussian \
#   --SEED 42



# python -u svhn.py \
#     --DATASET svhn \
#     --AA_norm Linf \
#     --PGD \
#     --BB_attack False \
#     --VERSION kernel \
#     --MC \
#     --Kernel_Name laplacian \
#     --SEED 42