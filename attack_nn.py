import dataset
import os
import json
import argparse
from typing import Dict, Any, List
from autoattack import AutoAttack
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from best_nn import FCNN2  # keep as before
from torchkernels.feature_maps import LaplacianRFF, GaussianRFF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
print("RUNNING:", os.path.abspath(__file__))

import re

def extract_classes_from_path(path):
    m = re.search(r"\((\d+),\s*(\d+)\)", path)
    if m is None:
        raise ValueError(f"Could not extract classes from path: {path}")
    return (int(m.group(1)), int(m.group(2)))

# -------------------------
# Models
# -------------------------
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # fixed
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def select_kernel(
    input_dim,
    num_features,
    kernel_type="Laplace",
    length_scale=1.0,
    seed=42,
    device=torch.device("cpu"),
):
    def normalize(x):
        return x / x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

    if kernel_type.lower() == "laplace":
        phi_obj = LaplacianRFF(
            input_dim=input_dim,
            num_features=num_features,
            length_scale=length_scale,
            seed=seed,
            bias_term=False,
            device=device,
            dtype=torch.float32,
        )
    elif kernel_type.lower() == "gaussian":
        phi_obj = GaussianRFF(
            input_dim=input_dim,
            num_features=num_features,
            length_scale=length_scale,
            seed=seed,
            bias_term=False,
            device=device,
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    def phi(x):
        return phi_obj(normalize(x))

    return phi


class RFFClassifier(nn.Module):
    def __init__(self, phi, U):
        super().__init__()
        self.phi = phi
        self.register_buffer("U", U)

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        return self.phi(x) @ self.U


def forward_model(net, x):
    if isinstance(net, LeNet):
        x = x.view(x.size(0), 1, 28, 28)
    return net(x)

# For adversarial lenet:-
# Dataset = "kmnist"
# AA_NORM = "Linf"
# norm_name = AA_NORM.lower()   # "Linf" -> "linf", "L2" -> "l2"

# pgd = True
# bb_attack = "spsa"
# mc = True

# model_type = "lenet"
# rff_kernel = None

# adv_lenet = True
# adv_eps = 6

# JSON_PATH = None

# if model_type == "lenet":

#     if adv_lenet:
#         if AA_NORM == "Linf":
#             adv_folder = "linf_eps_6"
#         elif AA_NORM == "L2":
#             adv_folder = "l2_eps_1.0"
#         else:
#             raise ValueError(f"Unsupported AA_NORM for adv_lenet: {AA_NORM}")

#         CKPT_PATH = (
#             f"/janaki/backup/users/student/rs/rahulky/water/retrain/"
#             f"FCNN_lenet_{Dataset}_mc/"
#             f"{adv_folder}/"
#             f"fcnn_sweep_sgd_{Dataset}_mc/"
#             f"best_model.pt"
#         )

#     else:
#         CKPT_PATH = (
#             f"/janaki/backup/users/student/rs/rahulky/water/retrain/"
#             f"FCNN_lenet_{Dataset}_mc/"
#             f"fcnn_sweep_sgd_{Dataset}_mc/"
#             f"best_model.pt"
#         )

# For adversarial lenet:-
Dataset = "qmnist"
AA_NORM = "L2"   # pass "Linf" or "L2"
norm_name = AA_NORM.lower()

pgd = True
bb_attack = "spsa"
mc = True

model_type = "lenet"
rff_kernel = None

adv_lenet = True
JSON_PATH = None

if adv_lenet and model_type == "lenet":
    if AA_NORM == "Linf":
        adv_folder = "linf_eps_6"
    elif AA_NORM == "L2":
        adv_folder = "l2_eps_1.0"
    else:
        raise ValueError(f"Unsupported AA_NORM: {AA_NORM}")
else:
    adv_folder = None

if model_type == "lenet":
    if adv_lenet:
        CKPT_PATH = (
            f"/janaki/backup/users/student/rs/rahulky/water/retrain/"
            f"FCNN_lenet_{Dataset}_mc/"
            f"{adv_folder}/"
            f"fcnn_sweep_sgd_{Dataset}_mc/"
            f"best_model.pt"
        )
    else:
        CKPT_PATH = (
            f"/janaki/backup/users/student/rs/rahulky/water/retrain/"
            f"FCNN_lenet_{Dataset}_mc/"
            f"fcnn_sweep_sgd_{Dataset}_mc/"
            f"best_model.pt"
        )

print(f"Dataset: {Dataset}, AA_NORM: {AA_NORM}, PGD: {pgd}, BB_ATTACK: {bb_attack}, MC: {mc}, MODEL_TYPE: {model_type}, RFF_KERNEL: {rff_kernel}, CKPT_PATH: {CKPT_PATH}, JSON_PATH: {JSON_PATH}")

results_out: List[Dict[str, Any]] = []
errors_out: List[Dict[str, str]] = []


def load_summary():
    if JSON_PATH is not None:
        with open(JSON_PATH, "r") as f:
            return json.load(f)

    if CKPT_PATH is not None:
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        return {
            "results": [
                {
                    "experiment": "direct_ckpt",
                    "path": CKPT_PATH,
                    "classes": [0, 1],
                    "best_run": ckpt.get("hparams", {}),
                }
            ]
        }

    raise ValueError("Set either JSON_PATH or CKPT_PATH")

    if model_type == "fcnn":
        if not mc:
            path = f"./FCNN_nn_{Dataset}_per_class/fcnn_sgd_bin_best_{Dataset}_class_pair_runs.json"
        else:
            path = f"./FCNN_nn_{Dataset}_mc/fcnn_sgd_bin_best_{Dataset}_mc_runs.json"

    elif model_type == "lenet":
        if not mc:
            path = f"./FCNN_lenet_{Dataset}_per_class/fcnn_sgd_bin_best_{Dataset}_class_pair_runs.json"
        else:
            path = f"./FCNN_lenet_{Dataset}_mc/fcnn_sgd_bin_best_{Dataset}_mc_runs.json"

    elif model_type == "rff":
        raise ValueError("For RFF, pass --CKPT_PATH to attack a saved RFF checkpoint directly.")

    with open(path, "r") as f:
        return json.load(f)


summary = load_summary()


# -------------------------
# Data loading
# -------------------------
def load_dataset_svhn(classes):
    return dataset.get_svhn(classes=classes)


def get_loaders(Dataset, mc, classes):
    if Dataset == "svhn":
        return load_dataset_svhn(classes)

    elif Dataset == "kmnist":
        if not mc:
            return dataset.load_fmnist(classes=classes, which_mnist='kmnist', multiclass=mc)
        return dataset.load_fmnist(classes=(0, 1), which_mnist='kmnist', multiclass=mc)

    elif Dataset == "qmnist":
        if not mc:
            return dataset.load_fmnist(classes=classes, which_mnist='qmnist', multiclass=mc)
        return dataset.load_fmnist(classes=(0, 1), which_mnist='qmnist', multiclass=mc)

    elif Dataset == "emnist":
        if not mc:
            return dataset.load_fmnist(classes=classes, which_mnist='emnist', multiclass=mc)
        return dataset.load_fmnist(classes=(0, 1), which_mnist='emnist', multiclass=mc)

    elif Dataset == "fmnist":
        if not mc:
            return dataset.load_fmnist(classes=classes, which_mnist='fmnist', multiclass=mc)
        return dataset.load_fmnist(classes=(0, 1), which_mnist='fmnist', multiclass=mc)

    else:
        raise ValueError(f"Unsupported dataset: {Dataset}")


# -------------------------
# SPSA
# -------------------------
def spsa(loss_fn, x, y, epsilon=0.03, eta=0.01, lr=0.01, iterations=40, sample_size=128, norm="Linf"):
    x_adv = x.detach().clone()
    print("Shape of x_adv:", x_adv.shape)
    eta = np.clip(epsilon / 10, 1e-3, 1e-2)
    print(f"SPSA attack with epsilon={epsilon}, eta={eta}, lr={lr}, iterations={iterations}, sample_size={sample_size}, norm={norm}")

    if norm == "L2":
        delta = torch.randn_like(x_adv)
        delta = delta / delta.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        r = torch.rand(x.size(0), 1, device=x.device)
        delta = delta * r * epsilon
        x_adv = (x_adv + delta).clamp(0, 1)

    for i in range(iterations):
        grad_estimate = torch.zeros_like(x_adv)
        for _ in range(sample_size):
            if norm == "Linf":
                v = torch.empty_like(x_adv).bernoulli_(0.5).mul_(2).sub_(1)
            else:
                v = torch.randn_like(x_adv)
                if v.dim() == 1:
                    v = v / v.norm(p=2).clamp_min(1e-12)
                else:
                    v = v / v.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

            with torch.no_grad():
                loss_plus = loss_fn((x_adv + eta * v).clamp(0, 1), y)
                loss_minus = loss_fn((x_adv - eta * v).clamp(0, 1), y)
            diff = (loss_plus - loss_minus)
            diff = diff.view(-1, 1)
            grad_estimate += (diff / (2 * eta)) * v

        grad_estimate /= sample_size

        if norm == "Linf":
            x_adv = x_adv + lr * torch.sign(grad_estimate)
        else:
            x_adv = x_adv + lr * grad_estimate / (
                torch.norm(grad_estimate, p=2, dim=1, keepdim=True).clamp_min(1e-12)
            )

        delta = x_adv - x

        if norm == "Linf":
            delta = torch.clamp(delta, min=-epsilon, max=epsilon)
        else:
            delta_norm = torch.norm(delta, p=2, dim=1, keepdim=True).clamp_min(1e-12)
            scale = (epsilon / delta_norm).clamp(max=1.0)
            delta = delta * scale

        x_adv = x + delta

        if torch.isnan(delta).any():
            print("NaNs in delta")
        if torch.isnan(x_adv).any():
            print("NaNs in x_adv")

        x_adv = torch.clamp(x_adv, min=0, max=1)

    delta = x_adv - x
    print("L2 =", delta.view(delta.size(0), -1).norm(p=2, dim=1).max().item())
    return x_adv

# -------------------------
# Main loop
# -------------------------
for entry in summary.get("results", []):
    try:
        exp_name = entry["experiment"]
        exp_path = entry["path"]
        if model_type == "fcnn":
            if not mc:
                classes_str = entry["classes"]

            best_run = entry.get("best_run", {})

            if not mc:
                c1, c2 = int(classes_str[0]), int(classes_str[1])
                # if Dataset == "fmnist":
                #     if (c1 == 7 and c2 == 9) or (c1 == 8 and c2 == 9):
                #         classes = (c2, c1)
                #     else:
                #         classes = (c1, c2)
                # else:
                #     classes = (c1, c2)
                classes = (c1, c2)
            else:
                classes = (0, 1)
        else:
            best_run = entry.get("best_run", {})

            if not mc:
                c1, c2 = extract_classes_from_path(entry["path"])
                classes = (c1, c2)
            else:
                classes = (0, 1)

        # input_dim = int(best_run.get("input_dim", 784))
        # width = int(best_run.get("width", 64))
        # depth = int(best_run.get("depth", 3))
        # weight_decay = float(best_run.get("weight_decay", 0.0))
        # lr = float(best_run.get("lr", 0.0))
        # epoch = int(best_run.get("best_epoch", 0))

        trainloader, valloader, testloader = get_loaders(Dataset, mc, classes)

        # if "ckpt_path" in entry:
        #     ckpt_path = entry["ckpt_path"]
        # else:
        #     if model_type == "rff":
        #         raise ValueError("RFF requires explicit --CKPT_PATH")
        #     ckpt_path = os.path.join(exp_path, "best_model.pt")

        ckpt_path = entry["path"]

        print(ckpt_path)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint at: {ckpt_path}")

        # -------------------------
        # Build model
        # -------------------------
        if model_type == "fcnn":
            ckpt1 = torch.load(ckpt_path, map_location=device)

            hp = ckpt1.get("hparams", {})
            input_dim = int(ckpt1.get("input_dim", hp.get("input_dim", 784)))
            width = int(hp["width"])
            depth = int(hp["depth"])
            num_classes = int(ckpt1.get("num_classes", 10 if mc else 2))
            dropout = float(hp.get("dropout", 0.0))

            if not mc:
                model = FCNN2(input_dim=input_dim, width=width, depth=depth, dropout=0, num_classes=2).to(device)
            else:
                model = FCNN2(input_dim=input_dim, width=width, depth=depth, dropout=0, num_classes=10).to(device)

            # ckpt1 = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt1["state_dict"])

        elif model_type == "lenet":
            if not mc:
                model = LeNet(num_classes=2).to(device)
            else:
                model = LeNet(num_classes=10).to(device)

            ckpt1 = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt1["state_dict"])

            input_dim= 784
            width= 64
            depth= 3

        elif model_type == "rff":
            ckpt1 = torch.load(ckpt_path, map_location=device)
            params = ckpt1["phi_params"]

            input_dim=params["input_dim"]
            num_features=params["num_features"]
            kernel_type=params["kernel_type"]
            length_scale=params["length_scale"]
            seed=params["seed"]
            width = 64
            depth = 3

            print("Input Dim:", input_dim, "num_features:", num_features,
            "kernel_type:", kernel_type, "length_scale:",length_scale, "seed:", seed)

            phi = select_kernel(
                input_dim=params["input_dim"],
                num_features=params["num_features"],
                kernel_type=params["kernel_type"],
                length_scale=params["length_scale"],
                seed=params["seed"],
                device=device,
            )

            U = ckpt1["U"].to(device)
            model = RFFClassifier(phi, U).to(device)

        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")

        model.eval()

        # -------------------------
        # Attacks
        # -------------------------
        def pgd_attack(net, X, y, epsilon, alpha, num_iters, loss_fn='mse'):
            net.eval()
            X_nat = X.detach().to(device)
            y = y.detach().to(device)

            if loss_fn == 'cross_entropy':
                y = y.argmax(dim=1).to(device)

            X_adv = X_nat.clone().detach().to(device)

            if pgd is True and AA_NORM == "Linf":
                # print("We are in the PGD l-inf attack")
                for _ in range(num_iters):
                    X_adv.requires_grad_(True)
                    if X_adv.grad is not None:
                        X_adv.grad.zero_()
                    with torch.enable_grad():
                        out = forward_model(net, X_adv)
                        if loss_fn == 'mse':
                            loss = F.mse_loss(out, y)
                        else:
                            loss = F.cross_entropy(out, y)
                        net.zero_grad()
                        loss.backward()

                    with torch.no_grad():
                        X_adv = X_adv + alpha * X_adv.grad.sign()
                        X_adv = torch.max(torch.min(X_adv, X_nat + epsilon), X_nat - epsilon)
                        X_adv = torch.clamp(X_adv, 0.0, 1.0)

                return X_adv.detach()

            elif pgd is True and AA_NORM == "L2":
                print("We are in the PGD l-2 attack")
                print(X_adv.max(), X_adv.min())

                def forw(x):
                    return forward_model(net, x)

                adversary = AutoAttack(forw, norm='L2', eps=epsilon, version='standard')
                adversary.attacks_to_run = ['apgd-ce']
                X_adv = adversary.run_standard_evaluation(X_adv, y, bs=X_adv.shape[0])
                return X_adv.detach()

            elif pgd is False and bb_attack == "spsa":
                print("We are in the SPSA black-box attack")
                loss_fn_local = lambda xb, yb: F.cross_entropy(forward_model(net, xb), yb, reduction="none")
                spsa_sample_size = 500
                if epsilon > 0:
                    X_adv = spsa(
                        loss_fn_local,
                        X_adv,
                        y,
                        epsilon=epsilon,
                        eta=0.01,
                        lr=alpha,
                        iterations=num_iters,
                        sample_size=spsa_sample_size,
                        norm=AA_NORM
                    )
                return X_adv.detach()

        def get_adv_acc(net, loader, epsilon=4/255, alpha=1/255, num_iters=15, loss_fn='mse'):
            net.eval()
            correct = 0
            total = 0
            for X, y in loader:
                X_adv = pgd_attack(net, X, y, epsilon, alpha, num_iters, loss_fn)
                with torch.no_grad():
                    preds = torch.argmax(forward_model(net, X_adv), dim=-1)
                    labels = torch.argmax(y.to(device), dim=-1)
                    correct += torch.sum(preds == labels).item()
                    total += X.size(0)

            return 100.0 * correct / total

        if AA_NORM == "Linf":
            # epsilons = [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0]
            epsilons = [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 32.0, 40.0, 55.0]
        elif AA_NORM == "L2":
            epsilons = [0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
        else:
            raise ValueError(f"Unsupported AA_norm: {AA_NORM}")

        adv_accs = []

        for j in epsilons:
            adv_results = []

            if AA_NORM == "Linf":
                eps_float = j / 255.0
            else:
                eps_float = j

            alpha = eps_float / 10

            # Only for spsa debug.
            # if eps_float < 4.0:
            #     acc = get_adv_acc(model, testloader, eps_float, alpha, 150, 'cross_entropy')
            # else:
            #     alpha = min(eps_float / 20, 0.1)
            #     acc = get_adv_acc(model, testloader, eps_float, alpha, 400, 'cross_entropy')
            acc = get_adv_acc(model, testloader, eps_float, alpha, 100, 'cross_entropy')


            adv_accs.append({
                "eps_float": float(eps_float),
                "alpha": float(alpha),
                "pgd_acc": float(acc),
            })

            adv_results.append({
                "eps_float": float(eps_float),
                "adv_accuracy": float(acc),
            })

            if not mc:
                adv_results[-1]["Class"] = classes

            if model_type == "fcnn":
                base_dir = f"./FCNN_nn_{Dataset}_{'mc' if mc else 'per_class'}"
                suffix = "nn"
            elif model_type == "lenet":
                base_dir = f"./FCNN_lenet_{Dataset}_{'mc' if mc else 'per_class'}"
                suffix = "lenet"
            else:
                base_dir = f"./rff_{rff_kernel}_{Dataset}_{'mc' if mc else 'pairwise'}"
                suffix = "rff"

            if AA_NORM == "Linf":
                if not mc:
                    with open(f"{base_dir}/adv_accuracy_results_linf_{Dataset}_class_pair_{suffix}.json", "w") as f:
                        json.dump(adv_results, f, indent=2)
                else:
                    with open(f"{base_dir}/adv_accuracy_results_linf_{Dataset}_mc_{suffix}.json", "w") as f:
                        json.dump(adv_results, f, indent=2)
            else:
                if not mc:
                    with open(f"{base_dir}/adv_accuracy_results_l2_{Dataset}_class_pair_{suffix}.json", "w") as f:
                        json.dump(adv_results, f, indent=2)
                else:
                    with open(f"{base_dir}/adv_accuracy_results_l2_{Dataset}_mc_{suffix}.json", "w") as f:
                        json.dump(adv_results, f, indent=2)

        results_out.append({
            "experiment": exp_name,
            "path": exp_path,
            "classes": [c1, c2] if not mc else [0, 1],
            "hyperparams": {
                "input_dim": input_dim,
                "width": width,
                "depth": depth,
            },
            "model_type": model_type,
            "pgd": adv_accs,
        })

    except Exception as e:
        errors_out.append({
            "experiment": entry.get("experiment", "UNKNOWN"),
            "path": entry.get("path", "UNKNOWN"),
            "error": f"{type(e).__name__}: {e}",
        })


out_obj = {
    "results": results_out,
    "errors": errors_out,
}

if model_type == "fcnn":
    base_dir = f"./FCNN_nn_{Dataset}_{'mc' if mc else 'per_class'}"
    suffix = "nn"
elif model_type == "lenet":
    base_dir = f"./FCNN_lenet_{Dataset}_{'mc' if mc else 'per_class'}"
    suffix = "lenet"
else:
    base_dir = f"./rff_{rff_kernel}_{Dataset}_{'mc' if mc else 'pairwise'}"
    suffix = "rff"


norm_name = AA_NORM.lower()

if pgd is True:
    attack_prefix = f"PGD_result_{norm_name}"
else:
    attack_prefix = f"PGD_result_{norm_name}_bb"

if not mc:
    task_suffix = f"{Dataset}_class_pair_{suffix}"
else:
    task_suffix = f"{Dataset}_mc_{suffix}"

if adv_lenet and model_type == "lenet":
    out_path = f"{base_dir}/{adv_folder}/{attack_prefix}_{task_suffix}.json"
else:
    out_path = f"{base_dir}/{attack_prefix}_{task_suffix}.json"

os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w") as f:
    json.dump(out_obj, f, indent=2)

print(f"Saved PGD results to: {out_path}")
if errors_out:
    print(f"WARNING: {len(errors_out)} experiments failed. See 'errors' in the output JSON.")
