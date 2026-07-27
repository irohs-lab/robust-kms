# For multi-class

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

import argparse

def save_json_with_seed(path_without_seed, obj):
    """
    Saves JSON inside a seed_{SEED} subdirectory.

    Example:
        ./FCNN_lenet_qmnist_mc/l2_eps_1.0/PGD_result_l2_qmnist_mc_lenet.json

    becomes:
        ./FCNN_lenet_qmnist_mc/l2_eps_1.0/seed_40/PGD_result_l2_qmnist_mc_lenet.json
    """
    parent = os.path.dirname(path_without_seed)
    fname = os.path.basename(path_without_seed)

    final_dir = os.path.join(parent, f"seed_{SEED}")
    os.makedirs(final_dir, exist_ok=True)

    final_path = os.path.join(final_dir, fname)

    with open(final_path, "w") as f:
        json.dump(obj, f, indent=2)

    return final_path

def extract_classes_from_path(path):
    m = re.search(r"\((\d+),\s*(\d+)\)", path)
    if m is None:
        raise ValueError(f"Could not extract classes from path: {path}")
    return (int(m.group(1)), int(m.group(2)))


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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
        # x = x.view(x.size(0), 1, 28, 28)
        x = x.view(x.size(0), 3, 32, 32)
    return net(x)


parser = argparse.ArgumentParser(description="Attack FCNN / LeNet / RFF models")

parser.add_argument("--DATASET", type=str, required=True,
                    choices=["fmnist", "kmnist", "qmnist", "emnist", "svhn", "cifar10"])

parser.add_argument("--AA_norm", type=str, required=True,
                    choices=["Linf", "L2"])

parser.add_argument("--PGD", action="store_true",
                    help="Use AutoAttack APGD-CE")

parser.add_argument("--BB_attack", type=str, default="False",
                    choices=["False", "spsa"])

parser.add_argument("--MC", action="store_true",
                    help="Use multiclass setting")

parser.add_argument("--MODEL_TYPE", type=str, required=True,
                    choices=["fcnn", "lenet", "rff"])

parser.add_argument("--RFF_KERNEL", type=str, default=None,
                    choices=["Laplace", "gaussian", "Gaussian", "laplace", None])

parser.add_argument("--ADV_LENET", action="store_true",
                    help="Use adversarially trained LeNet checkpoint folder")

parser.add_argument("--ADV_EPS", type=float, default=None,
                    help="Training epsilon folder value, e.g. 6 for Linf or 1.0 for L2")

parser.add_argument("--JSON_PATH", type=str, default=None,
                    help="Optional summary JSON path")

parser.add_argument("--CKPT_PATH", type=str, default=None,
                    help="Optional direct checkpoint path")

parser.add_argument("--SEED", type=int, default=42)

parser.add_argument("--TRN_TYPE", type=str, default="square",
                    choices=["square", "ce", "adv_ce"],
                    help="Training type: square, ce, or adv_ce")

parser.add_argument("--NUM_FEATURES", type=int, default=15000,
                    help="Number of random features to use for RFF models")

args = parser.parse_args()

SEED = args.SEED
Dataset = args.DATASET
AA_NORM = args.AA_norm
norm_name = AA_NORM.lower()

pgd = args.PGD
bb_attack = args.BB_attack
mc = args.MC

model_type = args.MODEL_TYPE
rff_kernel = args.RFF_KERNEL.lower() if args.RFF_KERNEL is not None else None
adv_lenet = args.ADV_LENET
adv_eps = args.ADV_EPS

# Comment this out if you are not attacking RFF models.
num_features = args.NUM_FEATURES

JSON_PATH = args.JSON_PATH
CKPT_PATH = args.CKPT_PATH

trn_type = args.TRN_TYPE

if adv_lenet and model_type == "lenet":
    if AA_NORM == "Linf":
        adv_folder = "linf_eps_6"
    elif AA_NORM == "L2":
        adv_folder = "l2_eps_1.0"
    else:
        raise ValueError(f"Unsupported AA_NORM: {AA_NORM}")
else:
    adv_folder = None


if CKPT_PATH is None and model_type == "lenet":
    if adv_lenet:
        if mc:
            CKPT_PATH = (
                f"/janaki/backup/users/student/rs/rahulky/water/retrain/"
                f"FCNN_lenet_{Dataset}_mc/"
                f"{adv_folder}/"
                f"fcnn_sweep_sgd_{Dataset}_mc/"
                f"best_model.pt"
            )
        else:
            CKPT_PATH = None
    else:
        if mc:
            CKPT_PATH = (
                f"/janaki/backup/users/student/rs/rahulky/water/retrain/"
                f"FCNN_lenet_{Dataset}_mc/"
                f"fcnn_sweep_sgd_{Dataset}_mc/"
                f"best_model.pt"
            )
        else:
            CKPT_PATH = None

print(f"Dataset: {Dataset}, AA_NORM: {AA_NORM}, PGD: {pgd}, BB_ATTACK: {bb_attack}, MC: {mc}, MODEL_TYPE: {model_type}, RFF_KERNEL: {rff_kernel}, CKPT_PATH: {CKPT_PATH}, JSON_PATH: {JSON_PATH}")

results_out: List[Dict[str, Any]] = []
errors_out: List[Dict[str, str]] = []

def load_summary():
    print("JSON_PATH:", JSON_PATH)
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

    # if model_type == "fcnn":
    #     if not mc:
    #         path = f"./FCNN_nn_{Dataset}_per_class/fcnn_sgd_bin_best_{Dataset}_class_pair_runs.json"
    #     else:
    #         path = f"./FCNN_nn_{Dataset}_mc/fcnn_sgd_bin_best_{Dataset}_mc_runs.json"
# Directory where attack_nn.py lives
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    if model_type == "fcnn":
        if not mc:
            path = os.path.join(SCRIPT_DIR, f"FCNN_nn_{Dataset}_per_class", 
                                f"FCNN_nn_{Dataset}_per_class.json")
        else:
            path = os.path.join(SCRIPT_DIR, f"FCNN_nn_{Dataset}_mc", 
                                f"FCNN_nn_{Dataset}_mc.json")
    elif model_type == "lenet":
        if not mc:
            path = f"./FCNN_lenet_{Dataset}_per_class/lenet_{Dataset}_per_class.json"
        else:
            path = f"./FCNN_lenet_{Dataset}_mc/fcnn_sgd_bin_best_{Dataset}_mc_runs.json"
    # elif model_type == "rff":
    #     raise ValueError("For RFF, pass --CKPT_PATH or --JSON_PATH.")

    elif model_type == "rff":
        pass
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if model_type != "rff":
        with open(path, "r") as f:
            return json.load(f)

summary = load_summary()

print(summary)

# breakpoint()


        # if ce or adv_ce, rewrite paths to point into subfolder
if model_type == "rff" and trn_type in ["ce", "adv_ce"] and not mc:
    json_dir = os.path.dirname(os.path.abspath(JSON_PATH))
    for entry in summary.get("results", []):
        original_path = entry["path"]
        # get just the filename e.g. qmnist_(0,1)_rff_best.pt
        fname = os.path.basename(original_path)
        # build new path inside ce/ or adv_ce/
        entry["path"] = os.path.join(
            json_dir, trn_type, fname
        )

# -------------------------
# Data loading
# -------------------------
def load_dataset_svhn(mc):

    return dataset.get_svhn(multiclass=mc)

                # trainloader, valloader, testloader = load_dataset_svhn()


def get_loaders(Dataset, mc, classes):
    if Dataset == "svhn":
        return load_dataset_svhn(mc)

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
    elif Dataset == "cifar10":
        if not mc:
            return dataset.get_cifar(classes=classes, multiclass = mc)
        return dataset.get_cifar(classes=(0,1),multiclass=mc)
    else:
        raise ValueError(f"Unsupported dataset: {Dataset}")

def spsa_fast(loss_fn, x, y, epsilon=0.03, eta=0.01, lr=0.01,
              iterations=40, sample_size=128, norm="Linf", seed=None):

    x_adv = x.detach().clone()
    B, D = x_adv.shape
    eta = float(np.clip(epsilon / 2, 1e-3, 0.01))

    generator = None
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)

    print("I'm in spsa fast", flush=True)

    for _ in range(iterations):
        print(_,"I am rademacher", flush=True)

        # if norm == "Linf":
        v = torch.empty(sample_size, B, D, device = x.device).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
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

# -------------------------
# Main loop
# -------------------------
for entry in summary.get("results", []):
    try:
        exp_name = entry["experiment"]
        exp_path = entry["path"]
        print("Experiment Path:", exp_path)
        # breakpoint()
        if model_type == "fcnn":
            if not mc:
                c1, c2 = extract_classes_from_path(entry["path"])
                classes = (c1, c2)
            else:
                classes = (0, 1)
            # if not mc:
            #     classes_str = entry["classes"]

            # best_run = entry.get("best_run", {})

            # if not mc:
            #     c1, c2 = int(classes_str[0]), int(classes_str[1])
            #     # if Dataset == "fmnist":
            #     #     if (c1 == 7 and c2 == 9) or (c1 == 8 and c2 == 9):
            #     #         classes = (c2, c1)
            #     #     else:
            #     #         classes = (c1, c2)
            #     # else:
            #     #     classes = (c1, c2)
            #     classes = (c1, c2)
            # else:
            #     classes = (0, 1)
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

        # elif model_type == "rff":
        #     ckpt1 = torch.load(ckpt_path, map_location=device)
        #     params = ckpt1["phi_params"]

        #     input_dim=params["input_dim"]
        #     num_features=params["num_features"]
        #     kernel_type=params["kernel_type"]
        #     length_scale=params["length_scale"]
        #     seed=params["seed"]
        #     width = 64
        #     depth = 3

        #     print("Input Dim:", input_dim, "num_features:", num_features,
        #     "kernel_type:", kernel_type, "length_scale:",length_scale, "seed:", seed)

        #     phi = select_kernel(
        #         input_dim=params["input_dim"],
        #         num_features=params["num_features"],
        #         kernel_type=params["kernel_type"],
        #         length_scale=params["length_scale"],
        #         seed=params["seed"],
        #         device=device,
        #     )

        #     U = ckpt1["U"].to(device)
        #     model = RFFClassifier(phi, U).to(device)

        elif model_type == "rff":

            X_sample, y_sample = next(iter(testloader))
            print(f"X range: min={X_sample.min():.4f} max={X_sample.max():.4f}")
            print(f"X dtype: {X_sample.dtype}")
            print(f"X shape: {X_sample.shape}")
            print(f"y sample: {y_sample[:5]}")
            print(f"classes: {torch.unique(torch.argmax(y_sample, dim=-1))}")
            # Fingerprint the actual data
            print(f"X first sample first 10: {X_sample[0].flatten()[:10].tolist()}")

            # breakpoint()

            ckpt1 = torch.load(ckpt_path, map_location=device)
            params = ckpt1["phi_params"]

            input_dim = params["input_dim"]
            num_features = params["num_features"]
            width = num_features
            depth = 1

            print(f"[DEBUG] phi_params: {params}")
            print(f"[DEBUG] trn_type: {trn_type}")
            print(f"[DEBUG] checkpoint keys: {list(ckpt1.keys())}")

            if trn_type in ["square" ,"ce", "adv_ce"]:
                num_classes = int(ckpt1.get("num_classes", 10 if mc else 2))

                phi = select_kernel(
                    input_dim=params["input_dim"],
                    num_features=params["num_features"],
                    kernel_type=params["kernel_type"],
                    length_scale=params["length_scale"],
                    seed=params["seed"],
                    device=device,
                )

                # Load the exact random weights used during training — ignore re-seeding
                # phi.load_state_dict({
                #     k: v.to(device)
                #     for k, v in ckpt1["phi_state_dict"].items()   # ← add this
                # })
                # phi.eval()  # ← add this

                probe = torch.zeros(1, params["input_dim"], device=device)
                print("phi probe:", phi(probe)[:, :5])  # first 5 features

                # breakpoint()

                clf = nn.Linear(num_features, num_classes).to(device)
                clf.load_state_dict({
                    k: v.to(device)
                    for k, v in ckpt1["linear_state_dict"].items()
                })
                clf.eval()

                class RFFClassifierCE(nn.Module):
                    def __init__(self, phi, clf):
                        super().__init__()
                        self.phi = phi
                        self.clf = clf

                    def forward(self, x):
                        if x.ndim > 2:
                            x = x.view(x.size(0), -1)
                        return self.clf(self.phi(x))

                model = RFFClassifierCE(phi, clf).to(device)
        else:
            raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")

        model.eval()

        # DEBUG: check clean accuracy first
        def get_clean_acc(net, loader):
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in loader:
                    out = forward_model(net, X.to(device))
                    preds = torch.argmax(out, dim=-1)
                    labels = torch.argmax(y.to(device), dim=-1)
                    correct += (preds == labels).sum().item()
                    total += X.size(0)
            return 100.0 * correct / total

        clean_acc = get_clean_acc(model, testloader)
        print(f"[DEBUG] Clean accuracy (no attack): {clean_acc:.2f}%")
        print(f"[DEBUG] Expected: should be high (>80%) before any attack")

        # -------------------------
        # Attacks
        # -------------------------
        def pgd_attack(net, X, y, epsilon, alpha, num_iters, loss_fn='mse', seed = None):
            net.eval()
            X_nat = X.detach().to(device)
            y = y.detach().to(device)

            if loss_fn == 'cross_entropy':
                y = y.argmax(dim=1).to(device)

            X_adv = X_nat.clone().detach().to(device)

            if pgd is True and AA_NORM == "Linf":
                # print("We are in the PGD l-inf attack")
                print("We are in the PGD l-inf attack")
                print(X_adv.max(), X_adv.min())

                def forw(x):
                    return forward_model(net, x)

                adversary = AutoAttack(forw, norm='Linf', eps=epsilon, version='standard', seed=SEED)
                adversary.attacks_to_run = ['apgd-ce']
                X_adv = adversary.run_standard_evaluation(X_adv, y, bs=X_adv.shape[0])
                return X_adv.detach()
                # for _ in range(num_iters):
                #     X_adv.requires_grad_(True)
                #     if X_adv.grad is not None:
                #         X_adv.grad.zero_()
                #     with torch.enable_grad():
                #         out = forward_model(net, X_adv)
                #         if loss_fn == 'mse':
                #             loss = F.mse_loss(out, y)
                #         else:
                #             loss = F.cross_entropy(out, y)
                #         net.zero_grad()
                #         loss.backward()

                #     with torch.no_grad():
                #         X_adv = X_adv + alpha * X_adv.grad.sign()
                #         X_adv = torch.max(torch.min(X_adv, X_nat + epsilon), X_nat - epsilon)
                #         X_adv = torch.clamp(X_adv, 0.0, 1.0)

                # return X_adv.detach()

            elif pgd is True and AA_NORM == "L2":
                print("We are in the PGD l-2 attack")
                print(X_adv.max(), X_adv.min())

                def forw(x):
                    return forward_model(net, x)

                adversary = AutoAttack(forw, norm='L2', eps=epsilon, version='standard', seed=SEED)
                adversary.attacks_to_run = ['apgd-ce']
                X_adv = adversary.run_standard_evaluation(X_adv, y, bs=X_adv.shape[0])
                return X_adv.detach()

            elif pgd is False and bb_attack == "spsa":
                print("We are in the SPSA black-box attack")
                loss_fn_local = lambda xb, yb: F.cross_entropy(forward_model(net, xb), yb, reduction="none")
                # spsa_sample_size = 500
                spsa_sample_size =16
                # if epsilon > 0:
                #     X_adv = spsa(
                #         loss_fn_local,
                #         X_adv,
                #         y,
                #         epsilon=epsilon,
                #         eta=0.01,
                #         lr=alpha,
                #         iterations=num_iters,
                #         sample_size=spsa_sample_size,
                #         norm=AA_NORM
                #     )
                # return X_adv.detach()
                if epsilon > 0:
                    print("Running SPSA attack with epsilon =", epsilon)
                    X_adv = spsa_fast(
                        loss_fn_local,
                        X_adv,
                        y,
                        epsilon=epsilon,
                        eta=0.01,
                        lr=alpha,
                        iterations=num_iters,
                        sample_size=spsa_sample_size,
                        norm=AA_NORM,
                        seed=seed
                    )
                return X_adv.detach()

        # def get_adv_acc(net, loader, epsilon=4/255, alpha=1/255, num_iters=15, loss_fn='mse'):
        #     net.eval()
        #     correct = 0
        #     total = 0
        #     for X, y in loader:
        #         X_adv = pgd_attack(net, X, y, epsilon, alpha, num_iters, loss_fn)
        #         with torch.no_grad():
        #             preds = torch.argmax(forward_model(net, X_adv), dim=-1)
        #             labels = torch.argmax(y.to(device), dim=-1)
        #             correct += torch.sum(preds == labels).item()
        #             total += X.size(0)

        #     return 100.0 * correct / total

        def get_adv_acc(net, loader, epsilon=4/255, alpha=1/255, num_iters=15, loss_fn='mse'):
            """classification accuracy on PGD-generated inputs."""
            net.eval()
            correct = 0
            total = 0

            for batch_id, (X, y) in enumerate(loader):
                print(f"Batch {batch_id} start", flush=True)
                X_adv = pgd_attack(
                    net, X, y, epsilon, alpha, num_iters, loss_fn,
                    seed=SEED + batch_id,
                )
                print(f"Batch {batch_id} pgd done", flush=True)
                torch.cuda.synchronize()
                print(f"Batch {batch_id} sync done", flush=True)
                with torch.no_grad():
                    preds  = torch.argmax(forward_model(net, X_adv), dim=-1)
                    labels = torch.argmax(y.to(device), dim=-1)
                    correct += torch.sum(preds == labels).item()
                    total   += X.size(0)
                print(f"Batch {batch_id} done | correct={correct} total={total}", flush=True)

        #     for batch_id, (X, y) in enumerate(loader):
        #         print(f"Batch {batch_id} | "
        #   f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB | "
        #   f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")
        #         torch.cuda.empty_cache()   
        #         X_adv = pgd_attack(
        #                         net,
        #                         X,
        #                         y,
        #                         epsilon,
        #                         alpha,
        #                         num_iters,
        #                         loss_fn,
        #                         seed=SEED + batch_id,
        #                     )
        #         # X_adv = pgd_attack(net, X, y, epsilon, alpha, num_iters, loss_fn)
        #         with torch.no_grad():
        #             preds = torch.argmax(forward_model(net, X_adv), dim=-1)
        #             labels = torch.argmax(y.to(device), dim=-1)
        #             correct += torch.sum(preds == labels).item()
        #             total += X.size(0)

            return 100.0 * correct / total

        if AA_NORM == "Linf":
            # epsilons = [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0]
            if mc:
                epsilons = [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 32.0, 40.0, 55.0]
            else:
                epsilons = [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0]
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
            # else:
            #     base_dir = f"./rff_{rff_kernel}_{Dataset}_{'mc' if mc else 'pairwise'}"
            #     suffix = "rff"
            else:
                # New path
                if trn_type == "square":
                    base_dir = f"./rff_code/rff_{rff_kernel}_{Dataset}_{'mc' if mc else 'pairwise'}"
                else:
                    base_dir = f"./rff_code/rff_{rff_kernel}_{Dataset}_{'mc' if mc else 'pairwise'}/{trn_type}"
                suffix = "rff"
                # if trn_type == "square":
                #     base_dir = f"./rff_{rff_kernel}_{Dataset}_{'mc' if mc else 'pairwise'}"
                # else:
                #     base_dir = f"./rff_{rff_kernel}_{Dataset}_{'mc' if mc else 'pairwise'}/{trn_type}"
                # suffix = "rff"

            if AA_NORM == "Linf":
                if not mc:
                    save_path = (
                        f"{base_dir}/seed_{SEED}/"
                        f"adv_accuracy_results_linf_{Dataset}_class_pair_{suffix}.json"
                    )
                else:
                    save_path = (
                        f"{base_dir}/seed_{SEED}/"
                        f"adv_accuracy_results_linf_{Dataset}_mc_{suffix}.json"
                    )
            else:
                if not mc:
                    save_path = (
                        f"{base_dir}/seed_{SEED}/"
                        f"adv_accuracy_results_l2_{Dataset}_class_pair_{suffix}.json"
                    )
                else:
                    save_path = (
                        f"{base_dir}/seed_{SEED}/"
                        f"adv_accuracy_results_l2_{Dataset}_mc_{suffix}.json"
                    )

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "w") as f:
                json.dump(adv_results, f, indent=2)

        results_out.append({
            "experiment": exp_name,
            "path": exp_path,
            "classes": [c1, c2] if not mc else [0, 1],
            # "hyperparams": {
            #     "input_dim": input_dim,
            #     "width": width,
            #     "depth": depth,
            # },
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

# if model_type == "fcnn":
#     base_dir = f"./FCNN_nn_{Dataset}_{'mc' if mc else 'per_class'}"
#     suffix = "nn"
# elif model_type == "lenet":
#     base_dir = f"./FCNN_lenet_{Dataset}_{'mc' if mc else 'per_class'}"
#     suffix = "lenet"
# else:
#     if trn_type == "square":
#         base_dir = f"./rff_{rff_kernel}_{Dataset}_{'mc' if mc else 'pairwise'}"
#     else:
#         base_dir = f"./rff_{rff_kernel}_{Dataset}_{'mc' if mc else 'pairwise'}/{trn_type}"
#     suffix = "rff"

norm_name = AA_NORM.lower()

if model_type == "rff":
    if pgd is True:
        attack_prefix = f"PGD_result_{norm_name}_sd_{params['seed']}_num_features_{num_features}"
    else:
        attack_prefix = f"PGD_result_{norm_name}_bb_sd_{params['seed']}_num_features_{num_features}"
    base_dir = "rff_code"
else:  # fcnn or lenet
    if pgd is True:
        attack_prefix = f"PGD_result_{norm_name}"
    else:
        attack_prefix = f"PGD_result_{norm_name}_bb"
    base_dir = f"./FCNN_nn_{Dataset}" if model_type == "fcnn" else f"./FCNN_lenet_{Dataset}"
    base_dir += "_mc" if mc else "_per_class"

suffix = "rff" if model_type == "rff" else model_type

if not mc:
    task_suffix = f"{Dataset}_class_pair_{suffix}"
else:
    task_suffix = f"{Dataset}_mc_{suffix}"

if adv_lenet and model_type == "lenet":
    out_path = f"{base_dir}/{adv_folder}/seed_{SEED}/{attack_prefix}_{task_suffix}.json"
else:
    out_path = f"{base_dir}/seed_{SEED}/{attack_prefix}_{task_suffix}.json"

os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w") as f:
    json.dump(out_obj, f, indent=2)

print(f"Saved PGD results to: {out_path}")
if errors_out:
    print(f"WARNING: {len(errors_out)} experiments failed. See 'errors' in the output JSON.")

# norm_name = AA_NORM.lower()

# if model_type == "rff":
#     if pgd is True:
#         attack_prefix = f"PGD_result_{norm_name}_sd_{params['seed']}_num_features_{num_features}"
#     else:
#         attack_prefix = f"PGD_result_{norm_name}_bb_sd_{params['seed']}"
# else:  # fcnn or lenet
#     if pgd is True:
#         attack_prefix = f"PGD_result_{norm_name}_"
#     else:
#         attack_prefix = f"PGD_result_{norm_name}_"

# suffix = "rff" if model_type == "rff" else model_type

# if not mc:
#     task_suffix = f"{Dataset}_class_pair_{suffix}"
# else:
#     task_suffix = f"{Dataset}_mc_{suffix}"


# base_dir = "rff_code"
# if adv_lenet and model_type == "lenet":
#     out_path = f"{base_dir}/{adv_folder}/seed_{SEED}/{attack_prefix}_{task_suffix}.json"
# else:
#     out_path = f"{base_dir}/seed_{SEED}/{attack_prefix}_{task_suffix}.json"

# os.makedirs(os.path.dirname(out_path), exist_ok=True)

# with open(out_path, "w") as f:
#     json.dump(out_obj, f, indent=2)

# print(f"Saved PGD results to: {out_path}")
# if errors_out:
#     print(f"WARNING: {len(errors_out)} experiments failed. See 'errors' in the output JSON.")

# python -u attack_nn.py \
#   --DATASET fmnist \
#   --AA_norm Linf \
#   --BB_attack spsa \
#   --MC \
#   --MODEL_TYPE lenet \
#   --SEED 42 \
#   --ADV_LENET  
# SEED 42, 100, 120
