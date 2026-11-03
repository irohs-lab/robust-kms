
import dataset

import random

import os
import json
import argparse
from typing import Dict, Any, Tuple, List
from autoattack import AutoAttack

import numpy as np

import torch
import torch.nn.functional as F

from best_nn import FCNN2, sweep_fcnn_sgd # must exist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse

parser = argparse.ArgumentParser(description="Adversarial Attack Evaluation")

parser.add_argument("--DATASET", type=str)
parser.add_argument("--AA_norm", type=str)
parser.add_argument("--PGD", action = "store_true")
parser.add_argument("--BB_attack", type=str)
parser.add_argument("--MC", action="store_true")
parser.add_argument("--eps_trn", type=float)
parser.add_argument("--SEED", type=int, default=42)
args = parser.parse_args()

Dataset = args.DATASET

tr_norm = args.AA_norm


AA_NORM = args.AA_norm


pgd = args.PGD


bb_attack = args.BB_attack 

mc = args.MC


eps_trn = args.eps_trn

SEED = args.SEED

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

if not mc:
    with open(f"./FCNN_nn_{Dataset}_per_class/file_paths_{tr_norm}_eps_{eps_trn}.json", "r") as f:
        summary = json.load(f)
else:
    with open(f"./FCNN_nn_{Dataset}_mc/file_paths_{tr_norm}_eps_{eps_trn}.json", "r") as f:
        summary = json.load(f)

results_out: List[Dict[str, Any]] = []
errors_out: List[Dict[str, str]] = []

import re

def seed_dir(base_dir):
    return os.path.join(base_dir, f"seed_{SEED}")


def save_json_with_seed(path_without_seed, obj):
    """
    Takes a normal path like:
        ./FCNN_nn_qmnist_per_class/l2_eps_1.0/PGD_result_l2_qmnist_class_pair_nn.json

    Saves it as:
        ./FCNN_nn_qmnist_per_class/l2_eps_1.0/seed_42/PGD_result_l2_qmnist_class_pair_nn.json
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

def load_dataset_svhn(classes):

    return dataset.get_svhn(classes = classes)

def spsa(loss_fn,x,y,epsilon=0.03, eta=0.01, lr=0.01, iterations=40, sample_size=128,norm="Linf"):
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
            # Gaussian for l-2, rademacher for l-inf
            if norm == "Linf":
                print("I am Rademacher direction")
                v = torch.empty_like(x_adv).bernoulli_(0.5).mul_(2).sub_(1)
            else:
                v = torch.randn_like(x_adv)
                if v.dim() == 1:
                    v = v / v.norm(p=2).clamp_min(1e-12)
                else:
                    v = v / v.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

            with torch.no_grad():
                loss_plus  = loss_fn((x_adv + eta * v).clamp(0, 1), y)
                loss_minus = loss_fn((x_adv - eta * v).clamp(0, 1), y)
            diff = (loss_plus - loss_minus)          # shape: (B,) or scalar
            diff = diff.view(-1, 1)            # for images

            grad_estimate += (diff / (2 * eta)) * v

        
        grad_estimate /= sample_size

        if norm=="linf":
            x_adv = x_adv + lr * torch.sign(grad_estimate)
        else:
            x_adv = x_adv + lr * grad_estimate / (torch.norm(grad_estimate, p=2, dim=1, keepdim=True).clamp_min(1e-12))        
        
        delta=x_adv-x
        
        if norm=="linf":
            delta=torch.clamp(delta,min=-epsilon,max=epsilon)
        else:
            delta_norm = torch.norm(delta, p=2, dim=1, keepdim=True).clamp_min(1e-12)
            scale = (epsilon / delta_norm).clamp(max=1.0)
            delta = delta * scale
        
        x_adv=x+delta

        if torch.isnan(delta).any():
            print("NaNs in delta")

        if torch.isnan(x_adv).any():
            print("NaNs in x_adv")
        x_adv=torch.clamp(x_adv,min=0,max=1)
    delta = x_adv - x
    print("L2 =", delta.view(delta.size(0), -1).norm(p=2, dim=1).max().item())    
    return x_adv

def spsa_fast(loss_fn, x, y, epsilon=0.03, eta=0.01, lr=0.01,
              iterations=40, sample_size=128, norm="Linf", seed=None):

    x_adv = x.detach().clone()
    B, D = x_adv.shape
    eta = float(np.clip(epsilon / 2, 1e-3, 0.01))

    generator = None
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)

    print("I'm in spsa fast")

    for _ in range(iterations):
        print("I am rademacher")

        v = torch.empty(sample_size, B, D, device = x.device).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)


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

        if norm == "linf":
            x_adv = x_adv + lr * torch.sign(grad_estimate)
        else:
            g_norm = grad_estimate.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            x_adv = x_adv + lr * grad_estimate / g_norm

        delta = x_adv - x

        if norm == "linf":
            delta = torch.clamp(delta, -epsilon, epsilon)
        else:
            d_norm = delta.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            delta = delta * (epsilon / d_norm).clamp(max=1.0)

        x_adv = torch.clamp(x + delta, 0, 1).detach()

    return x_adv

for entry in summary.get("results", []):
    try:
        exp_name = entry["experiment"]
        exp_path = entry["path"]


        if not mc:
            c1, c2 = extract_classes_from_path(entry["path"])
            classes = (c1, c2)
        else:
            classes = (0, 1)

        ckpt_path = entry["path"]

        ckpt = torch.load(ckpt_path, map_location=device)

        hparams = ckpt["hparams"]

        input_dim = int(ckpt.get("input_dim", 784))
        width = int(hparams["width"])
        depth = int(hparams["depth"])
        weight_decay = float(hparams["weight_decay"])
        lr = float(hparams["lr"])
        epoch = int(ckpt["best_epoch"])
        
        if Dataset == "svhn":
            trainloader, valloader, testloader = load_dataset_svhn(classes)
        elif Dataset == "kmnist":
            if not mc:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=classes , which_mnist='kmnist', multiclass=mc)
            else:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=(0,1) , which_mnist='kmnist', multiclass=mc)
        elif Dataset == "qmnist":
            if not mc:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=classes , which_mnist='qmnist', multiclass=mc)
            else:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=(0,1) , which_mnist='qmnist', multiclass=mc)
        elif Dataset == "emnist":
            if not mc:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=classes , which_mnist='emnist', multiclass=mc)
            else:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=(0,1) , which_mnist='emnist', multiclass=mc)
        elif Dataset == "fmnist":
            if not mc:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=classes , which_mnist='fmnist', multiclass=mc)
            else:
                trainloader, valloader, testloader = dataset.load_fmnist(classes=(0,1) , which_mnist='fmnist', multiclass=mc)
        elif Dataset == "cifar10":
            if not mc:
                trainloader, valloader, testloader = dataset.get_cifar(classes=classes, multiclass = mc)
            else:
                trainloader, valloader, testloader = dataset.get_cifar(classes=(0,1),multiclass=mc)

        print(ckpt_path)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing best_model.pt at: {ckpt_path}")

        if not mc:
            model = FCNN2(input_dim=input_dim, width=width, depth=depth, dropout=0, num_classes=2).to(device)
        else:
            model = FCNN2(input_dim=input_dim, width=width, depth=depth, dropout=0, num_classes=10).to(device)
        ckpt1 = torch.load(ckpt_path, map_location=device) 
        model.load_state_dict(ckpt1['state_dict'])

        def pgd_attack(net, X, y, epsilon, alpha, num_iters,loss_fn = 'mse',seed=None):
            """
            L-infinity PGD attack that always runs with gradients on.
            """
            net.eval()
            X_nat = X.detach().to(device)
            y     = y.detach().to(device)

            if loss_fn == 'cross_entropy':                      # one‑hot case
                y = y.argmax(dim=1).to(device)

            X_adv = X_nat.clone().detach().to(device)

            if pgd == True and AA_NORM == "linf":
                print("We are in the PGD l-inf attack")
                X_nat=X_nat.flatten(1)
                print(X_adv.max(), X_adv.min())
                def forw(x): return net(x)
                adversary=AutoAttack(forw, norm='Linf', eps=epsilon, version='standard',seed=SEED)
                adversary.attacks_to_run = ['apgd-ce']
                print(X_adv.shape)
                X_adv = adversary.run_standard_evaluation(X_adv, y, bs=X_adv.shape[0])
                return X_adv.detach()
            elif pgd == True and AA_NORM == "l2":
                print("We are in the PGD l-2 attack")
                X_nat=X_nat.flatten(1)
                print(X_adv.max(), X_adv.min())
                def forw(x): return net(x)
                adversary=AutoAttack(forw, norm='L2', eps=epsilon, version='standard',seed=SEED)
                adversary.attacks_to_run = ['apgd-ce']
                print(X_adv.shape)
                X_adv = adversary.run_standard_evaluation(X_adv, y, bs=X_adv.shape[0])
                return X_adv.detach()
            elif pgd == False and bb_attack == "spsa":
                print("We are in the SPSA black-box attack")

                loss_fn = lambda xb, yb: F.cross_entropy(net(xb) , yb, reduction="none")                      # forward

                spsa_sample_size = 16 

                if epsilon > 0:
                    X_adv = spsa_fast(
                        loss_fn,
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
        def get_adv_acc(net, loader, epsilon=4/255, alpha=1/255, num_iters=15, loss_fn='mse'):
            """classification accuracy on PGD-generated inputs."""
            net.eval()
            correct = 0
            total = 0
            for batch_id, (X, y) in enumerate(loader):
                X_adv = pgd_attack(
                                net,
                                X,
                                y,
                                epsilon,
                                alpha,
                                num_iters,
                                loss_fn,
                                seed=SEED + batch_id,
                            )
                with torch.no_grad():
                    preds = torch.argmax(net(X_adv), dim=-1)
                    labels = torch.argmax(y.to(device), dim=-1)
                    correct += torch.sum(preds == labels).item()
                    total += X.size(0)

            return 100.0 * correct / total


        if AA_NORM == "linf":
            if not mc:
                epsilons = [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0]
            else:
                epsilons = [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0, 55.0]

        elif AA_NORM == "l2":
            epsilons = [0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

        adv_accs = []

        for j in epsilons:
            adv_results = []
            if AA_NORM == "linf":
                eps_float = j/255.0   
            elif AA_NORM == "l2":             
                eps_float = j
            alpha = eps_float/10
            acc = get_adv_acc(model, testloader, eps_float, alpha, 100,'cross_entropy')

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
            


            if AA_NORM == "linf":
                if not mc:
                    tmp_path = (
                        f"./FCNN_nn_{Dataset}_per_class/"
                        f"{tr_norm}_eps_{eps_trn}/"
                        f"adv_accuracy_results_linf_{Dataset}_class_pair_nn.json"
                    )
                    save_json_with_seed(tmp_path, adv_results)

                else:
                    tmp_path = (
                        f"./FCNN_nn_{Dataset}_mc/"
                        f"{tr_norm}_eps_{eps_trn}/"
                        f"adv_accuracy_results_linf_{Dataset}_mc_nn.json"
                    )
                    save_json_with_seed(tmp_path, adv_results)

            elif AA_NORM == "l2":
                if not mc:
                    tmp_path = (
                        f"./FCNN_nn_{Dataset}_per_class/"
                        f"{tr_norm}_eps_{eps_trn}/"
                        f"adv_accuracy_results_l2_{Dataset}_class_pair_nn.json"
                    )
                    save_json_with_seed(tmp_path, adv_results)

                else:
                    tmp_path = (
                        f"./FCNN_nn_{Dataset}_mc/"
                        f"{tr_norm}_eps_{eps_trn}/"
                        f"adv_accuracy_results_l2_{Dataset}_mc_nn.json"
                    )
                    save_json_with_seed(tmp_path, adv_results)
        
        results_out.append({
                "experiment": exp_name,
                "path": exp_path,
                "classes": [c1, c2] if not mc else [0, 1],
                "hyperparams": {
                    "input_dim": input_dim,
                    "width": width,
                    "depth": depth,
                },
                "pgd": adv_accs,
            })

        if not mc:
            results_out[-1]["classes"] = [c1, c2]

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


if AA_NORM == "linf":
    if pgd == True:
        if not mc:
            save_path = (
                f"./FCNN_nn_{Dataset}_per_class/"
                f"{tr_norm}_eps_{eps_trn}/"
                f"PGD_result_linf_{Dataset}_class_pair_nn.json"
            )
        else:
            save_path = (
                f"./FCNN_nn_{Dataset}_mc/"
                f"{tr_norm}_eps_{eps_trn}/"
                f"PGD_result_linf_{Dataset}_mc_nn.json"
            )

    else:
        if not mc:
            save_path = (
                f"./FCNN_nn_{Dataset}_per_class/"
                f"{tr_norm}_eps_{eps_trn}/"
                f"PGD_result_linf_bb_{Dataset}_class_pair_nn.json"
            )
        else:
            save_path = (
                f"./FCNN_nn_{Dataset}_mc/"
                f"{tr_norm}_eps_{eps_trn}/"
                f"PGD_result_linf_bb_{Dataset}_mc_nn.json"
            )


elif AA_NORM == "l2":
    if pgd == True:
        if not mc:
            save_path = (
                f"./FCNN_nn_{Dataset}_per_class/"
                f"{tr_norm}_eps_{eps_trn}/"
                f"PGD_result_l2_{Dataset}_class_pair_nn.json"
            )
        else:
            save_path = (
                f"./FCNN_nn_{Dataset}_mc/"
                f"{tr_norm}_eps_{eps_trn}/"
                f"PGD_result_l2_{Dataset}_mc_nn.json"
            )

    else:
        if not mc:
            save_path = (
                f"./FCNN_nn_{Dataset}_per_class/"
                f"{tr_norm}_eps_{eps_trn}/"
                f"PGD_result_l2_bb_{Dataset}_class_pair_nn.json"
            )
        else:
            save_path = (
                f"./FCNN_nn_{Dataset}_mc/"
                f"{tr_norm}_eps_{eps_trn}/"
                f"PGD_result_l2_bb_{Dataset}_mc_nn.json"
            )


save_path = save_json_with_seed(save_path, out_obj)

print(f"Saved PGD results to: {save_path}")

if errors_out:
    print(f"WARNING: {len(errors_out)} experiments failed. See 'errors' in the output JSON.")
