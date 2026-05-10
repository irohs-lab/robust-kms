# For multi-class

# import sys
# sys.path.append("/users/student/rs/rahulky/robust-rfms/nfa_src/src")
import dataset

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

# parser.add_argument("--DATASET", type=str)
# parser.add_argument("--AA_norm", type=str)
# parser.add_argument("--PGD", action = "store_true")
# parser.add_argument("--BB_attack", type=str)
# parser.add_argument("--MC", action="store_true")
# parser.add_argument("--eps_trn", type=float)

args = parser.parse_args()

Dataset = "qmnist"

tr_norm = "l2"

AA_NORM = "l2"

pgd = True

bb_attack = "False"
 
version = "rfm"
 
mc = False

eps_trn = 1.0

# Dataset = args.DATASET
# AA_NORM = args.AA_norm
# pgd = args.PGD
# bb_attack = args.BB_attack
# # version = args.VERSION
# mc = args.MC
# eps_trn = args.eps_trn

# with open("fcnn_sgd_bin_fmnist_mc.json", "r") as f:
#     summary = json.load(f)
# if not mc:
#     with open(f"./FCNN_nn_{Dataset}_per_class/fcnn_sgd_bin_best_{Dataset}_class_pair_runs.json", "r") as f:
#         summary = json.load(f)
# else:
#     with open(f"./FCNN_nn_{Dataset}_mc/fcnn_sgd_bin_best_{Dataset}_mc_runs.json", "r") as f:
#         summary = json.load(f)

if not mc:
    with open(f"./FCNN_nn_{Dataset}_per_class/file_paths_{tr_norm}_eps_{eps_trn}.json", "r") as f:
        summary = json.load(f)
else:
    with open(f"./FCNN_nn_{Dataset}_mc/file_paths_{tr_norm}_eps_{eps_trn}.json", "r") as f:
        summary = json.load(f)

results_out: List[Dict[str, Any]] = []
errors_out: List[Dict[str, str]] = []

#  Only for 2-class experiments
# def load_fmnist(clas):
# def load_fmnist():
#     # return dataset.load_fmnist(classes = clas)
#     return dataset.load_fmnist()

import re

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
    # eta=1e-2
    # eta = 1e-3
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
                # v = v / torch.norm(v, p=2, dim=1, keepdim=True).clamp_min(1e-12)  # normalize to unit l-2 norm
            else:
                # print("I am Gaussian direction")
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

            # grad_estimate += 784*(diff / (2 * eta)) * v
            grad_estimate += (diff / (2 * eta)) * v

        
        grad_estimate /= sample_size

        if norm=="Linf":
            # lr needs to be smaller for huge epsilons, for FCNN.
            x_adv = x_adv + lr * torch.sign(grad_estimate)
        else:
            x_adv = x_adv + lr * grad_estimate / (torch.norm(grad_estimate, p=2, dim=1, keepdim=True).clamp_min(1e-12))        
        
        delta=x_adv-x
        
        if norm=="Linf":
            delta=torch.clamp(delta,min=-epsilon,max=epsilon)
        else:
            # delta=epsilon*delta/torch.norm(delta,p=2,dim=1, keepdim=True)
            delta_norm = torch.norm(delta, p=2, dim=1, keepdim=True).clamp_min(1e-12)
            scale = (epsilon / delta_norm).clamp(max=1.0)
            delta = delta * scale
        
        x_adv=x+delta
        # print(f"Iteration {i+1}/{iterations}, max perturbation: {delta.view(delta.size(0), -1).abs().max(dim=1)[0].mean().item():.4f}")

        if torch.isnan(delta).any():
            print("NaNs in delta")

        if torch.isnan(x_adv).any():
            print("NaNs in x_adv")
        x_adv=torch.clamp(x_adv,min=0,max=1)
    delta = x_adv - x
    print("L2 =", delta.view(delta.size(0), -1).norm(p=2, dim=1).max().item())    
    return x_adv

for entry in summary.get("results", []):
    try:
        exp_name = entry["experiment"]
        exp_path = entry["path"]

        # Only for 2-class experiments
        # if not mc:
        #     classes_str = entry["classes"]  # ["0", "1"]
        # best_run = entry["best_run"]

        # # classes tuple for your dataloader lookup
        # if not mc:
        #     c1, c2 = int(classes_str[0]), int(classes_str[1])
        #     if Dataset == "fmnist":
        #         if c1 == 7 and c2 == 9 or c1 == 8 and c2 == 9:
        #             classes = (c2, c1)
        #         else:
        #             classes = (c1, c2)
        #     else:
        #         classes = (c1, c2)
        
        # best_run = entry.get("best_run", {})

        if not mc:
            c1, c2 = extract_classes_from_path(entry["path"])
            classes = (c1, c2)
        else:
            classes = (0, 1)

        # input_dim = int(best_run.get("input_dim", 784))
        # width = int(best_run["width"])
        # depth = int(best_run["depth"])
        # weight_decay = float(best_run["weight_decay"])
        # lr = float(best_run["lr"])
        # epoch = int(best_run["best_epoch"])

        ckpt_path = entry["path"]

        ckpt = torch.load(ckpt_path, map_location=device)

        hparams = ckpt["hparams"]

        input_dim = int(ckpt.get("input_dim", 784))
        width = int(hparams["width"])
        depth = int(hparams["depth"])
        weight_decay = float(hparams["weight_decay"])
        lr = float(hparams["lr"])
        epoch = int(ckpt["best_epoch"])
        
        # trainloader, testloader = load_fmnist(classes)
        # trainloader, testloader = load_fmnist()
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

        # best = sweep_fcnn_sgd(trainloader, valloader, num_classes=2, out_dir= f"./FCNN_svhn_per_class/fcnn_sweep_sgd_SVHN_{classes}",depths = (depth,), widths = (width,), weight_decays=(weight_decay,), lrs = (lr,), epochs=epoch, momentum= 0.9, nesterov= True, dropout = 0.0, seed= 42)

                    # model path
        # ckpt_path = os.path.join(exp_path, "best_model.pt")

        print(ckpt_path)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing best_model.pt at: {ckpt_path}")

    # final_net = neural_model.Net(dim=dim, depth=5, width=64, num_classes=2, act_name='relu').to(device)
        if not mc:
            model = FCNN2(input_dim=input_dim, width=width, depth=depth, dropout=0, num_classes=2).to(device)
        else:
            model = FCNN2(input_dim=input_dim, width=width, depth=depth, dropout=0, num_classes=10).to(device)
    # 5. Load the trained weights
        ckpt1 = torch.load(ckpt_path, map_location=device) 
        model.load_state_dict(ckpt1['state_dict'])

        def pgd_attack(net, X, y, epsilon, alpha, num_iters,loss_fn = 'mse'):
            """
            L-infinity PGD attack that always runs with gradients on.
            """
            net.eval()
            X_nat = X.detach().to(device)
            y     = y.detach().to(device)

            if loss_fn == 'cross_entropy':                      # one‑hot case
                y = y.argmax(dim=1).to(device)
            # print(f"X_nat shape: {X_nat.shape}, y shape: {y.shape}")

            # start from the natural images
            X_adv = X_nat.clone().detach().to(device)

            # if pgd == True and AA_NORM == "Linf":
            #     print("We are in the PGD l-inf attack")

            #     # ensure we can take grad w.r.t. X_adv
            #     for _ in range(num_iters):
            #         # re-enable grad for X_adv
            #         X_adv.requires_grad_(True)
            #         if X_adv.grad is not None:
            #             X_adv.grad.zero_()
            #         with torch.enable_grad():
            #             out  = net(X_adv)                       # forward
                        
            #             # print("We are in the PGD l-inf attack")

            #             # print(f"out shape: {out.shape}, y shape: {y.shape}")
            #             if loss_fn == 'mse':
            #                 loss = F.mse_loss(out, y)               # squared-error
            #             else:
            #                 loss = F.cross_entropy(out, y)
            #             net.zero_grad()
            #             loss.backward()                         # grad w.r.t. X_adv

            #         # PGD update (no grad needed here)
            #         with torch.no_grad():
            #             X_adv = X_adv + alpha * X_adv.grad.sign()
            #             X_adv = torch.max(torch.min(X_adv, X_nat + epsilon),
            #                             X_nat - epsilon)
            #             X_adv = torch.clamp(X_adv, 0.0, 1.0)

            #     return X_adv.detach()
            if pgd == True and AA_NORM == "linf":
                print("We are in the PGD l-inf attack")
                X_nat=X_nat.flatten(1)
                # print("Max Value:",max(X_nat).item(), "Min Value:",min(X_nat).item())
                print(X_adv.max(), X_adv.min())
                def forw(x): return net(x)
                adversary=AutoAttack(forw, norm='Linf', eps=epsilon, version='standard')
                adversary.attacks_to_run = ['apgd-ce']
                print(X_adv.shape)
                X_adv = adversary.run_standard_evaluation(X_adv, y, bs=X_adv.shape[0])
                return X_adv.detach()
            elif pgd == True and AA_NORM == "l2":
                print("We are in the PGD l-2 attack")
                X_nat=X_nat.flatten(1)
                # print("Max Value:",max(X_nat).item(), "Min Value:",min(X_nat).item())
                print(X_adv.max(), X_adv.min())
                def forw(x): return net(x)
                adversary=AutoAttack(forw, norm='L2', eps=epsilon, version='standard')
                adversary.attacks_to_run = ['apgd-ce']
                print(X_adv.shape)
                X_adv = adversary.run_standard_evaluation(X_adv, y, bs=X_adv.shape[0])
                return X_adv.detach()
            elif pgd == False and bb_attack == "spsa":
                print("We are in the SPSA black-box attack")
                # loss_fn=torch.nn.CrossEntropyLoss()
                # spsa_lambda=lambda x,y: spsa(loss_fn,x,y, epsilon=epsilon, eta=perturbation, lr=alpha, iterations=num_iters, sample_size=X_adv.shape[0], norm=AA_NORM)
                # X_adv=vmap(spsa_lambda)(X_adv,y)
                loss_fn = lambda xb, yb: F.cross_entropy(net(xb) , yb, reduction="none")                      # forward

            # sample_size is SPSA MC samples (n), NOT batch size.
            # pick something like 64/128/256; do NOT set it to X_adv.shape[0].
            # 100 for l-2 and 500 for l-inf
                spsa_sample_size = 500  # or set to a fixed number like 128
                if epsilon > 0:
                    X_adv = spsa(
                        loss_fn,
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
            """classification accuracy on PGD-generated inputs."""
            net.eval()
            correct = 0
            total = 0
            for X, y in loader:
                X_adv = pgd_attack(net, X, y, epsilon, alpha, num_iters, loss_fn)
                with torch.no_grad():
                    preds = torch.argmax(net(X_adv), dim=-1)
                    labels = torch.argmax(y.to(device), dim=-1)
                    correct += torch.sum(preds == labels).item()
                    total += X.size(0)

            return 100.0 * correct / total

        if AA_NORM == "linf":
            # , 32.0, 40.0, 55.0
            epsilons = [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0]

        elif AA_NORM == "l2":
            epsilons = [0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

        adv_accs = []

        for j in epsilons:
            adv_results = []
            if AA_NORM == "linf":
                eps_float = j/255.0   
            elif AA_NORM == "l2":             # actual epsilon used in attack
                eps_float = j
            # alpha = eps_float / 4                  # your step size rule
            alpha = eps_float/10
            # For pgd:-
            acc = get_adv_acc(model, testloader, eps_float, alpha, 100,'cross_entropy')

            # For spsa:-
            # if eps_float < 4.0:
            #     acc = get_adv_acc(model, testloader, eps_float, alpha, 150,'cross_entropy')
            # else:
            #     alpha = min(eps_float/20, 0.1)
            #     acc = get_adv_acc(model, testloader, eps_float, alpha, 400,'cross_entropy')

            adv_accs.append({
                # "eps_int": int(eps_int),
                        "eps_float": float(eps_float),
                        "alpha": float(alpha),
                        "pgd_acc": float(acc),
                    })
            adv_results.append({
                    # "eps_int": int(j),
                    "eps_float": float(eps_float),
                    "adv_accuracy": float(acc),
                    # Only for 2-class experiments
                    # "Class": classes,
                    })
            if not mc:
                adv_results[-1]["Class"] = classes
            

            # print(f"Adversarial accuracy for eps={j} (i.e., {eps_float:.5f} in [0,1] scale): {acc:.2f}")
            if AA_NORM == "linf":
                if not mc:
                    with open(f"./FCNN_nn_{Dataset}_per_class/{tr_norm}_eps_{eps_trn}/adv_accuracy_results_linf_{Dataset}_class_pair_nn.json", "w") as f:
                        json.dump(adv_results, f, indent=2)
                else:
                    with open(f"./FCNN_nn_{Dataset}_mc/{tr_norm}_eps_{eps_trn}/adv_accuracy_results_linf_{Dataset}_mc_nn.json", "w") as f:
                        json.dump(adv_results, f, indent=2)
            elif AA_NORM == "l2":
                if not mc:
                    with open(f"./FCNN_nn_{Dataset}_per_class/{tr_norm}_eps_{eps_trn}/adv_accuracy_results_l2_{Dataset}_class_pair_nn.json", "w") as f:
                        json.dump(adv_results, f, indent=2)
                else:
                    with open(f"./FCNN_nn_{Dataset}_mc/{tr_norm}_eps_{eps_trn}/adv_accuracy_results_l2_{Dataset}_mc_nn.json", "w") as f:
                        json.dump(adv_results, f, indent=2)
        
        results_out.append({
                "experiment": exp_name,
                "path": exp_path,
                # Only for 2-class experiments
                "classes": [c1, c2] if not mc else [0,1],
                "hyperparams": {
                    "input_dim": input_dim,
                    "width": width,
                    "depth": depth,
                    # "dropout": dropout,
                },
                # "best_val_acc": float(best_run.get("best_val_acc", best_run.get("best_val_accuracy", -1))),
                # "clean_acc": float(clean),
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
        # "device": str(device),
        # "json_path": args.json_path,
        # "num_done": len(results_out),
        # "num_errors": len(errors_out),
        "results": results_out,
        "errors": errors_out,
    }

if AA_NORM == "linf":
    global save_path
    if pgd == True:
        if not mc:
            save_path = f"./FCNN_nn_{Dataset}_per_class/{tr_norm}_eps_{eps_trn}/PGD_result_linf_{Dataset}_class_pair_nn.json"
            with open(f"./FCNN_nn_{Dataset}_per_class/{tr_norm}_eps_{eps_trn}/PGD_result_linf_{Dataset}_class_pair_nn.json", "w") as f:
                json.dump(out_obj, f, indent=2)
        else:
            save_path = f"./FCNN_nn_{Dataset}_mc/{tr_norm}_eps_{eps_trn}/PGD_result_linf_{Dataset}_mc_nn.json"
            with open(f"./FCNN_nn_{Dataset}_mc/{tr_norm}_eps_{eps_trn}/PGD_result_linf_{Dataset}_mc_nn.json", "w") as f:
                json.dump(out_obj, f, indent=2)
    else:
        if not mc:
            save_path = f"./FCNN_nn_{Dataset}_per_class/{tr_norm}_eps_{eps_trn}/PGD_result_linf_bb_{Dataset}_class_pair_nn.json"
            with open(f"./FCNN_nn_{Dataset}_per_class/{tr_norm}_eps_{eps_trn}/PGD_result_linf_bb_{Dataset}_class_pair_nn.json", "w") as f:
                json.dump(out_obj, f, indent=2)
        else:
            save_path = f"./FCNN_nn_{Dataset}_mc/{tr_norm}_eps_{eps_trn}/PGD_result_linf_bb_{Dataset}_mc_nn.json"
            with open(f"./FCNN_nn_{Dataset}_mc/{tr_norm}_eps_{eps_trn}/PGD_result_linf_bb_{Dataset}_mc_nn.json", "w") as f:
                json.dump(out_obj, f, indent=2)
elif AA_NORM == "l2":
    if pgd == True:
        if not mc:
            save_path = f"./FCNN_nn_{Dataset}_per_class/{tr_norm}_eps_{eps_trn}/PGD_result_l2_{Dataset}_class_pair_nn.json"
            with open(f"./FCNN_nn_{Dataset}_per_class/{tr_norm}_eps_{eps_trn}/PGD_result_l2_{Dataset}_class_pair_nn.json", "w") as f:
                json.dump(out_obj, f, indent=2)
        else:
            save_path = f"./FCNN_nn_{Dataset}_mc/{tr_norm}_eps_{eps_trn}/PGD_result_l2_{Dataset}_mc_nn.json"
            with open(f"./FCNN_nn_{Dataset}_mc/{tr_norm}_eps_{eps_trn}/PGD_result_l2_{Dataset}_mc_nn.json", "w") as f:
                json.dump(out_obj, f, indent=2)
    else:
        if not mc:
            save_path = f"./FCNN_nn_{Dataset}_per_class/{tr_norm}_eps_{eps_trn}/PGD_result_l2_bb_{Dataset}_class_pair_nn.json"
            with open(f"./FCNN_nn_{Dataset}_per_class/{tr_norm}_eps_{eps_trn}/PGD_result_l2_bb_{Dataset}_class_pair_nn.json", "w") as f:
                json.dump(out_obj, f, indent=2)
        else:
            save_path = f"./FCNN_nn_{Dataset}_mc/{tr_norm}_eps_{eps_trn}/PGD_result_l2_bb_{Dataset}_mc_nn.json"
            with open(f"./FCNN_nn_{Dataset}_mc/{tr_norm}_eps_{eps_trn}/PGD_result_l2_bb_{Dataset}_mc_nn.json", "w") as f:
                json.dump(out_obj, f, indent=2)

print(f"Saved PGD results to: {save_path}")
if errors_out:
    print(f"WARNING: {len(errors_out)} experiments failed. See 'errors' in the output JSON.")

# if __name__ == "__main__":
#     main()
