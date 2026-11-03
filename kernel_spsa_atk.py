"""
SPSA black-box attack for KLR / RKHS checkpoints (binary AND multiclass).

Supported checkpoint formats (auto-detected):
  1. Sweep-script format (klr_sweep.py):
       {"model": <KernelLogisticRegression>, "model_config", "dataset_config", "metrics"}
       stored at  <root>/klr_sweep_<dataset>/<kernel>/binary_<a>vs<b>/best_klr_model.pt
       (multiclass: <root>/klr_sweep_<dataset>/<kernel>/best_klr_model.pt)
  2. Old robust_kernels multiclass format:
       {"model_type": "multiclass", "centers", "state", "options", "class_ids", "config"}

Examples:
  # one binary pair, located automatically
  python attack_spsa.py --dataset fmnist --kernel gaussian --classes 0 1 --norm Linf

  # many pairs (consecutive pairs: (0,1) (0,2) ...)
  python attack_spsa.py --dataset fmnist --kernel gaussian --classes 0 1 0 2 0 3

  # explicit checkpoint(s)
  python attack_spsa.py --checkpoint path/to/best_klr_model.pt --norm Linf

  # multiclass sweep-format model
  python attack_spsa.py --dataset fmnist --kernel gaussian --multiclass
"""

import argparse
import json
import pathlib
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("/janaki/backup/users/student/rs/rahulky/water/retrain")

import dataset
import klr  # noqa: F401  (needed so the pickled KernelLogisticRegression can be loaded)

LINF_DEFAULT = [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 32.0, 40.0, 55.0]
L2_DEFAULT = [0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]


# ============================================================
# Data
# ============================================================

def load_test(dataset_name, classes, multiclass, device):
    """Returns flattened test images and integer labels (0..C-1)."""
    classes = tuple(classes) if classes else (0, 1)

    if dataset_name in ("fmnist", "kmnist", "qmnist", "emnist"):
        _, _, loader = dataset.load_fmnist(
            classes=classes, which_mnist=dataset_name, multiclass=multiclass
        )
    elif dataset_name == "svhn":
        _, _, loader = dataset.get_svhn(classes=classes, multiclass=multiclass)
    elif dataset_name in ("cifar", "cifar10"):
        _, _, loader = dataset.get_cifar(classes=classes, multiclass=multiclass)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    xs, ys = [], []
    for x, y in loader:
        xs.append(x.reshape(x.shape[0], -1))
        ys.append(y.argmax(1) if y.ndim == 2 else y.reshape(-1))

    x = torch.cat(xs).to(device, torch.float32)
    y = torch.cat(ys).to(device).long().reshape(-1)
    return x, y


# ============================================================
# Model adapters -> predict(x_flat) -> logits [N, C]
# ============================================================

def make_predict_rk(model, device):
    import robust_kernels as rk

    centers = model["centers"].to(device)
    state = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model["state"].items()}
    options = model["options"]

    def predict(images):
        x = images.reshape(images.shape[0], -1)
        out = rk.multioutput_eval_and_grad(
            x, centers, state["alpha"], state["factors"], gradients=False, **options
        )
        return out[0] if isinstance(out, (tuple, list)) else out

    return predict


def make_predict_klr(klr_model, probe_x, num_classes):
    """
    The fitted KernelLogisticRegression only has a documented .predict (labels).
    SPSA needs a continuous loss, so look for a method returning logits/probabilities.
    Each candidate is probed on a small input and checked for shape [N, C] (or [N]/[N,1]).
    """
    candidates = [
        ("decision_function", False),
        ("predict_logits", False),
        ("logits", False),
        ("forward", False),
        ("predict_proba", True),
    ]
    probe = probe_x[:2]

    for name, is_proba in candidates:
        fn = getattr(klr_model, name, None)
        if not callable(fn):
            continue
        try:
            with torch.no_grad():
                out = torch.as_tensor(fn(probe)).float()
        except Exception:
            continue
        if out.shape[0] != probe.shape[0]:
            continue

        def predict(images, fn=fn, is_proba=is_proba):
            with torch.no_grad():
                out = torch.as_tensor(fn(images), device=images.device).float()
            if out.ndim == 1 or out.shape[1] == 1:  # single logit -> two-class logits
                z = out.reshape(-1, 1)
                out = torch.cat([-z, z], dim=1)
            if is_proba:
                out = torch.log(out.clamp_min(1e-12))
            return out

        check = predict(probe)
        if check.shape[1] == num_classes:
            print(f"Using KLR method '{name}' for logits.")
            return predict

    raise RuntimeError(
        "Could not find a method on the KLR object that returns logits/probabilities.\n"
        "Inspect it with:\n"
        "  python -c \"import torch,klr; m=torch.load('best_klr_model.pt',weights_only=False)['model']; "
        "print([a for a in dir(m) if not a.startswith('__')])\"\n"
        "and add the right method name to `candidates` in make_predict_klr."
    )


# ============================================================
# Metrics / checks
# ============================================================

@torch.no_grad()
def accuracy(predict, images, labels, batch_size):
    correct = 0
    for s in range(0, len(images), batch_size):
        e = min(s + batch_size, len(images))
        correct += (predict(images[s:e]).argmax(1) == labels[s:e]).sum().item()
    return 100.0 * correct / len(images)


def check_adversarial(adv, images, eps, norm):
    diff = (adv - images).reshape(len(images), -1)
    if norm == "Linf":
        size = diff.abs().max().item()
        ok = size <= eps + 1e-6
    else:
        size = diff.norm(p=2, dim=1).max().item()
        ok = size <= eps + 1e-4
    if not ok:
        raise RuntimeError(f"attack exceeded {norm} radius: {size} > {eps}")
    if adv.min() < -1e-6 or adv.max() > 1 + 1e-6:
        raise RuntimeError("attack left valid pixel range [0,1]")
    return size


# ============================================================
# SPSA
# ============================================================

def spsa_fast(loss_fn, x, y, epsilon, lr, iterations, sample_size, norm, seed=None):
    x = x.detach()
    x_adv = x.clone()
    B, D = x_adv.shape
    eta = float(np.clip(epsilon / 2, 1e-3, 0.01))

    generator = None
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)

    for it in range(iterations):
        print(f"    SPSA iteration {it + 1}/{iterations}", flush=True)

        v = torch.empty(sample_size, B, D, device=x.device).bernoulli_(0.5, generator=generator)
        v = v.mul_(2).sub_(1)

        x_plus = (x_adv.unsqueeze(0) + eta * v).clamp(0, 1)
        x_minus = (x_adv.unsqueeze(0) - eta * v).clamp(0, 1)
        x_pm = torch.cat([x_plus, x_minus], dim=0).reshape(2 * sample_size * B, D)
        y_rep = y.repeat(2 * sample_size)

        with torch.no_grad():
            losses = loss_fn(x_pm, y_rep).view(2 * sample_size, B)

        diff = (losses[:sample_size] - losses[sample_size:]).unsqueeze(2)
        grad_estimate = ((diff / (2 * eta)) * v).mean(dim=0)

        if norm == "Linf":
            x_adv = x_adv + lr * torch.sign(grad_estimate)
        else:
            g = grad_estimate.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            x_adv = x_adv + lr * grad_estimate / g

        delta = x_adv - x
        if norm == "Linf":
            delta = delta.clamp(-epsilon, epsilon)
        else:
            d = delta.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            delta = delta * (epsilon / d).clamp(max=1.0)

        x_adv = (x + delta).clamp(0, 1).detach()

    return x_adv


def attack_spsa(predict, images, labels, epsilon, lr, args, seed):
    if epsilon == 0:
        return images.clone()

    def loss_fn(xq, yq):
        return F.cross_entropy(predict(xq), yq, reduction="none")

    out = []
    n_batches = (len(images) + args.batch_size - 1) // args.batch_size
    for b, s in enumerate(range(0, len(images), args.batch_size)):
        e = min(s + args.batch_size, len(images))
        print(f"\nSPSA batch {b + 1}/{n_batches} samples {s}:{e}", flush=True)
        out.append(
            spsa_fast(
                loss_fn, images[s:e].clone().detach(), labels[s:e],
                epsilon=epsilon, lr=lr, iterations=args.iters,
                sample_size=args.sample_size, norm=args.norm, seed=seed + b,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(out, dim=0)


# ============================================================
# One checkpoint
# ============================================================

def run_checkpoint(ckpt_path, args):
    ckpt_path = pathlib.Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    device = args.device

    if ckpt.get("model_type") == "multiclass":  # old robust_kernels format
        cfg = ckpt["config"]
        dataset_name, kernel, bandwidth = cfg["dataset"], cfg["kernel"], cfg["length_scale"]
        multiclass = True
        classes = ckpt["class_ids"].tolist()
        x, y = load_test(dataset_name, cfg.get("classes"), True, device)
        y = torch.searchsorted(ckpt["class_ids"].to(device), y)  # same mapping as training
        predict = make_predict_rk(ckpt, device)
        saved = ckpt.get("metrics", {}).get("test_accuracy")
        seed = cfg.get("seed", 0) if args.seed is None else args.seed

    elif "model" in ckpt and "dataset_config" in ckpt:  # klr_sweep.py format
        dc, mc = ckpt["dataset_config"], ckpt["model_config"]
        dataset_name, kernel, bandwidth = dc["dataset"], mc["kernel"], mc["length_scale"]
        multiclass = bool(dc["multiclass"])
        classes = list(range(10)) if multiclass else list(dc["classes"])
        x, y = load_test(dataset_name, dc["classes"], multiclass, device)
        predict = make_predict_klr(ckpt["model"], x, num_classes=mc["num_classes"])
        saved = ckpt.get("metrics", {}).get("test_accuracy")
        seed = 0 if args.seed is None else args.seed

    else:
        raise ValueError(f"Unrecognised checkpoint format: {ckpt_path}")

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Dataset: {dataset_name} | kernel: {kernel} | classes: {classes} | multiclass: {multiclass}")
    print(f"Test samples: {len(x)} | input shape: {tuple(x.shape)} | range: {x.min().item()}, {x.max().item()}")

    clean = accuracy(predict, x, y, batch_size=1000)
    print(f"Clean test accuracy: {clean:.2f}%")
    if saved is not None:
        print(f"Saved test accuracy: {100 * saved:.2f}%  (should match)")

    budgets = args.budgets or (LINF_DEFAULT if args.norm == "Linf" else L2_DEFAULT)
    tag = "mc" if multiclass else f"{classes[0]}vs{classes[1]}"
    out_file = (
        pathlib.Path(args.output) if args.output
        else ckpt_path.resolve().parent / f"SPSA_result_{args.norm}_{kernel}_{dataset_name}_{tag}.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records, errors = [], []

    for budget in budgets:
        eps = budget / 255.0 if args.norm == "Linf" else float(budget)
        print("\n" + "=" * 70)
        print(f"SPSA {args.norm} budget={budget} (epsilon={eps})")

        if eps == 0:
            robust, max_pert, lr = clean, 0.0, 0.0
        else:
            lr = args.lr if args.lr is not None else eps / 10.0
            try:
                adv = attack_spsa(predict, x, y, eps, lr, args, seed)
                max_pert = check_adversarial(adv, x, eps, args.norm)
                robust = accuracy(predict, adv, y, batch_size=1000)
            except Exception as e:
                traceback.print_exc()
                errors.append({"budget": budget, "error": f"{type(e).__name__}: {e}"})
                continue

        print(f"[{args.norm} eps={budget}] SPSA robust accuracy: {robust:.2f}%")

        records.append({
            "eps_float": float(eps),
            "eps_budget": float(budget),
            "norm": args.norm,
            "attack": "spsa",
            "pgd_acc": float(robust),
            "clean_acc": float(clean),
            "max_perturbation": float(max_pert),
            "iterations": args.iters,
            "sample_size": args.sample_size,
            "batch_size": args.batch_size,
            "eta": float(np.clip(eps / 2, 1e-3, 0.01)) if eps > 0 else 0.0,
            "lr": float(lr),
        })

        out_obj = {
            "results": [{
                "path": str(ckpt_path.resolve()),
                "kernel": kernel,
                "classes": classes,
                "bandwidth": bandwidth,
                "attack": "spsa",
                "pgd": records,
            }],
            "errors": errors,
        }
        out_file.write_text(json.dumps(out_obj, indent=2) + "\n")  # saved after every epsilon

    print(f"\n[DONE] Saved SPSA results to: {out_file}")
    if errors:
        print(f"[WARN] {len(errors)} budgets failed.")


def parse_args():
    p = argparse.ArgumentParser()

    # which model(s)
    p.add_argument("--checkpoint", nargs="+", default=None, help="Explicit checkpoint path(s).")
    p.add_argument("--dataset", default=None, help="fmnist/kmnist/qmnist/emnist/svhn/cifar")
    p.add_argument("--kernel", default=None, choices=["gaussian", "laplacian"])
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   help="Binary pairs, e.g. 0 1 0 2 -> (0,1),(0,2).")
    p.add_argument("--multiclass", action="store_true")
    p.add_argument("--root", default=".", help="Folder containing klr_sweep_<dataset>/.")

    # attack
    p.add_argument("--output", default=None, help="Only valid with a single checkpoint.")
    p.add_argument("--norm", default="Linf", choices=["Linf", "L2"])
    p.add_argument("--budgets", nargs="+", type=float, default=None,
                   help="Linf: units of 1/255. L2: raw epsilon.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--sample_size", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="Default: epsilon/10.")
    return p.parse_args()


def resolve_checkpoints(args):
    if args.checkpoint:
        return args.checkpoint

    if args.dataset is None or args.kernel is None:
        raise ValueError("Provide --checkpoint, or --dataset and --kernel (plus --classes or --multiclass).")

    base = pathlib.Path(args.root) / f"klr_sweep_{args.dataset}" / args.kernel

    if args.multiclass:
        return [str(base / "best_klr_model.pt")]

    if not args.classes or len(args.classes) % 2:
        raise ValueError("--classes must contain an even number of values for binary attacks.")

    pairs = [(args.classes[i], args.classes[i + 1]) for i in range(0, len(args.classes), 2)]
    return [str(base / f"binary_{a}vs{b}" / "best_klr_model.pt") for a, b in pairs]


def main():
    args = parse_args()
    paths = resolve_checkpoints(args)

    if args.output and len(paths) > 1:
        raise ValueError("--output can only be used with a single checkpoint.")

    for path in paths:
        try:
            run_checkpoint(path, args)
        except Exception:
            print(f"\n[ERROR] failed on {path}")
            traceback.print_exc()


if __name__ == "__main__":
    main()