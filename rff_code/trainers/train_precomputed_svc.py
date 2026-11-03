#!/usr/bin/env python3
"""Tune Gaussian and Laplacian SVCs using precomputed Gram matrices.

The kernels are evaluated with torchkernels on the requested torch device.
The completed Gram matrices are transferred to CPU and fitted with
``sklearn.svm.SVC(kernel="precomputed")``.  By default, one complete
hyperparameter search is run for each of the Gaussian and Laplacian kernels.
"""

import argparse
import gc
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.svm import SVC
from torchkernels.kernels.radial import gaussian, laplacian

import dataset


KERNEL_FUNCTIONS = {
    "gaussian": gaussian,
    "laplacian": laplacian,
}


def ensure_labels(y):
    """Convert integer or one-hot targets to a one-dimensional label tensor."""
    if y.ndim == 2:
        return y.argmax(dim=1).long()
    return y.long().reshape(-1)


def normalize(x):
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)


def canonical_kernel_name(name):
    name = name.lower()
    if name == "laplace":
        return "laplacian"
    if name not in KERNEL_FUNCTIONS:
        raise ValueError(f"Unsupported kernel: {name}")
    return name


def save_pickle_atomic(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(value, handle)
    os.replace(temporary_path, path)


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def dataloader_to_tensors(loader):
    xs = []
    ys = []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError("The data loader is empty.")
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def compute_gram(kernel_fn, reference, samples, gamma, metric):
    """Return K(samples, reference), the layout sklearn expects for predict."""
    return kernel_fn(
        reference,
        samples,
        gamma,
        metric,
        in_place=False,
    ).T.contiguous()


def release_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fit_precomputed_svc(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    *,
    kernel,
    Cs,
    gammas,
    max_iter,
    tol,
    cache_size,
    model_path,
    json_path,
    training_data_path,
    device,
):
    """Tune and save one precomputed-kernel SVC."""
    kernel = canonical_kernel_name(kernel)
    kernel_fn = KERNEL_FUNCTIONS[kernel]

    x_train = x_train.to(device=device, dtype=torch.float32).contiguous()
    x_val = x_val.to(device=device, dtype=torch.float32).contiguous()
    x_test = x_test.to(device=device, dtype=torch.float32).contiguous()
    y_train = ensure_labels(y_train).cpu().numpy()
    y_val = ensure_labels(y_val).cpu().numpy()
    y_test = ensure_labels(y_test).cpu().numpy()

    for split, x, y in (
        ("training", x_train, y_train),
        ("validation", x_val, y_val),
        ("test", x_test, y_test),
    ):
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"{split} size mismatch: X has {x.shape[0]} rows, "
                f"but y has {y.shape[0]} labels."
            )
        if not torch.isfinite(x).all():
            raise ValueError(f"{split} inputs contain NaN or Inf values.")

    classes = np.unique(y_train)
    if classes.size < 2:
        raise ValueError(f"At least two training classes are required; found {classes}.")

    xh_train = normalize(x_train)
    xh_val = normalize(x_val)
    xh_test = normalize(x_test)
    metric = torch.eye(x_train.shape[1], device=device, dtype=torch.float32)

    training_data_path = Path(training_data_path).resolve()
    training_data_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x_train": x_train.detach().cpu(),
            "y_train": torch.from_numpy(y_train),
            "normalized": True,
            "kernel": kernel,
        },
        training_data_path,
    )

    model_path = Path(model_path).resolve()
    json_path = Path(json_path).resolve()
    best = None
    all_results = []

    print(f"\nTraining {kernel} SVC with precomputed kernels")
    print(f"Device used to construct kernels: {device}")
    print(f"Classes: {classes.tolist()}")

    for gamma in gammas:
        print(f"Computing {kernel} Gram matrices for gamma={gamma:g}")
        K_train_t = compute_gram(kernel_fn, xh_train, xh_train, gamma, metric)
        K_val_t = compute_gram(kernel_fn, xh_train, xh_val, gamma, metric)

        expected_train = (x_train.shape[0], x_train.shape[0])
        expected_val = (x_val.shape[0], x_train.shape[0])
        if tuple(K_train_t.shape) != expected_train:
            raise ValueError(
                f"K_train must have shape {expected_train}; got {tuple(K_train_t.shape)}."
            )
        if tuple(K_val_t.shape) != expected_val:
            raise ValueError(
                f"K_val must have shape {expected_val}; got {tuple(K_val_t.shape)}."
            )

        K_train = K_train_t.detach().cpu().numpy()
        K_val = K_val_t.detach().cpu().numpy()

        for C in Cs:
            clf = SVC(
                kernel="precomputed",
                C=float(C),
                max_iter=int(max_iter),
                tol=float(tol),
                cache_size=float(cache_size),
            )
            clf.fit(K_train, y_train)
            train_acc = float(np.mean(clf.predict(K_train) == y_train))
            val_acc = float(np.mean(clf.predict(K_val) == y_val))
            num_support = int(np.asarray(clf.n_support_).sum())

            result = {
                "kernel": kernel,
                "backend": "sklearn_precomputed",
                "C": float(C),
                "gamma": float(gamma),
                "max_iter": int(max_iter),
                "tol": float(tol),
                "cache_size": float(cache_size),
                "num_support_vectors": num_support,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": None,
                "num_classes": int(classes.size),
            }
            all_results.append(result)

            candidate_key = (val_acc, -num_support, -float(C))
            best_key = None if best is None else (
                best["val_acc"],
                -best["num_support_vectors"],
                -best["C"],
            )
            if best_key is None or candidate_key > best_key:
                save_pickle_atomic(clf, model_path)
                best = {
                    **result,
                    "model_path": str(model_path),
                    "training_data_path": str(training_data_path),
                    "input_dim": int(x_train.shape[1]),
                    "classes": classes.tolist(),
                }
                print(
                    f"New best: C={C:g}, gamma={gamma:g}, "
                    f"train={train_acc:.4f}, val={val_acc:.4f}, "
                    f"support={num_support}"
                )

            del clf

        del K_train, K_val, K_train_t, K_val_t
        release_cuda_memory()

    if best is None:
        raise RuntimeError(f"No {kernel} SVC was fitted.")

    best_clf = load_pickle(model_path)
    best_gamma = float(best["gamma"])
    print(f"Computing test kernel once for selected gamma={best_gamma:g}")
    K_train_t = compute_gram(kernel_fn, xh_train, xh_train, best_gamma, metric)
    K_val_t = compute_gram(kernel_fn, xh_train, xh_val, best_gamma, metric)
    K_test_t = compute_gram(kernel_fn, xh_train, xh_test, best_gamma, metric)

    expected_test = (x_test.shape[0], x_train.shape[0])
    if tuple(K_test_t.shape) != expected_test:
        raise ValueError(
            f"K_test must have shape {expected_test}; got {tuple(K_test_t.shape)}."
        )

    K_train = K_train_t.detach().cpu().numpy()
    K_val = K_val_t.detach().cpu().numpy()
    K_test = K_test_t.detach().cpu().numpy()
    best["train_acc"] = float(np.mean(best_clf.predict(K_train) == y_train))
    best["val_acc"] = float(np.mean(best_clf.predict(K_val) == y_val))
    best["test_acc"] = float(np.mean(best_clf.predict(K_test) == y_test))

    for result in all_results:
        if result["C"] == best["C"] and result["gamma"] == best["gamma"]:
            result["test_acc"] = best["test_acc"]
            break

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"best": best, "all_results": all_results}, handle, indent=2)

    print(
        f"Best {kernel}: C={best['C']:g}, gamma={best['gamma']:g}, "
        f"train={best['train_acc']:.4f}, val={best['val_acc']:.4f}, "
        f"test={best['test_acc']:.4f}"
    )
    print(f"Saved model: {model_path}")
    print(f"Saved results: {json_path}")
    print(f"Saved fit-time data: {training_data_path}")

    del K_train, K_val, K_test, K_train_t, K_val_t, K_test_t
    release_cuda_memory()
    return best, all_results


def load_dataset(dataset_name, classes, multiclass):
    if dataset_name == "svhn":
        return dataset.get_svhn(classes=classes)
    if dataset_name in {"fmnist", "qmnist", "kmnist", "emnist"}:
        return dataset.load_fmnist(
            classes=classes,
            which_mnist=dataset_name,
            multiclass=multiclass,
        )
    if dataset_name == "bank":
        trainloader, testloader = dataset.uci_bank()
        return trainloader, testloader, testloader
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Gaussian and Laplacian SVCs from precomputed kernels."
    )
    parser.add_argument("--MC", action="store_true", help="Run multiclass training.")
    parser.add_argument(
        "--DATASET",
        default="kmnist",
        choices=["svhn", "fmnist", "qmnist", "kmnist", "emnist", "bank"],
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["gaussian", "laplacian"],
        choices=["gaussian", "laplacian", "laplace"],
        help="Kernels to train; the default trains both.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=Path("svc_cv"))
    parser.add_argument(
        "--Cs",
        nargs="+",
        type=float,
        default=[1e-2, 1e-1,1.0, 1e1,1e2, 1e3],
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0],
        
    )
    parser.add_argument("--max-iter", type=int, default=-1)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--cache-size", type=float, default=4096.0)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.DATASET.lower()
    multiclass = bool(args.MC or dataset_name == "bank")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; constructing kernels on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    kernels = list(dict.fromkeys(canonical_kernel_name(k) for k in args.kernels))
    class_pairs = [(0, 1)] if multiclass else [
        (class1, class2)
        for class1 in range(9)
        for class2 in range(class1 + 1, 10)
    ]

    for class1, class2 in class_pairs:
        classes = (class1, class2)
        trainloader, valloader, testloader = load_dataset(
            dataset_name, classes, multiclass
        )
        x_train, y_train = dataloader_to_tensors(trainloader)
        x_val, y_val = dataloader_to_tensors(valloader)
        x_test, y_test = dataloader_to_tensors(testloader)

        run_name = "mc" if multiclass else f"{class1}_{class2}"
        print(
            f"\nDataset={dataset_name}, run={run_name}, "
            f"train={tuple(x_train.shape)}, val={tuple(x_val.shape)}, "
            f"test={tuple(x_test.shape)}"
        )

        for kernel in kernels:
            output_dir = (
                args.output_root
                / f"precomputed_{kernel}_{dataset_name}_{'mc' if multiclass else 'pairwise'}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{dataset_name}_{run_name}_{kernel}_svc"

            fit_precomputed_svc(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                kernel=kernel,
                Cs=args.Cs,
                gammas=args.gammas,
                max_iter=args.max_iter,
                tol=args.tol,
                cache_size=args.cache_size,
                model_path=output_dir / f"{prefix}_best.pkl",
                json_path=output_dir / f"{prefix}_results.json",
                training_data_path=output_dir / f"{prefix}_xtrain.pt",
                device=device,
            )


if __name__ == "__main__":
    main()
