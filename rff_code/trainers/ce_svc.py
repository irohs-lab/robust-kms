def ensure_labels(y):
    if y.ndim == 2:
        return y.argmax(dim=1).long()
    return y.long()

import argparse
import gc
import os
import pickle
import dataset
# from pathlib import Path

# import cupy as cp
# import torch
# from cuml.svm import SVC
import json
# import numpy as np

def _cupy_to_python_float(value):
    """
    Convert a CuPy scalar or zero-dimensional array to a Python float.
    """
    return float(cp.asarray(value).item())


def _accuracy_from_predictions(predictions_cp, labels_cp):
    """
    predictions_cp: CuPy array of shape [N]
    labels_cp:      CuPy array of shape [N]
    """
    return _cupy_to_python_float(
        cp.mean(predictions_cp.reshape(-1) == labels_cp.reshape(-1))
    )


def _number_of_support_vectors(clf):
    """
    Return the total number of support vectors when available.
    """
    if hasattr(clf, "n_support_"):
        return int(cp.asarray(clf.n_support_).sum().item())

    if hasattr(clf, "support_"):
        return int(cp.asarray(clf.support_).size)

    if hasattr(clf, "support_vectors_"):
        return int(cp.asarray(clf.support_vectors_).shape[0])

    return None


def _save_cuml_model(clf, model_path):
    """
    Save the complete fitted cuML SVC.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = model_path.with_suffix(model_path.suffix + ".tmp")

    with open(temporary_path, "wb") as file:
        pickle.dump(clf, file)

    os.replace(temporary_path, model_path)


def load_cuml_svc(model_path):
    """
    Load a previously saved cuML SVC model.
    """
    with open(model_path, "rb") as file:
        clf = pickle.load(file)

    return clf

import gc
import json
from pathlib import Path

import cupy as cp
import numpy as np
import torch
from cuml.svm import SVC

from torchkernels.kernels.radial import laplacian

def normalize(x): return x/x.norm(p=2,dim=-1,keepdim=True).clamp_min(1e-12)

class OneVsRestKernelSVC:
    """
    Multiclass kernel SVM built from cuML's binary SVC, using a
    One-vs-Rest (OvR) decomposition.

    Rationale
    ---------
    cuML's built-in multiclass SVC path is broken in your current
    installation, but the binary path works fine. OvR sidesteps this by
    training K independent binary SVCs (class k vs. everyone else) and
    combining their decision scores at predict time. This keeps you on
    the known-good binary code path while supporting any number of
    classes.

    Interface mirrors what fit_kernel_svc_cuml expects from a cuML SVC:
    .fit(X, y), .predict(X), plus .decision_function(X) and
    .n_support_ (aggregated) for bookkeeping.
    """

    def __init__(
        self,
       classes,
        kernel="rbf",
        gamma=1e-2,
        C=1.0,
        max_iter=-1,
        tol=1e-3,
        cache_size=4096.0,
    ):
        self.classes_ = cp.asarray(sorted(int(c) for c in cp.asnumpy(classes)))
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.cache_size = cache_size

        self.estimators_ = []

    def fit(self, X_cp, y_cp):
        self.estimators_ = []

        for cls in self.classes_.tolist():
            y_bin = (y_cp == cls).astype(cp.int32)

            clf = SVC(
                kernel=self.kernel,
                gamma=float(self.gamma),
                C=float(self.C),
                max_iter=int(self.max_iter),
                tol=float(self.tol),
                cache_size=float(self.cache_size),
                output_type="cupy",
            )
            clf.fit(X_cp, y_bin)
            self.estimators_.append(clf)

        return self

    def decision_function(self, X_cp):
        """
        Returns an (n_samples, n_classes) matrix of per-class margin
        scores. Column k is the "class k vs. rest" decision score.
        """
        scores = cp.stack(
            [est.decision_function(X_cp).reshape(-1) for est in self.estimators_],
            axis=1,
        )
        return scores

    def predict(self, X_cp):
        scores = self.decision_function(X_cp)
        best_idx = cp.argmax(scores, axis=1)
        return self.classes_[best_idx]

    @property
    def n_support_total(self):
        """Sum of support vectors across all binary sub-estimators."""
        total = 0
        for est in self.estimators_:
            n_support = getattr(est, "n_support_", None)
            if n_support is not None:
                total += int(n_support)
        return total


def _number_of_support_vectors_generic(clf):
    """
    Generalized version of your existing _number_of_support_vectors
    that also handles the OvR wrapper.
    """
    if isinstance(clf, OneVsRestKernelSVC):
        return clf.n_support_total

    n_support = getattr(clf, "n_support_", None)
    if n_support is None:
        return None

    if isinstance(n_support, (list, tuple)):
        return int(sum(int(v) for v in n_support))

    if isinstance(n_support, (int, np.integer)):
        return int(n_support)

    try:
        return int(cp.asnumpy(cp.sum(cp.asarray(n_support))))
    except Exception:
        try:
            return int(np.asarray(n_support).sum())
        except Exception:
            return None


def fit_kernel_svc_cuml(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    Cs=(1e4, 1e3, 1e2, 1e1, 1e0, 1e-1),
    gammas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    kernel="rbf",
    max_iter=-1,
    tol=1e-3,
    cache_size=4096.0,
    model_path="svc_cv/best_kernel_svc.pt",
    json_path="svc_cv/kernel_svc_results.json",
    device=torch.device("cuda"),
):
    """
    Tune a true cuML kernel SVM using validation accuracy.

    Now supports both binary and multiclass classification. Multiclass
    is handled via a One-vs-Rest wrapper (OneVsRestKernelSVC) built on
    top of cuML's binary SVC, since the cuML multiclass path is broken
    in this environment.

    Selection order:
        1. Higher validation accuracy
        2. Fewer support vectors
        3. Smaller C

    Notes
    -----
    - x_train/x_val/x_test are the original input features, not RFF features.
    """

    print("Using cuML kernel SVC on GPU")
    print(f"Kernel: {kernel}")

    use_precomputed_laplacian = kernel.lower() in {
        "laplace", "laplacian", "precomputed_laplacian"
    }

    x_train = x_train.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    x_val = x_val.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
    x_test = x_test.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()

    y_train = ensure_labels(y_train).to(device=device, dtype=torch.int32, non_blocking=True).reshape(-1).contiguous()
    y_val = ensure_labels(y_val).to(device=device, dtype=torch.int32, non_blocking=True).reshape(-1).contiguous()
    y_test = ensure_labels(y_test).to(device=device, dtype=torch.int32, non_blocking=True).reshape(-1).contiguous()

    print("X_train:", x_train.shape, x_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("Training classes:", torch.unique(y_train))

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Training-size mismatch: X has {x_train.shape[0]} samples but y has {y_train.shape[0]} labels.")
    if x_val.shape[0] != y_val.shape[0]:
        raise ValueError(f"Validation-size mismatch: X has {x_val.shape[0]} samples but y has {y_val.shape[0]} labels.")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Test-size mismatch: X has {x_test.shape[0]} samples but y has {y_test.shape[0]} labels.")

    if not torch.isfinite(x_train).all():
        raise ValueError("x_train contains NaN or Inf values.")
    if not torch.isfinite(x_val).all():
        raise ValueError("x_val contains NaN or Inf values.")
    if not torch.isfinite(x_test).all():
        raise ValueError("x_test contains NaN or Inf values.")

    train_classes = torch.unique(y_train)
    num_classes = train_classes.numel()

    if num_classes < 2:
        raise ValueError(f"Need at least 2 classes, found {num_classes}: {train_classes.tolist()}.")

    is_multiclass = num_classes > 2
    print(f"Detected {num_classes} classes -> {'multiclass (OvR)' if is_multiclass else 'binary'} mode")

    X_train_cp = cp.from_dlpack(x_train)
    X_val_cp = cp.from_dlpack(x_val)
    X_test_cp = cp.from_dlpack(x_test)

    y_train_cp = cp.from_dlpack(y_train)
    y_val_cp = cp.from_dlpack(y_val)
    y_test_cp = cp.from_dlpack(y_test)

    classes_cp = cp.asarray(sorted(train_classes.detach().cpu().tolist()))

    if use_precomputed_laplacian:
        d = int(x_train.shape[1])
        M = torch.eye(d, device=device, dtype=torch.float32)
        Xh_train = normalize(x_train)
        Xh_val = normalize(x_val)
        Xh_test = normalize(x_test)

    best = None
    all_results = []

    model_path = str(Path(model_path).resolve())

    for gamma in gammas:
        if use_precomputed_laplacian:
            print(f"\nComputing precomputed Laplacian kernels for gamma={gamma}")

            K_train_t = laplacian(Xh_train, Xh_train, gamma, M).contiguous()
            K_val_t = laplacian(Xh_train, Xh_val, gamma, M).T.contiguous()

            if K_train_t.shape != (x_train.shape[0], x_train.shape[0]):
                raise ValueError(
                    f"K_train must have shape "
                    f"({x_train.shape[0]}, {x_train.shape[0]}), "
                    f"got {tuple(K_train_t.shape)}."
                )
            if K_val_t.shape != (x_val.shape[0], x_train.shape[0]):
                raise ValueError(
                    f"K_val must have shape "
                    f"({x_val.shape[0]}, {x_train.shape[0]}), "
                    f"got {tuple(K_val_t.shape)}."
                )

            fit_train_cp = cp.from_dlpack(K_train_t)
            predict_train_cp = fit_train_cp
            predict_val_cp = cp.from_dlpack(K_val_t)
            svc_kernel = "precomputed"
        else:
            fit_train_cp = X_train_cp
            predict_train_cp = X_train_cp
            predict_val_cp = X_val_cp
            svc_kernel = kernel

        for C in Cs:
            print("\n" + "=" * 70)
            print(f"Training kernel={kernel}, gamma={gamma}, C={C}")

            if is_multiclass:
                clf = OneVsRestKernelSVC(
                    classes=classes_cp,
                    kernel=svc_kernel,
                    gamma=float(gamma),
                    C=float(C),
                    max_iter=int(max_iter),
                    tol=float(tol),
                    cache_size=float(cache_size),
                )
            else:
                clf = SVC(
                    kernel=svc_kernel,
                    gamma=float(gamma),
                    C=float(C),
                    max_iter=int(max_iter),
                    tol=float(tol),
                    cache_size=float(cache_size),
                    output_type="cupy",
                )

            clf.fit(fit_train_cp, y_train_cp)

            pred_train_cp = clf.predict(predict_train_cp).reshape(-1)
            pred_val_cp = clf.predict(predict_val_cp).reshape(-1)

            train_acc = _accuracy_from_predictions(pred_train_cp, y_train_cp)
            val_acc = _accuracy_from_predictions(pred_val_cp, y_val_cp)

            num_support_vectors = _number_of_support_vectors_generic(clf)

            result = {
                "kernel": str(kernel),
                "C": float(C),
                "gamma": float(gamma),
                "max_iter": int(max_iter),
                "tol": float(tol),
                "cache_size": float(cache_size),
                "num_support_vectors": num_support_vectors,
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "num_classes": int(num_classes),
                "test_acc": None,
            }

            all_results.append(result)

            print(
                f"kernel={kernel:<8} "
                f"gamma={gamma:<10.4g} "
                f"C={C:<10.4g} "
                f"train_acc={train_acc * 100:.2f}% "
                f"val_acc={val_acc * 100:.2f}% "
                f"n_support={num_support_vectors}"
            )

            if best is None:
                is_best = True
            elif val_acc > best["val_acc"]:
                is_best = True
            elif val_acc < best["val_acc"]:
                is_best = False
            else:
                current_nsv = num_support_vectors if num_support_vectors is not None else float("inf")
                best_nsv = best["num_support_vectors"] if best["num_support_vectors"] is not None else float("inf")

                if current_nsv < best_nsv:
                    is_best = True
                elif current_nsv > best_nsv:
                    is_best = False
                else:
                    is_best = float(C) < best["C"]

            if is_best:
                print(f"New best model: val_acc={val_acc * 100:.2f}%, C={C}, gamma={gamma}")

                _save_cuml_model(clf, model_path)

                best = {
                    "kernel": str(kernel),
                    "C": float(C),
                    "gamma": float(gamma),
                    "max_iter": int(max_iter),
                    "tol": float(tol),
                    "cache_size": float(cache_size),
                    "num_support_vectors": num_support_vectors,
                    "train_acc": float(train_acc),
                    "val_acc": float(val_acc),
                    "test_acc": None,
                    "num_classes": int(num_classes),
                    "is_multiclass": bool(is_multiclass),
                    "model_path": model_path,
                    "kernel_params": {
                        "kernel": str(kernel),
                        "svc_kernel": str(svc_kernel),
                        "gamma": float(gamma),
                        "C": float(C),
                    },
                    "data_params": {
                        "input_dim": int(x_train.shape[1]),
                        "num_train_samples": int(x_train.shape[0]),
                        "classes": train_classes.detach().cpu(),
                    },
                }

            del clf, pred_train_cp, pred_val_cp
            gc.collect()
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()

        if use_precomputed_laplacian:
            del fit_train_cp, predict_train_cp, predict_val_cp
            del K_train_t, K_val_t
            gc.collect()
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()

    if best is None:
        raise RuntimeError("No SVC model was successfully fitted.")

    best_clf = load_cuml_svc(best["model_path"])

    if use_precomputed_laplacian:
        best_gamma = float(best["gamma"])
        print(f"Recomputing Laplacian kernels for best gamma={best_gamma}")

        best_K_train_t = laplacian(
            Xh_train, Xh_train, best_gamma, M
        ).contiguous()
        best_K_val_t = laplacian(
            Xh_train, Xh_val, best_gamma, M
        ).T.contiguous()
        best_K_test_t = laplacian(
            Xh_train, Xh_test, best_gamma, M
        ).T.contiguous()

        if best_K_test_t.shape != (x_test.shape[0], x_train.shape[0]):
            raise ValueError(
                f"K_test must have shape "
                f"({x_test.shape[0]}, {x_train.shape[0]}), "
                f"got {tuple(best_K_test_t.shape)}."
            )

        best_K_train_cp = cp.from_dlpack(best_K_train_t)
        best_K_val_cp = cp.from_dlpack(best_K_val_t)
        best_K_test_cp = cp.from_dlpack(best_K_test_t)
    else:
        best_K_train_cp = X_train_cp
        best_K_val_cp = X_val_cp
        best_K_test_cp = X_test_cp

    pred_train_cp = best_clf.predict(best_K_train_cp).reshape(-1)
    pred_val_cp = best_clf.predict(best_K_val_cp).reshape(-1)
    pred_test_cp = best_clf.predict(best_K_test_cp).reshape(-1)

    best["train_acc"] = _accuracy_from_predictions(pred_train_cp, y_train_cp)
    best["val_acc"] = _accuracy_from_predictions(pred_val_cp, y_val_cp)
    best["test_acc"] = _accuracy_from_predictions(pred_test_cp, y_test_cp)

    for result in all_results:
        same_hyperparameters = (
            result["kernel"] == best["kernel"]
            and result["C"] == best["C"]
            and result["gamma"] == best["gamma"]
        )
        if same_hyperparameters:
            result["test_acc"] = best["test_acc"]
            break

    print("\n" + "=" * 70)
    print("BEST KERNEL SVC")
    print("=" * 70)
    print(f"Mode:               {'multiclass (OvR)' if is_multiclass else 'binary'}")
    print(f"Kernel:             {best['kernel']}")
    print(f"C:                  {best['C']}")
    print(f"Gamma:              {best['gamma']}")
    print(f"Support vectors:    {best['num_support_vectors']}")
    print(f"Training accuracy:  {best['train_acc'] * 100:.2f}%")
    print(f"Validation accuracy:{best['val_acc'] * 100:.2f}%")
    print(f"Test accuracy:      {best['test_acc'] * 100:.2f}%")
    print(f"Saved model:        {best['model_path']}")

    if json_path is not None:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)

        json_dict = {
            "best": {
                "kernel": best["kernel"],
                "C": best["C"],
                "gamma": best["gamma"],
                "train_acc": best["train_acc"],
                "val_acc": best["val_acc"],
                "test_acc": best["test_acc"],
                "num_support_vectors": best["num_support_vectors"],
                "num_classes": best["num_classes"],
                "model_path": best["model_path"],
            },
            "all_results": all_results,
        }

        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=4)

        print(f"Saved search results to {json_path}")

    return best, all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--MC", action="store_true", help="Run multiclass instead of pairwise")
    parser.add_argument("--DATASET", type=str, default="kmnist")
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    Dataset = args.DATASET.lower()
    kernel_type = args.kernel.lower()
    mc = args.MC
    device_str = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if mc or Dataset in ["bank"]:
        print("Running multiclass...")
        classes1 = [0]
        classes2 = [1]
    else:
        print("Running per-class pairwise...")
        classes1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        classes2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    out_dir = os.path.join("svc_cv", f"rff_{kernel_type}_{Dataset}_{'mc' if mc else 'pairwise'}")
    os.makedirs(out_dir, exist_ok=True)

    def dataloader_to_tensors(loader):
        x_list = []
        y_list = []

        for x, y in loader:
            x_list.append(x)
            y_list.append(y)

        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)

        return x, y

    for cl1 in classes1:
        for cl2 in range(cl1 + 1, len(classes2) + 1):
            classes = (cl1, cl2)

            if Dataset == "svhn":
                def load_dataset_svhn(classes):
                    return dataset.get_svhn(classes=classes)

                trainloader, valloader, testloader = load_dataset_svhn(classes=(cl1, cl2))
                print("I am SVHN")

            elif Dataset == "fmnist":
                def ld_fmnist():
                    return dataset.load_fmnist(
                        classes=(cl1, cl2),
                        which_mnist="fmnist",
                        multiclass=mc,
                    )

                trainloader, valloader, testloader = ld_fmnist()
                print("I am FMNIST")

            elif Dataset == "qmnist":
                def ld_fmnist():
                    return dataset.load_fmnist(
                        classes=(cl1, cl2),
                        which_mnist="qmnist",
                        multiclass=mc,
                    )

                trainloader, valloader, testloader = ld_fmnist()
                print("I am QMNIST")

            elif Dataset == "kmnist":
                def ld_fmnist():
                    return dataset.load_fmnist(
                        classes=(cl1, cl2),
                        which_mnist="kmnist",
                        multiclass=mc,
                    )

                trainloader, valloader, testloader = ld_fmnist()
                print("I am KMNIST")

            elif Dataset == "emnist":
                def ld_fmnist():
                    return dataset.load_fmnist(
                        classes=(cl1, cl2),
                        which_mnist="emnist",
                        multiclass=mc,
                    )

                trainloader, valloader, testloader = ld_fmnist()
                print("I am EMNIST")

            elif Dataset == "bank":
                def ld_bank():
                    return dataset.uci_bank()

                trainloader, testloader = ld_bank()
                valloader = testloader
                print("I am Bank")

            else:
                raise ValueError(f"Unsupported dataset: {Dataset}")

            x_train, y_train = dataloader_to_tensors(trainloader)
            x_val, y_val = dataloader_to_tensors(valloader)
            x_test, y_test = dataloader_to_tensors(testloader)

            print("x_train:", x_train.shape, y_train.shape)
            print("x_val:", x_val.shape, y_val.shape)
            print("x_test:", x_test.shape, y_test.shape)

            if mc:
                save_name = f"{Dataset}_mc_svc_results.json"
                save_model = f"{Dataset}_mc_svc_best.pt"
            else:
                save_name = f"{Dataset}_{cl1}_{cl2}_svc_results.json"
                save_model = f"{Dataset}_{cl1}_{cl2}_svc_best.pt"

            if mc:
                print(f"Training multiclass SVC for {Dataset}")
            else:
                print(f"Training binary SVC for classes {classes}")

            fit_kernel_svc_cuml(
                x_train=x_train.float(),
                y_train=y_train.long(),
                x_val=x_val.float(),
                y_val=y_val.long(),
                x_test=x_test.float(),
                y_test=y_test.long(),
                Cs=(1e3, 1e2, 1e1, 1.0, 1e-1),
                gammas=(1e-3, 1e-2, 1e-1, 1.0,10.0),
                kernel=args.kernel,
                max_iter=-1,
                tol=1e-3,
                cache_size=4096,
                model_path=os.path.join(out_dir, save_model),
                json_path=os.path.join(out_dir, save_name),
                device=device,
            )