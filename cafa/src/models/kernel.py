# src/models/kernel.py

import logging
import os
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchkernels.kernels.radial import laplacian, gaussian

logger = logging.getLogger(__name__)


def normalize(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


def dataset_to_tensors(ds: torch.utils.data.Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for x, y in ds:
        xs.append(x.view(-1).float())
        ys.append(y)
    X = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0).long()
    return X, y


def labels_to_one_hot(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    return F.one_hot(y.long(), num_classes=n_classes).float()


class KernelArtifactModel(nn.Module):
    """
    Inference wrapper over the saved kernel artifact.
    """

    def __init__(self, payload: Dict[str, Any]):
        super().__init__()

        kernel_name = payload["kernel"]
        L = float(payload.get("bandwidth", payload.get("L")))
        X_train = payload["X_train"].float()
        Ms = payload["Ms"]
        sols = payload["sols"]
        best_iter = int(payload["iteration"])

        # M_best = Ms[best_iter]
        # sol_best = sols[best_iter]

        # if not isinstance(M_best, torch.Tensor):
        #     M_best = torch.tensor(M_best, dtype=torch.float32)
        # else:
        #     M_best = M_best.float()

        # if not isinstance(sol_best, torch.Tensor):
        #     sol_best = torch.tensor(sol_best, dtype=torch.float32)
        # else:
        #     sol_best = sol_best.float()

        # self.kernel_name = kernel_name
        # self.bandwidth = L

        # self.register_buffer("X_train_ref", X_train)
        # self.register_buffer("M", M_best)
        # self.register_buffer("sol", sol_best)
        M_best = Ms[best_iter]
        sol_best = sols[best_iter]

        if not isinstance(M_best, torch.Tensor):
            M_best = torch.tensor(M_best, dtype=torch.float32)
        else:
            M_best = M_best.float()

        if not isinstance(sol_best, torch.Tensor):
            sol_best = torch.tensor(sol_best, dtype=torch.float32)
        else:
            sol_best = sol_best.float()

        M_best = torch.nan_to_num(M_best, nan=0.0, posinf=1.0, neginf=-1.0)
        sol_best = torch.nan_to_num(sol_best, nan=0.0, posinf=1.0, neginf=-1.0)

        self.kernel_name = kernel_name
        self.bandwidth = L

        self.register_buffer("X_train_ref", X_train)
        self.register_buffer("M", M_best)
        self.register_buffer("sol", sol_best)

    def _kernel(self, X_ref: torch.Tensor, X_query: torch.Tensor) -> torch.Tensor:
        if self.kernel_name == "laplacian":
            return laplacian(X_ref, X_query, self.bandwidth, self.M, in_place=False)
        elif self.kernel_name == "gaussian":
            return gaussian(X_ref, X_query, self.bandwidth, self.M, in_place=False)
        raise ValueError(f"Unsupported kernel: {self.kernel_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            x = x.view(x.shape[0], -1)

        X_ref = normalize(self.X_train_ref)
        X_query = normalize(x)

        K = self._kernel(X_ref, X_query)          # (N_train, B)
        logits = (self.sol @ K).transpose(0, 1)  # (B, C)

        if not torch.isfinite(logits).all():
            print("Bad logits detected")
            print("nan logits:", torch.isnan(logits).sum().item())
            print("inf logits:", torch.isinf(logits).sum().item())
        return logits


def load_kernel_artifact(saved_model_path: str) -> KernelArtifactModel:
    payload = torch.load(saved_model_path, map_location="cpu")
    model = KernelArtifactModel(payload)
    model.eval()
    return model


# ---------------------------
# Your RFM / kernel internals
# ---------------------------

from torch.func import vmap, jacrev


def compute_EGOP_with_jacrev_laplacian(
    X: torch.Tensor,
    sol: torch.Tensor,
    L: float,
    M: torch.Tensor,
    num_samples: Optional[int] = 20000,
    batch_size: int = 2500,
) -> np.ndarray:
    device = X.device
    dtype = X.dtype
    n, d = X.shape
    sol = sol.to(device=device, dtype=dtype)
    M = M.to(device=device, dtype=dtype)

    if (num_samples is None) or (num_samples >= n):
        x_eval = X
    else:
        idx = torch.randint(0, n, (num_samples,), device=device)
        x_eval = X[idx]
    m = x_eval.shape[0]

    X_hat = normalize(X)

    def f_all_classes(x):
        x_hat = normalize(x.unsqueeze(0))
        K = laplacian(X_hat, x_hat, L, M, in_place=False)
        return (sol @ K).squeeze()

    grad_fn = jacrev(f_all_classes)
    batched_grad_fn = vmap(grad_fn)

    EGOP = torch.zeros(d, d, device=device, dtype=torch.float32)
    for i in range(0, m, batch_size):
        batch = x_eval[i:i + batch_size]
        G_batch = torch.nan_to_num(
            batched_grad_fn(batch), nan=0.0, posinf=0.0, neginf=0.0
        )
        EGOP += torch.einsum("bcd,bce->de", G_batch, G_batch)

    EGOP /= m
    return EGOP.detach().cpu().numpy().astype("float32")


def compute_EGOP_with_jacrev_gaussian(
    X: torch.Tensor,
    sol: torch.Tensor,
    L: float,
    M: torch.Tensor,
    num_samples: Optional[int] = 20000,
    batch_size: int = 2500,
) -> np.ndarray:
    device = X.device
    dtype = X.dtype
    n, d = X.shape
    sol = sol.to(device=device, dtype=dtype)
    M = M.to(device=device, dtype=dtype)

    if (num_samples is None) or (num_samples >= n):
        x_eval = X
    else:
        idx = torch.randint(0, n, (num_samples,), device=device)
        x_eval = X[idx]
    m = x_eval.shape[0]

    X_hat = normalize(X)

    def f_all_classes(x):
        x_hat = normalize(x.unsqueeze(0))
        K = gaussian(X_hat, x_hat, L, M, in_place=False)
        return (sol @ K).squeeze()

    grad_fn = jacrev(f_all_classes)
    batched_grad_fn = vmap(grad_fn)

    EGOP = torch.zeros(d, d, device=device, dtype=torch.float32)
    for i in range(0, m, batch_size):
        batch = x_eval[i:i + batch_size]
        G_batch = torch.nan_to_num(
            batched_grad_fn(batch), nan=0.0, posinf=0.0, neginf=0.0
        )
        EGOP += torch.einsum("bcd,bce->de", G_batch, G_batch)

    EGOP /= m
    return EGOP.detach().cpu().numpy().astype("float32")


def rfm_laplacian_from_tensors(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    iters: int = 0,
    reg: float = 1e-3,
    L: float = 1.0,
    train_acc: bool = False,
    classif: bool = True,
    egop_samples: Optional[int] = 20000,
    egop_batch_size: int = 2500,
):
    X_train = X_train.float().cpu()
    X_test = X_test.float().cpu()
    y_train = y_train.float().cpu()
    y_test = y_test.float().cpu()

    n, d = X_train.shape
    M = np.eye(d, dtype="float32")

    mses, Ms, sols = [], [], []
    Ms.append(M.copy())

    best_acc = -float("inf")
    best_iter = 0
    kernel_acc = 0.0

    for i in range(iters):
        K_train = laplacian(normalize(X_train), normalize(X_train), L, torch.from_numpy(M)).numpy()
        sol = np.linalg.solve(K_train + reg * np.eye(len(K_train)), y_train.numpy()).T
        sols.append(sol)

        if train_acc and classif:
            preds_train = (sol @ K_train).T
            train_pred = np.argmax(preds_train, axis=-1)
            train_true = torch.argmax(y_train, dim=-1).numpy()
            logger.info(f"Round {i} Train Acc: {(train_pred == train_true).mean() * 100:.2f}")

        K_test = laplacian(normalize(X_train), normalize(X_test), L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_test).T
        mse_ = float(((preds - y_test.numpy()) ** 2).mean())
        mses.append(mse_)

        if classif:
            y_pred = np.argmax(preds, axis=-1)
            labels = torch.argmax(y_test, dim=-1).numpy()
            acc = 100.0 * float((y_pred == labels).mean())
            logger.info(f"Round {i} Acc: {acc:.2f}")

            if i == 0:
                kernel_acc = acc
            if acc > best_acc:
                best_acc = acc
                best_iter = i

        sol_t = torch.tensor(sol, dtype=torch.float32)
        M = compute_EGOP_with_jacrev_laplacian(
            X_train,
            sol_t,
            L,
            torch.from_numpy(M).float(),
            num_samples=egop_samples,
            batch_size=egop_batch_size,
        )
        Ms.append(M.copy())

    K_train = laplacian(normalize(X_train), normalize(X_train), L, torch.from_numpy(M).float()).numpy()
    sol = np.linalg.solve(K_train + reg * np.eye(len(K_train)), y_train.numpy()).T
    sols.append(sol)

    K_test = laplacian(normalize(X_train), normalize(X_test), L, torch.from_numpy(M).float()).numpy()
    preds = (sol @ K_test).T
    final_mse = float(((preds - y_test.numpy()) ** 2).mean())

    if classif:
        y_pred = np.argmax(preds, axis=-1)
        labels = torch.argmax(y_test, dim=-1).numpy()
        final_acc = 100.0 * float((y_pred == labels).mean())
        logger.info(f"Final Acc: {final_acc:.2f}")
        if final_acc > best_acc:
            best_acc = final_acc
            best_iter = iters

    diff_acc = best_acc - kernel_acc

    return {
        "Ms": Ms,
        "mses": mses + [final_mse],
        "sols": sols,
        "bandwidth": L,
        "X_train": X_train,
        "y_train": y_train,
        "best_round_accuracy": best_acc,
        "iteration": best_iter,
        "diff_rfm_kernel_acc": diff_acc,
        "kernel_acc": kernel_acc,
        "kernel": "laplacian",
        "L": L,
        "reg": reg,
    }


def rfm_gaussian_from_tensors(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    iters: int = 0,
    reg: float = 1e-3,
    L: float = 1.0,
    train_acc: bool = False,
    classif: bool = True,
    egop_samples: Optional[int] = 20000,
    egop_batch_size: int = 2500,
):
    X_train = X_train.float().cpu()
    X_test = X_test.float().cpu()
    y_train = y_train.float().cpu()
    y_test = y_test.float().cpu()

    n, d = X_train.shape
    M = np.eye(d, dtype="float32")

    mses, Ms, sols = [], [], []
    Ms.append(M.copy())

    best_acc = -float("inf")
    best_iter = 0
    kernel_acc = 0.0

    for i in range(iters):
        K_train = gaussian(normalize(X_train), normalize(X_train), L, torch.from_numpy(M)).numpy()
        sol = np.linalg.solve(K_train + reg * np.eye(len(K_train)), y_train.numpy()).T
        sols.append(sol)

        K_test = gaussian(normalize(X_train), normalize(X_test), L, torch.from_numpy(M)).numpy()
        preds = (sol @ K_test).T
        mse_ = float(((preds - y_test.numpy()) ** 2).mean())
        mses.append(mse_)

        if classif:
            y_pred = np.argmax(preds, axis=-1)
            labels = torch.argmax(y_test, dim=-1).numpy()
            acc = 100.0 * float((y_pred == labels).mean())
            logger.info(f"Round {i} Acc: {acc:.2f}")

            if i == 0:
                kernel_acc = acc
            if acc > best_acc:
                best_acc = acc
                best_iter = i

        sol_t = torch.tensor(sol, dtype=torch.float32)
        M = compute_EGOP_with_jacrev_gaussian(
            X_train,
            sol_t,
            L,
            torch.from_numpy(M).float(),
            num_samples=egop_samples,
            batch_size=egop_batch_size,
        )
        Ms.append(M.copy())

    K_train = gaussian(normalize(X_train), normalize(X_train), L, torch.from_numpy(M).float()).numpy()
    sol = np.linalg.solve(K_train + reg * np.eye(len(K_train)), y_train.numpy()).T
    sols.append(sol)

    K_test = gaussian(normalize(X_train), normalize(X_test), L, torch.from_numpy(M).float()).numpy()
    preds = (sol @ K_test).T
    final_mse = float(((preds - y_test.numpy()) ** 2).mean())

    if classif:
        y_pred = np.argmax(preds, axis=-1)
        labels = torch.argmax(y_test, dim=-1).numpy()
        final_acc = 100.0 * float((y_pred == labels).mean())
        logger.info(f"Final Acc: {final_acc:.2f}")
        if final_acc >= best_acc:
            best_acc = final_acc
            best_iter = iters

    diff_acc = best_acc - kernel_acc

    return {
        "Ms": Ms,
        "mses": mses + [final_mse],
        "sols": sols,
        "bandwidth": L,
        "X_train": X_train,
        "y_train": y_train,
        "best_round_accuracy": best_acc,
        "iteration": best_iter,
        "diff_rfm_kernel_acc": diff_acc,
        "kernel_acc": kernel_acc,
        "kernel": "gaussian",
        "L": L,
        "reg": reg,
    }


def train(
    hyperparameters: Dict[str, Any],
    trainset: torch.utils.data.Dataset,
    testset: torch.utils.data.Dataset,
    tab_dataset,
    model_artifact_path: str = None,
    additional_callbacks: Optional[List[Any]] = None,
):
    """
    attack-tabular-compatible training entrypoint.
    """
    kernel = hyperparameters["kernel"]
    L = float(hyperparameters["L"])
    reg = float(hyperparameters.get("reg", 1e-3))
    version = hyperparameters.get("version", "kernel")
    # iters = int(hyperparameters.get("iters", 0 if version == "kernel" else 5))
    iters = int(hyperparameters.get("iters", 0 if version == "kernel" else 2))
    egop_samples = hyperparameters.get("egop_samples", 20000)
    egop_batch_size = int(hyperparameters.get("egop_batch_size", 2500))

    X_train, y_train_idx = dataset_to_tensors(trainset)
    X_test, y_test_idx = dataset_to_tensors(testset)

    y_train = labels_to_one_hot(y_train_idx, tab_dataset.n_classes)
    y_test = labels_to_one_hot(y_test_idx, tab_dataset.n_classes)

    logger.info(
        f"Training kernel model: kernel={kernel}, L={L}, reg={reg}, "
        f"iters={iters}, n_train={len(X_train)}, n_test={len(X_test)}"
    )

    if kernel == "laplacian":
        payload = rfm_laplacian_from_tensors(
            X_train, y_train, X_test, y_test,
            iters=iters,
            reg=reg,
            L=L,
            classif=True,
            egop_samples=egop_samples,
            egop_batch_size=egop_batch_size,
        )
    elif kernel == "gaussian":
        payload = rfm_gaussian_from_tensors(
            X_train, y_train, X_test, y_test,
            iters=iters,
            reg=reg,
            L=L,
            classif=True,
            egop_samples=egop_samples,
            egop_batch_size=egop_batch_size,
        )
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    # if model_artifact_path is None:
    #     raise ValueError("model_artifact_path must be provided for kernel training.")

    # os.makedirs(os.path.dirname(model_artifact_path), exist_ok=True)
    # torch.save(payload, model_artifact_path, pickle_protocol=4)

    # logger.info(f"Saved kernel artifact to {model_artifact_path}")

    if model_artifact_path is None:
        raise ValueError("model_artifact_path must be provided for kernel training.")

    kernel_name = hyperparameters["kernel"]
    version = hyperparameters.get("version", "kernel")
    dataset_name = tab_dataset.data_parameters["dataset_name"]

    base_dir = os.path.dirname(model_artifact_path)
    os.makedirs(base_dir, exist_ok=True)

    final_artifact_path = os.path.join(
        base_dir,
        f"{dataset_name}-{kernel_name}-{version}.pt"
    )

    torch.save(payload, final_artifact_path, pickle_protocol=4)

    logger.info(f"Saved kernel artifact to {final_artifact_path}")

    return {
        # "best_model_path": model_artifact_path,
        "best_model_path": final_artifact_path,
        "best_val_acc": payload["best_round_accuracy"],
        "best_val_hp_metric": payload["best_round_accuracy"],
        "best_iteration": payload["iteration"],
        "kernel_acc": payload["kernel_acc"],
    }


def grid_search_hyperparameters(
    trainset: torch.utils.data.Dataset,
    testset: torch.utils.data.Dataset,
    tab_dataset,
):
    """
    Minimal internal grid search.
    """
    search_space = {
        "kernel": ["gaussian"],
        "L": [0.05, 0.125, 0.25, 0.5, 1, 2.0, 4.0, 6.0],
        # "L": [0.05, 0.125],
        "reg": [1e-2, 1e-3, 1e-4],
        # "reg": [1e-2,1e-3],
        "version": ["rfm"],
        "iters": [5],
    }

    best_hparams = None
    best_score = -float("inf")

    for kernel in search_space["kernel"]:
        for L in search_space["L"]:
            for reg in search_space["reg"]:
                hparams = {
                    "kernel": kernel,
                    "L": L,
                    "reg": reg,
                    "version": "rfm",
                    "iters": 5,
                }

                tmp_artifact = "outputs/tmp_kernel_bank_gaus.pt"
                results = train(
                    hparams,
                    trainset=trainset,
                    testset=testset,
                    tab_dataset=tab_dataset,
                    model_artifact_path=tmp_artifact,
                )

                score = results["best_val_hp_metric"]
                logger.info(f"kernel={kernel} L={L} reg={reg} => score={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_hparams = hparams.copy()

    logger.info(f"Best kernel hyperparameters: {best_hparams}")
    return best_hparams