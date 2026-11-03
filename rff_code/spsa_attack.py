#!/usr/bin/env python3

import argparse
import gc
import json
import pickle
import random
from pathlib import Path
from typing import Optional, Sequence

import cupy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoattack import AutoAttack
from torchkernels.kernels.radial import laplacian
import dataset

from cuml.svm import SVC
from sklearn.svm import SVC as SklearnSVC


torch.autograd.set_detect_anomaly(True)

class OneVsRestKernelSVC:
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
        self.classes_ = cp.asarray(
            sorted(int(c) for c in cp.asnumpy(classes))
        )
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
        return cp.stack(
            [
                est.decision_function(X_cp).reshape(-1)
                for est in self.estimators_
            ],
            axis=1,
        )

    def predict(self, X_cp):
        scores = self.decision_function(X_cp)
        return self.classes_[cp.argmax(scores, axis=1)]

class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "OneVsRestKernelSVC":
            return OneVsRestKernelSVC

        return super().find_class(module, name)


def load_svc_checkpoint(path: Path):
    with open(path, "rb") as f:
        return _CheckpointUnpickler(f).load()

def cp_to_torch(x, device, dtype=None):
    arr = cp.asnumpy(cp.asarray(x))

    t = torch.as_tensor(
        arr,
        device=device,
    )

    if dtype is not None:
        t = t.to(dtype=dtype)

    return t


def labels_to_indices(y):
    if y.ndim == 2:
        y = y.argmax(dim=1)

    return y.reshape(-1).long()


def dataloader_to_tensors(loader):
    xs = []
    ys = []

    for x, y in loader:
        xs.append(x)
        ys.append(y)

    return (
        torch.cat(xs, dim=0),
        torch.cat(ys, dim=0),
    )


def load_multiclass_dataset(name):
    name = name.lower()

    if name == "svhn":
        return dataset.get_svhn(
            multiclass=True
        )

    if name in {
        "fmnist",
        "kmnist",
        "qmnist",
        "emnist",
    }:
        return dataset.load_fmnist(
            classes=(0, 1),
            which_mnist=name,
            multiclass=True,
        )

    if name == "cifar10":
        return dataset.get_cifar(
            classes=(0, 1),
            multiclass=True,
        )

    raise ValueError(
        f"Unsupported dataset: {name}"
    )


def normalize_rows(x, eps=1e-12):
    return (
        x
        / x.norm(
            p=2,
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)
    )


def infer_model_dir(
    model_root,
    dataset_name,
    kernel_name,
):
    return (
        model_root
        / f"svc_{kernel_name}_{dataset_name}_mc"
    )


def find_model_file(
    model_dir,
    dataset_name,
    kernel_name,
):
    stem_kernel = (
        "laplace"
        if kernel_name == "laplacian"
        else kernel_name
    )

    expected = (
        model_dir
        / f"{dataset_name}_{stem_kernel}_mc_svc_best.pt"
    )

    if expected.exists():
        return expected

    candidates = sorted(
        model_dir.glob("*_svc_best.pt")
    )

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No *_svc_best.pt found in {model_dir}"
        )

    raise RuntimeError(
        "Multiple model files found: "
        f"{[p.name for p in candidates]}; "
        "pass --MODEL_FILE"
    )


def find_xtrain_file(
    model_dir,
    dataset_name,
    kernel_name,
    explicit_path=None,
):
    if explicit_path is not None:
        p = (
            Path(explicit_path)
            .expanduser()
            .resolve()
        )

        if not p.exists():
            raise FileNotFoundError(
                f"--XTRAIN_FILE not found: {p}"
            )

        return p

    stem_kernel = (
        "laplace"
        if kernel_name == "laplacian"
        else kernel_name
    )

    expected = (
        model_dir
        / f"{dataset_name}_{stem_kernel}_mc_xtrain.pt"
    )

    if expected.exists():
        return expected

    candidates = sorted(
        model_dir.glob("*_xtrain.pt")
    )

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        return None

    raise RuntimeError(
        "Multiple xtrain files found: "
        f"{[p.name for p in candidates]}; "
        "pass --XTRAIN_FILE"
    )


def find_results_json(
    model_dir,
    dataset_name,
    kernel_name,
):
    stem_kernel = (
        "laplace"
        if kernel_name == "laplacian"
        else kernel_name
    )

    expected = (
        model_dir
        / f"{dataset_name}_{stem_kernel}_mc_svc_results.json"
    )

    if expected.exists():
        return expected

    candidates = sorted(
        model_dir.glob("*_svc_results.json")
    )

    if len(candidates) == 1:
        return candidates[0]

    return None


def read_best_gamma(results_json: Path):
    if results_json is None:
        raise FileNotFoundError(
            "Could not locate the SVC results JSON. "
            "For a precomputed Laplacian SVC, gamma "
            "must be obtained from the training results. "
            "Pass --GAMMA explicitly if necessary."
        )

    with open(results_json, "r") as f:
        obj = json.load(f)

    if "best" not in obj:
        raise KeyError(
            f"'best' is missing from {results_json}"
        )

    if "gamma" not in obj["best"]:
        raise KeyError(
            f"'best.gamma' is missing from {results_json}"
        )

    return float(obj["best"]["gamma"])


def read_expected_test_accuracy(results_json):
    if results_json is None:
        return None

    try:
        with open(results_json, "r") as f:
            obj = json.load(f)

        value = obj["best"]["test_acc"]

        if value is None:
            return None

        value = float(value)

        if value <= 1.0:
            value *= 100.0

        return value

    except Exception:
        return None


def estimator_gamma(estimator):
    g = getattr(
        estimator,
        "gamma",
        None,
    )

    if isinstance(g, str) or g is None:
        raise RuntimeError(
            "Saved estimator has no numeric gamma; "
            "pass --GAMMA."
        )

    return float(g)

class DifferentiableOvRSVC(nn.Module):
    def __init__(
        self,
        fitted_ovr,
        device,
        gamma_override=None,
        rbf_chunk_size=8192,
    ):
        super().__init__()

        if not hasattr(
            fitted_ovr,
            "estimators_",
        ):
            raise TypeError(
                "Expected a OneVsRestKernelSVC checkpoint."
            )

        self.device = device
        self.rbf_chunk_size = int(
            rbf_chunk_size
        )

        classes = (
            cp.asnumpy(
                cp.asarray(
                    fitted_ovr.classes_
                )
            )
            .astype(np.int64)
        )

        self.register_buffer(
            "class_labels",
            torch.as_tensor(
                classes,
                device=device,
                dtype=torch.long,
            ),
        )

        self.num_classes = len(
            fitted_ovr.estimators_
        )

        self.estimators_meta = []

        for class_idx, est in enumerate(
            fitted_ovr.estimators_
        ):
            dual = cp_to_torch(
                est.dual_coef_,
                device,
                torch.float32,
            ).reshape(-1)

            intercept = cp_to_torch(
                est.intercept_,
                device,
                torch.float32,
            ).reshape(-1)[0]

            gamma = (
                float(gamma_override)
                if gamma_override is not None
                else estimator_gamma(est)
            )

            support_vectors = cp_to_torch(
                est.support_vectors_,
                device,
                torch.float32,
            )

            if (
                support_vectors.ndim != 2
                or support_vectors.shape[0] == 0
            ):
                raise RuntimeError(
                    f"Class {class_idx}: "
                    "no feature-space support vectors found."
                )

            if (
                dual.numel()
                != support_vectors.shape[0]
            ):
                raise RuntimeError(
                    f"Class {class_idx}: "
                    f"dual count {dual.numel()} != "
                    f"support count "
                    f"{support_vectors.shape[0]}"
                )

            self.register_buffer(
                f"dual_{class_idx}",
                dual,
            )

            self.register_buffer(
                f"bias_{class_idx}",
                intercept,
            )

            self.register_buffer(
                f"sv_{class_idx}",
                support_vectors,
            )

            self.estimators_meta.append({
                "gamma": gamma,
                "num_support": int(
                    support_vectors.shape[0]
                ),
            })

    def _rbf_score(
        self,
        x,
        class_idx,
    ):
        sv = getattr(
            self,
            f"sv_{class_idx}",
        )

        dual = getattr(
            self,
            f"dual_{class_idx}",
        )

        bias = getattr(
            self,
            f"bias_{class_idx}",
        )

        gamma = self.estimators_meta[
            class_idx
        ]["gamma"]

        out = torch.zeros(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

        x_sq = (
            (x * x)
            .sum(
                dim=1,
                keepdim=True,
            )
        )

        for start in range(
            0,
            sv.shape[0],
            self.rbf_chunk_size,
        ):
            sv_c = sv[
                start:
                start + self.rbf_chunk_size
            ]

            dual_c = dual[
                start:
                start + self.rbf_chunk_size
            ]

            sv_sq = (
                (sv_c * sv_c)
                .sum(dim=1)
                .unsqueeze(0)
            )

            dist2 = (
                x_sq
                + sv_sq
                - 2.0 * (x @ sv_c.T)
            ).clamp_min(0.0)

            K = torch.exp(
                -gamma * dist2
            )

            out = out + K @ dual_c

        return out + bias

    def forward(self, x):
        if x.ndim > 2:
            x = x.flatten(1)

        x = x.to(
            dtype=torch.float32
        )

        scores = [
            self._rbf_score(x, i)
            for i in range(
                self.num_classes
            )
        ]

        return torch.stack(
            scores,
            dim=1,
        )

class DifferentiableSklearnPrecomputedSVC(
    nn.Module
):
    def __init__(
        self,
        fitted_svc,
        x_train,
        gamma,
        device,
        laplacian_chunk_size=4096,
    ):
        super().__init__()

        if not isinstance(
            fitted_svc,
            SklearnSVC,
        ):
            raise TypeError(
                "Expected sklearn.svm.SVC."
            )

        if fitted_svc.kernel != "precomputed":
            raise ValueError(
                "This class is only for "
                "sklearn SVC(kernel='precomputed')."
            )

        self.device = device
        self.gamma = float(gamma)

        self.laplacian_chunk_size = int(
            laplacian_chunk_size
        )

        classes = np.asarray(
            fitted_svc.classes_,
            dtype=np.int64,
        )

        self.register_buffer(
            "class_labels",
            torch.as_tensor(
                classes,
                device=device,
                dtype=torch.long,
            ),
        )

        self.num_classes = int(
            len(classes)
        )

        if x_train.ndim > 2:
            x_train = x_train.flatten(1)

        x_train = x_train.to(
            device=device,
            dtype=torch.float32,
        )

        support_idx = torch.as_tensor(
            np.asarray(
                fitted_svc.support_,
                dtype=np.int64,
            ),
            device=device,
            dtype=torch.long,
        )

        if (
            int(support_idx.max().item())
            >= x_train.shape[0]
        ):
            raise RuntimeError(
                "support_ contains index "
                f"{int(support_idx.max().item())} "
                "but x_train only has "
                f"{x_train.shape[0]} rows. "
                "x_train does not match the "
                "fit-time training tensor."
            )

        self.register_buffer(
            "support_indices",
            support_idx,
        )

        support_vectors = (
            x_train[support_idx]
            .detach()
        )

        support_vectors = normalize_rows(
            support_vectors
        )

        self.register_buffer(
            "support_vectors",
            support_vectors,
        )

        n_support = np.asarray(
            fitted_svc.n_support_,
            dtype=np.int64,
        )

        self.register_buffer(
            "n_support",
            torch.as_tensor(
                n_support,
                device=device,
                dtype=torch.long,
            ),
        )

        starts = np.zeros(
            self.num_classes,
            dtype=np.int64,
        )

        if self.num_classes > 1:
            starts[1:] = np.cumsum(
                n_support[:-1]
            )

        ends = starts + n_support

        self.starts = [
            int(v)
            for v in starts
        ]

        self.ends = [
            int(v)
            for v in ends
        ]

        dual_coef = torch.as_tensor(
            np.asarray(
                fitted_svc.dual_coef_,
                dtype=np.float32,
            ),
            device=device,
            dtype=torch.float32,
        )

        intercept = torch.as_tensor(
            np.asarray(
                fitted_svc.intercept_,
                dtype=np.float32,
            ),
            device=device,
            dtype=torch.float32,
        )

        self.register_buffer(
            "dual_coef",
            dual_coef,
        )

        self.register_buffer(
            "intercept",
            intercept,
        )

        expected_pairs = (
            self.num_classes
            * (self.num_classes - 1)
            // 2
        )

        if (
            self.intercept.numel()
            != expected_pairs
        ):
            raise RuntimeError(
                f"Expected {expected_pairs} "
                "pairwise intercepts, got "
                f"{self.intercept.numel()}."
            )

        d = int(
            support_vectors.shape[1]
        )

        self.register_buffer(
            "M",
            torch.eye(
                d,
                device=device,
                dtype=torch.float32,
            ),
        )

        self.estimators_meta = [
            {
                "gamma": self.gamma,
                "num_support": int(
                    n_support[i]
                ),
            }
            for i in range(
                self.num_classes
            )
        ]

    def _kernel_to_all_support_vectors(
        self,
        x,
    ):
        xh = normalize_rows(x)

        n_sv = self.support_vectors.shape[0]

        chunks = []

        for start in range(
            0,
            n_sv,
            self.laplacian_chunk_size,
        ):
            sv_c = self.support_vectors[
                start:
                start + self.laplacian_chunk_size
            ]

            K_sv_x = laplacian(
                sv_c,
                xh,
                self.gamma,
                self.M,
                in_place=False,
            )

            chunks.append(
                K_sv_x
            )

        return (
            torch.cat(
                chunks,
                dim=0,
            )
            .T
            .contiguous()
        )

    def _ovo_decision_function(
        self,
        K_x_sv,
    ):
        pair_scores = []
        pair_idx = 0

        for i in range(
            self.num_classes
        ):
            si = self.starts[i]
            ei = self.ends[i]

            for j in range(
                i + 1,
                self.num_classes,
            ):
                sj = self.starts[j]
                ej = self.ends[j]

                coef_i = self.dual_coef[
                    j - 1,
                    si:ei,
                ]

                coef_j = self.dual_coef[
                    i,
                    sj:ej,
                ]

                score = (
                    self.intercept[
                        pair_idx
                    ]
                    .expand(
                        K_x_sv.shape[0]
                    )
                )

                if ei > si:
                    score = (
                        score
                        + K_x_sv[
                            :,
                            si:ei,
                        ]
                        @ coef_i
                    )

                if ej > sj:
                    score = (
                        score
                        + K_x_sv[
                            :,
                            sj:ej,
                        ]
                        @ coef_j
                    )

                pair_scores.append(
                    score
                )

                pair_idx += 1

        return torch.stack(
            pair_scores,
            dim=1,
        )

    def _ovo_to_ovr_scores(
        self,
        ovo,
    ):
        batch = ovo.shape[0]

        votes = torch.zeros(
            batch,
            self.num_classes,
            device=ovo.device,
            dtype=ovo.dtype,
        )

        sum_conf = torch.zeros_like(
            votes
        )

        k = 0

        for i in range(
            self.num_classes
        ):
            for j in range(
                i + 1,
                self.num_classes,
            ):
                dec = ovo[:, k]

                sum_conf[:, i] = (
                    sum_conf[:, i]
                    + dec
                )

                sum_conf[:, j] = (
                    sum_conf[:, j]
                    - dec
                )

                vote_j = (
                    dec < 0
                ).to(
                    dtype=ovo.dtype
                )

                vote_i = (
                    1.0 - vote_j
                )

                votes[:, i] = (
                    votes[:, i]
                    + vote_i
                )

                votes[:, j] = (
                    votes[:, j]
                    + vote_j
                )

                k += 1

        transformed = (
            sum_conf
            / (
                3.0
                * (
                    torch.abs(sum_conf)
                    + 1.0
                )
            )
        )

        return (
            votes
            + transformed
        )

    def forward(self, x):
        if x.ndim > 2:
            x = x.flatten(1)

        x = x.to(
            dtype=torch.float32
        )

        K_x_sv = (
            self
            ._kernel_to_all_support_vectors(
                x
            )
        )

        ovo = (
            self
            ._ovo_decision_function(
                K_x_sv
            )
        )

        return (
            self
            ._ovo_to_ovr_scores(
                ovo
            )
        )

@torch.no_grad()
def accuracy(
    model,
    x,
    y,
    batch_size,
):
    correct = 0
    total = 0

    for start in range(
        0,
        x.shape[0],
        batch_size,
    ):
        xb = x[
            start:
            start + batch_size
        ]

        yb = y[
            start:
            start + batch_size
        ]

        pred_idx = (
            model(xb)
            .argmax(dim=1)
        )

        pred = (
            model.class_labels[
                pred_idx
            ]
            if hasattr(
                model,
                "class_labels",
            )
            else pred_idx
        )

        correct += int(
            (pred == yb)
            .sum()
            .item()
        )

        total += int(
            yb.numel()
        )

    return (
        100.0
        * correct
        / max(total, 1)
    )


@torch.no_grad()
def compare_reconstructed_with_sklearn(
    fitted_svc,
    reconstructed_model,
    x_train,
    x_test,
    gamma,
    batch_size,
    max_samples=256,
):
    n = min(
        int(max_samples),
        int(x_test.shape[0]),
    )

    if n <= 0:
        return None, 0, n

    x_small = x_test[:n]

    xh_train = (
        normalize_rows(
            x_train
        )
        .double()
    )

    xh_small = (
        normalize_rows(
            x_small
        )
        .double()
    )

    d = int(
        x_train.shape[1]
    )

    M = torch.eye(
        d,
        device=x_train.device,
        dtype=torch.float64,
    )

    K_small = (
        laplacian(
            xh_train,
            xh_small,
            gamma,
            M,
        )
        .T
        .contiguous()
    )

    pred_sk = fitted_svc.predict(
        K_small
        .detach()
        .cpu()
        .numpy()
    )

    pred_sk = torch.as_tensor(
        pred_sk,
        device=x_test.device,
        dtype=torch.long,
    )

    logits = reconstructed_model(
        x_small
    )

    pred_rec_idx = (
        logits.argmax(dim=1)
    )

    pred_rec = (
        reconstructed_model
        .class_labels[
            pred_rec_idx
        ]
    )

    n_mismatch = int(
        (pred_rec != pred_sk)
        .sum()
        .item()
    )

    agreement = (
        100.0
        * (
            1.0
            - n_mismatch / n
        )
    )

    return (
        agreement,
        n_mismatch,
        n,
    )


def parse_epsilons(
    values: Optional[
        Sequence[float]
    ],
    norm,
):
    if values:
        return [
            float(v)
            for v in values
        ]

    if norm == "Linf":
        return [
            0,
            0.1,
            1.0,
            2.0,
            4.0,
            6.0,
            8.0,
            12.0,
            20.0,
            32.0,
            40.0,
            55.0,
        ]

    return [
        0,
        0.5,
        1.0,
        2.0,
        4.0,
        6.0,
        8.0,
    ]

def spsa_fast(
    model,
    x,
    y,
    epsilon,
    lr,
    iterations=40,
    sample_size=128,
    norm="L2",
    seed=42,
):
    x_adv = (
        x.detach()
        .clone()
    )

    B, D = x_adv.shape

    print(
        f"SPSA input batch size: {B}"
    )
    print(
        f"SPSA samples/directions: "
        f"{sample_size}"
    )
    print(
        f"SPSA iterations: "
        f"{iterations}"
    )
    print(
        f"SPSA learning rate: {lr}"
    )

    eta = float(
        np.clip(
            epsilon / 2.0,
            1e-3,
            0.01,
        )
    )

    print(
        f"SPSA eta: {eta}"
    )

    generator = torch.Generator(
        device=x.device
    )

    generator.manual_seed(
        seed
    )

    for iteration in range(
        iterations
    ):
        print(
            f"SPSA iteration "
            f"{iteration + 1}/{iterations}"
        )

        v = (
            torch.empty(
                sample_size,
                B,
                D,
                device=x.device,
                dtype=x.dtype,
            )
            .bernoulli_(
                0.5,
                generator=generator,
            )
            .mul_(2)
            .sub_(1)
        )

        x_plus = (
            x_adv.unsqueeze(0)
            + eta * v
        ).clamp(
            0.0,
            1.0,
        )

        x_minus = (
            x_adv.unsqueeze(0)
            - eta * v
        ).clamp(
            0.0,
            1.0,
        )

        x_pm = torch.cat(
            [
                x_plus,
                x_minus,
            ],
            dim=0,
        )

        x_pm = x_pm.reshape(
            2
            * sample_size
            * B,
            D,
        )

        y_rep = y.repeat(
            2 * sample_size
        )

        with torch.no_grad():
            logits = model(
                x_pm
            )

            losses = (
                F.cross_entropy(
                    logits,
                    y_rep,
                    reduction="none",
                )
            )

        losses = losses.view(
            2 * sample_size,
            B,
        )

        loss_plus = losses[
            :sample_size
        ]

        loss_minus = losses[
            sample_size:
        ]

        diff = (
            loss_plus
            - loss_minus
        ).unsqueeze(2)

        grad_estimate = (
            (
                diff
                / (2.0 * eta)
            )
            * v
        ).mean(
            dim=0
        )

        if norm == "Linf":

            x_adv = (
                x_adv
                + lr
                * torch.sign(
                    grad_estimate
                )
            )

        elif norm == "L2":

            g_norm = (
                grad_estimate
                .norm(
                    p=2,
                    dim=1,
                    keepdim=True,
                )
                .clamp_min(
                    1e-12
                )
            )

            x_adv = (
                x_adv
                + lr
                * grad_estimate
                / g_norm
            )

        else:
            raise ValueError(
                "Unsupported SPSA norm: "
                f"{norm}"
            )

        delta = (
            x_adv
            - x
        )

        if norm == "Linf":

            delta = torch.clamp(
                delta,
                -epsilon,
                epsilon,
            )

        else:

            delta_norm = (
                delta
                .norm(
                    p=2,
                    dim=1,
                    keepdim=True,
                )
                .clamp_min(
                    1e-12
                )
            )

            scale = (
                epsilon
                / delta_norm
            ).clamp(
                max=1.0
            )

            delta = (
                delta
                * scale
            )

        x_adv = torch.clamp(
            x + delta,
            0.0,
            1.0,
        ).detach()

        # Free large SPSA intermediates
        del (
            v,
            x_plus,
            x_minus,
            x_pm,
            y_rep,
            logits,
            losses,
            loss_plus,
            loss_minus,
            diff,
            grad_estimate,
            delta,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return x_adv
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--DATASET",
        required=True,
        choices=[
            "fmnist",
            "kmnist",
            "qmnist",
            "emnist",
            "svhn",
            "cifar10",
        ],
    )

    parser.add_argument(
        "--KERNEL",
        required=True,
        choices=[
            "gaussian",
            "rbf",
            "laplacian",
        ],
    )

    parser.add_argument(
        "--AA_norm",
        required=True,
        choices=[
            "Linf",
            "L2",
        ],
    )

    parser.add_argument(
        "--MODEL_ROOT",
        default="./svc_cv",
    )

    parser.add_argument(
        "--MODEL_DIR",
        default=None,
    )

    parser.add_argument(
        "--MODEL_FILE",
        default=None,
    )

    parser.add_argument(
        "--RESULTS_JSON",
        default=None,
    )

    parser.add_argument(
        "--XTRAIN_FILE",
        default=None,
        help=(
            "Exact fit-time x_train file. "
            "Required for Laplacian unless "
            "auto-detected."
        ),
    )

    parser.add_argument(
        "--SEED",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--BATCH_SIZE",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--NUM_TEST",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--EPSILONS",
        nargs="*",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--GAMMA",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--LAPLACIAN_CHUNK",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--RBF_CHUNK",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--SAVE_ADV",
        action="store_true",
    )

    parser.add_argument(
        "--SKIP_SKLEARN_CHECK",
        action="store_true",
    )

    parser.add_argument(
        "--CHECK_SAMPLES",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--MIN_AGREEMENT",
        type=float,
        default=99.0,
    )

    parser.add_argument(
        "--ATTACK",
        default="apgd-ce",
        choices=[
            "apgd-ce",
            "spsa",
        ],
    )

    parser.add_argument(
        "--SPSA_ITERS",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--SPSA_SAMPLES",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--SPSA_LR",
        type=float,
        default=None,
        help=(
            "SPSA step size. "
            "Default is epsilon/4."
        ),
    )

    args = parser.parse_args()

    dataset_name = (
        args.DATASET.lower()
    )

    kernel_name = (
        args.KERNEL.lower()
    )

    kernel_folder_name = (
        "gaussian"
        if kernel_name == "rbf"
        else kernel_name
    )

    norm = args.AA_norm

    random.seed(
        args.SEED
    )

    np.random.seed(
        args.SEED
    )

    torch.manual_seed(
        args.SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.SEED
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if args.MODEL_DIR:

        model_dir = (
            Path(
                args.MODEL_DIR
            )
            .expanduser()
            .resolve()
        )

    else:

        model_dir = infer_model_dir(
            Path(
                args.MODEL_ROOT
            )
            .expanduser()
            .resolve(),
            dataset_name,
            kernel_folder_name,
        )

    if args.MODEL_FILE:

        model_file = (
            Path(
                args.MODEL_FILE
            )
            .expanduser()
            .resolve()
        )

    else:

        model_file = find_model_file(
            model_dir,
            dataset_name,
            kernel_folder_name,
        )

    if args.RESULTS_JSON:

        results_json = (
            Path(
                args.RESULTS_JSON
            )
            .expanduser()
            .resolve()
        )

    else:

        results_json = (
            find_results_json(
                model_dir,
                dataset_name,
                kernel_folder_name,
            )
        )

    print(
        "Model directory:",
        model_dir,
    )

    print(
        "Model file:",
        model_file,
    )

    print(
        "Results JSON:",
        results_json,
    )

    _, _, testloader = (
        load_multiclass_dataset(
            dataset_name
        )
    )

    x_test, y_test = (
        dataloader_to_tensors(
            testloader
        )
    )

    if x_test.ndim > 2:
        x_test = x_test.flatten(1)

    y_test = labels_to_indices(
        y_test
    )

    x_test = x_test.to(
        device=device,
        dtype=torch.float32,
    )

    y_test = y_test.to(
        device=device,
        dtype=torch.long,
    )

    if args.NUM_TEST > 0:

        n = min(
            args.NUM_TEST,
            x_test.shape[0],
        )

        x_test = x_test[:n]
        y_test = y_test[:n]

    print(
        "x_test:",
        tuple(
            x_test.shape
        ),
    )

    print(
        "Input min/max:",
        float(
            x_test.min()
        ),
        float(
            x_test.max()
        ),
    )

    if (
        x_test.min() < -1e-6
        or x_test.max() > 1.0 + 1e-6
    ):
        raise ValueError(
            "Attack code assumes "
            "inputs are in [0,1]."
        )

    fitted = load_svc_checkpoint(
        model_file
    )

    print(
        "Loaded checkpoint:",
        type(fitted),
    )

    expected_test_acc = (
        read_expected_test_accuracy(
            results_json
        )
    )

    if isinstance(
        fitted,
        SklearnSVC,
    ):

        print(
            "Detected sklearn "
            "precomputed SVC."
        )

        print(
            "classes_:",
            fitted.classes_,
        )

        print(
            "n_support_:",
            fitted.n_support_,
        )

        print(
            "total support vectors:",
            len(
                fitted.support_
            ),
        )

        if kernel_name != "laplacian":
            raise ValueError(
                "Loaded sklearn "
                "precomputed SVC but "
                "--KERNEL is not laplacian."
            )

        if args.GAMMA is not None:
            gamma = float(
                args.GAMMA
            )
        else:
            gamma = read_best_gamma(
                results_json
            )

        print(
            "Using Laplacian gamma:",
            gamma,
        )

        xtrain_file = find_xtrain_file(
            model_dir,
            dataset_name,
            kernel_folder_name,
            args.XTRAIN_FILE,
        )

        if xtrain_file is None:
            raise FileNotFoundError(
                "No saved fit-time x_train "
                "found. Expected "
                f"{dataset_name}_laplace_mc_xtrain.pt "
                f"in {model_dir}, or pass "
                "--XTRAIN_FILE."
            )

        print(
            "Loading exact fit-time "
            "x_train from:",
            xtrain_file,
        )

        xtrain_blob = torch.load(
            xtrain_file,
            map_location=device,
        )

        x_train = (
            xtrain_blob["x_train"]
            .to(
                device=device,
                dtype=torch.float32,
            )
        )

        if x_train.ndim > 2:
            x_train = x_train.flatten(1)

        print(
            "x_train:",
            tuple(
                x_train.shape
            ),
        )

        attack_model = (
            DifferentiableSklearnPrecomputedSVC(
                fitted_svc=fitted,
                x_train=x_train,
                gamma=gamma,
                device=device,
                laplacian_chunk_size=(
                    args.LAPLACIAN_CHUNK
                ),
            )
            .to(device)
            .eval()
        )

        for cls, meta in zip(
            attack_model
            .class_labels
            .tolist(),
            attack_model
            .estimators_meta,
        ):
            print(
                f"class={cls}: "
                f"n_support="
                f"{meta['num_support']} "
                f"gamma="
                f"{meta['gamma']}"
            )

        if not args.SKIP_SKLEARN_CHECK:

            (
                agreement,
                n_mismatch,
                n_checked,
            ) = (
                compare_reconstructed_with_sklearn(
                    fitted_svc=fitted,
                    reconstructed_model=attack_model,
                    x_train=x_train,
                    x_test=x_test,
                    gamma=gamma,
                    batch_size=args.BATCH_SIZE,
                    max_samples=args.CHECK_SAMPLES,
                )
            )

            print(
                "Reconstruction vs sklearn "
                "prediction agreement: "
                f"{agreement:.4f}% "
                f"({n_mismatch}/{n_checked} "
                "disagree)"
            )

            if (
                agreement
                < args.MIN_AGREEMENT
            ):
                raise RuntimeError(
                    "Reconstruction agreement "
                    f"{agreement:.4f}% is below "
                    "--MIN_AGREEMENT "
                    f"({args.MIN_AGREEMENT}%)."
                )
    else:

        if not hasattr(
            fitted,
            "estimators_",
        ):
            raise TypeError(
                "Unsupported checkpoint "
                f"type: {type(fitted)}"
            )

        print(
            "Detected custom "
            "cuML OvR SVC."
        )

        print(
            "Binary estimators:",
            len(
                fitted.estimators_
            ),
        )

        if kernel_name not in {
            "gaussian",
            "rbf",
        }:
            raise ValueError(
                "Custom cuML OvR "
                "checkpoint expected "
                "for Gaussian/RBF."
            )

        attack_model = (
            DifferentiableOvRSVC(
                fitted_ovr=fitted,
                device=device,
                gamma_override=args.GAMMA,
                rbf_chunk_size=args.RBF_CHUNK,
            )
            .to(device)
            .eval()
        )

        for cls, meta in zip(
            attack_model
            .class_labels
            .tolist(),
            attack_model
            .estimators_meta,
        ):
            print(
                f"class={cls}: "
                f"n_support="
                f"{meta['num_support']} "
                f"gamma="
                f"{meta['gamma']}"
            )

    clean_acc = accuracy(
        attack_model,
        x_test,
        y_test,
        args.BATCH_SIZE,
    )

    print(
        f"Clean accuracy: "
        f"{clean_acc:.4f}%"
    )

    if expected_test_acc is not None:

        print(
            "Training JSON test accuracy: "
            f"{expected_test_acc:.4f}%"
        )

        print(
            "Absolute clean-accuracy "
            "difference: "
            f"{abs(clean_acc - expected_test_acc):.6f}%"
        )

    out_dir = (
        model_dir
        / f"seed_{args.SEED}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    attack_name = args.ATTACK

    results = []

    output_json = (
        out_dir
        / (
            f"{attack_name}_"
            f"{norm}_"
            f"{dataset_name}_"
            f"{kernel_folder_name}_"
            f"mc.json"
        )
    )

    for eps_value in parse_epsilons(
        args.EPSILONS,
        norm,
    ):

        if norm == "Linf":
            eps = (
                float(eps_value)
                / 255.0
            )
        else:
            eps = float(
                eps_value
            )

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"{attack_name.upper()} | "
            f"{dataset_name} | "
            f"{kernel_name} | "
            f"{norm} | "
            f"eps_input={eps_value} | "
            f"eps={eps}"
        )

        print(
            "=" * 80
        )

        if eps == 0.0:

            robust_acc = (
                clean_acc
            )

            x_adv_all = (
                x_test
                .detach()
                .cpu()
                .clone()
                if args.SAVE_ADV
                else None
            )

        elif attack_name == "apgd-ce":

            adversary = AutoAttack(
                attack_model,
                norm=norm,
                eps=eps,
                version="standard",
                seed=args.SEED,
                device=device,
            )

            adversary.attacks_to_run = [
                "apgd-ce"
            ]

            x_adv = (
                adversary
                .run_standard_evaluation(
                    x_test,
                    y_test,
                    bs=args.BATCH_SIZE,
                )
            )

            robust_acc = accuracy(
                attack_model,
                x_adv,
                y_test,
                args.BATCH_SIZE,
            )

            x_adv_all = (
                x_adv
                .detach()
                .cpu()
                if args.SAVE_ADV
                else None
            )

            del x_adv

        elif attack_name == "spsa":

            spsa_lr = (
                float(
                    args.SPSA_LR
                )
                if args.SPSA_LR is not None
                else eps / 4.0
            )

            print(
                "Running SPSA on entire "
                f"test set: {x_test.shape[0]} "
                "samples"
            )

            print(
                "SPSA configuration:"
            )

            print(
                f"  batch size : "
                f"{x_test.shape[0]}"
            )

            print(
                f"  samples    : "
                f"{args.SPSA_SAMPLES}"
            )

            print(
                f"  iterations : "
                f"{args.SPSA_ITERS}"
            )

            print(
                f"  lr         : "
                f"{spsa_lr}"
            )

            x_adv = spsa_fast(
                model=attack_model,
                x=x_test,
                y=y_test,
                epsilon=eps,
                lr=spsa_lr,
                iterations=args.SPSA_ITERS,
                sample_size=args.SPSA_SAMPLES,
                norm=norm,
                seed=args.SEED,
            )

            robust_acc = accuracy(
                attack_model,
                x_adv,
                y_test,
                args.BATCH_SIZE,
            )

            x_adv_all = (
                x_adv
                .detach()
                .cpu()
                if args.SAVE_ADV
                else None
            )

            del x_adv

        else:

            raise ValueError(
                "Unknown attack: "
                f"{attack_name}"
            )

        print(
            f"epsilon={eps:.8f}, "
            f"robust accuracy="
            f"{robust_acc:.4f}%"
        )

        adv_path = None

        if args.SAVE_ADV:

            adv_path = (
                out_dir
                / (
                    f"adv_"
                    f"{attack_name}_"
                    f"{norm}_"
                    f"{dataset_name}_"
                    f"{kernel_folder_name}_"
                    f"eps_"
                    f"{str(eps_value).replace('.', 'p')}"
                    f".pt"
                )
            )

            torch.save(
                {
                    "x_adv": x_adv_all,
                    "y": (
                        y_test
                        .detach()
                        .cpu()
                    ),
                    "eps_input": float(
                        eps_value
                    ),
                    "eps": float(
                        eps
                    ),
                    "norm": norm,
                    "attack": attack_name,
                },
                adv_path,
            )
        results.append({
            "eps_input": float(
                eps_value
            ),
            "eps_float": float(
                eps
            ),
            "norm": norm,
            "attack": attack_name,
            "robust_acc": float(
                robust_acc
            ),
            "adv_path": (
                str(adv_path)
                if adv_path is not None
                else None
            ),
        })

        out_obj = {
            "dataset": dataset_name,
            "kernel": kernel_folder_name,
            "model_file": str(
                model_file
            ),
            "results_json": (
                str(results_json)
                if results_json is not None
                else None
            ),
            "seed": int(
                args.SEED
            ),
            "norm": norm,
            "attack": attack_name,
            "clean_acc": float(
                clean_acc
            ),
            "training_json_test_acc": (
                float(
                    expected_test_acc
                )
                if expected_test_acc is not None
                else None
            ),
            "num_test": int(
                x_test.shape[0]
            ),

            "batch_size": (
                int(
                    x_test.shape[0]
                )
                if attack_name == "spsa"
                else int(
                    args.BATCH_SIZE
                )
            ),

            "spsa_iterations": (
                int(
                    args.SPSA_ITERS
                )
                if attack_name == "spsa"
                else None
            ),

            "spsa_samples": (
                int(
                    args.SPSA_SAMPLES
                )
                if attack_name == "spsa"
                else None
            ),

            "spsa_lr": (
                float(
                    spsa_lr
                )
                if (
                    attack_name == "spsa"
                    and eps != 0.0
                )
                else None
            ),

            "results": results,
        }

        with open(
            output_json,
            "w",
        ) as f:
            json.dump(
                out_obj,
                f,
                indent=2,
            )

        del x_adv_all

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cp.get_default_memory_pool().free_all_blocks()

    print(
        "\n[DONE] Saved:",
        output_json,
    )


if __name__ == "__main__":
    main()