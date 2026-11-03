#!/usr/bin/env python3
"""Attack Gaussian or Laplacian precomputed-kernel sklearn SVCs.

The saved sklearn estimator only contains support-row indices for a
precomputed kernel.  This script therefore requires the exact fit-time
``x_train`` companion file, reconstructs the selected kernel in PyTorch, and
checks its predictions against sklearn before running APGD-CE or SPSA.
"""

import argparse
import gc
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from autoattack import AutoAttack
from sklearn.svm import SVC as SklearnSVC
from torchkernels.kernels.radial import gaussian, laplacian


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import dataset
from spsa_utils import add_spsa_arguments, spsa_attack, spsa_options


KERNEL_FUNCTIONS = {
    "gaussian": gaussian,
    "laplacian": laplacian,
}


def canonical_kernel_name(name):
    name = name.lower()
    if name == "rbf":
        return "gaussian"
    if name == "laplace":
        return "laplacian"
    if name not in KERNEL_FUNCTIONS:
        raise ValueError(f"Unsupported kernel: {name}")
    return name


def normalize_rows(x, eps=1e-12):
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)


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
    if not xs:
        raise ValueError("The data loader is empty.")
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def load_test_loader(name, classes):
    multiclass = classes is None
    selected_classes = (0, 1) if multiclass else tuple(classes)

    if name == "svhn":
        return dataset.get_svhn(
            classes=selected_classes,
            multiclass=multiclass,
        )[2]
    if name in {"fmnist", "kmnist", "qmnist", "emnist"}:
        return dataset.load_fmnist(
            classes=selected_classes,
            which_mnist=name,
            multiclass=multiclass,
        )[2]
    if name == "cifar10":
        return dataset.get_cifar(
            classes=selected_classes,
            multiclass=multiclass,
        )[2]
    raise ValueError(f"Unsupported dataset: {name}")


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def unique_match(directory, patterns, description):
    matches = []
    for pattern in patterns:
        matches.extend(directory.glob(pattern))
    matches = sorted(set(path.resolve() for path in matches))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {description} found in {directory}.")
    raise RuntimeError(
        f"Multiple {description} files found in {directory}: "
        f"{[path.name for path in matches]}. Pass the explicit path."
    )


def infer_model_dir(model_root, dataset_name, kernel_name, pair):
    mode = "mc" if pair is None else "pairwise"
    candidates = [
        model_root / f"precomputed_{kernel_name}_{dataset_name}_{mode}",
        model_root / f"svc_{kernel_name}_{dataset_name}_{mode}",
    ]
    if kernel_name == "laplacian":
        candidates.extend(
            [
                model_root / f"precomputed_laplace_{dataset_name}_{mode}",
                model_root / f"svc_laplace_{dataset_name}_{mode}",
            ]
        )
    existing = [path for path in candidates if path.is_dir()]
    if len(existing) == 1:
        return existing[0].resolve()
    if not existing:
        raise FileNotFoundError(
            "Could not infer a model directory. Tried: "
            + ", ".join(str(path) for path in candidates)
        )
    raise RuntimeError(
        f"Multiple possible model directories found: {existing}. Pass --MODEL_DIR."
    )


def run_prefix(dataset_name, kernel_name, pair):
    run_name = "mc" if pair is None else f"{pair[0]}_{pair[1]}"
    return f"{dataset_name}_{run_name}_{kernel_name}_svc"


def find_model_file(model_dir, dataset_name, kernel_name, pair):
    prefix = run_prefix(dataset_name, kernel_name, pair)
    exact = [model_dir / f"{prefix}_best.pkl", model_dir / f"{prefix}_best.pt"]
    for path in exact:
        if path.is_file():
            return path.resolve()
    return unique_match(
        model_dir,
        ["*_svc_best.pkl", "*_svc_best.pt", "*_best.pkl", "*_best.pt"],
        "SVC model",
    )


def find_xtrain_file(model_dir, dataset_name, kernel_name, pair):
    prefix = run_prefix(dataset_name, kernel_name, pair)
    exact = model_dir / f"{prefix}_xtrain.pt"
    if exact.is_file():
        return exact.resolve()
    return unique_match(model_dir, ["*_xtrain.pt"], "fit-time x_train companion")


def find_results_json(model_dir, dataset_name, kernel_name, pair):
    prefix = run_prefix(dataset_name, kernel_name, pair)
    exact = model_dir / f"{prefix}_results.json"
    if exact.is_file():
        return exact.resolve()
    try:
        return unique_match(model_dir, ["*_svc_results.json", "*_results.json"], "results JSON")
    except FileNotFoundError:
        return None


def read_best_gamma(results_json):
    if results_json is None:
        raise FileNotFoundError(
            "No results JSON was found. A precomputed SVC does not store the "
            "kernel-construction gamma; pass --GAMMA explicitly."
        )
    with Path(results_json).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    try:
        return float(result["best"]["gamma"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing numeric best.gamma in {results_json}.") from exc


def read_expected_test_accuracy(results_json):
    if results_json is None:
        return None
    try:
        with Path(results_json).open("r", encoding="utf-8") as handle:
            value = json.load(handle)["best"]["test_acc"]
        if value is None:
            return None
        value = float(value)
        return value * 100.0 if value <= 1.0 else value
    except (OSError, KeyError, TypeError, ValueError):
        return None


class DifferentiablePrecomputedSVC(nn.Module):
    """Differentiable reconstruction of an sklearn precomputed-kernel SVC."""

    def __init__(
        self,
        fitted_svc,
        x_train,
        gamma,
        kernel,
        device,
        kernel_chunk_size=4096,
    ):
        super().__init__()
        if not isinstance(fitted_svc, SklearnSVC):
            raise TypeError(f"Expected sklearn.svm.SVC, got {type(fitted_svc)}.")
        if fitted_svc.kernel != "precomputed":
            raise ValueError(
                f"Expected SVC(kernel='precomputed'), got kernel={fitted_svc.kernel!r}."
            )

        self.kernel_name = canonical_kernel_name(kernel)
        self.kernel_fn = KERNEL_FUNCTIONS[self.kernel_name]
        self.gamma = float(gamma)
        self.kernel_chunk_size = int(kernel_chunk_size)
        if self.kernel_chunk_size <= 0:
            raise ValueError("kernel_chunk_size must be positive.")

        classes = np.asarray(fitted_svc.classes_, dtype=np.int64)
        self.num_classes = int(classes.size)
        self.register_buffer(
            "class_labels",
            torch.as_tensor(classes, device=device, dtype=torch.long),
        )

        if x_train.ndim > 2:
            x_train = x_train.flatten(1)
        x_train = x_train.to(device=device, dtype=torch.float32)
        support_indices = torch.as_tensor(
            np.asarray(fitted_svc.support_, dtype=np.int64),
            device=device,
            dtype=torch.long,
        )
        if support_indices.numel() == 0:
            raise RuntimeError("The fitted SVC has no support indices.")
        maximum_index = int(support_indices.max().item())
        if maximum_index >= x_train.shape[0]:
            raise RuntimeError(
                f"SVC support index {maximum_index} exceeds the saved x_train size "
                f"({x_train.shape[0]}). The companion file is not the exact fit-time data."
            )

        self.register_buffer("support_indices", support_indices)
        self.register_buffer(
            "support_vectors",
            normalize_rows(x_train[support_indices]).detach(),
        )

        n_support = np.asarray(fitted_svc.n_support_, dtype=np.int64)
        starts = np.zeros(self.num_classes, dtype=np.int64)
        if self.num_classes > 1:
            starts[1:] = np.cumsum(n_support[:-1])
        self.starts = [int(value) for value in starts]
        self.ends = [int(value) for value in starts + n_support]

        self.register_buffer(
            "dual_coef",
            torch.as_tensor(
                np.asarray(fitted_svc.dual_coef_, dtype=np.float32),
                device=device,
            ),
        )
        self.register_buffer(
            "intercept",
            torch.as_tensor(
                np.asarray(fitted_svc.intercept_, dtype=np.float32),
                device=device,
            ),
        )
        expected_pairs = self.num_classes * (self.num_classes - 1) // 2
        if self.intercept.numel() != expected_pairs:
            raise RuntimeError(
                f"Expected {expected_pairs} pairwise intercepts, "
                f"got {self.intercept.numel()}."
            )
        self.register_buffer(
            "metric",
            torch.eye(
                self.support_vectors.shape[1],
                device=device,
                dtype=torch.float32,
            ),
        )

    def _kernel_to_support_vectors(self, x):
        x = normalize_rows(x)
        chunks = []
        for start in range(0, self.support_vectors.shape[0], self.kernel_chunk_size):
            support_chunk = self.support_vectors[start : start + self.kernel_chunk_size]
            chunks.append(
                self.kernel_fn(
                    support_chunk,
                    x,
                    self.gamma,
                    self.metric,
                    in_place=False,
                )
            )
        return torch.cat(chunks, dim=0).T.contiguous()

    def _ovo_decision_function(self, kernel_values):
        decisions = []
        pair_index = 0
        for class_i in range(self.num_classes):
            start_i, end_i = self.starts[class_i], self.ends[class_i]
            for class_j in range(class_i + 1, self.num_classes):
                start_j, end_j = self.starts[class_j], self.ends[class_j]
                coef_i = self.dual_coef[class_j - 1, start_i:end_i]
                coef_j = self.dual_coef[class_i, start_j:end_j]
                score = self.intercept[pair_index].expand(kernel_values.shape[0])
                if end_i > start_i:
                    score = score + kernel_values[:, start_i:end_i] @ coef_i
                if end_j > start_j:
                    score = score + kernel_values[:, start_j:end_j] @ coef_j
                decisions.append(score)
                pair_index += 1
        return torch.stack(decisions, dim=1)

    def _ovo_to_ovr_scores(self, decisions):
        batch_size = decisions.shape[0]
        votes = torch.zeros(
            batch_size,
            self.num_classes,
            device=decisions.device,
            dtype=decisions.dtype,
        )
        confidence_sums = torch.zeros_like(votes)
        pair_index = 0
        for class_i in range(self.num_classes):
            for class_j in range(class_i + 1, self.num_classes):
                decision = decisions[:, pair_index]
                confidence_sums[:, class_i] += decision
                confidence_sums[:, class_j] -= decision
                vote_for_j = (decision < 0).to(decisions.dtype)
                votes[:, class_i] += 1.0 - vote_for_j
                votes[:, class_j] += vote_for_j
                pair_index += 1
        transformed = confidence_sums / (3.0 * (confidence_sums.abs() + 1.0))
        return votes + transformed

    def forward(self, x):
        if x.ndim > 2:
            x = x.flatten(1)
        x = x.to(dtype=torch.float32)
        kernel_values = self._kernel_to_support_vectors(x)
        decisions = self._ovo_decision_function(kernel_values)

        # sklearn binary SVC: negative -> class 0, positive -> class 1
        if self.num_classes == 2:
            binary_margin = decisions[:, 0]
            return torch.stack((-binary_margin, binary_margin), dim=1)

        return self._ovo_to_ovr_scores(decisions)


@torch.no_grad()
def accuracy(model, x, y, batch_size):
    correct = 0
    total = 0
    for start in range(0, x.shape[0], batch_size):
        labels = y[start : start + batch_size]
        predicted_indices = model(x[start : start + batch_size]).argmax(dim=1)
        predictions = model.class_labels[predicted_indices]
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def compare_with_sklearn(
    fitted_svc,
    reconstructed_model,
    x_train,
    x_test,
    gamma,
    kernel,
    max_samples,
):
    sample_count = min(int(max_samples), int(x_test.shape[0]))
    if sample_count <= 0:
        return 100.0, 0, 0

    kernel_fn = KERNEL_FUNCTIONS[canonical_kernel_name(kernel)]
    xh_train = normalize_rows(x_train).double()
    xh_test = normalize_rows(x_test[:sample_count]).double()
    metric = torch.eye(
        xh_train.shape[1],
        device=xh_train.device,
        dtype=torch.float64,
    )
    gram = kernel_fn(
        xh_train,
        xh_test,
        float(gamma),
        metric,
        in_place=False,
    ).T.contiguous()
    sklearn_predictions = torch.as_tensor(
        fitted_svc.predict(gram.cpu().numpy()),
        device=x_test.device,
        dtype=torch.long,
    )
    predicted_indices = reconstructed_model(x_test[:sample_count]).argmax(dim=1)
    reconstructed_predictions = reconstructed_model.class_labels[predicted_indices]
    mismatches = int((reconstructed_predictions != sklearn_predictions).sum().item())
    agreement = 100.0 * (1.0 - mismatches / sample_count)
    return agreement, mismatches, sample_count


def parse_epsilons(values: Optional[Sequence[float]], norm):
    if values:
        return [float(value) for value in values]
    if norm == "Linf":
        # return [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 32.0, 40.0, 55.0]
        return [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0]
    return [0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Attack Gaussian or Laplacian precomputed sklearn SVCs."
    )
    parser.add_argument(
        "--DATASET",
        required=True,
        choices=["fmnist", "kmnist", "qmnist", "emnist", "svhn", "cifar10"],
    )
    parser.add_argument(
        "--KERNEL",
        required=True,
        choices=["gaussian", "rbf", "laplacian", "laplace"],
    )
    parser.add_argument("--AA_norm", required=True, choices=["Linf", "L2"])
    parser.add_argument(
        "--CLASSES",
        nargs=2,
        type=int,
        metavar=("CLASS1", "CLASS2"),
        help="Attack a pairwise model. Omit for a multiclass model.",
    )
    parser.add_argument("--MODEL_ROOT", type=Path, default=Path("svc_cv"))
    parser.add_argument("--MODEL_DIR", type=Path)
    parser.add_argument("--MODEL_FILE", type=Path)
    parser.add_argument("--RESULTS_JSON", type=Path)
    parser.add_argument("--XTRAIN_FILE", type=Path)
    parser.add_argument("--OUTPUT_PATH", type=Path)
    parser.add_argument("--DEVICE", default=None)
    parser.add_argument("--SEED", type=int, default=42)
    parser.add_argument("--BATCH_SIZE", type=int, default=256)
    parser.add_argument("--NUM_TEST", type=int, default=0)
    parser.add_argument("--EPSILONS", nargs="*", type=float)
    parser.add_argument("--GAMMA", type=float)
    parser.add_argument("--KERNEL_CHUNK", type=int, default=4096)
    parser.add_argument("--SAVE_ADV", action="store_true")
    parser.add_argument("--SKIP_SKLEARN_CHECK", action="store_true")
    parser.add_argument("--CHECK_SAMPLES", type=int, default=128)
    parser.add_argument("--MIN_AGREEMENT", type=float, default=99.0)
    add_spsa_arguments(parser)
    return parser.parse_args()


def resolve_optional_path(path):
    return None if path is None else path.expanduser().resolve()


def main():
    args = parse_args()
    dataset_name = args.DATASET.lower()
    kernel_name = canonical_kernel_name(args.KERNEL)
    pair = None if args.CLASSES is None else tuple(args.CLASSES)
    if pair is not None and (pair[0] == pair[1] or min(pair) < 0 or max(pair) > 9):
        raise ValueError("--CLASSES must contain two distinct labels from 0 through 9.")

    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.SEED)

    if args.DEVICE is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.DEVICE)
    print("Device:", device)

    if args.MODEL_DIR is not None:
        model_dir = args.MODEL_DIR.expanduser().resolve()
    elif args.MODEL_FILE is not None:
        model_dir = args.MODEL_FILE.expanduser().resolve().parent
    else:
        model_dir = infer_model_dir(
            args.MODEL_ROOT.expanduser().resolve(),
            dataset_name,
            kernel_name,
            pair,
        )

    model_file = resolve_optional_path(args.MODEL_FILE) or find_model_file(
        model_dir, dataset_name, kernel_name, pair
    )
    xtrain_file = resolve_optional_path(args.XTRAIN_FILE) or find_xtrain_file(
        model_dir, dataset_name, kernel_name, pair
    )
    results_json = resolve_optional_path(args.RESULTS_JSON)
    if results_json is None:
        results_json = find_results_json(model_dir, dataset_name, kernel_name, pair)

    gamma = float(args.GAMMA) if args.GAMMA is not None else read_best_gamma(results_json)
    print("Model:", model_file)
    print("Fit-time data:", xtrain_file)
    print("Results:", results_json)
    print(f"Kernel: {kernel_name}; gamma: {gamma:g}")

    fitted_svc = load_pickle(model_file)
    if not isinstance(fitted_svc, SklearnSVC):
        raise TypeError(
            f"Expected a sklearn SVC checkpoint, got {type(fitted_svc)}. "
            "This script only attacks precomputed-kernel sklearn models."
        )
    if fitted_svc.kernel != "precomputed":
        raise ValueError(
            f"Checkpoint kernel is {fitted_svc.kernel!r}, not 'precomputed'."
        )

    xtrain_blob = torch.load(xtrain_file, map_location="cpu")
    if not isinstance(xtrain_blob, dict) or "x_train" not in xtrain_blob:
        raise ValueError(f"{xtrain_file} must contain a dictionary with 'x_train'.")
    x_train = xtrain_blob["x_train"].to(device=device, dtype=torch.float32)
    if x_train.ndim > 2:
        x_train = x_train.flatten(1)

    testloader = load_test_loader(dataset_name, pair)
    x_test, y_test = dataloader_to_tensors(testloader)
    if x_test.ndim > 2:
        x_test = x_test.flatten(1)
    x_test = x_test.to(device=device, dtype=torch.float32)
    y_test = labels_to_indices(y_test).to(device=device)
    if args.NUM_TEST > 0:
        sample_count = min(args.NUM_TEST, x_test.shape[0])
        x_test = x_test[:sample_count]
        y_test = y_test[:sample_count]
    if x_test.min() < -1e-6 or x_test.max() > 1.0 + 1e-6:
        raise ValueError("Attack input clipping assumes test inputs are in [0, 1].")

    attack_model = DifferentiablePrecomputedSVC(
        fitted_svc=fitted_svc,
        x_train=x_train,
        gamma=gamma,
        kernel=kernel_name,
        device=device,
        kernel_chunk_size=args.KERNEL_CHUNK,
    ).to(device).eval()

    label_matches = y_test[:, None].eq(attack_model.class_labels[None, :])
    if not label_matches.any(dim=1).all():
        raise ValueError("Test labels are absent from the saved SVC classes.")
    attack_labels = label_matches.long().argmax(dim=1)

    if not args.SKIP_SKLEARN_CHECK:
        agreement, mismatches, checked = compare_with_sklearn(
            fitted_svc,
            attack_model,
            x_train,
            x_test,
            gamma,
            kernel_name,
            args.CHECK_SAMPLES,
        )
        print(
            f"Reconstruction agreement: {agreement:.4f}% "
            f"({mismatches}/{checked} disagree)"
        )
        if agreement < args.MIN_AGREEMENT:
            raise RuntimeError(
                f"Agreement {agreement:.4f}% is below {args.MIN_AGREEMENT:.4f}%. "
                "Check the kernel, gamma, normalization, and exact x_train row order."
            )

    clean_acc = accuracy(attack_model, x_test, y_test, args.BATCH_SIZE)
    expected_acc = read_expected_test_accuracy(results_json)
    print(f"Clean accuracy: {clean_acc:.4f}%")
    if expected_acc is not None:
        print(f"Training JSON test accuracy: {expected_acc:.4f}%")
        print(f"Absolute difference: {abs(clean_acc - expected_acc):.6f}%")

    mode_name = "mc" if pair is None else f"{pair[0]}_{pair[1]}"
    output_path = resolve_optional_path(args.OUTPUT_PATH)
    if output_path is None:
        output_path = (
            model_dir
            / f"seed_{args.SEED}"
            / f"{args.ATTACK.replace('-', '_')}_{args.AA_norm}_{dataset_name}_{mode_name}_{kernel_name}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adversarial_dir = output_path.parent

    results = []
    for epsilon_input in parse_epsilons(args.EPSILONS, args.AA_norm):
        epsilon = (
            float(epsilon_input) / 255.0
            if args.AA_norm == "Linf"
            else float(epsilon_input)
        )
        print(
            f"\n{args.ATTACK.upper()} | {dataset_name} | {mode_name} | {kernel_name} | "
            f"{args.AA_norm} | epsilon={epsilon_input}"
        )
        if epsilon == 0.0:
            x_adv = x_test.detach().clone()
            robust_acc = clean_acc
        elif args.ATTACK == "spsa":
            x_adv = spsa_attack(attack_model, x_test, attack_labels, epsilon,
                                args.AA_norm, **spsa_options(args))
            robust_acc = accuracy(attack_model, x_adv, y_test, args.BATCH_SIZE)
        else:
            adversary = AutoAttack(
                attack_model,
                norm=args.AA_norm,
                eps=epsilon,
                version="standard",
                seed=args.SEED,
                device=device,
            )
            adversary.attacks_to_run = ["apgd-ce"]
            x_adv = adversary.run_standard_evaluation(
                x_test,
                attack_labels,
                bs=args.BATCH_SIZE,
            )
            robust_acc = accuracy(attack_model, x_adv, y_test, args.BATCH_SIZE)

        adversarial_path = None
        if args.SAVE_ADV:
            epsilon_name = str(epsilon_input).replace(".", "p")
            adversarial_path = adversarial_dir / (
                f"adv_{args.ATTACK.replace('-', '_')}_{args.AA_norm}_{dataset_name}_{mode_name}_"
                f"{kernel_name}_eps_{epsilon_name}.pt"
            )
            torch.save(
                {
                    "x_adv": x_adv.detach().cpu(),
                    "y": y_test.detach().cpu(),
                    "eps_input": float(epsilon_input),
                    "eps": epsilon,
                    "norm": args.AA_norm,
                    "kernel": kernel_name,
                    "gamma": gamma,
                    "attack": args.ATTACK,
                },
                adversarial_path,
            )

        print(f"Robust accuracy: {robust_acc:.4f}%")
        results.append(
            {
                "eps_input": float(epsilon_input),
                "eps_float": epsilon,
                "robust_acc": float(robust_acc),
                "adv_path": (
                    str(adversarial_path) if adversarial_path is not None else None
                ),
            }
        )
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "dataset": dataset_name,
                    "classes": list(pair) if pair is not None else None,
                    "kernel": kernel_name,
                    "gamma": gamma,
                    "model_file": str(model_file),
                    "xtrain_file": str(xtrain_file),
                    "results_json": str(results_json) if results_json else None,
                    "seed": args.SEED,
                    "norm": args.AA_norm,
                    "attack": args.ATTACK,
                    "spsa_options": spsa_options(args) if args.ATTACK == "spsa" else None,
                    "clean_acc": clean_acc,
                    "training_json_test_acc": expected_acc,
                    "num_test": int(x_test.shape[0]),
                    "batch_size": args.BATCH_SIZE,
                    "results": results,
                },
                handle,
                indent=2,
            )

        del x_adv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
