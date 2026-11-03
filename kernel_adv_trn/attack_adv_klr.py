import os
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoattack import AutoAttack

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset
import robust_kernels as rk
from spsa_utils import add_spsa_arguments, spsa_attack, spsa_options

def parse_args():

    parser = argparse.ArgumentParser(
        description="Attack adversarially-trained RKHS/KLR checkpoints"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to one checkpoint, e.g. "
            "kernel_adv_trn/runs/kmnist_rho20_klr_gaussian/"
            "kmnist_0_vs_1_best.pt"
        ),
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Directory containing multiple checkpoints. "
            "Used together with --CLASS_PAIRS."
        ),
    )

    parser.add_argument(
        "--DATASET",
        type=str,
        default=None,
        choices=[
            "fmnist",
            "kmnist",
            "qmnist",
            "emnist",
        ],
        help=(
            "Optional. Normally read from checkpoint config."
        ),
    )

    parser.add_argument(
        "--CLASS_PAIRS",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Pairs sequentially. Example: "
            "--CLASS_PAIRS 0 1 0 2 0 3"
        ),
    )

    parser.add_argument(
        "--NORM",
        type=str,
        default="Linf",
        choices=[
            "Linf",
            "L2",
        ],
    )

    parser.add_argument(
        "--EPSILONS",
        type=float,
        nargs="+",
        default=[
            0,
            0.1,
            1,
            2,
            4,
            6,
            8,
            12,
            16,
            20,
            24,
            32,
            40,
            55,
            70,
            80,
            90,
            100,
            110,
        ],
        help=(
            "For Linf these are pixel-scale budgets and "
            "will be divided by 255."
        ),
    )

    parser.add_argument(
        "--BATCH_SIZE",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--NUM_SAMPLES",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--SEED",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--DEVICE",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--OUTPUT",
        type=str,
        default=None,
        help=(
            "Optional output JSON path for single checkpoint. "
            "For multiple checkpoints, results are also saved "
            "inside checkpoint directory."
        ),
    )

    add_spsa_arguments(parser)
    return parser.parse_args()


class KernelValueFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        centers,
        alpha,
        beta,
        options,
    ):

        values, gradients = rk.kernel_eval_and_grad(
            x,
            centers,
            alpha,
            beta,
            **options,
        )

        ctx.save_for_backward(
            gradients
        )

        return values


    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):

        (gradients,) = ctx.saved_tensors

        grad_x = (
            grad_output
            .reshape(-1, 1)
            * gradients
        )

        return (
            grad_x,
            None,
            None,
            None,
            None,
        )


class AttackModel(nn.Module):

    def __init__(
        self,
        centers,
        alpha,
        beta,
        bias,
        options,
    ):
        super().__init__()

        self.register_buffer(
            "centers",
            centers,
        )

        self.register_buffer(
            "alpha",
            alpha,
        )

        self.register_buffer(
            "beta",
            beta,
        )

        self.bias = float(bias)

        self.options = dict(options)

    def forward(self, x):

        x = x.reshape(
            x.shape[0],
            -1,
        )
        values = KernelValueFunction.apply(
            x,
            self.centers,
            self.alpha,
            self.beta,
            self.options,
        )

        values = values.reshape(-1)

        if self.bias != 0.0:
            values = values + self.bias

        logits = torch.stack(
            [
                torch.zeros_like(values),
                values,
            ],
            dim=1,
        )

        return logits

def prepare_labels(
    y,
    positive_class,
):
    if (
        y.ndim == 2
        and y.shape[1] == 2
    ):

        y = y.argmax(
            dim=1
        )

    else:

        y = y.reshape(-1)

        unique = torch.unique(
            y
        )
        already_binary = torch.all(
            torch.isin(
                unique,
                torch.tensor(
                    [0, 1],
                    device=y.device,
                    dtype=y.dtype,
                ),
            )
        ).item()

        if not already_binary:

            y = (
                y == positive_class
            ).long()

    return (
        y.long()
        .reshape(-1)
    )

def load_test_data(
    dataset_name,
    classes,
    positive_class,
    batch_size,
):

    if dataset_name in [
        "fmnist",
        "kmnist",
        "qmnist",
        "emnist",
    ]:

        _, _, test_loader = (
            dataset.load_fmnist(
                classes=classes,
                which_mnist=dataset_name,
                multiclass=False,
                batch_size=batch_size,
            )
        )

    else:

        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    X_test, y_test = (
        test_loader.dataset.tensors
    )

    y_test = prepare_labels(
        y_test,
        positive_class=positive_class,
    )

    return (
        X_test,
        y_test,
    )

@torch.no_grad()
def accuracy(
    model,
    X,
    y,
    batch_size,
):

    correct = 0
    total = 0

    for start in range(
        0,
        len(X),
        batch_size,
    ):

        xb = X[
            start:
            start + batch_size
        ]

        yb = y[
            start:
            start + batch_size
        ]

        logits = model(
            xb
        )

        pred = logits.argmax(
            dim=1
        )

        correct += (
            pred == yb
        ).sum().item()

        total += yb.numel()

    return (
        correct / total
    )

def check_gradients(
    model,
    X,
    y,
):

    print(
        "\nChecking input gradients..."
    )

    n = min(
        2,
        len(X),
    )

    x_check = (
        X[:n]
        .clone()
        .detach()
        .requires_grad_(True)
    )

    y_check = y[:n]

    logits = model(
        x_check
    )

    loss = F.cross_entropy(
        logits,
        y_check,
    )

    grad = torch.autograd.grad(
        loss,
        x_check,
        retain_graph=False,
        create_graph=False,
    )[0]

    mean_grad = (
        grad.abs()
        .mean()
        .item()
    )

    max_grad = (
        grad.abs()
        .max()
        .item()
    )

    print(
        "Logits shape:",
        logits.shape,
    )

    print(
        "Mean absolute gradient:",
        mean_grad,
    )

    print(
        "Max absolute gradient:",
        max_grad,
    )

    if not torch.isfinite(
        grad
    ).all():

        raise RuntimeError(
            "Gradient contains NaN or Inf."
        )

    if max_grad == 0:

        raise RuntimeError(
            "Input gradient is exactly zero. "
            "APGD cannot reliably attack this model."
        )

    print(
        "Gradient check passed."
    )

def attack_checkpoint(
    checkpoint_path,
    args,
):

    device = torch.device(
        args.DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n" + "#" * 80
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Device:",
        device,
    )

    print(
        "#" * 80
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = [
        "centers",
        "alpha",
        "beta",
        "options",
        "classes",
        "positive_class",
        "config",
    ]

    for key in required_keys:

        if key not in checkpoint:

            raise KeyError(
                f"Missing checkpoint key: {key}"
            )

    centers = (
        checkpoint["centers"]
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    alpha = (
        checkpoint["alpha"]
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    beta = (
        checkpoint["beta"]
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    bias = checkpoint.get(
        "bias",
        0.0,
    )

    options = dict(
        checkpoint["options"]
    )

    classes = tuple(
        checkpoint["classes"]
    )

    positive_class = int(
        checkpoint["positive_class"]
    )

    config = checkpoint[
        "config"
    ]

    metrics = checkpoint.get(
        "metrics",
        {},
    )

    dataset_name = config[
        "dataset"
    ]

    rho = config.get(
        "rho",
        None,
    )

    best_lam = checkpoint.get(
        "best_lam",
        config.get(
            "lam",
            None,
        ),
    )

    print(
        "\nCheckpoint information:"
    )

    print(
        "Dataset       :",
        dataset_name,
    )

    print(
        "Classes       :",
        classes,
    )

    print(
        "Positive class:",
        positive_class,
    )

    print(
        "Kernel        :",
        options.get(
            "kernel"
        ),
    )

    print(
        "Length scale  :",
        options.get(
            "length_scale"
        ),
    )

    print(
        "rho           :",
        rho,
    )

    print(
        "Best lambda   :",
        best_lam,
    )

    if (
        args.DATASET is not None
        and args.DATASET != dataset_name
    ):

        raise ValueError(
            f"--DATASET={args.DATASET} "
            f"but checkpoint dataset={dataset_name}"
        )

    X_test, y_test = load_test_data(
        dataset_name=dataset_name,
        classes=classes,
        positive_class=positive_class,
        batch_size=args.BATCH_SIZE,
    )

    X_test = X_test.to(
        device=device,
        dtype=torch.float32,
    )

    y_test = y_test.to(
        device=device,
        dtype=torch.long,
    )

    if args.NUM_SAMPLES is not None:

        X_test = X_test[
            :args.NUM_SAMPLES
        ]

        y_test = y_test[
            :args.NUM_SAMPLES
        ]

    print(
        "\nTest data:"
    )

    print(
        "X_test shape:",
        X_test.shape,
    )

    print(
        "y_test shape:",
        y_test.shape,
    )

    print(
        "Input range:",
        X_test.min().item(),
        X_test.max().item(),
    )

    print(
        "Labels:",
        torch.unique(
            y_test,
            return_counts=True,
        ),
    )

    attack_model = AttackModel(
        centers=centers,
        alpha=alpha,
        beta=beta,
        bias=bias,
        options=options,
    ).to(
        device
    )

    attack_model.eval()

    with torch.no_grad():

        n_check = min(
            args.BATCH_SIZE,
            len(X_test),
        )

        x_check = X_test[
            :n_check
        ]

        flat_check = x_check.reshape(
            x_check.shape[0],
            -1,
        )

        values, _ = (
            rk.kernel_eval_and_grad(
                flat_check,
                centers,
                alpha,
                beta,
                **options,
            )
        )

        original_pred = (
            values.reshape(-1)
            + float(bias)
            > 0
        ).long()

        wrapped_pred = (
            attack_model(
                x_check
            )
            .argmax(
                dim=1
            )
        )

        same = torch.equal(
            original_pred,
            wrapped_pred,
        )

        print(
            "\nWrapper matches original decision rule:",
            same,
        )

        if not same:

            raise RuntimeError(
                "AttackModel does not match "
                "the original prediction function."
            )

    if args.ATTACK == "apgd-ce":
        check_gradients(attack_model, X_test, y_test)

    clean_acc = accuracy(
        attack_model,
        X_test,
        y_test,
        args.BATCH_SIZE,
    )

    print(
        f"\nClean test accuracy: "
        f"{100 * clean_acc:.2f}%"
    )

    if (
        "test_accuracy"
        in metrics
    ):

        print(
            "Saved checkpoint test accuracy:",
            f"{100 * metrics['test_accuracy']:.2f}%"
        )

    result = {

        "checkpoint":
            os.path.abspath(
                checkpoint_path
            ),

        "dataset":
            dataset_name,

        "classes":
            list(
                classes
            ),

        "positive_class":
            positive_class,

        "kernel":
            options.get(
                "kernel"
            ),

        "length_scale":
            options.get(
                "length_scale"
            ),

        "rho":
            rho,

        "best_lam":
            best_lam,

        "norm":
            args.NORM,

        "attack":
            args.ATTACK,

        "spsa_options": spsa_options(args) if args.ATTACK == "spsa" else None,

        "seed":
            args.SEED,

        "num_test_samples":
            len(
                X_test
            ),

        "clean_test_accuracy":
            clean_acc,

        "saved_metrics":
            metrics,

        "pgd":
            [],

        "errors":
            [],
    }

    for eps_pixel in args.EPSILONS:

        print(
            "\n" + "=" * 70
        )

        print(
            f"Budget: {eps_pixel}"
        )

        if args.NORM == "Linf":

            epsilon = (
                float(
                    eps_pixel
                )
                / 255.0
            )

        else:

            epsilon = float(
                eps_pixel
            )

        print(
            "Actual epsilon:",
            epsilon,
        )

        if epsilon == 0:

            result[
                "pgd"
            ].append(
                {
                    "eps_float":
                        epsilon,

                    "budget_pixels":
                        float(
                            eps_pixel
                        ),

                    "norm":
                        args.NORM,

                    "attack":
                        "clean",

                    "pgd_acc":
                        100
                        * clean_acc,

                    "max_linf":
                        0.0,

                    "max_l2":
                        0.0,

                    "attack_time_seconds":
                        0.0,
                }
            )

            continue

        try:

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

                torch.cuda.synchronize()

            start = time.perf_counter()

            if args.ATTACK == "spsa":
                X_adv = spsa_attack(attack_model, X_test, y_test, epsilon,
                                    args.NORM, **spsa_options(args))
            else:
                adversary = AutoAttack(
                    attack_model, norm=args.NORM, eps=epsilon,
                    version="custom", seed=args.SEED, device=device,
                )
                adversary.attacks_to_run = ["apgd-ce"]
                X_adv = adversary.run_standard_evaluation(
                    X_test, y_test, bs=args.BATCH_SIZE,
                )

            if torch.cuda.is_available():

                torch.cuda.synchronize()

            attack_time = (
                time.perf_counter()
                - start
            )

            robust_acc = accuracy(
                attack_model,
                X_adv,
                y_test,
                args.BATCH_SIZE,
            )

            delta = (
                X_adv
                - X_test
            )

            delta_flat = delta.reshape(
                delta.shape[0],
                -1,
            )

            max_linf = (
                delta_flat
                .abs()
                .max(
                    dim=1
                )
                .values
                .max()
                .item()
            )

            max_l2 = (
                delta_flat
                .norm(
                    p=2,
                    dim=1,
                )
                .max()
                .item()
            )

            print(
                f"Clean accuracy : "
                f"{100 * clean_acc:.2f}%"
            )

            print(
                f"Robust accuracy: "
                f"{100 * robust_acc:.2f}%"
            )

            print(
                f"Max Linf       : "
                f"{max_linf:.8f}"
            )

            print(
                f"Max L2         : "
                f"{max_l2:.8f}"
            )

            print(
                f"Attack time    : "
                f"{attack_time:.2f} sec"
            )

            result[
                "pgd"
            ].append(
                {
                    "eps_float":
                        float(
                            epsilon
                        ),

                    "budget_pixels":
                        float(
                            eps_pixel
                        ),

                    "norm":
                        args.NORM,

                    "attack":
                        args.ATTACK,


                    "pgd_acc":
                        100
                        * float(
                            robust_acc
                        ),

                    "clean_acc":
                        100
                        * float(
                            clean_acc
                        ),

                    "max_linf":
                        float(
                            max_linf
                        ),

                    "max_l2":
                        float(
                            max_l2
                        ),

                    "attack_time_seconds":
                        float(
                            attack_time
                        ),
                }
            )

        except Exception as e:

            print(
                "\n[ERROR]",
                type(e).__name__,
                str(e),
            )

            result[
                "errors"
            ].append(
                {
                    "eps_float":
                        float(
                            epsilon
                        ),

                    "budget_pixels":
                        float(
                            eps_pixel
                        ),

                    "error":
                        (
                            f"{type(e).__name__}: "
                            f"{e}"
                        ),
                }
            )

    return result

def main():

    args = parse_args()

    if (
        args.checkpoint is None
        and args.checkpoint_dir is None
    ):

        raise ValueError(
            "Provide either --checkpoint or --checkpoint_dir"
        )

    if (
        args.checkpoint is not None
        and args.checkpoint_dir is not None
    ):

        raise ValueError(
            "Use only one of --checkpoint or --checkpoint_dir"
        )

    if args.checkpoint is not None:

        result = attack_checkpoint(
            args.checkpoint,
            args,
        )

        if args.OUTPUT is None:

            checkpoint_path = Path(
                args.checkpoint
            )

            output_path = (
                checkpoint_path.parent
                /
                (
                    checkpoint_path.stem
                    + f"_{'apgd' if args.ATTACK == 'apgd-ce' else 'spsa'}_{args.NORM}.json"
                )
            )

        else:

            output_path = Path(
                args.OUTPUT
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(
            output_path,
            "w",
        ) as f:

            json.dump(
                result,
                f,
                indent=4,
            )

        print(
            "\nResults saved to:"
        )

        print(
            output_path
        )

        return

    if args.CLASS_PAIRS is None:

        raise ValueError(
            "--CLASS_PAIRS is required "
            "when using --checkpoint_dir"
        )

    if (
        len(
            args.CLASS_PAIRS
        )
        % 2
        != 0
    ):

        raise ValueError(
            "--CLASS_PAIRS must contain "
            "an even number of class IDs."
        )

    class_pairs = list(
        zip(
            args.CLASS_PAIRS[::2],
            args.CLASS_PAIRS[1::2],
        )
    )

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    all_results = []

    for cl1, cl2 in class_pairs:

        if args.DATASET is None:

            raise ValueError(
                "--DATASET is required when "
                "using --checkpoint_dir"
            )

        checkpoint_path = (
            checkpoint_dir
            /
            (
                f"{args.DATASET}_"
                f"{cl1}_vs_{cl2}_"
                f"best.pt"
            )
        )

        if not checkpoint_path.exists():

            print(
                "\n[WARNING] Missing checkpoint:"
            )

            print(
                checkpoint_path
            )

            continue

        try:

            result = attack_checkpoint(
                str(
                    checkpoint_path
                ),
                args,
            )

            all_results.append(
                result
            )

        except Exception as e:

            print(
                "\n[ERROR] Failed:",
                checkpoint_path,
            )

            print(
                type(e).__name__,
                str(e),
            )

    combined = {

        "dataset":
            args.DATASET,

        "norm":
            args.NORM,

        "attack":
            args.ATTACK,

        "results":
            all_results,
    }

    combined_output = (
        checkpoint_dir
        /
        (
            f"all_binary_"
            f"{'apgd' if args.ATTACK == 'apgd-ce' else 'spsa'}_{args.NORM}.json"
        )
    )

    with open(
        combined_output,
        "w",
    ) as f:

        json.dump(
            combined,
            f,
            indent=4,
        )

    print(
        "\nCombined results saved to:"
    )

    print(
        combined_output
    )

if __name__ == "__main__":
    main()
