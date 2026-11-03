import os
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoattack import AutoAttack

import dataset

import functools
from torchkernels.kernels.radial import laplacian, gaussian


parser = argparse.ArgumentParser(
    description="Attack trained KLR models"
)


parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=[
        "fmnist",
        "kmnist",
        "qmnist",
        "emnist",
        "svhn",
        "cifar"
    ]
)


parser.add_argument(
    "--kernel",
    type=str,
    required=True,
    choices=[
        "gaussian",
        "laplacian"
    ]
)


parser.add_argument(
    "--multiclass",
    action="store_true",
    help="Attack multiclass model"
)


parser.add_argument(
    "--classes",
    type=int,
    nargs="+",
    default=None,
    help=(
        "Binary class pairs. "
        "Example: --classes 0 1 2 3 4 5"
    )
)


parser.add_argument(
    "--norm",
    type=str,
    choices=["Linf", "L2"],
    required=True
)


parser.add_argument(
    "--batch_size",
    type=int,
    default=256
)


parser.add_argument(
    "--seed",
    type=int,
    default=42
)


parser.add_argument(
    "--num_samples",
    type=int,
    default=None
)


parser.add_argument(
    "--root",
    type=str,
    default=None,
    help=(
        "Optional experiment root. "
        "Default: klr_sweep_{dataset}"
    )
)


args = parser.parse_args()


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Device:",
    device
)


# ============================================================
# Basic configuration
# ============================================================

DATASET = args.dataset

KERNEL = args.kernel

MULTICLASS = args.multiclass


if args.root is None:

    EXPERIMENT_ROOT = (
        f"klr_sweep_{DATASET}"
    )

else:

    EXPERIMENT_ROOT = args.root


# ============================================================
# Determine experiments
# ============================================================

if MULTICLASS:

    # dataset.py expects classes, even though these
    # are ignored when multiclass=True.
    EXPERIMENTS = [
        (0, 1)
    ]


else:

    if args.classes is None:

        EXPERIMENTS = [
            (0, 1)
        ]

    else:

        if len(args.classes) % 2 != 0:

            raise ValueError(
                "--classes must contain an even number "
                "of class labels.\n"
                "Example: --classes 0 1 2 3 4 5"
            )


        EXPERIMENTS = [

            tuple(
                args.classes[i:i + 2]
            )

            for i in range(
                0,
                len(args.classes),
                2
            )
        ]


        for c0, c1 in EXPERIMENTS:

            if c0 == c1:

                raise ValueError(
                    f"Invalid class pair "
                    f"({c0}, {c1})"
                )


if args.norm == "Linf":

    EPSILONS = [
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
        55
        ,70, 80, 90, 100, 110
    ]


else:

    EPSILONS = [
        0,
        0.5,
        1,
        2,
        4,
        6,
        8
    ]

def to_class_indices(y):

    if y.ndim > 1:

        return (
            y.argmax(dim=1)
            .long()
        )

    return (
        y.long()
        .view(-1)
    )
def load_test_data(
    dataset_name,
    classes,
    multiclass,
    batch_size
):

    if dataset_name in [
        "fmnist",
        "kmnist",
        "qmnist",
        "emnist"
    ]:

        _, _, testloader = (
            dataset.load_fmnist(

                classes=classes,

                which_mnist=dataset_name,

                multiclass=multiclass,

                batch_size=batch_size
            )
        )


    elif dataset_name == "svhn":

        _, _, testloader = (
            dataset.get_svhn(

                classes=classes,

                multiclass=multiclass,

                batch_size=batch_size
            )
        )


    elif dataset_name == "cifar":

        _, _, testloader = (
            dataset.get_cifar(

                classes=classes,

                multiclass=multiclass,

                batch_size=batch_size
            )
        )


    else:

        raise ValueError(
            f"Unsupported dataset: "
            f"{dataset_name}"
        )


    X_test, y_test = (
        testloader.dataset.tensors
    )


    y_test = to_class_indices(
        y_test
    )


    return (
        X_test,
        y_test
    )

class AttackModel(nn.Module):

    def __init__(
        self,
        klr_model
    ):

        super().__init__()

        self.klr_model = (
            klr_model
        )


    def forward(
        self,
        x
    ):

        logits = (
            self.klr_model
            ._logits(x)
        )
        if logits.ndim == 1:

            logits = (
                logits.unsqueeze(1)
            )


        if logits.shape[1] == 1:

            logits = torch.cat(
                [
                    -logits,
                    logits
                ],
                dim=1
            )


        return logits

@torch.no_grad()
def accuracy(
    model,
    X,
    y,
    batch_size
):

    correct = 0

    total = 0


    for start in range(
        0,
        len(X),
        batch_size
    ):

        xb = X[
            start:start + batch_size
        ]


        yb = y[
            start:start + batch_size
        ]


        logits = model(
            xb
        )


        preds = (
            logits.argmax(dim=1)
        )


        correct += (
            preds == yb
        ).sum().item()


        total += len(
            yb
        )


    return (
        correct / total
    )

all_attack_results = []
for CLASSES in EXPERIMENTS:

    if MULTICLASS:

        checkpoint_dir = os.path.join(

            EXPERIMENT_ROOT,

            KERNEL
        )


    else:

        checkpoint_dir = os.path.join(

            EXPERIMENT_ROOT,

            KERNEL,

            (
                f"binary_"
                f"{CLASSES[0]}"
                f"vs"
                f"{CLASSES[1]}"
            )
        )


    checkpoint_path = os.path.join(

        checkpoint_dir,

        "best_klr_model.pt"
    )

    if not os.path.exists(
        checkpoint_path
    ):

        print(
            "\n[ERROR] Checkpoint not found:"
        )

        print(
            checkpoint_path
        )

        continue

    print(
        "\n" + "#" * 80
    )


    if MULTICLASS:

        print(
            "ATTACKING MULTICLASS MODEL"
        )

    else:

        print(
            "ATTACKING BINARY MODEL"
        )

        print(
            f"Classes: "
            f"{CLASSES[0]} "
            f"vs "
            f"{CLASSES[1]}"
        )


    print(
        f"Dataset: "
        f"{DATASET}"
    )


    print(
        f"Kernel: "
        f"{KERNEL}"
    )


    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )


    print(
        "#" * 80
    )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device,

        weights_only=False
    )


    model = (
        checkpoint["model"]
    )


    model_config = (
        checkpoint["model_config"]
    )


    dataset_config = (
        checkpoint["dataset_config"]
    )


    training_metrics = (
        checkpoint["metrics"]
    )


    print(
        "\nModel config:"
    )

    print(
        model_config
    )


    print(
        "\nDataset config:"
    )

    print(
        dataset_config
    )

    SAVED_DATASET = (
        dataset_config["dataset"]
    )


    SAVED_MULTICLASS = (
        dataset_config.get(
            "multiclass",
            MULTICLASS
        )
    )


    if SAVED_MULTICLASS:

        SAVED_CLASSES = (
            0,
            1
        )

    else:

        SAVED_CLASSES = tuple(

            dataset_config.get(
                "classes",
                CLASSES
            )
        )


    if SAVED_DATASET != DATASET:

        raise ValueError(

            f"Checkpoint dataset "
            f"{SAVED_DATASET} "

            f"does not match requested "
            f"dataset {DATASET}."
        )


    if (
        model_config["kernel"]
        != KERNEL
    ):

        raise ValueError(

            f"Checkpoint kernel "
            f"{model_config['kernel']} "

            f"does not match requested "
            f"kernel {KERNEL}."
        )


    model.X_train_ = (
        model.X_train_.to(
            device
        )
    )

    model.alpha_ = (
        model.alpha_.to(
            device
        )
    )

    original_kernel = model.kernel
    ls = model_config["length_scale"]

    if KERNEL == "laplacian":
        model.kernel = functools.partial(
            laplacian, length_scale=ls, in_place=False
        )
    else:
        model.kernel = functools.partial(
            gaussian, length_scale=ls, in_place=False
        )

    X_test, y_test = load_test_data(

        dataset_name=SAVED_DATASET,

        classes=SAVED_CLASSES,

        multiclass=SAVED_MULTICLASS,

        batch_size=args.batch_size
    )


    X_test = X_test.to(
        device
    )


    y_test = (
        y_test
        .to(device)
        .long()
    )

    if args.num_samples is not None:

        X_test = X_test[
            :args.num_samples
        ]

        y_test = y_test[
            :args.num_samples
        ]

    print(
        "\nTest data:"
    )


    print(
        "X_test:",
        X_test.shape
    )


    print(
        "y_test:",
        y_test.shape
    )


    print(
        "Classes:",
        torch.unique(
            y_test,
            return_counts=True
        )
    )


    print(
        "Input range:",
        X_test.min().item(),
        X_test.max().item()
    )

    attack_model = AttackModel(
        model
    )


    attack_model.eval()

    print(
        "\nChecking gradients..."
    )


    x_check = (

        X_test[:2]

        .clone()

        .detach()

        .requires_grad_(True)
    )


    logits = attack_model(
        x_check
    )


    loss = F.cross_entropy(

        logits,

        y_test[:2]
    )


    grad = torch.autograd.grad(

        loss,

        x_check

    )[0]


    print(
        "Logits shape:",
        logits.shape
    )


    print(
        "Mean absolute gradient:",
        grad.abs().mean().item()
    )


    print(
        "Gradient check passed."
    )

    clean_acc = accuracy(

        attack_model,

        X_test,

        y_test,

        args.batch_size
    )


    print(

        f"\nClean test accuracy: "
        f"{100 * clean_acc:.2f}%"
    )

    results = {

        "checkpoint":
            os.path.abspath(
                checkpoint_path
            ),

        "dataset":
            SAVED_DATASET,

        "multiclass":
            SAVED_MULTICLASS,

        "classes":
            (
                None
                if SAVED_MULTICLASS
                else list(
                    SAVED_CLASSES
                )
            ),

        "kernel":
            model_config[
                "kernel"
            ],

        "length_scale":
            model_config[
                "length_scale"
            ],

        "learning_rate":
            model_config[
                "learning_rate"
            ],

        "training_batch_size":
            model_config[
                "batch_size"
            ],

        "num_classes":
            model_config[
                "num_classes"
            ],

        "max_steps":
            model_config[
                "max_steps"
            ],

        "min_delta":
            model_config[
                "min_delta"
            ],

        "norm":
            args.norm,

        "attack":
            "apgd-ce",

        "seed":
            args.seed,

        "attack_batch_size":
            args.batch_size,

        "num_test_samples":
            len(
                X_test
            ),

        "clean_test_accuracy":
            clean_acc,

        "training_metrics":
            training_metrics,

        "results":
            [],

        "errors":
            []
    }

    for eps_pixel in EPSILONS:

        print(
            "\n" + "=" * 70
        )


        print(

            f"Norm = "
            f"{args.norm}, "

            f"budget = "
            f"{eps_pixel}"
        )

        if args.norm == "Linf":

            epsilon = (
                eps_pixel / 255.0
            )

        else:

            epsilon = float(
                eps_pixel
            )


        print(
            f"Actual epsilon = "
            f"{epsilon}"
        )

        if epsilon == 0:

            results[
                "results"
            ].append({

                "budget_pixels":
                    float(
                        eps_pixel
                    ),

                "epsilon":
                    float(
                        epsilon
                    ),

                "clean_test_accuracy":
                    float(
                        clean_acc
                    ),

                "robust_test_accuracy":
                    float(
                        clean_acc
                    ),

                "max_linf":
                    0.0,

                "max_l2":
                    0.0,

                "samples":
                    len(
                        X_test
                    ),

                "attack":
                    "clean",

                "attack_time_seconds":
                    0.0
            })

            continue
        try:

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

                torch.cuda.synchronize()


            start = (
                time.perf_counter()
            )


            adversary = AutoAttack(

                attack_model,

                norm=args.norm,

                eps=epsilon,

                version="standard",

                seed=args.seed,

                device=device
            )


            adversary.attacks_to_run = [
                "apgd-ce"
            ]


            X_adv = (

                adversary

                .run_standard_evaluation(

                    X_test,

                    y_test,
                    bs = X_test.shape[0]

                    # bs=args.batch_size
                )
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

                args.batch_size
            )

            delta = (
                X_adv
                - X_test
            )


            delta_flat = delta.reshape(

                len(
                    delta
                ),

                -1
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
                    dim=1
                )

                .max()

                .item()
            )


            print(

                f"Clean accuracy: "
                f"{100 * clean_acc:.2f}%"
            )


            print(

                f"Robust accuracy: "
                f"{100 * robust_acc:.2f}%"
            )


            print(

                f"Max Linf: "
                f"{max_linf:.8f}"
            )


            print(

                f"Max L2: "
                f"{max_l2:.8f}"
            )


            print(

                f"Attack time: "
                f"{attack_time:.2f} sec"
            )
            results[
                "results"
            ].append({

                "budget_pixels":
                    float(
                        eps_pixel
                    ),

                "epsilon":
                    float(
                        epsilon
                    ),

                "clean_test_accuracy":
                    float(
                        clean_acc
                    ),

                "robust_test_accuracy":
                    float(
                        robust_acc
                    ),

                "max_linf":
                    float(
                        max_linf
                    ),

                "max_l2":
                    float(
                        max_l2
                    ),

                "samples":
                    len(
                        X_test
                    ),

                "attack":
                    "apgd-ce",

                "attack_time_seconds":
                    float(
                        attack_time
                    )
            })


        except Exception as e:

            print(

                "[ERROR]",

                type(e).__name__,

                str(e)
            )


            results[
                "errors"
            ].append({

                "budget_pixels":
                    float(
                        eps_pixel
                    ),

                "epsilon":
                    float(
                        epsilon
                    ),

                "error":
                    (
                        f"{type(e).__name__}: "
                        f"{e}"
                    )
            })

    output_path = os.path.join(

        checkpoint_dir,

        (
            f"attack_"
            f"apgd_ce_"
            f"{args.norm}"
            f".json"
        )
    )


    with open(

        output_path,

        "w"

    ) as f:

        json.dump(

            results,

            f,

            indent=4
        )

    all_attack_results.append(results)
    print(
        "\n" + "=" * 70
    )


    print(
        "Results saved to:",
        output_path
    )


    print(
        "=" * 70
    )

if MULTICLASS:

    combined_output_path = os.path.join(
        EXPERIMENT_ROOT,
        KERNEL,
        f"all_multiclass_attack_results_{args.norm}.json"
    )

else:

    combined_output_path = os.path.join(
        EXPERIMENT_ROOT,
        KERNEL,
        f"all_binary_attack_results_{args.norm}.json"
    )


combined_results = {
    "dataset": DATASET,
    "kernel": KERNEL,
    "norm": args.norm,
    "attack": "apgd-ce",

    "results": all_attack_results,

    "errors": []
}


with open(
    combined_output_path,
    "w"
) as f:

    json.dump(
        combined_results,
        f,
        indent=4
    )


print(
    "\nCombined attack results saved to:",
    combined_output_path
)

print(
    "\nALL ATTACKS FINISHED"
)