"""
SPSA black-box attack for multiclass RKHS/KLR checkpoints.

Purpose:
    Compare black-box SPSA robustness against APGD-CE robustness
    to investigate possible gradient masking.

For Linf:
    budgets are supplied in pixel units and divided by 255.

Example:
    python attack_multiclass_spsa.py \
        --checkpoint runs/model.pt \
        --norm Linf \
        --budgets 8 12 16 20 \
        --sample_size 16 \
        --iters 100 \
        --batch_size 32 \
        --device cuda:0
"""

import argparse
import json
import pathlib
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import dataset
import robust_kernels as rk
LINF_DEFAULT = [
    0,
    0.1,
    1.0,
    2.0,
    4.0,
    6.0,
    8.0,
    12.0,
    16.0,
    20.0,
    24.0,
    32.0,
    40.0,
    55.0,
]

L2_DEFAULT = [
    0,
    0.5,
    1.0,
    2.0,
    4.0,
    6.0,
    8.0,
]

def eval_multiclass(
    x,
    centers,
    state,
    options,
):
    """
    Only return logits.

    SPSA does NOT require analytical gradients.
    """

    out = rk.multioutput_eval_and_grad(
        x,
        centers,
        state["alpha"],
        state["factors"],
        gradients=False,
        **options,
    )

    if isinstance(
        out,
        (tuple, list),
    ):
        values = out[0]
    else:
        values = out

    return values

def make_predict(
    model,
    device,
):

    centers = (
        model["centers"]
        .to(device)
    )

    state = model["state"]

    state_device = {}

    for key, value in state.items():

        if torch.is_tensor(value):

            state_device[key] = (
                value.to(device)
            )

        else:

            state_device[key] = value

    options = model[
        "options"
    ]

    def predict(
        images,
    ):

        x = images.reshape(
            images.shape[0],
            -1,
        )

        return eval_multiclass(
            x,
            centers,
            state_device,
            options,
        )

    return predict

def load_test(
    config,
    class_ids,
    device,
):

    classes = (
        tuple(
            config["classes"]
        )
        if config.get(
            "classes"
        )
        else None
    )

    _, _, test_loader = (
        dataset.load_fmnist(
            classes=classes,
            which_mnist=config[
                "dataset"
            ],
            multiclass=True,
        )
    )

    xs = []
    ys = []

    for x, y in test_loader:

        xs.append(
            x.reshape(
                x.shape[0],
                -1,
            )
        )

        if y.ndim == 2:

            ys.append(
                y.argmax(1)
            )

        else:

            ys.append(y)

    x = (
        torch.cat(xs)
        .to(
            device,
            torch.float32,
        )
    )

    y = (
        torch.cat(ys)
        .to(device)
        .long()
        .reshape(-1)
    )

    y = torch.searchsorted(
        class_ids.to(device),
        y,
    )

    return x, y

@torch.no_grad()
def accuracy(
    predict,
    images,
    labels,
    batch_size,
):

    correct = 0

    for start in range(
        0,
        len(images),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(images),
        )

        logits = predict(
            images[
                start:end
            ]
        )

        predictions = (
            logits.argmax(1)
        )

        correct += (
            predictions
            == labels[
                start:end
            ]
        ).sum().item()

    return (
        100.0
        * correct
        / len(images)
    )

def spsa_fast(
    loss_fn,
    x,
    y,
    epsilon=0.03,
    eta=0.01,
    lr=0.01,
    iterations=40,
    sample_size=16,
    norm="Linf",
    seed=None,
):

    x = x.detach()

    x_adv = (
        x.clone()
        .detach()
    )

    B, D = x_adv.shape
    eta = float(
        np.clip(
            epsilon / 2,
            1e-3,
            0.01,
        )
    )

    generator = None

    if seed is not None:

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
            f"    SPSA iteration "
            f"{iteration + 1}/{iterations}",
            flush=True,
        )

        v = (
            torch.empty(
                sample_size,
                B,
                D,
                device=x.device,
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
            0,
            1,
        )

        x_minus = (
            x_adv.unsqueeze(0)
            - eta * v
        ).clamp(
            0,
            1,
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
            2
            * sample_size
        )

        with torch.no_grad():

            losses = loss_fn(
                x_pm,
                y_rep,
            )

        losses = losses.view(
            2
            * sample_size,
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
                / (2 * eta)
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
                f"Unsupported norm: "
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

            d_norm = (
                delta.norm(
                    p=2,
                    dim=1,
                    keepdim=True,
                )
                .clamp_min(
                    1e-12
                )
            )

            delta = (
                delta
                * (
                    epsilon
                    / d_norm
                ).clamp(
                    max=1.0
                )
            )

        x_adv = (
            x
            + delta
        ).clamp(
            0,
            1,
        ).detach()

    return x_adv
def attack_spsa(
    predict,
    images,
    labels,
    epsilon,
    args,
    seed,
    lr,
):

    if epsilon == 0:

        return (
            images.clone()
        )

    adversarial_batches = []

    total_batches = (
        len(images)
        + args.batch_size
        - 1
    ) // args.batch_size

    for batch_idx, start in enumerate(
        range(
            0,
            len(images),
            args.batch_size,
        )
    ):

        end = min(
            start
            + args.batch_size,
            len(images),
        )

        print(
            f"\nSPSA batch "
            f"{batch_idx + 1}/{total_batches} "
            f"samples {start}:{end}",
            flush=True,
        )

        xb = (
            images[
                start:end
            ]
            .clone()
            .detach()
        )

        yb = labels[
            start:end
        ]

        def loss_fn(
            x_query,
            y_query,
        ):

            logits = predict(
                x_query
            )

            return F.cross_entropy(
                logits,
                y_query,
                reduction="none",
            )

        x_adv = spsa_fast(
            loss_fn=loss_fn,
            x=xb,
            y=yb,
            epsilon=epsilon,
            eta=args.eta,
            lr=lr,
            iterations=args.iters,
            sample_size=args.sample_size,
            norm=args.norm,
            seed=seed + batch_idx,
        )

        adversarial_batches.append(
            x_adv.detach()
        )

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    return torch.cat(
        adversarial_batches,
        dim=0,
    )

def check_adversarial(
    adversarial,
    images,
    eps,
    norm,
):

    diff = (
        adversarial
        - images
    ).reshape(
        len(images),
        -1,
    )

    if norm == "Linf":

        size = (
            diff.abs()
            .max()
            .item()
        )

        ok = (
            size
            <= eps
            + 1e-6
        )

    else:

        size = (
            diff.norm(
                p=2,
                dim=1,
            )
            .max()
            .item()
        )

        ok = (
            size
            <= eps
            + 1e-4
        )

    if not ok:

        raise RuntimeError(
            f"attack exceeded "
            f"{norm} radius: "
            f"{size} > {eps}"
        )

    if (
        adversarial.min()
        < -1e-6
        or adversarial.max()
        > 1 + 1e-6
    ):

        raise RuntimeError(
            "attack left valid "
            "pixel range [0,1]"
        )

    return size

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional exact output JSON. "
            "Default: checkpoint folder/"
            "SPSA_result_<norm>.json"
        ),
    )

    parser.add_argument(
        "--norm",
        default="Linf",
        choices=[
            "Linf",
            "L2",
        ],
    )

    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Linf: values in units of 1/255. "
            "L2: raw epsilon."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of SPSA iterations",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=16,
        help=(
            "Rademacher directions "
            "per SPSA iteration"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help=(
            "Number of test examples "
            "attacked simultaneously"
        ),
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.01,
        help=(
            "Nominal SPSA finite-difference "
            "radius. spsa_fast clips it using "
            "epsilon/2 as in your reference."
        ),
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "SPSA attack step size. "
            "Default: epsilon/10."
        ),
    )

    return parser.parse_args()

def main():

    args = parse_args()

    checkpoint = pathlib.Path(
        args.checkpoint
    )

    model = torch.load(
        checkpoint,
        map_location=args.device,
        weights_only=False,
    )

    if (
        model.get(
            "model_type"
        )
        != "multiclass"
    ):

        raise ValueError(
            "This script expects a "
            "multiclass checkpoint."
        )

    config = model[
        "config"
    ]

    seed = (
        config.get(
            "seed",
            0,
        )
        if args.seed is None
        else args.seed
    )

    x, y = load_test(
        config,
        model["class_ids"],
        args.device,
    )
    images = x.reshape(
        len(x),
        -1,
    )

    predict = make_predict(
        model,
        args.device,
    )

    print(
        f"\nDataset: "
        f"{config['dataset']}"
    )

    print(
        "Classes:",
        model[
            "class_ids"
        ].tolist(),
    )

    print(
        "Test samples:",
        len(images),
    )

    print(
        "Input shape:",
        images.shape,
    )

    print(
        "Input range:",
        images.min().item(),
        images.max().item(),
    )

    clean = accuracy(
        predict,
        images,
        y,
        batch_size=1000,
    )

    saved = (
        model.get(
            "metrics",
            {}
        )
        .get(
            "test_accuracy"
        )
    )

    print(
        f"\nClean test accuracy: "
        f"{clean:.2f}%"
    )

    if saved is not None:

        print(
            f"Saved test accuracy: "
            f"{100 * saved:.2f}%"
        )

    budgets = (
        args.budgets
        or (
            LINF_DEFAULT
            if args.norm == "Linf"
            else L2_DEFAULT
        )
    )

    records = []
    errors = []

    for budget in budgets:

        eps = (
            budget / 255.0
            if args.norm
            == "Linf"
            else float(
                budget
            )
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"SPSA {args.norm} "
            f"budget={budget}"
        )

        print(
            f"epsilon={eps}"
        )

        if eps == 0:

            robust = clean
            max_perturbation = 0.0

        else:
            current_lr = (
                args.lr
                if args.lr
                is not None
                else eps / 10.0
            )

            try:

                adversarial = attack_spsa(
                    predict=predict,
                    images=images,
                    labels=y,
                    epsilon=eps,
                    args=args,
                    seed=seed,
                    lr = current_lr, 
                )

                max_perturbation = (
                    check_adversarial(
                        adversarial,
                        images,
                        eps,
                        args.norm,
                    )
                )

                robust = accuracy(
                    predict,
                    adversarial,
                    y,
                    batch_size=1000,
                )

            except Exception as e:

                traceback.print_exc()

                errors.append(
                    {
                        "budget":
                            budget,

                        "error":
                            (
                                f"{type(e).__name__}: "
                                f"{e}"
                            ),
                    }
                )

                continue

        current_lr = (
            0.0
            if eps == 0
            else (
                args.lr
                if args.lr
                is not None
                else eps / 10.0
            )
        )

        print(
            f"\n[{args.norm} "
            f"eps={budget}] "
            f"SPSA robust accuracy: "
            f"{robust:.2f}%"
        )

        records.append(
            {
                "eps_float":
                    float(
                        eps
                    ),

                "eps_budget":
                    float(
                        budget
                    ),

                "norm":
                    args.norm,

                "attack":
                    "spsa",

                "pgd_acc":
                    float(
                        robust
                    ),

                "clean_acc":
                    float(
                        clean
                    ),

                "max_perturbation":
                    float(
                        max_perturbation
                    ),

                "iterations":
                    args.iters,

                "sample_size":
                    args.sample_size,

                "batch_size":
                    args.batch_size,

                "eta":
                    float(
                        np.clip(
                            eps / 2,
                            1e-3,
                            0.01,
                        )
                    )
                    if eps > 0
                    else 0.0,

                "lr":
                    float(
                        current_lr
                    ),
            }
        )

        out_obj = {

            "results": [
                {
                    "path":
                        str(
                            checkpoint.resolve()
                        ),

                    "kernel":
                        config[
                            "kernel"
                        ],

                    "classes":
                        model[
                            "class_ids"
                        ].tolist(),

                    "bandwidth":
                        config[
                            "length_scale"
                        ],

                    "attack":
                        "spsa",

                    "pgd":
                        records,
                }
            ],

            "errors":
                errors,
        }

        if args.output:

            out_file = pathlib.Path(
                args.output
            )

        else:

            out_file = (
                checkpoint.resolve()
                .parent
                / (
                    f"SPSA_result_"
                    f"{args.norm}_"
                    f"{config['kernel']}_"
                    f"{config['dataset']}_mc.json"
                )
            )

        out_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_file.write_text(
            json.dumps(
                out_obj,
                indent=2,
            )
            + "\n"
        )

    print(
        "\n[DONE] Saved SPSA results to:"
    )

    print(
        out_file
    )

    if errors:

        print(
            f"[WARN] "
            f"{len(errors)} budgets failed."
        )


if __name__ == "__main__":
    main()