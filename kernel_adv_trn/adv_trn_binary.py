import argparse
import json
from pathlib import Path
import sys

import torch
import dataset
import robust_kernels as rk


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--DATASET",
        type=str,
        default="fmnist",
    )

    # Example:
    # --CLASS_PAIRS 0 1 2 3 4 5
    # means:
    # (0,1), (2,3), (4,5)
    parser.add_argument(
        "--CLASS_PAIRS",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Pairs sequentially, e.g. --CLASS_PAIRS 0 1 2 3 4 5",
    )

    parser.add_argument(
        "--RHO",
        type=lambda x: float(eval(x)),
        default=0.0,
    )

    # Hyperparameter sweep
    parser.add_argument(
        "--LAM",
        type=float,
        nargs="+",
        default=[1.0],
        help="Lambda sweep, e.g. --LAM 1e-4 1e-3 1e-2 1e-1 1",
    )

    parser.add_argument(
        "--STEP_SIZE",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--ITERATIONS",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--BATCH_SIZE",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--DECAY",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--KERNEL",
        type=str,
        default="gaussian",
    )

    parser.add_argument(
        "--LENGTH_SCALE",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--QUERY_TILE",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--CENTER_TILE",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--SEED",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--DEVICE",
        type=str,
        default="cuda:0",
    )

    # OUTPUT is now a directory
    parser.add_argument(
        "--OUTPUT",
        type=str,
        default="runs/binary_models",
    )

    return parser.parse_args()

def loader_to_tensor(loader, device):

    xs = []
    ys = []

    for x, y in loader:
        xs.append(x)
        ys.append(y)

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    x = x.reshape(x.shape[0], -1)
    if y.ndim == 2 and y.shape[1] == 2:
        y = y.argmax(dim=1)

    x = x.to(
        device=device,
        dtype=torch.float32,
    )

    y = y.to(
        device=device,
        dtype=torch.float32,
    ).reshape(-1)

    return x, y

@torch.no_grad()
def evaluate_accuracy(
    loader,
    centers,
    alpha,
    beta,
    *,
    positive_class,
    remap_labels=False,
    **options,
):

    correct = 0
    total = 0

    for x, y in loader:

        x = x.reshape(
            x.shape[0],
            -1,
        ).to(
            device=centers.device,
            dtype=centers.dtype,
        )

        if y.ndim == 2 and y.shape[1] == 2:
            y = y.argmax(dim=1)

        elif remap_labels:
            y = (y == positive_class).long()

        y = y.to(
            device=centers.device,
            dtype=torch.long,
        ).reshape(-1)

        if not torch.all(
            (y == 0) | (y == 1)
        ):
            raise ValueError(
                "Accuracy evaluation requires binary labels 0/1"
            )

        values, _ = rk.kernel_eval_and_grad(
            x,
            centers,
            alpha,
            beta,
            **options,
        )

        if not torch.isfinite(values).all():
            raise FloatingPointError(
                "Nonfinite predictions during accuracy evaluation"
            )

        predictions = (values > 0).long()

        correct += (
            predictions == y
        ).sum().item()

        total += y.numel()

    if total == 0:
        raise ValueError(
            "Cannot evaluate accuracy on an empty loader"
        )

    return correct / total, total

def main():

    args = parse_args()

    if len(args.CLASS_PAIRS) % 2 != 0:
        raise ValueError(
            "--CLASS_PAIRS must contain an even number of class IDs.\n"
            "Example: --CLASS_PAIRS 0 1 2 3 4 5"
        )

    class_pairs = list(
        zip(
            args.CLASS_PAIRS[::2],
            args.CLASS_PAIRS[1::2],
        )
    )

    print("Class pairs:", class_pairs)
    print("Lambda sweep:", args.LAM)

    output_dir = Path(args.OUTPUT)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_runs = []
    for cl1, cl2 in class_pairs:

        print("\n")
        print("=" * 75)
        print(f"CLASS PAIR: {cl1} vs {cl2}")
        print("=" * 75)

        torch.manual_seed(args.SEED)
        torch.cuda.manual_seed_all(args.SEED)

        train_loader, val_loader, test_loader = (
            dataset.load_fmnist(
                classes=(cl1, cl2),
                which_mnist=args.DATASET,
                multiclass=False,
            )
        )

        x_train, y_train = loader_to_tensor(
            train_loader,
            args.DEVICE,
        )

        unique_labels = torch.unique(
            y_train
        )

        remap_labels = not torch.all(
            torch.isin(
                unique_labels,
                torch.tensor(
                    [0.0, 1.0],
                    device=args.DEVICE,
                ),
            )
        ).item()

        if remap_labels:

            y_train = (
                y_train == cl2
            ).float()

        if args.LENGTH_SCALE is None:

            length_scale = (
                x_train.shape[1] ** 0.5
            )

        else:

            length_scale = (
                args.LENGTH_SCALE
            )

        options = {
            "kernel": args.KERNEL,
            "length_scale": length_scale,
            "query_tile": args.QUERY_TILE,
            "center_tile": args.CENTER_TILE,
        }

        print(
            f"Training shape : {x_train.shape}"
        )

        print(
            f"Labels         : {torch.unique(y_train)}"
        )

        print(
            f"rho            : {args.RHO}"
        )

        print(
            f"kernel         : {args.KERNEL}"
        )

        print(
            f"length scale   : {length_scale}"
        )

        print(
            f"device         : {x_train.device}"
        )

        print(
            "Training type  :",
            "benign"
            if args.RHO == 0
            else "robust",
        )

        best_val_accuracy = -1.0

        best_alpha = None
        best_beta = None
        best_lam = None

        sweep_results = []

        for lam in args.LAM:

            print("\n")
            print("-" * 65)
            print(
                f"Classes {cl1} vs {cl2} | lambda = {lam}"
            )
            print("-" * 65)

            # Same initialization/random sequence for each lambda
            torch.manual_seed(args.SEED)
            torch.cuda.manual_seed_all(
                args.SEED
            )

            state = rk.fit(
                x_train,
                y_train,

                rho=args.RHO,
                lam=lam,

                step_size=args.STEP_SIZE,
                iterations=args.ITERATIONS,
                batch_size=args.BATCH_SIZE,
                decay=args.DECAY,

                kernel=args.KERNEL,
                length_scale=length_scale,

                query_tile=args.QUERY_TILE,
                center_tile=args.CENTER_TILE,

                seed=args.SEED,
            )

            print(
                "alpha norm       :",
                state["alpha"].norm().item(),
            )

            print(
                "avg alpha norm   :",
                state["average_alpha"].norm().item(),
            )

            print(
                "beta norm        :",
                state["beta"].norm().item(),
            )

            print(
                "avg beta norm    :",
                state["average_beta"].norm().item(),
            )

            print(
                "delta norm       :",
                state["delta"].norm().item(),
            )

            if args.RHO > 0:

                boundary = (
                    state["delta"].abs()
                    >= args.RHO - 1e-8
                ).float().mean()

                print(
                    "fraction delta at boundary:",
                    boundary.item(),
                )

            val_accuracy, val_count = (
                evaluate_accuracy(
                    val_loader,
                    x_train,
                    state["average_alpha"],
                    state["average_beta"],

                    positive_class=cl2,
                    remap_labels=remap_labels,

                    **options,
                )
            )

            print(
                f"Validation accuracy: "
                f"{val_accuracy:.4%}"
            )

            sweep_results.append(
                {
                    "lam": lam,
                    "val_accuracy": val_accuracy,
                    "val_samples": val_count,
                }
            )

            if val_accuracy > best_val_accuracy:

                best_val_accuracy = (
                    val_accuracy
                )

                best_lam = lam

                best_alpha = (
                    state["average_alpha"]
                    .detach()
                    .cpu()
                    .clone()
                )

                best_beta = (
                    state["average_beta"]
                    .detach()
                    .cpu()
                    .clone()
                )

                print(
                    ">>> New best model"
                )

        print("\n")
        print("=" * 65)
        print(
            f"BEST MODEL FOR {cl1} vs {cl2}"
        )
        print("=" * 65)

        print(
            f"Best lambda       : {best_lam}"
        )

        print(
            f"Best val accuracy : "
            f"{best_val_accuracy:.4%}"
        )

        # Put best parameters back on GPU
        alpha = best_alpha.to(
            args.DEVICE
        )

        beta = best_beta.to(
            args.DEVICE
        )

        metrics = {}

        print(
            "\nClean accuracy of best model:"
        )

        for split, loader in (
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        ):

            accuracy, count = (
                evaluate_accuracy(
                    loader,
                    x_train,
                    alpha,
                    beta,

                    positive_class=cl2,
                    remap_labels=remap_labels,

                    **options,
                )
            )

            metrics[
                f"{split}_accuracy"
            ] = accuracy

            metrics[
                f"{split}_samples"
            ] = count

            print(
                f"  {split:5s} accuracy: "
                f"{accuracy:.4%} "
                f"({count} samples)"
            )
        
        pair_results = {
        "dataset": args.DATASET,

        "class_pair": [
            cl1,
            cl2,
        ],

        "rho": args.RHO,

        "kernel": args.KERNEL,
        "length_scale": length_scale,

        "step_size": args.STEP_SIZE,
        "iterations": args.ITERATIONS,
        "batch_size": args.BATCH_SIZE,
        "decay": args.DECAY,

        "seed": args.SEED,

        "runs": sweep_results,

        "best_hyperparameters": {
            "lam": best_lam,
            "val_accuracy": best_val_accuracy,
        },

        "best_model_metrics": metrics,
    }

        all_runs.append(pair_results)

        config = {

            "dataset": args.DATASET,

            "classes": [
                cl1,
                cl2,
            ],

            "positive_class": cl2,

            "rho": args.RHO,

            "lam": best_lam,

            "lambda_sweep": args.LAM,

            "step_size": (
                args.STEP_SIZE
            ),

            "iterations": (
                args.ITERATIONS
            ),

            "batch_size": (
                args.BATCH_SIZE
            ),

            "decay": (
                args.DECAY
            ),

            "kernel": (
                args.KERNEL
            ),

            "length_scale": (
                length_scale
            ),

            "query_tile": (
                args.QUERY_TILE
            ),

            "center_tile": (
                args.CENTER_TILE
            ),

            "seed": (
                args.SEED
            ),

            "epsilon": (
                args.RHO
            ),

            "data_root": (
                "/janaki/common/Datasets/rahulky"
            ),

            "autoattack_commit": (
                "unknown"
            ),
        }

        output = output_dir / (
            f"{args.DATASET}_"
            f"{cl1}_vs_{cl2}_"
            f"best.pt"
        )

        torch.save(
            {

                "centers": (
                    x_train
                    .detach()
                    .cpu()
                ),

                "alpha": (
                    best_alpha
                ),

                "beta": (
                    best_beta
                ),

                "bias": 0.0,

                "options": options,

                "classes": [
                    cl1,
                    cl2,
                ],

                "positive_class": (
                    cl2
                ),

                "config": config,

                "metrics": metrics,

                "best_lam": (
                    best_lam
                ),

                "best_val_accuracy": (
                    best_val_accuracy
                ),

                "sweep_results": (
                    sweep_results
                ),
            },

            output,
        )

        json_output = output_dir / (
            f"{args.DATASET}_"
            f"{cl1}_vs_{cl2}_"
            f"results.json"
        )

        with open(json_output, "w") as f:
            json.dump(
                pair_results,
                f,
                indent=4,
            )

        print(
            f"Saved results JSON to:"
            f"\n{json_output}"
        )

        print(
            f"\nSaved best model to:"
            f"\n{output}"
        )

        print("\nSweep results:")

        for result in sweep_results:

            print(
                f"  lambda={result['lam']:<10g} "
                f"val={result['val_accuracy']:.4%}"
            )
        del alpha
        del beta

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    all_runs_output = output_dir / "all_runs.json"

    with open(all_runs_output, "w") as f:
        json.dump(
            {
                "dataset": args.DATASET,

                "class_pairs": [
                    list(pair)
                    for pair in class_pairs
                ],

                "rho": args.RHO,

                "lambda_sweep": args.LAM,

                "results": all_runs,
            },
            f,
            indent=4,
        )

    print(
        f"\nSaved complete experiment JSON to:"
        f"\n{all_runs_output}"
    )


if __name__ == "__main__":
    main()


# Use experiments.attack_budget to attack.