import os
import subprocess
from pathlib import Path

import torch

import argparse


parser = argparse.ArgumentParser()

#Only to be uncommented while using fmnist
# parser.add_argument('--class1', type=int, default=4, help='Class data to use from fmnist dataset')
# parser.add_argument('--class2', type=int, default=2, help='Class data to use from fmnist dataset')
parser.add_argument('--Feature_IDX', type=int, default=31, help='Feature index to use from celeba dataset')

args = parser.parse_args()

# Adjust these grids to sweep different values per kernel.
SWEEPS = {
    # "gaussian": {
    #     "length_scales": [0.1, 0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0],
    #     # "length_scales": [0.1],
    #     # "regularizers": [1e-2],
    #     "regularizers": [1e-4, 1e-3, 1e-2],
    # },
    "laplacian": {
        "length_scales": [0.1, 0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0],
        "regularizers": [1e-4, 1e-3, 1e-2],
        # "regularizers": [1e-1],
    },
}
# GPU_DEVICE = "6"


def artifact_path(script_path: Path, kernel: str, length_str: str, reg: float) -> Path:
    return script_path.with_name(
        # f"rfm_results_{kernel}_fmnist_L{length_str}_reg{reg}_classes({args.class1}_{args.class2}).pt"
        f"rfm_results_{kernel}_celeba_L{length_str}_reg{reg}_fidx({args.Feature_IDX}).pt"
    )


def main() -> None:
    script_path = Path(__file__).with_name("save_rfms.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Expected save_rfms.py at {script_path}")

    # env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE
    any_results = False

    for kernel, sweep in SWEEPS.items():
        length_scales = sweep["length_scales"]
        regularizers = sweep["regularizers"]

        results = []

        print(f"\n=== Sweeping {kernel} kernel ===")
        for length in length_scales:
            length_str = str(length)
            for reg in regularizers:
                cmd = [
                    "python",
                    str(script_path),
                    "--kernel",
                    kernel,
                    "--L",
                    length_str,
                    "--reg",
                    str(reg),
                    "--Feature_IDX",
                    str(args.Feature_IDX),
                    # "--class1",
                    # str(args.class1),
                    # "--class2",
                    # str(args.class2),
                ]
                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

                candidate = artifact_path(script_path, kernel, length_str, reg)
                if not candidate.exists():
                    print(f"  -> Expected results file {candidate.name} not found.")
                    continue

                payload = torch.load(candidate, map_location="cpu")
                accuracy = payload.get("best_round_accuracy")
                iteration = payload.get("iteration")
                print(f"  -> best_round_accuracy={accuracy}")
                results.append(
                    {
                        "kernel": kernel,
                        "length": length,
                        "length_str": length_str,
                        "reg": reg,
                        "accuracy": accuracy,
                        "iteration": iteration,
                        "best_round_accuracy": payload.get("best_round_accuracy"),
                        "diff_rfm_kernel_acc": payload.get("diff_rfm_kernel_acc"),
                        "artifact": candidate,
                        "Feature_IDX": args.Feature_IDX,
                        #Only to be uncommented while using fmnist
                        # "class1": args.class1,
                        # "class2": args.class2,
                    }
                )

        if not results:
            print(f"No results for {kernel}; skipping log write.")
            continue

        any_results = True
        results_log = Path(__file__).with_name(f"tune_{kernel}_rfm_reg{reg}_fidx{args.Feature_IDX}_lengthscale.txt")
        best_log = Path(__file__).with_name(f"tune_{kernel}_rfm_reg{reg}_fidx{args.Feature_IDX}.txt")

        lines = [
            f"kernel={item['kernel']} L={item['length']} reg={item['reg']} fidx{args.Feature_IDX} accuracy={item['accuracy']}"
            for item in results
        ]
        results_log.write_text("\n".join(lines) + "\n")

        numeric_results = [item for item in results if item["accuracy"] is not None]
        if not numeric_results:
            best_log.write_text("No valid accuracy values were produced.\n")
            print(f"No valid accuracy values were produced for {kernel}.")
            continue

        best_item = max(numeric_results, key=lambda item: item["accuracy"])
        best_log.write_text(
            (
                "Best configuration\n"
                f"kernel={best_item['kernel']}\n"
                f"L={best_item['length']}\n"
                f"reg={best_item['reg']}\n"
                f"accuracy={best_item['accuracy']}\n"
                f"iteration={best_item['iteration']}\n"
                f"best_round_accuracy ={best_item['best_round_accuracy']}\n"
                f"diff_rfm_kernel_acc = {best_item['diff_rfm_kernel_acc']}\n"
                f"artifact={best_item['artifact']}\n"
                # f"Feature_IDX={best_item['Feature_IDX']}\n"
            )
        )
        print(
            "Best configuration:"
            f" kernel={best_item['kernel']} L={best_item['length']} reg={best_item['reg']}"
            f" accuracy={best_item['accuracy']}" f" iteration={best_item['iteration']}" f" artifact={best_item['artifact']}"
            f" best_round_accuracy ={best_item['best_round_accuracy']}" f" diff_rfm_kernel_acc = {best_item['diff_rfm_kernel_acc']}"
        )

    if not any_results:
        print("No runs produced results; no logs written.")


if __name__ == "__main__":
    main()
