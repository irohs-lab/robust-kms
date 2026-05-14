import os
import subprocess
from pathlib import Path

import torch

import argparse
from typing import List, Dict, Any
import json

parser = argparse.ArgumentParser()


parser.add_argument('--DATASET', type=str, default="fmnist", help='Dataset to use for tuning')
parser.add_argument('--MC', action='store_true', help='Whether to do multiclass tuning (only class1 vs rest)')
parser.add_argument('--VERSION', type=str, default="rfm", help='Version tag to differentiate results files')

args = parser.parse_args()

Dataset = args.DATASET

multiclass = args.MC
Version = args.VERSION

if not multiclass:
    classes1 = [0,1,2,3,4,5,6,7,8]
    classes2 = [1,2,3,4,5,6,7,8,9]
else:
    classes1 = [0]
    classes2 = [1]
# classes1 = [2,3,4]
# classes2 = [8,9]

# Adjust these grids to sweep different values per kernel.
SWEEPS = {
    "gaussian": {
        "length_scales": [0.05, 0.125, 0.25, 0.5, 1, 2.0, 4.0, 6.0],
        # "length_scales": [0.05, 0.125],
        # "length_scales": [0.1],
        # "regularizers": [1e-2],
        "regularizers": [1e-4, 1e-3, 1e-2],
    }
    # }, 

    # "laplacian": {
    #     # "length_scales": [0.1, 0.05, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 6.0],
    #     "length_scales": [0.05, 0.125, 0.25, 0.5, 1, 2.0, 4.0, 6.0],
    #     # "length_scales": [1.0],
    #     "regularizers": [1e-4, 1e-3, 1e-2],
    #     # "regularizers": [1e-2],
    # }
    
}


def artifact_path(script_path: Path, kernel: str, length_str: str, reg: float, class1, class2) -> Path:
    if not multiclass:
        return script_path.parent / f"{Dataset}2" / f"{Version}_results_{kernel}_{Dataset}_L{length_str}_reg{reg}_{class1}_{class2}.pt"
    else:
        return script_path.parent / f"{Dataset}2" / f"{Version}_results_{kernel}_{Dataset}_L{length_str}_reg{reg}_mc.pt"

# Metrics-only JSON output
def metrics_path(script_path: Path, kernel: str, length_str: str, reg: float, class1, class2) -> Path:
    if not multiclass:
        return script_path.parent / f"{Dataset}2" / f"metrics_{kernel}_{Version}_{Dataset}_L{length_str}_reg{reg}_{class1}_{class2}.json"
    else:
        return script_path.parent / f"{Dataset}2" / f"metrics_{kernel}_{Version}_{Dataset}_L{length_str}_reg{reg}_mc.json"



def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def main() -> None:
    script_path = Path(__file__).with_name("save_rfms.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Expected save_rfms.py at {script_path}")

    # env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = GPU_DEVICE
    any_results = False

    for class1 in classes1:
        for class2 in range(class1+1,len(classes2)+1):
        # for class2 in classes2:
            for kernel, sweep in SWEEPS.items():
                length_scales = sweep["length_scales"]
                regularizers = sweep["regularizers"]

                results = []

                print(f"\n=== Sweeping {kernel} kernel ===")
                for length in length_scales:
                    length_str = str(length)
                    for reg in regularizers:
                        # Previous sweep command without --metrics_only (kept for reference):
                        # cmd = [
                        #     "python",
                        #     str(script_path),
                        #     "--kernel",
                        #     kernel,
                        #     "--L",
                        #     length_str,
                        #     "--reg",
                        #     str(reg),
                        #     # "--Feature_IDX",
                        #     # str(args.Feature_IDX),
                        #     "--class1",
                        #     str(class1),
                        #     "--class2",
                        #     str(class2),
                        # ]
                        cmd = [
                            "python",
                            str(script_path),
                            "--kernel",
                            kernel,
                            "--L",
                            length_str,
                            "--reg",
                            str(reg),
                            "--class1",
                            str(class1),
                            "--class2",
                            str(class2),
                            "--dataset",
                            Dataset,
                            "--metrics_only",
                            "--version",
                            Version,
                            # "--Feature_IDX",
                            # str(args.Feature_IDX),
                        ]
                        if multiclass:
                            cmd.append("--mc")
                        print(f"Running: {' '.join(cmd)}")
                        subprocess.run(cmd, check=True)


                        candidate = metrics_path(script_path, kernel, length_str, reg, class1, class2)
                        if not candidate.exists():
                            print(f"  -> Expected results file {candidate.name} not found.")
                            continue

                        payload = json.loads(candidate.read_text())
                        rfm_best_acc = payload.get("best_round_accuracy")  # best across iterations
                        kernel_acc = payload.get("kernel_acc")            # iteration-0 base kernel
                        diff = payload.get("diff_rfm_kernel_acc")         # (best - kernel)
                        iteration = payload.get("iteration")

                        print(f"  rfm_best_acc={rfm_best_acc:.2f}  best_iter={iteration}")
                        results.append(
                            {
                                "kernel": kernel,
                                "length": length,
                                "length_str": length_str,
                                "reg": reg,
                                "kernel_acc": kernel_acc,
                                "rfm_best_acc": rfm_best_acc,
                                "best_iteration": iteration,
                                "diff_rfm_kernel_acc": diff,
                                "artifact": str(artifact_path(script_path, kernel, length_str, reg, class1, class2)),
                                "metrics_artifact": str(candidate),

                                "class1": class1,
                                "class2": class2,
                            }
                        )


                if not results:
                    print(f"No results for {kernel}; skipping log write.")
                    continue

                any_results = True

                if not multiclass:
                    results_log = Path(__file__).with_name(f"tune_idx_{kernel}_{Dataset}_L{length_str}_rfm_reg{reg}_lengthscale_{class1}_{class2}.txt")
                    best_log = Path(__file__).with_name(f"tune_idx_{kernel}_{Dataset}_L{length_str}_rfm_reg{reg}_{class1}_{class2}.txt")
                else:
                    results_log = Path(__file__).with_name(f"tune_idx_{kernel}_{Dataset}_L{length_str}_rfm_reg{reg}_lengthscale_mc.txt")
                    best_log = Path(__file__).with_name(f"tune_idx_{kernel}_{Dataset}_L{length_str}_rfm_reg{reg}_mc.txt")
        # Only to be uncommented while using celeba
                # lines = [
                #     f"kernel={item['kernel']} L={item['length']} reg={item['reg']} fidx{args.Feature_IDX} accuracy={item['accuracy']}"
                #     for item in results
                # ]

        # Only to be uncommented while using svhn2-class
                lines = [
                    f"kernel={item['kernel']} L={item['length']} reg={item['reg']}"
                    f"rfm_best_acc={item['rfm_best_acc']} "
                    f"best_iter={item['best_iteration']} diff={item['diff_rfm_kernel_acc']}"
                    for item in results
                ]
                results_log.write_text("\n".join(lines) + "\n")

                # numeric_results = [item for item in results if item["accuracy"] is not None]
                numeric_results = [item for item in results if item["rfm_best_acc"] is not None]
                if not numeric_results:
                    best_log.write_text("No valid accuracy values were produced.\n")
                    print(f"No valid accuracy values were produced for {kernel}.")
                    continue
                best_item = max(numeric_results, key=lambda item: item["rfm_best_acc"])



                # best_item = max(numeric_results, key=lambda item: item["accuracy"])
                best_log.write_text(
                    (
                        "Best configuration\n"
                        f"kernel={best_item['kernel']}\n"
                        f"L={best_item['length']}\n"
                        f"reg={best_item['reg']}\n"
                        f"rfm_best_acc={best_item['rfm_best_acc']}\n"
                        f"best_iteration={best_item['best_iteration']}\n"
                        f"diff_rfm_kernel_acc={best_item['diff_rfm_kernel_acc']}\n"
                        f"artifact={best_item['artifact']}\n"
                        f"classes={class1}_{class2}"
                    )
                )

                # Rerun best config without --metrics_only to save the full .pt
                best_cmd = [
                    "python",
                    str(script_path),
                    "--kernel",
                    str(best_item["kernel"]),
                    "--L",
                    str(best_item["length_str"]),
                    "--reg",
                    str(best_item["reg"]),
                    "--class1",
                    str(class1),
                    "--class2",
                    str(class2),
                    "--dataset",
                    Dataset,
                    "--version",
                    Version,
                ]
                if multiclass:
                    best_cmd.append("--mc")
                print(f"Saving best artifact: {' '.join(best_cmd)}")
                subprocess.run(best_cmd, check=True)

                # json_out = Path(__file__).with_name("results") / f"tune_0idx_{kernel}_svhn.json"
                if not multiclass:
                    json_out = Path(__file__).with_name("results") / f"tune_{kernel}_{Version}_{Dataset}_{class1}_{class2}.json"
                else:
                    json_out = Path(__file__).with_name("results") / f"tune_{kernel}_{Version}_{Dataset}_mc.json"
                write_json(json_out, results)

                print(
                    "Best configuration:"
                    f" kernel={best_item['kernel']} L={best_item['length']} reg={best_item['reg']}"
                    f" iteration={best_item['best_iteration']}" f" artifact={best_item['artifact']}"
                    f" best_round_accuracy ={best_item['rfm_best_acc']}" f" diff_rfm_kernel_acc = {best_item['diff_rfm_kernel_acc']}"
                    f"classes={class1}_{class2}"
                )

            if not any_results:
                print("No runs produced results; no logs written.")


if __name__ == "__main__":
    main()
