"""Run the model attacks from svhn_og.py concurrently on two GPUs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run svhn_og.py attacks in parallel on two GPUs."
    )
    parser.add_argument("--DATASET", required=True)
    parser.add_argument("--AA_norm", required=True, choices=("L2", "Linf"))
    parser.add_argument("--PGD", action="store_true")
    parser.add_argument("--BB_attack", default="False")
    parser.add_argument("--VERSION", required=True)
    parser.add_argument("--MC", action="store_true")
    parser.add_argument("--Kernel_Name", required=True)
    parser.add_argument("--SEED", type=int, default=42)
    parser.add_argument(
        "--GPUS",
        default="0,1",
        help="Two visible GPU indices, in order (default: 0,1).",
    )
    return parser.parse_args()


def paths_for(args, base_dir):
    directory = base_dir / f"{args.DATASET}2" / f"{args.Kernel_Name}_{args.VERSION}"
    if args.MC:
        directory = base_dir / f"{args.DATASET}2" / f"{args.Kernel_Name}_{args.VERSION}_mc"

    input_json = directory / f"{args.Kernel_Name}_{args.VERSION}.json"
    attack_prefix = "bb_" if args.BB_attack != "False" else ""
    seed_dir = directory / f"seed_{args.SEED}"
    if args.MC:
        output_name = (
            f"PGD_result_{args.AA_norm}_{attack_prefix}{args.Kernel_Name}_"
            f"{args.VERSION}_{args.DATASET}_mc_over_iters1.json"
        )
    else:
        output_name = (
            f"PGD_result_{args.AA_norm}_{attack_prefix}{args.Kernel_Name}_"
            f"{args.VERSION}_{args.DATASET}_class_pair.json"
        )
    return input_json, seed_dir / output_name


def child_command(args, script, indices, output_json):
    command = [
        sys.executable, "-u", str(script),
        "--DATASET", args.DATASET,
        "--AA_norm", args.AA_norm,
        "--BB_attack", args.BB_attack,
        "--VERSION", args.VERSION,
        "--Kernel_Name", args.Kernel_Name,
        "--SEED", str(args.SEED),
        "--ENTRY_INDICES", ",".join(map(str, indices)),
        "--OUT_JSON", str(output_json),
        "--NO_PROGRESS_JSON",
    ]
    if args.PGD:
        command.append("--PGD")
    if args.MC:
        command.append("--MC")
    return command


def physical_gpu_token(visible_index, original_visible_devices):
    """Map a visible CUDA index through an existing scheduler allocation."""
    if not original_visible_devices:
        return str(visible_index)
    allocated = [token.strip() for token in original_visible_devices.split(",")]
    if visible_index < 0 or visible_index >= len(allocated):
        raise ValueError(
            f"GPU index {visible_index} is unavailable: CUDA_VISIBLE_DEVICES="
            f"{original_visible_devices!r}"
        )
    return allocated[visible_index]


def main():
    args = parse_args()
    gpu_indices = [int(value.strip()) for value in args.GPUS.split(",") if value.strip()]
    if len(gpu_indices) != 2 or len(set(gpu_indices)) != 2:
        raise ValueError("--GPUS must contain exactly two distinct visible indices")

    base_dir = Path(__file__).resolve().parent
    input_json, output_json = paths_for(args, base_dir)
    with input_json.open("r") as handle:
        summary = json.load(handle)
    entries = summary.get("results", summary)
    if not isinstance(entries, list):
        raise TypeError(f"Expected a result list in {input_json}")
    if not entries:
        raise ValueError(f"No model entries found in {input_json}")

    # Contiguous chunks preserve input order when the two outputs are merged.
    split_at = (len(entries) + 1) // 2
    chunks = [range(0, split_at), range(split_at, len(entries))]
    original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="svhn_parallel_", dir=output_json.parent
    ) as temporary_directory:
        temporary_directory = Path(temporary_directory)
        workers = []
        partial_paths = []
        for worker_id, (gpu_index, indices) in enumerate(zip(gpu_indices, chunks)):
            if not indices:
                continue
            partial_path = temporary_directory / f"worker_{worker_id}.json"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = physical_gpu_token(
                gpu_index, original_visible_devices
            )
            command = child_command(args, base_dir / "svhn_og.py", indices, partial_path)
            print(
                f"[worker {worker_id}] GPU {gpu_index}: attacking {len(indices)} "
                f"model(s), entries {indices.start}..{indices.stop - 1}",
                flush=True,
            )
            workers.append(
                (worker_id, subprocess.Popen(command, cwd=base_dir, env=environment))
            )
            partial_paths.append(partial_path)

        failures = []
        for worker_id, process in workers:
            return_code = process.wait()
            if return_code != 0:
                failures.append((worker_id, return_code))
        if failures:
            details = ", ".join(
                f"worker {worker_id} exited {code}" for worker_id, code in failures
            )
            raise RuntimeError(f"Parallel attack failed: {details}")

        combined = {"results": [], "errors": []}
        for partial_path in partial_paths:
            with partial_path.open("r") as handle:
                partial = json.load(handle)
            combined["results"].extend(partial.get("results", []))
            combined["errors"].extend(partial.get("errors", []))

        with output_json.open("w") as handle:
            json.dump(combined, handle, indent=2)

    print(f"[DONE] Saved merged results to: {output_json}")
    if combined["errors"]:
        print(f"[WARN] {len(combined['errors'])} entries failed; see the output JSON.")


if __name__ == "__main__":
    main()
