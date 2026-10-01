"""Benchmark KLR packed arithmetic on a synthetic symmetric matrix.

This is not FashionMNIST training or a radial-kernel construction benchmark.
The matrix is K[i,j] = 0.1 + x[i]*x[j], with x equally spaced in [0,1].
KLR's normal tiled construction and packed operator are timed separately.

Example (KLR must be importable)::

  python -m experiments.multiclass_storage_benchmark --device cuda:2 \
    --output runs/packed-storage-benchmark.json

The default float32 54,000-center construction temporarily holds both dense
and packed matrices, requiring at least 16.3 GiB before other allocations.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time


def _git(path):
  def read(*args):
    try:
      return subprocess.check_output(["git", "-C", str(path), *args],
        text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
      return None
  return dict(commit=read("rev-parse", "HEAD"),
    branch=read("branch", "--show-current"),
    dirty=None if (status := read("status", "--porcelain")) is None else bool(status))


def run(args):
  import torch
  try:
    import klr
    from klr.utils.kernel_operator import _kernel_matrix
    from klr.utils.linear_operator import KernelLinearOperator
  except ImportError as error:
    raise RuntimeError("This optional benchmark requires KLR's new_api branch "
      "on PYTHONPATH (including klr.utils.linear_operator).") from error

  for name in ("size", "outputs", "batch_size", "block_size", "repeats", "warmup"):
    if getattr(args, name) < 1:
      raise ValueError(f"{name} must be positive")
  torch.set_num_threads(1)
  torch.backends.cuda.matmul.allow_tf32 = False
  device = torch.device(args.device)
  if device.type not in ("cpu", "cuda"):
    raise ValueError("device must be cpu or cuda")
  if device.type == "cuda":
    torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())
  dtype = torch.float32

  def sync():
    if device.type == "cuda":
      torch.cuda.synchronize(device)

  def allocated():
    return torch.cuda.memory_allocated(device) if device.type == "cuda" else None

  def clear():
    gc.collect()
    if device.type == "cuda":
      torch.cuda.empty_cache()

  def construct(function):
    sync()
    before = allocated()
    if device.type == "cuda":
      torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    value = function()
    sync()
    return value, dict(seconds=time.perf_counter() - started,
      allocated_before_bytes=before, allocated_after_bytes=allocated(),
      peak_allocated_bytes=(torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda" else None))

  def timed(function):
    for _ in range(args.warmup):
      value = function()
    sync()
    times = []
    for _ in range(args.repeats):
      sync()
      started = time.perf_counter()
      value = function()
      sync()
      times.append(time.perf_counter() - started)
    return value, dict(mean_seconds=statistics.mean(times),
      median_seconds=statistics.median(times), minimum_seconds=min(times),
      repeats_seconds=times)

  def check(value, expected):
    # The two summation orders differ; both must match the analytic answer.
    torch.testing.assert_close(value, expected, rtol=2e-4, atol=1e-4)
    return dict(max_absolute_error=(value - expected).abs().max().item(),
      relative_frobenius_error=((value - expected).norm() /
        expected.norm().clamp_min(torch.finfo(dtype).tiny)).item(),
      rtol=2e-4, atol=1e-4, passed=True)

  generator = torch.Generator(device=device).manual_seed(args.seed)
  x = torch.linspace(0, 1, args.size, dtype=dtype, device=device).reshape(-1, 1)
  weights = torch.randn(args.size, args.outputs, dtype=dtype, device=device,
    generator=generator) * .01
  rows = torch.randperm(args.size, device=device, generator=generator)[
    :min(args.batch_size, args.size)]
  expected = .1 * weights.sum(0)[None, :] + x * (x.T @ weights)
  expected_rows = expected[rows]

  def kernel(left, right):
    return (left @ right.T).add_(.1)

  def build_dense():
    # The same bounded evaluator used by kernel_operator(storage="packed").
    return _kernel_matrix(kernel, x, x, args.block_size)

  with torch.no_grad():
    # Initialize arithmetic libraries before measuring construction.
    kernel(x[:min(64, args.size)], x[:min(64, args.size)])
    clear()
    matrix, dense_construction = construct(build_dense)
    packed, packing = construct(lambda: KernelLinearOperator(matrix,
      sym=True, block_size=args.block_size, device=device))
    packed_values = int(packed.data.packed.numel())
    expected_values = args.size * (args.size + 1) // 2
    if packed_values != expected_values:
      raise AssertionError("KLR packed storage did not retain exactly one triangle")
    packed_bytes = int(packed.data.packed.untyped_storage().nbytes())
    del matrix
    clear()
    packed_live = allocated()
    packed_rows, packed_rows_timing = timed(lambda: packed[rows, :] @ weights)
    packed_full, packed_full_timing = timed(lambda: packed @ weights)
    packed_checks = dict(rows=check(packed_rows, expected_rows),
                         full=check(packed_full, expected))
    packed_diagonals = len(packed.data.diagonals)
    packed_rectangles = len(packed.data.rectangles)
    del packed
    clear()

    # Recreate the identical dense matrix after releasing packed storage.
    # Small retained outputs allow a direct dense-versus-packed comparison.
    matrix, dense_reconstruction = construct(build_dense)
    dense_bytes = int(matrix.untyped_storage().nbytes())
    dense_live = allocated()
    dense_rows, dense_rows_timing = timed(lambda: matrix[rows] @ weights)
    dense_full, dense_full_timing = timed(lambda: matrix @ weights)
    dense_checks = dict(rows=check(dense_rows, expected_rows),
                        full=check(dense_full, expected))
    agreement = dict(rows=check(packed_rows, dense_rows),
                     full=check(packed_full, dense_full))

  klr_source = Path(klr.__file__).resolve()
  return dict(
    description="Synthetic symmetric float32 matrix arithmetic benchmark; "
      "not a FashionMNIST training or radial-kernel construction benchmark",
    matrix="K[i,j] = 0.1 + x[i]*x[j]; x = linspace(0, 1, n)",
    timestamp_utc=datetime.now(timezone.utc).isoformat(),
    configuration=dict(size=args.size, outputs=args.outputs,
      batch_size=len(rows), block_size=args.block_size, warmup=args.warmup,
      repeats=args.repeats, seed=args.seed, device=str(device), dtype=str(dtype)),
    environment=dict(python=platform.python_version(), torch=torch.__version__,
      cuda=torch.version.cuda, tf32_matmul=False, torch_threads=torch.get_num_threads(),
      device_name=(torch.cuda.get_device_name(device) if device.type == "cuda"
                   else platform.processor()),
      klr_source=str(klr_source), klr=_git(klr_source.parent),
      benchmark_repository=_git(Path(__file__).resolve().parent)),
    construction=dict(dense_kernel=dense_construction, packing=packing,
      dense_reconstruction=dense_reconstruction,
      note="Construction uses KLR's tiled dense evaluator then its packed "
        "constructor, matching kernel_operator(storage='packed'). Packing holds "
        "the dense source plus packed destination. Dense reconstruction is "
        "needed only to compare products after releasing packed storage."),
    memory=dict(dense_matrix_storage_bytes=dense_bytes,
      packed_matrix_storage_bytes=packed_bytes, packed_values=packed_values,
      packed_stage_allocated_bytes=packed_live, dense_stage_allocated_bytes=dense_live,
      theoretical_dense_plus_packed_bytes=dense_bytes + packed_bytes,
      note="Storage bytes count each matrix allocation only. CUDA allocated "
        "and peak bytes are this process's live PyTorch allocations, including "
        "inputs and workspace; peaks include existing allocations at stage start. "
        "They exclude caching-allocator reserve, CUDA context and other processes. "
        "Packed-stage memory is sampled after releasing the dense source; "
        "dense-stage memory includes retained small packed product outputs. "
        "CUDA allocator measurements are null on CPU."),
    layout=dict(diagonal_blocks=packed_diagonals, rectangles=packed_rectangles),
    timings=dict(packed_rows=packed_rows_timing, packed_full=packed_full_timing,
      dense_rows=dense_rows_timing, dense_full=dense_full_timing),
    correctness=dict(packed_vs_analytic=packed_checks, dense_vs_analytic=dense_checks,
                     packed_vs_dense=agreement))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--size", type=int, default=54000)
  parser.add_argument("--outputs", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--block-size", type=int, default=2048)
  parser.add_argument("--device", default="cuda:2")
  parser.add_argument("--warmup", type=int, default=1)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = run(args)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
