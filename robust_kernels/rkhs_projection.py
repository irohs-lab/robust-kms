import math
import time
import torch
import robust_kernels.lowrank_factors as factors
import robust_kernels.lowrank_cache as factor_packing
import robust_kernels.eigenpro_solve as eigenpro
import robust_kernels.multioutput_evaluate as evaluation
import robust_kernels.projection_residual as residual
import robust_kernels.projection_search as search


@torch.no_grad()
def project(centers, state, *, max_steps=20, tolerance=1e-6, step_size=1.,
            solve_rtol=1e-6, solve_atol=1e-8, solve_max_epochs=100,
            eigenpro_samples=1024, eigenpro_rank=100, eigenpro_batch_size=128,
            seed=0, kernel="gaussian",
            length_scale=1., query_tile=128, center_tile=1024):
  """Approximate the coupled RKHS rank-one projection; mutate only on success.

  Alpha is eliminated by EigenPro2. A line search improves the factorized initializer.
  No full derivative Gram matrix or n*d*C coefficient tensor is allocated.
  """
  if type(max_steps) is not int or max_steps < 0 or not math.isfinite(step_size) or step_size <= 0:
    raise ValueError("max_steps must be nonnegative and step_size finite positive")
  if not math.isfinite(tolerance) or tolerance < 0:
    raise ValueError("tolerance must be finite and nonnegative")
  options = dict(kernel=kernel, length_scale=length_scale,
                 query_tile=query_tile, center_tile=center_tile)
  solver = dict(rtol=solve_rtol, atol=solve_atol, max_epochs=solve_max_epochs)
  started = time.perf_counter()
  target = state["factors"]
  pending = target["pending"]
  buffer_bytes = sum(t.numel() * t.element_size()
    for block in pending.values() for t in block.values())
  candidate = factors.rank_one(target)
  if all(min(block["core"].shape) <= 1 for block in pending.values()):
    # The target is already feasible. Only consolidate its exact factorization.
    state["factors"] = candidate
    state["last_projection"] = state.get("iterations", 0)
    state["last_projection_epoch"] = state.get("epochs", 0)
    report = dict(error_squared=0., initial_error_squared=0., iterations=0,
      converged=True, reason="already_rank_one", max_value_error=None,
      kernel_solve=dict(method="not_needed", epochs=0),
      pending_blocks_before=len(pending), buffer_bytes=buffer_bytes,
      kernel_solve_calls=0, kernel_solve_epochs=0,
      max_qr_width=max((max(b["core"].shape) for b in pending.values()), default=0),
      elapsed_seconds=time.perf_counter() - started)
    state["projection"] = report
    return report
  zeros = torch.zeros_like(state["alpha"])
  target_cache = factor_packing.prepare(target, center_tile)
  values, _ = evaluation.kernel_eval_and_grad(
    centers, centers, zeros, target, gradients=False, factor_cache=target_cache, **options)
  workspace = eigenpro.prepare(centers, options, samples=eigenpro_samples,
    rank=eigenpro_rank, batch_size=eigenpro_batch_size, seed=seed)
  workspace["target_factor_cache"] = target_cache
  workspace["target_query_cache"] = factor_packing.prepare(target, query_tile)
  current = residual.measure(centers, target, candidate, values, options, solver, workspace)
  initial, iterations, rate, reason = current[0], 0, step_size, "max_steps"
  for _ in range(max_steps):
    norm = (current[1].square().sum() + current[2].square().sum()).sqrt().item()
    if norm <= tolerance or current[0] == 0:
      reason = "stationary"
      break
    proposal, rate = search.trial(
      centers, target, candidate, values, options, solver, workspace, current, rate)
    if proposal is None:
      reason = "line_search_failed"
      break
    candidate = proposal
    current = residual.measure(centers, target, candidate, values, options, solver, workspace)
    iterations += 1
  state["alpha"] = state["alpha"] + current[3]
  state["factors"] = candidate
  state["last_projection"] = state.get("iterations", 0)
  state["last_projection_epoch"] = state.get("epochs", 0)
  report = dict(error_squared=2 * current[0], initial_error_squared=2 * initial,
    iterations=iterations, converged=reason == "stationary", reason=reason,
    max_value_error=current[4]["max_value_error"], kernel_solve=current[4],
    pending_blocks_before=len(pending), buffer_bytes=buffer_bytes,
    kernel_solve_calls=workspace["solve_calls"],
    kernel_solve_epochs=workspace["total_epochs"],
    max_qr_width=max((max(b["core"].shape) for b in pending.values()), default=0),
    elapsed_seconds=time.perf_counter() - started)
  state["projection"] = report
  return report
