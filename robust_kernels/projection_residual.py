import math
import torch
import robust_kernels.lowrank_factors as factors
import robust_kernels.multioutput_evaluate as evaluation
import robust_kernels.eigenpro_solve as eigenpro


@torch.no_grad()
def measure(centers, target, candidate, target_values, options, solver, workspace, gradient=True):
  """RKHS distance after eliminating alpha; stream derivative residuals."""
  zeros = torch.zeros_like(target_values)
  # Form G(target-candidate) directly, avoiding cancellation between two
  # large value evaluations before the scalar kernel solve.
  rhs, _ = evaluation.kernel_eval_and_grad(
    centers, centers, zeros, target, gradients=False,
    factor_cache=workspace.get("target_factor_cache"),
    subtract_factors=candidate, **options)
  correction, diagnostics = eigenpro.solve(workspace, rhs, **solver)
  du = torch.zeros_like(candidate["u"]) if gradient else None
  dv = torch.zeros_like(candidate["v"]) if gradient else None
  error, max_value_error = centers.new_zeros(()), 0.
  for start in range(0, len(centers), options["query_tile"]):
    stop = min(start + options["query_tile"], len(centers))
    rows = slice(start, stop)
    values_residual, residual = evaluation.kernel_eval_and_grad(
      centers[rows], centers, correction, candidate, subtract_factors=target,
      subtract_factor_cache=workspace.get("target_factor_cache"), **options)
    error += (correction[rows] * values_residual).sum()
    max_value_error = max(max_value_error, values_residual.abs().max().item())
    query_cache = workspace.get("target_query_cache")
    prepared = None if query_cache is None else query_cache[start // options["query_tile"]]
    for c in range(zeros.shape[1]):
      difference = (factors.column(candidate, start, stop, c) -
                    factors.column(target, start, stop, c, prepared=prepared))
      error += (difference * residual[:, :, c]).sum()
    if gradient:
      empty = (candidate["u"][rows].norm(dim=1) == 0) & (candidate["v"][rows].norm(dim=1) == 0)
      indices = empty.nonzero().flatten()
      if len(indices):
        coordinates = residual[indices].square().sum(2).argmax(1)
        candidate["u"][start + indices, coordinates] = 1.
      du[rows] = (residual * candidate["v"][rows, None, :]).sum(2)
      dv[rows] = (residual * candidate["u"][rows, :, None]).sum(1)
  diagnostics["max_value_error"] = max_value_error
  raw = error.item()
  if not math.isfinite(raw) or raw < -1e-6:
    raise RuntimeError("Invalid RKHS distance; use tighter solves or float64")
  return max(raw, 0.) / 2, du, dv, correction, diagnostics

