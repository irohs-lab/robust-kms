import math
import torch
import robust_kernels.lowrank_factors as factors
import robust_kernels.multioutput_evaluate as evaluation
import robust_kernels.eigenpro_solve as eigenpro


@torch.no_grad()
def measure(centers, target, candidate, target_values, options, solver, workspace, gradient=True):
  """RKHS distance after eliminating alpha; stream derivative residuals."""
  zeros = torch.zeros_like(target_values)
  values, _ = evaluation.kernel_eval_and_grad(
    centers, centers, zeros, candidate, gradients=False, **options)
  correction, diagnostics = eigenpro.solve(workspace, target_values - values, **solver)
  du = torch.zeros_like(candidate["u"]) if gradient else None
  dv = torch.zeros_like(candidate["v"]) if gradient else None
  error, max_value_error = centers.new_zeros(()), 0.
  for start in range(0, len(centers), options["query_tile"]):
    stop = min(start + options["query_tile"], len(centers))
    rows = slice(start, stop)
    v, jacobian = evaluation.kernel_eval_and_grad(
      centers[rows], centers, correction, candidate, **options)
    reference, reference_jacobian = evaluation.kernel_eval_and_grad(
      centers[rows], centers, zeros, target, **options)
    residual = jacobian - reference_jacobian
    error += (correction[rows] * (v - reference)).sum()
    max_value_error = max(max_value_error, (v - reference).abs().max().item())
    for c in range(zeros.shape[1]):
      difference = (factors.column(candidate, start, stop, c) -
                    factors.column(target, start, stop, c))
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

