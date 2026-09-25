import math
import torch
import robust_kernels.kernel_matmul as kernel_operator
import robust_kernels.eigenpro_iteration as iteration
from robust_kernels.eigenpro_setup import prepare


@torch.no_grad()
def solve(workspace, rhs, *, rtol=1e-6, atol=1e-8, max_epochs=100):
  """Solve K W=rhs by EigenPro2, stopping on the full true residual."""
  if not (math.isfinite(rtol) and math.isfinite(atol) and rtol >= 0 and atol >= 0):
    raise ValueError("solve tolerances must be finite and nonnegative")
  if type(max_epochs) is not int or max_epochs < 1:
    raise ValueError("solve_max_epochs must be a positive integer")
  workspace["solve_calls"] += 1
  apply = lambda w: kernel_operator.apply(workspace["centers"], w, **workspace["options"])
  threshold = atol + rtol * rhs.norm(dim=0)
  solution, residual = torch.zeros_like(rhs), rhs.norm(dim=0)
  warm = workspace["warm_start"]
  if warm is not None and warm.shape == rhs.shape:
    warm_residual = (apply(warm) - rhs).norm(dim=0)
    if warm_residual.max() < residual.max():
      solution, residual = warm.clone(), warm_residual
  for epoch in range(max_epochs + 1):
    if not torch.isfinite(residual).all():
      raise RuntimeError("EigenPro2 kernel solve produced nonfinite residuals")
    if (residual <= threshold).all():
      workspace["warm_start"] = solution.clone()
      return solution, dict(method="eigenpro2", epochs=epoch,
        max_residual=residual.max().item(), samples=workspace["samples"],
        rank=workspace["rank"], batch_size=workspace["batch_size"])
    if epoch < max_epochs:
      iteration.epoch(workspace, solution, rhs)
      workspace["total_epochs"] += 1
      residual = (apply(solution) - rhs).norm(dim=0)
  raise RuntimeError("EigenPro2 failed its residual tolerance; increase solve_max_epochs, "
                     "adjust the preconditioner, or use float64")
