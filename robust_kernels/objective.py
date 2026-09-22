import torch
import torch.nn.functional
import robust_kernels.evaluate as evaluation


@torch.no_grad()
def objective(centers, labels, alpha, beta, *, rho, lam, kernel="gaussian",
              length_scale=1.0, query_tile=128, center_tile=1024):
  """Stream the exact summed objective; this is a full-data diagnostic."""
  loss = centers.new_zeros(())
  penalty = centers.new_zeros(())
  norm_squared = centers.new_zeros(())
  for i in range(0, len(centers), query_tile):
    rows = slice(i, i + query_tile)
    values, gradients = evaluation.kernel_eval_and_grad(
      centers[rows], centers, alpha, beta, kernel=kernel,
      length_scale=length_scale, query_tile=query_tile, center_tile=center_tile)
    loss.add_(torch.nn.functional.binary_cross_entropy_with_logits(
      values, labels[rows].to(values.dtype), reduction="sum"))
    penalty.add_(gradients.abs().sum())
    norm_squared.add_(alpha[rows] @ values + (beta[rows] * gradients).sum())
  return {"loss": loss, "gradient_l1": penalty, "norm_squared": norm_squared,
          "total": loss + rho * penalty + 0.5 * lam * norm_squared}
