import torch
import torch.nn.functional
import robust_kernels as rk


@torch.no_grad()
def evaluate(queries, labels, centers, alpha, beta, options, training=False):
  correct = queries.new_zeros(())
  loss = queries.new_zeros(())
  gradient_l1 = queries.new_zeros(())
  norm_squared = queries.new_zeros(())
  for start in range(0, len(queries), 256):
    rows = slice(start, start + 256)
    values, gradients = rk.kernel_eval_and_grad(
      queries[rows], centers, alpha, beta, **options)
    correct.add_(((values > 0).long() == labels[rows]).sum())
    loss.add_(torch.nn.functional.binary_cross_entropy_with_logits(
      values, labels[rows].to(values.dtype), reduction="sum"))
    if training:
      gradient_l1.add_(gradients.abs().sum())
      norm_squared.add_(alpha[rows] @ values + (beta[rows] * gradients).sum())
  result = dict(accuracy=correct.item() / len(queries), loss=loss.item() / len(queries))
  if training:
    result.update(gradient_l1=gradient_l1.item(), norm_squared=norm_squared.item())
  return result
