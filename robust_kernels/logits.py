import torch
import robust_kernels.evaluate as evaluation


def binary_logits(inputs, centers, alpha, beta, **options):
  """Return [0, f(x)] logits with exact first input derivatives.

  The analytic Jacobian avoids retaining kernel tiles for backward.
  This interface supports first-order attacks, not higher derivatives.
  """
  flat = inputs.flatten(1)
  values, gradients = evaluation.kernel_eval_and_grad(
    flat.detach(), centers, alpha, beta, **options)
  if torch.is_grad_enabled() and flat.requires_grad:
    values = values + (gradients * (flat - flat.detach())).sum(dim=1)
  return torch.stack((torch.zeros_like(values), values), dim=1)
