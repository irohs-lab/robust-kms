import torch
import robust_kernels._radial as radial


def evaluate_tile(queries, centers, alpha, beta, kernel, length_scale):
  """Contract one tile without displacement tensors or Hessian blocks."""
  values, a, b = radial.radial_terms(queries, centers, kernel, length_scale)
  projections = queries @ beta.T
  projections.sub_(torch.sum(centers * beta, dim=1)[None, :])
  values_out = values @ alpha + torch.sum(a * projections, dim=1)
  b.mul_(projections)
  gradients = a @ (beta + alpha[:, None] * centers)
  gradients.addmm_(b, centers)
  gradients.sub_(queries * (a @ alpha + b.sum(dim=1))[:, None])
  return values_out, gradients
