import torch
import robust_kernels._radial as radial
import robust_kernels.lowrank_factors as factors_ops


def evaluate_tile(queries, centers, alpha, factors, start, stop, kernel,
                  length_scale, gradients):
  kernel_values, a, b = radial.radial_terms(queries, centers, kernel, length_scale)
  values = kernel_values @ alpha
  jacobian = (queries.new_empty((len(queries), centers.shape[1], alpha.shape[1]))
              if gradients else None)
  for c in range(alpha.shape[1]):
    if gradients:
      g = a @ (alpha[:, c, None] * centers)
      g.sub_(queries * (a @ alpha[:, c])[:, None])
    if factors is not None:
      beta = factors_ops.column(factors, start, stop, c)
      if (beta.shape != centers.shape or beta.dtype != centers.dtype
          or beta.device != centers.device):
        raise ValueError("factor columns must match the centers' shape and type")
      projections = queries @ beta.T
      projections.sub_(torch.sum(centers * beta, dim=1)[None, :])
      values[:, c].add_(torch.sum(a * projections, dim=1))
      if gradients:
        weighted = b * projections
        g.addmm_(a, beta)
        g.addmm_(weighted, centers)
        g.sub_(queries * weighted.sum(dim=1)[:, None])
    if gradients:
      jacobian[:, :, c] = g
  return values, jacobian
