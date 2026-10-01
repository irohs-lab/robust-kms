import torch
import robust_kernels.multioutput_inputs as inputs
import robust_kernels.multioutput_tile as tile
import robust_kernels.lowrank_cache as cache


@torch.no_grad()
def kernel_eval_and_grad(queries, centers, alpha, factors=None, *,
                         kernel="gaussian", length_scale=1.0,
                         query_tile=128, center_tile=1024, gradients=True,
                         factor_cache=None):
  """Evaluate vector logits and optional Jacobians from factored coefficients.

  Returns (m, C) values and (m, d, C) Jacobians, or None for the latter.
  Derivative coefficients are materialized for one center tile and class only.
  Pending skinny factors are packed once and shared across all query tiles.
  factor_cache may reuse lowrank_cache.prepare output for unchanged factors
  with the same center_tile; it is scratch storage, not checkpoint state.
  """
  inputs.validate(queries, centers, alpha, factors, kernel, length_scale,
                  query_tile, center_tile, gradients)
  values = queries.new_zeros((len(queries), alpha.shape[1]))
  jacobian = (queries.new_zeros((len(queries), centers.shape[1], alpha.shape[1]))
              if gradients else None)
  prepared = factor_cache
  if factors is not None and prepared is None:
    prepared = cache.prepare(factors, center_tile)
  for i in range(0, len(queries), query_tile):
    rows = slice(i, i + query_tile)
    for j in range(0, len(centers), center_tile):
      stop = min(j + center_tile, len(centers))
      v, g = tile.evaluate_tile(queries[rows], centers[j:stop], alpha[j:stop],
                                factors, j, stop, kernel, length_scale,
                                gradients, None if prepared is None else
                                prepared[j // center_tile])
      values[rows].add_(v)
      if gradients:
        jacobian[rows].add_(g)
  return values, jacobian
