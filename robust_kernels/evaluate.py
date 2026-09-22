import torch
import robust_kernels._eval_inputs as inputs
import robust_kernels._tile as tile


@torch.no_grad()
def kernel_eval_and_grad(queries, centers, alpha, beta, *, kernel="gaussian",
                         length_scale=1.0, query_tile=128, center_tile=1024):
  """Evaluate E*alpha + D*beta and its input gradient at queries.

  Results have shapes (b,) and (b, d). No autograd graph is retained.
  Workspace scales with the tile sizes, never with n*n*d or n*n*d*d.
  """
  inputs.validate(queries, centers, alpha, beta, kernel, length_scale,
                  query_tile, center_tile)
  values = queries.new_zeros(len(queries))
  gradients = torch.zeros_like(queries)
  for i in range(0, len(queries), query_tile):
    rows = slice(i, i + query_tile)
    for j in range(0, len(centers), center_tile):
      cols = slice(j, j + center_tile)
      v, g = tile.evaluate_tile(queries[rows], centers[cols], alpha[cols],
                                beta[cols], kernel, length_scale)
      values[rows].add_(v)
      gradients[rows].add_(g)
  return values, gradients
