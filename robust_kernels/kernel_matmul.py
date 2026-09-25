import torch
import robust_kernels._radial as radial


@torch.no_grad()
def apply(centers, weights, *, kernel="gaussian", length_scale=1.,
          query_tile=128, center_tile=1024):
  """Apply the value Gram matrix without storing it."""
  result = torch.zeros_like(weights)
  for i in range(0, len(centers), query_tile):
    rows = slice(i, i + query_tile)
    for j in range(0, len(centers), center_tile):
      cols = slice(j, j + center_tile)
      values, _, _ = radial.radial_terms(
        centers[rows], centers[cols], kernel, length_scale)
      result[rows].addmm_(values, weights[cols])
  return result
