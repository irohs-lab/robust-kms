import torch


def block(factors, index):
  """Return left/core/right with physical scaling included in core."""
  if index in factors["pending"]:
    item = factors["pending"][index]
    return item["left"], factors["scale"] * item["core"], item["right"]
  left, right = factors["u"][index], factors["v"][index]
  nl, nr = torch.linalg.vector_norm(left), torch.linalg.vector_norm(right)
  if nl == 0 or nr == 0:
    return left[:, None][:, :0], left.new_zeros((0, 0)), right[:, None][:, :0]
  core = (factors["scale"] * nl * nr).reshape(1, 1)
  return left[:, None] / nl, core, right[:, None] / nr


def column(factors, start, stop, output):
  """Reconstruct one output's derivative weights for a slice of centers."""
  result = factors["u"][start:stop] * factors["v"][start:stop, output, None]
  pending = factors["pending"]
  indices = range(start, stop) if stop - start < len(pending) else pending
  for index in indices:
    if start <= index < stop and index in pending:
      item = pending[index]
      result[index - start] = item["left"] @ (item["core"] @ item["right"][output])
  return result * factors["scale"]
