import math
import torch


@torch.no_grad()
def scale(factors, scalar):
  """Scale all blocks lazily, flushing extreme scales to avoid division loss."""
  scalar = float(scalar)
  if not math.isfinite(scalar):
    raise ValueError("scale must be finite")
  if scalar == 0:
    factors["u"].zero_()
    factors["v"].zero_()
    factors["pending"].clear()
    factors["scale"] = 1.0
    return factors
  updated = factors["scale"] * scalar
  if not math.isfinite(updated):
    raise ValueError("combined scale must be finite")
  factors["scale"] = updated
  if abs(updated) < 1e-8 or abs(updated) > 1e8:
    factors["u"].mul_(updated)
    for item in factors["pending"].values():
      item["core"].mul_(updated)
    factors["scale"] = 1.0
  return factors
