import torch


def initialize(centers, outputs):
  """Create zero derivative coefficients without an (n, d, C) allocation."""
  if centers.ndim != 2 or min(centers.shape) == 0:
    raise ValueError("centers must be nonempty and two-dimensional")
  if centers.dtype not in (torch.float32, torch.float64):
    raise ValueError("centers must use float32 or float64")
  if not isinstance(outputs, int) or isinstance(outputs, bool) or outputs < 1:
    raise ValueError("outputs must be a positive integer")
  return {"u": torch.zeros_like(centers),
          "v": centers.new_zeros((len(centers), outputs)),
          "scale": 1.0, "pending": {}}


def clone(factors):
  """Copy coefficients and pending factors without sharing tensor storage."""
  return {"u": factors["u"].clone(), "v": factors["v"].clone(),
          "scale": factors["scale"],
          "pending": {i: {key: value.clone() for key, value in block.items()}
                      for i, block in factors["pending"].items()}}
