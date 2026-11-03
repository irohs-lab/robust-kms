import torch


def initialize(centers, *, average=True):
  """Allocate zero coefficients and feasible dual blocks on centers.device."""
  if centers.ndim != 2 or min(centers.shape) == 0:
    raise ValueError("centers must be a nonempty (n, d) tensor")
  if centers.dtype not in (torch.float32, torch.float64):
    raise ValueError("centers must use float32 or float64")
  state = {
    "alpha": centers.new_zeros(len(centers)),
    "beta": torch.zeros_like(centers),
    "delta": torch.zeros_like(centers),
    "iterations": 0,
    "weight_sum": 0.0,
  }
  if average:
    state["average_alpha"] = torch.zeros_like(state["alpha"])
    state["average_beta"] = torch.zeros_like(state["beta"])
  return state
