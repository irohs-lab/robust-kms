import math
import torch


def validate(centers, labels, state, batch, rho, lam, eta):
  if not math.isfinite(rho) or rho < 0:
    raise ValueError("rho must be finite and nonnegative")
  if not math.isfinite(lam) or lam <= 0:
    raise ValueError("lam must be finite and positive")
  if not math.isfinite(eta) or not 0 < eta <= 1 / lam:
    raise ValueError("eta must satisfy 0 < eta <= 1 / lam")
  if labels.shape != (len(centers),) or labels.device != centers.device:
    raise ValueError("labels must have shape (n,) on centers.device")
  if batch.ndim != 1 or batch.dtype != torch.long or batch.numel() == 0:
    raise ValueError("batch must be a nonempty vector of int64 indices")
  if batch.device != centers.device:
    raise ValueError("batch and centers must share a device")
  if torch.any((batch < 0) | (batch >= len(centers))):
    raise ValueError("batch indices are out of bounds")
  if torch.unique(batch).numel() != batch.numel():
    raise ValueError("batch must be sampled without replacement")
  if state["delta"].shape != centers.shape:
    raise ValueError("delta must have shape (n, d)")
  if (state["delta"].device != centers.device
      or state["delta"].dtype != centers.dtype):
    raise ValueError("delta must share centers' dtype and device")
