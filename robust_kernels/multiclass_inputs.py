import math
import torch


def validate(centers, labels, state, batch, rho, lam, eta):
  if any(not math.isfinite(value) or value < 0 for value in (rho, lam, eta)):
    raise ValueError("rho and lam must be nonnegative; eta must be positive")
  if eta == 0 or eta * lam > 1:
    raise ValueError("require eta > 0 and eta*lam <= 1")
  if labels.shape != (len(centers),) or labels.device != centers.device:
    raise ValueError("labels must have shape (n,) on centers.device")
  if labels.dtype not in (torch.int32, torch.int64):
    raise ValueError("labels must be integer class indices")
  if batch.ndim != 1 or not batch.numel() or batch.dtype != torch.int64:
    raise ValueError("batch must be a nonempty int64 index vector")
  if batch.device != centers.device or (batch < 0).any() or (batch >= len(centers)).any():
    raise ValueError("batch indices must be valid and on centers.device")
  if batch.unique().numel() != batch.numel():
    raise ValueError("batch indices must be distinct")
  if (labels < 0).any() or (labels >= state["alpha"].shape[1]).any():
    raise ValueError("labels must be valid output indices")
