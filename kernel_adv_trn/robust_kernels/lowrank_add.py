import torch
import robust_kernels.lowrank_block as blocks
import robust_kernels.lowrank_qr as qr


@torch.no_grad()
def add(factors, index, left, right):
  """Add a physical outer product, updating only the touched block's QR."""
  if not isinstance(index, int) or not 0 <= index < len(factors["u"]):
    raise ValueError("index must identify one center")
  for value, base in ((left, factors["u"]), (right, factors["v"])):
    if value.shape != base.shape[1:] or value.device != base.device or value.dtype != base.dtype:
      raise ValueError("update vectors must match coefficient dimensions, dtype, and device")
    if not torch.isfinite(value).all():
      raise ValueError("update vectors must be finite")
  if not torch.any(left) or not torch.any(right):
    return factors
  old_left, core, old_right = blocks.block(factors, index)
  core = core / factors["scale"]
  new_left, lc = qr.append(old_left, left / factors["scale"])
  new_right, rc = qr.append(old_right, right)
  updated = lc[:, None] * rc[None, :]
  updated[:core.shape[0], :core.shape[1]] += core
  if max(updated.shape) > min(left.numel(), right.numel()):
    u, singular, vh = torch.linalg.svd(updated, full_matrices=False)
    new_left, new_right = new_left @ u, new_right @ vh.T
    updated = torch.diag(singular)
  factors["pending"][index] = {"left": new_left, "core": updated, "right": new_right}
  return factors
