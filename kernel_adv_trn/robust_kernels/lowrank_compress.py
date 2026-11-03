import torch


@torch.no_grad()
def rank_one(factors):
  """Euclidean coefficient compression; not the coupled RKHS projection."""
  left, right, scale = factors["u"], factors["v"], factors["scale"]
  ln, rn = torch.linalg.vector_norm(left, dim=1), torch.linalg.vector_norm(right, dim=1)
  size = (abs(scale) * ln * rn).sqrt()
  tiny = torch.finfo(left.dtype).tiny
  u = left * (size / ln.clamp_min(tiny))[:, None]
  v = right * (size / rn.clamp_min(tiny))[:, None] * (1 if scale >= 0 else -1)
  groups = {}
  for index, item in factors["pending"].items():
    groups.setdefault(tuple(item["core"].shape), []).append((index, item))
  for shape, entries in groups.items():
    # Bound scratch storage while batching the tiny SVDs and factor products.
    for start in range(0, len(entries), 4096):
      selected = entries[start:start + 4096]
      indices = torch.tensor([i for i, _ in selected], device=left.device)
      if min(shape) == 0:
        u[indices] = 0
        v[indices] = 0
        continue
      cores = torch.stack([item["core"] for _, item in selected]) * scale
      ql, singular, qrt = torch.linalg.svd(cores, full_matrices=False)
      roots = singular[:, :1].sqrt()
      left_batch = torch.stack([item["left"] for _, item in selected])
      u[indices] = torch.bmm(left_batch, ql[:, :, :1]).squeeze(2) * roots
      right_batch = torch.stack([item["right"] for _, item in selected])
      v[indices] = torch.bmm(right_batch, qrt[:, :1, :].transpose(1, 2)).squeeze(2) * roots
  return {"u": u, "v": v, "scale": 1.0, "pending": {}}


@torch.no_grad()
def frobenius_rank_error(factors):
  """Return the Frobenius norm discarded by independent rank-one SVDs."""
  squared = factors["u"].new_zeros(())
  for item in factors["pending"].values():
    singular = torch.linalg.svdvals(item["core"])
    squared += singular[1:].square().sum()
  return squared.sqrt() * abs(factors["scale"])
