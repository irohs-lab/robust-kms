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
  for index, item in factors["pending"].items():
    ql, singular, qrt = torch.linalg.svd(scale * item["core"], full_matrices=False)
    if singular.numel() == 0:
      u[index].zero_()
      v[index].zero_()
    else:
      u[index] = (item["left"] @ ql[:, 0]) * singular[0].sqrt()
      v[index] = (item["right"] @ qrt[0]) * singular[0].sqrt()
  return {"u": u, "v": v, "scale": 1.0, "pending": {}}


@torch.no_grad()
def frobenius_rank_error(factors):
  """Return the Frobenius norm discarded by independent rank-one SVDs."""
  squared = factors["u"].new_zeros(())
  for item in factors["pending"].values():
    singular = torch.linalg.svdvals(item["core"])
    squared += singular[1:].square().sum()
  return squared.sqrt() * abs(factors["scale"])
