import torch
import torchkernels.linalg


@torch.no_grad()
def matrix(x, z, length_scale):
  s = torchkernels.linalg.euclidean(x, z, squared=True)
  s.sqrt_().mul_(5 ** .5 / length_scale)
  return (1 + s + s.square() / 3) * torch.exp(-s)
