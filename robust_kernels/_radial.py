import torch
import torchkernels.linalg


def radial_terms(queries, centers, kernel, length_scale):
  """Return K, a, b with grad_2 K = a*delta, Hess_12 K = a*I-b*dd'."""
  distances = torchkernels.linalg.euclidean(queries, centers, squared=True)
  inv_scale2 = length_scale ** -2
  if kernel == "gaussian":
    values = distances.mul_(-0.5 * inv_scale2).exp_()
    a = values * inv_scale2
    return values, a, a * inv_scale2
  s = distances.sqrt_().mul_(5 ** 0.5 / length_scale)
  exponential = torch.exp(-s)
  values = (1 + s + s.square() / 3) * exponential
  a = (5 / 3 * inv_scale2) * (1 + s) * exponential
  b = (25 / 3 * inv_scale2 ** 2) * exponential
  return values, a, b
