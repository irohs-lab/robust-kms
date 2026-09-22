import torch


def scalar_kernel(x, z, kernel, length_scale):
  squared = (x - z).square().sum() / length_scale ** 2
  if kernel == "gaussian":
    return torch.exp(-squared / 2)
  if squared.detach().item() == 0:
    return 1 - 5 * squared / 6
  s = torch.sqrt(5 * squared)
  return (1 + s + s.square() / 3) * torch.exp(-s)


def dense_blocks(queries, centers, kernel, length_scale):
  n, d = centers.shape
  matrix = centers.new_zeros((len(queries) * (1 + d), n * (1 + d)))
  for i, query in enumerate(queries):
    for j, center in enumerate(centers):
      pair = torch.cat((query, center)).requires_grad_()
      fn = lambda t: scalar_kernel(t[:d], t[d:], kernel, length_scale)
      first = torch.autograd.functional.jacobian(fn, pair)
      second = torch.autograd.functional.hessian(fn, pair)
      rows = slice(len(queries) + i * d, len(queries) + (i + 1) * d)
      cols = slice(n + j * d, n + (j + 1) * d)
      matrix[i, j] = fn(pair).detach()
      matrix[i, cols] = first[d:]
      matrix[rows, j] = first[:d]
      matrix[rows, cols] = second[:d, d:]
  return matrix
