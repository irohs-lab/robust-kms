import torch
import torchkernels.linalg.eigh as eigensystem


def initialize(kernel, x, samples=1024, rank=100, batch_size=128):
  """EigenPro2 preconditioner, matching torchkernels.solvers.eigenpro2."""
  indices = torch.randperm(len(x), device=x.device)[:samples]
  vectors, values, next_value, beta = eigensystem.top_eigensystem(
    kernel, x[indices], rank, method="scipy.linalg.eigh")
  vectors.mul_(((1 - next_value / values) / values).sqrt())
  extended = kernel(x, x[indices]) @ vectors
  critical = int(beta * samples / next_value) + 1
  return dict(indices=indices, vectors=vectors, extended=extended,
    critical=critical, beta=beta, next_value=next_value,
    samples=samples, batch_size=batch_size)


@torch.no_grad()
def run(alpha, gram, targets, state):
  for rows in torch.randperm(len(alpha), device=alpha.device).split(state["batch_size"]):
    size = len(rows)
    rate = (1 / state["beta"] if size < state["critical"] else
      2 / (state["beta"] + (size - 1) * state["next_value"] / state["samples"]))
    residual = gram[rows] @ alpha - targets[rows]
    alpha[rows] -= rate * residual
    alpha[state["indices"]] += rate * state["vectors"] @ (
      state["extended"][rows].T @ residual)
