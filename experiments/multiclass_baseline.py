"""Unregularized cross-entropy kernel SGD using KLR models and derivatives."""
import math
import torch
from torch.func import jacrev, vmap


@torch.no_grad()
def initialize(x, y, *, kernel, outputs=10, batch_size=128, storage="dense",
               seed=42, kernel_batch_size=2048, eta=.01):
  """Prepare a KLR model and one reusable training kernel operator.

  The caller supplies the shared comparison kernel. The update is literally
  alpha[rows] -= eta * CE'(logits, labels), with no batch normalization factor.
  Validation-based early stopping belongs to the caller.
  """
  from klr.models import KernelModel
  from klr.solver import multiclass_loss
  from klr.utils.kernel_operator import kernel_operator

  if x.ndim != 2 or min(x.shape) < 1 or x.dtype not in (torch.float32, torch.float64):
    raise ValueError("x must be a nonempty real floating-point sample matrix")
  if type(outputs) is not int or outputs < 2:
    raise ValueError("outputs must be an integer >= 2")
  if y.shape != (len(x),) or y.dtype != torch.long or y.device != x.device:
    raise ValueError("y must be a long vector on the sample device")
  if not torch.isfinite(x).all() or not ((y >= 0) & (y < outputs)).all():
    raise ValueError("samples must be finite and labels must lie in [0, outputs)")
  for name, value in (("batch_size", batch_size), ("kernel_batch_size", kernel_batch_size)):
    if type(value) is not int or value < 1:
      raise ValueError(f"{name} must be a positive integer")
  if type(seed) is not int:
    raise ValueError("seed must be an integer")
  if not math.isfinite(eta) or eta <= 0:
    raise ValueError("eta must be finite and positive")
  batch_size = min(batch_size, len(x))
  model = KernelModel(kernel, x, num_outputs=outputs, fit_intercept=False,
                      batch_size=kernel_batch_size)
  operator = kernel_operator(kernel, x, storage=storage, batch_size=kernel_batch_size)
  return dict(model=model, operator=operator, labels=y, outputs=outputs,
    derivative=vmap(jacrev(multiclass_loss)), eta=eta,
    generator=torch.Generator(device=x.device).manual_seed(seed),
    batch_size=batch_size, storage=storage, iterations=0, epochs=0)


@torch.no_grad()
def step(state, rows, eta=None):
  """Update only the sampled coefficient rows using their old-logit CE gradient."""
  model = state["model"]
  eta = state["eta"] if eta is None else eta
  if (rows.ndim != 1 or not len(rows) or rows.dtype != torch.long
      or rows.device != model.centers.device):
    raise ValueError("rows must be a nonempty long vector on the sample device")
  if rows.min() < 0 or rows.max() >= len(model.centers) or len(rows.unique()) != len(rows):
    raise ValueError("rows must contain distinct valid sample indices")
  if not math.isfinite(eta) or eta <= 0:
    raise ValueError("eta must be finite and positive")
  logits = state["operator"][rows, :] @ model.weights
  targets = state["labels"][rows]
  residual = state["derivative"](logits, targets)
  model.weights[rows] -= eta * residual
  state["iterations"] += 1
  return dict(iteration=state["iterations"], eta=eta,
    cross_entropy=torch.nn.functional.cross_entropy(logits, targets).item(),
    accuracy=(logits.argmax(1) == targets).float().mean().item())


def epoch(state, eta=None):
  """Visit every sample once; use the same per-row rate for the shorter tail."""
  model = state["model"]
  order = torch.randperm(len(model.centers), device=model.centers.device,
                         generator=state["generator"])
  events = [step(state, rows, eta) for rows in order.split(state["batch_size"])]
  state["epochs"] += 1
  return events
