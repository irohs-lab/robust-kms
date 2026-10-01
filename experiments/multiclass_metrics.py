"""Common clean and first-order adversarial metrics for multiclass models."""

import math
import torch
import torch.nn.functional as functional
from robust_kernels import multioutput_eval_and_grad


def multiclass_logits(inputs, centers, alpha, factors=None, **options):
  """Evaluate logits with exact first input derivatives and bounded tile memory.

  The attached local linear expression supports first-order attacks. It does
  not provide higher derivatives or gradients of the model coefficients.
  Inference omits the analytic Jacobian entirely.
  """
  flat = inputs.flatten(1)
  gradients = torch.is_grad_enabled() and flat.requires_grad
  values, jacobian = multioutput_eval_and_grad(
    flat.detach(), centers, alpha, factors, gradients=gradients, **options)
  if gradients:
    values = values + torch.einsum("md,mdc->mc", flat - flat.detach(), jacobian)
  return values


def _validate(x, y, batch_size):
  if not len(x) or y.ndim != 1 or len(x) != len(y):
    raise ValueError("x and y must contain the same nonzero number of examples")
  if y.dtype != torch.long or x.device != y.device:
    raise ValueError("labels must be int64 and on the same device as inputs")
  if type(batch_size) is not int or batch_size < 1:
    raise ValueError("batch_size must be a positive integer")


def _check_logits(logits, labels):
  if logits.ndim != 2 or len(logits) != len(labels) or logits.shape[1] < 2:
    raise ValueError("predict must return a matrix of multiclass logits")
  if not torch.isfinite(logits).all():
    raise RuntimeError("model produced nonfinite logits")


def _summary(correct, loss, counts, correct_counts, samples):
  return {"accuracy": correct / samples,
          "cross_entropy": loss / samples,
          "per_class_accuracy": [good / total if total else None
                                 for good, total in zip(correct_counts, counts)],
          "samples": samples}


@torch.no_grad()
def evaluate(predict, x, y, *, batch_size=256):
  """Measure accuracy and mean cross-entropy with integer class labels."""
  _validate(x, y, batch_size)
  correct, loss, counts, correct_counts = 0, 0., None, None
  for start in range(0, len(x), batch_size):
    labels = y[start:start + batch_size]
    logits = predict(x[start:start + batch_size])
    _check_logits(logits, labels)
    matches = logits.argmax(1) == labels
    correct += matches.sum().item()
    loss += functional.cross_entropy(logits, labels, reduction="sum").item()
    batch_counts = torch.bincount(labels, minlength=logits.shape[1])
    batch_correct = torch.bincount(labels[matches], minlength=logits.shape[1])
    counts = batch_counts if counts is None else counts + batch_counts
    correct_counts = (batch_correct if correct_counts is None
                      else correct_counts + batch_correct)
  return _summary(correct, loss, counts.tolist(), correct_counts.tolist(), len(x))


def pgd(predict, x, y, *, epsilon=8 / 255, steps=20, step_size=2 / 255,
        batch_size=64, seed=42):
  """Single-restart random-start Linf PGD on the caller's fixed test subset.

  Attack candidates include the clean image, random start, and every iterate.
  A sample is robust only if all these candidates remain correctly classified;
  an already incorrect clean example can never improve the reported score.
  The function validates the perturbation and pixel bounds before reporting.
  """
  _validate(x, y, batch_size)
  if not math.isfinite(epsilon) or epsilon < 0:
    raise ValueError("epsilon must be finite and nonnegative")
  if not math.isfinite(step_size) or step_size <= 0:
    raise ValueError("step_size must be finite and positive")
  if type(steps) is not int or steps < 1:
    raise ValueError("steps must be a positive integer")
  if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
    raise ValueError("PGD inputs must contain finite raw pixels in [0, 1]")
  generator = torch.Generator(device="cpu").manual_seed(seed)
  correct, clean_correct, loss = 0, 0, 0.
  counts, correct_counts, max_linf = None, None, 0.
  for start in range(0, len(x), batch_size):
    clean = x[start:start + batch_size].detach()
    labels = y[start:start + batch_size]
    with torch.no_grad():
      clean_logits = predict(clean)
      _check_logits(clean_logits, labels)
      robust = clean_logits.argmax(1) == labels
      clean_correct += robust.sum().item()
      worst_loss = functional.cross_entropy(clean_logits, labels, reduction="none")
      noise = torch.rand(clean.shape, generator=generator, dtype=clean.dtype)
      adversarial = (clean + epsilon * (2 * noise.to(clean.device) - 1)).clamp(0, 1)
      lower, upper = (clean - epsilon).clamp(0, 1), (clean + epsilon).clamp(0, 1)
    for iteration in range(steps + 1):
      adversarial = adversarial.detach().requires_grad_(iteration < steps)
      with torch.enable_grad():
        logits = predict(adversarial)
        _check_logits(logits, labels)
        losses = functional.cross_entropy(logits, labels, reduction="none")
        if iteration < steps:
          gradient, = torch.autograd.grad(losses.sum(), adversarial)
      with torch.no_grad():
        robust.logical_and_(logits.argmax(1) == labels)
        worst_loss = torch.maximum(worst_loss, losses.detach())
        max_linf = max(max_linf, (adversarial - clean).abs().max().item())
        if adversarial.min() < -1e-6 or adversarial.max() > 1 + 1e-6:
          raise RuntimeError("attack left the valid pixel range")
        if iteration < steps:
          if not torch.isfinite(gradient).all():
            raise RuntimeError("attack encountered a nonfinite input gradient")
          adversarial = torch.maximum(lower, torch.minimum(
            upper, adversarial + step_size * gradient.sign()))
    correct += robust.sum().item()
    loss += worst_loss.sum().item()
    batch_counts = torch.bincount(labels, minlength=logits.shape[1])
    batch_correct = torch.bincount(labels[robust], minlength=logits.shape[1])
    counts = batch_counts if counts is None else counts + batch_counts
    correct_counts = (batch_correct if correct_counts is None
                      else correct_counts + batch_correct)
  if max_linf > epsilon + 1e-6:
    raise RuntimeError("attack exceeded the pixel-space Linf radius")
  result = _summary(correct, loss, counts.tolist(), correct_counts.tolist(), len(x))
  result.update(robust_accuracy=result["accuracy"],
                clean_accuracy=clean_correct / len(x), epsilon=epsilon,
                steps=steps, step_size=step_size, seed=seed, restarts=1,
                max_linf=max_linf, attack="pgd-ce-linf")
  return result
