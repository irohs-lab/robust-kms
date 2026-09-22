import math
import torch


def validate(centers, labels, iterations, batch_size, step_size, decay):
  if centers.ndim != 2 or min(centers.shape) == 0:
    raise ValueError("centers must be a nonempty (n, d) tensor")
  if labels.shape != (len(centers),) or labels.device != centers.device:
    raise ValueError("labels must have shape (n,) on centers.device")
  if not torch.isfinite(centers).all():
    raise ValueError("centers must be finite")
  if not torch.all((labels == 0) | (labels == 1)):
    raise ValueError("logistic labels must be 0 or 1")
  if type(iterations) is not int or iterations <= 0:
    raise ValueError("iterations must be a positive integer")
  if type(batch_size) is not int or not 0 < batch_size <= len(centers):
    raise ValueError("batch_size must be an integer between 1 and n")
  if not math.isfinite(step_size) or step_size <= 0:
    raise ValueError("step_size must be finite and positive")
  if not math.isfinite(decay) or not 0.5 < decay <= 1:
    raise ValueError("decay must satisfy 0.5 < decay <= 1")
