import math
import torch


def validate(queries, centers, alpha, factors, kernel, length_scale,
             query_tile, center_tile, gradients):
  if kernel not in ("gaussian", "matern52"):
    raise ValueError("kernel must be 'gaussian' or 'matern52'")
  if not math.isfinite(length_scale) or length_scale <= 0:
    raise ValueError("length_scale must be finite and positive")
  if any(type(size) is not int or size <= 0
         for size in (query_tile, center_tile)):
    raise ValueError("tile sizes must be positive integers")
  if type(gradients) is not bool:
    raise ValueError("gradients must be a boolean")
  if centers.ndim != 2 or centers.shape[1] == 0:
    raise ValueError("centers must have shape (n, d), with d > 0")
  if queries.ndim != 2 or queries.shape[1] != centers.shape[1]:
    raise ValueError("queries must have shape (m, d)")
  if alpha.ndim != 2 or alpha.shape[0] != len(centers) or alpha.shape[1] == 0:
    raise ValueError("alpha must have shape (n, C), with C > 0")
  if centers.dtype not in (torch.float32, torch.float64):
    raise ValueError("use float32 or float64 tensors")
  tensors = [queries, alpha]
  if factors is not None:
    if not isinstance(factors, dict) or not {"u", "v", "scale", "pending"} <= factors.keys():
      raise ValueError("factors must contain u, v, scale, and pending")
    if factors["u"].shape != centers.shape or factors["v"].shape != alpha.shape:
      raise ValueError("factor u and v must have shapes (n, d) and (n, C)")
    if not math.isfinite(factors["scale"]) or not isinstance(factors["pending"], dict):
      raise ValueError("factor scale must be finite and pending must be a dictionary")
    tensors.extend((factors["u"], factors["v"]))
  for tensor in tensors:
    if tensor.dtype != centers.dtype or tensor.device != centers.device:
      raise ValueError("all tensors must share a dtype and device")
