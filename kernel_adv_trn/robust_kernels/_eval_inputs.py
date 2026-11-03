import math
import torch


def validate(queries, centers, alpha, beta, kernel, length_scale,
             query_tile, center_tile):
  if kernel not in ("gaussian", "matern52"):
    raise ValueError("kernel must be 'gaussian' or 'matern52'")
  if not math.isfinite(length_scale) or length_scale <= 0:
    raise ValueError("length_scale must be finite and positive")
  if any(type(size) is not int or size <= 0
         for size in (query_tile, center_tile)):
    raise ValueError("tile sizes must be positive integers")
  if centers.ndim != 2 or centers.shape[1] == 0:
    raise ValueError("centers must have shape (n, d), with d > 0")
  if queries.ndim != 2 or queries.shape[1] != centers.shape[1]:
    raise ValueError("queries must have shape (b, d)")
  if alpha.shape != (len(centers),) or beta.shape != centers.shape:
    raise ValueError("alpha and beta must have shapes (n,) and (n, d)")
  if centers.dtype not in (torch.float32, torch.float64):
    raise ValueError("use float32 or float64 tensors")
  for tensor in (queries, alpha, beta):
    if tensor.dtype != centers.dtype or tensor.device != centers.device:
      raise ValueError("all tensors must share a dtype and device")
