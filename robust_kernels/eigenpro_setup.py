import torch
import torchkernels.linalg.eigh as eigensystem
import robust_kernels._radial as radial
import robust_kernels.multioutput_evaluate as evaluation


def prepare(centers, options, *, samples=1024, rank=100, batch_size=128, seed=0):
  """Cache the EigenPro2 preconditioner once for all solves in a projection."""
  if any(type(v) is not int for v in (samples, rank, batch_size, seed)):
    raise ValueError("EigenPro2 sizes and seed must be integers")
  if samples < 1 or rank < 0 or batch_size < 1:
    raise ValueError("EigenPro2 samples/batch_size must be positive and rank nonnegative")
  samples, batch_size = min(samples, len(centers)), min(batch_size, len(centers))
  rank = min(rank, samples - 1)
  generator = torch.Generator(device=centers.device).manual_seed(seed)
  indices = torch.randperm(len(centers), device=centers.device,
                           generator=generator)[:samples]
  kernel = lambda x, z: radial.radial_terms(
    x, z, options["kernel"], options["length_scale"])[0]
  vectors, values, next_value, beta = eigensystem.top_eigensystem(
    kernel, centers[indices], rank, method="scipy.linalg.eigh")
  if not torch.isfinite(next_value) or next_value <= 0 or beta <= 0:
    raise RuntimeError("EigenPro2 sample spectrum must be positive; reduce rank or use float64")
  vectors.mul_(((1 - next_value / values).clamp_min(0) / values).sqrt())
  if rank:
    extended, _ = evaluation.kernel_eval_and_grad(
      centers, centers[indices], vectors, gradients=False, **options)
  else:
    extended = centers.new_zeros((len(centers), 0))
  return dict(centers=centers, options=options, indices=indices, vectors=vectors,
    extended=extended, beta=beta, next_value=next_value, samples=samples,
    rank=rank, batch_size=batch_size, critical=int(beta * samples / next_value) + 1,
    generator=generator, warm_start=None, solve_calls=0, total_epochs=0)
