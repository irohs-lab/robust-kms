import unittest
import torch
import robust_kernels as rk


def test_gpu_zero_matches_cpu():
  if not torch.cuda.is_available():
    raise unittest.SkipTest("CUDA is not available")
  generator = torch.Generator().manual_seed(12)
  x = torch.randn(9, 4, generator=generator, dtype=torch.float64)
  y = torch.arange(9) % 2
  for kernel in ("gaussian", "matern52"):
    cpu, gpu = rk.initialize(x), rk.initialize(x.cuda(0))
    for m in range(12):
      batch = torch.randperm(9, generator=generator)[:4]
      options = dict(rho=0.1, lam=1., eta=0.03 / (m + 1) ** 0.75,
                     kernel=kernel, center_tile=3, query_tile=2)
      rk.step(x, y, cpu, batch, **options)
      rk.step(x.cuda(0), y.cuda(0), gpu, batch.cuda(0), **options)
    for key in ("alpha", "beta", "delta", "average_alpha", "average_beta"):
      torch.testing.assert_close(gpu[key].cpu(), cpu[key], atol=1e-11, rtol=1e-11)


def test_float32_gpu_evaluator_against_float64():
  if not torch.cuda.is_available():
    raise unittest.SkipTest("CUDA is not available")
  generator = torch.Generator().manual_seed(101)
  x = torch.randn(23, 11, generator=generator, dtype=torch.float64)
  q = torch.cat((x[:1], x[4:10] + 0.3))
  a = torch.randn(23, generator=generator, dtype=torch.float64)
  b = torch.randn(23, 11, generator=generator, dtype=torch.float64)
  for kernel in ("gaussian", "matern52"):
    expected = rk.kernel_eval_and_grad(q, x, a, b, kernel=kernel, length_scale=3)
    actual = rk.kernel_eval_and_grad(
      q.float().cuda(0), x.float().cuda(0), a.float().cuda(0), b.float().cuda(0),
      kernel=kernel, length_scale=3, query_tile=3, center_tile=7)
    for output, reference in zip(actual, expected):
      torch.testing.assert_close(output.cpu().double(), reference,
                                 atol=2e-5, rtol=2e-5)
