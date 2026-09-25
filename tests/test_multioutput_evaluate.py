import torch
import tests.reference as reference
import robust_kernels.lowrank_factors as factors_ops
import robust_kernels.multioutput_evaluate as evaluate


def test_multioutput_evaluate():
  generator = torch.Generator().manual_seed(19)
  centers = torch.randn(4, 2, generator=generator, dtype=torch.float64)
  queries = torch.cat((centers[:1], torch.randn(2, 2, generator=generator,
                                              dtype=torch.float64)))
  alpha = torch.randn(4, 3, generator=generator, dtype=torch.float64)
  u = torch.randn(4, 2, generator=generator, dtype=torch.float64)
  v = torch.randn(4, 3, generator=generator, dtype=torch.float64)
  left = torch.tensor([0.6, -0.3], dtype=torch.float64)
  right = torch.tensor([0.2, 0.8, -0.4], dtype=torch.float64)
  beta = 0.7 * u[:, :, None] * v[:, None, :]
  beta[1].add_(left[:, None] * right[None, :])
  devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
  for device in devices:
    dtype = torch.float64 if device == "cpu" else torch.float32
    x, z, a = [t.to(device=device, dtype=dtype) for t in (queries, centers, alpha)]
    factors = factors_ops.initialize(z, 3)
    factors["u"].copy_(u)
    factors["v"].copy_(v)
    factors_ops.scale(factors, 0.7)
    factors_ops.add(factors, 1, left.to(z), right.to(z))
    for kernel in ("gaussian", "matern52"):
      matrix = reference.dense_blocks(queries, centers, kernel, 1.3)
      coefficients = torch.cat((alpha, beta.reshape(-1, 3)))
      expected = (matrix @ coefficients).to(z)
      values, jacobian = evaluate.kernel_eval_and_grad(
        x, z, a, factors, kernel=kernel, length_scale=1.3,
        query_tile=2, center_tile=2)
      torch.testing.assert_close(values, expected[:len(x)], atol=2e-5, rtol=2e-5)
      torch.testing.assert_close(jacobian, expected[len(x):].reshape(3, 2, 3),
                                  atol=2e-5, rtol=2e-5)
      values_only, missing = evaluate.kernel_eval_and_grad(
        x, z, a, factors, kernel=kernel, length_scale=1.3, gradients=False)
      assert missing is None
      torch.testing.assert_close(values_only, values, atol=2e-5, rtol=2e-5)
