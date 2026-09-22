import torch
import torch.nn.functional
import robust_kernels as rk
import tests.reference as reference


def test_objective_includes_squared_rkhs_norm_and_cross_term():
  generator = torch.Generator().manual_seed(2)
  x = torch.randn(4, 2, generator=generator, dtype=torch.float64)
  y = torch.tensor([0., 1., 1., 0.], dtype=torch.float64)
  theta = torch.randn(12, generator=generator, dtype=torch.float64)
  for kernel in ("gaussian", "matern52"):
    matrix = reference.dense_blocks(x, x, kernel, 0.9)
    torch.testing.assert_close(matrix, matrix.T)
    evaluated = matrix @ theta
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
      evaluated[:4], y, reduction="sum")
    expected += 0.3 * evaluated[4:].abs().sum() + 0.2 * (theta @ evaluated)
    actual = rk.objective(x, y, theta[:4], theta[4:].reshape(4, 2),
                          rho=0.3, lam=0.4, kernel=kernel, length_scale=0.9,
                          query_tile=3, center_tile=2)
    torch.testing.assert_close(actual["total"], expected)
    torch.testing.assert_close(actual["norm_squared"], theta @ evaluated)
