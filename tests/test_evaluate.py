import torch
import robust_kernels as rk
import tests.reference as reference


def test_rectangular_tiles_against_autograd():
  generator = torch.Generator().manual_seed(7)
  for kernel in ("gaussian", "matern52"):
    for d in (1, 3):
      centers = torch.randn(5, d, generator=generator, dtype=torch.float64)
      queries = torch.cat((centers[:1], centers[2:4] + 0.2))
      alpha = torch.randn(5, generator=generator, dtype=torch.float64)
      beta = torch.randn(5, d, generator=generator, dtype=torch.float64)
      matrix = reference.dense_blocks(queries, centers, kernel, 1.7)
      expected = matrix @ torch.cat((alpha, beta.flatten()))
      for query_tile, center_tile in ((1, 2), (2, 3), (10, 10)):
        values, gradients = rk.kernel_eval_and_grad(
          queries, centers, alpha, beta, kernel=kernel, length_scale=1.7,
          query_tile=query_tile, center_tile=center_tile)
        actual = torch.cat((values, gradients.flatten()))
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        assert not values.requires_grad and not gradients.requires_grad


def test_coincident_diagonal_and_empty_queries():
  centers = torch.zeros(2, 3, dtype=torch.float64)
  alpha, beta = torch.ones(2, dtype=torch.float64), torch.ones_like(centers)
  for kernel, curvature in (("gaussian", 1.0), ("matern52", 5 / 3)):
    values, gradients = rk.kernel_eval_and_grad(
      centers, centers, alpha, beta, kernel=kernel, length_scale=2)
    torch.testing.assert_close(values, torch.full_like(values, 2))
    torch.testing.assert_close(gradients, torch.full_like(beta, curvature / 2))
    values, gradients = rk.kernel_eval_and_grad(centers[:0], centers, alpha, beta)
    assert values.shape == (0,) and gradients.shape == (0, 3)
