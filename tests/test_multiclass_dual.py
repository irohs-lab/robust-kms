import torch
import robust_kernels.multiclass_dual as dual


def test_multiclass_dual_attains_support_function_and_is_a_subgradient():
  generator = torch.Generator().manual_seed(917)
  # Internal storage is (batch, d, c), with unequal dimensions to catch transposes.
  jacobian = torch.randn(6, 5, 3, generator=generator, dtype=torch.float64)
  jacobian[0].zero_()
  jacobian[1, :, 1] = -jacobian[1, :, 0]
  jacobian[1, :, 2].zero_()
  other = torch.randn(6, 5, 3, generator=generator, dtype=torch.float64)
  devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
  for device in devices:
    j, z = jacobian.to(device), other.to(device)
    expected = torch.linalg.matrix_norm(j, ord=1)
    for rho in (0., .37):
      left, right, penalty = dual.maximize(j, rho)
      delta = left[:, :, None] * right[:, None, :]
      torch.testing.assert_close(penalty, expected)
      torch.testing.assert_close((delta * j).sum((1, 2)), rho * expected)
      assert torch.all(delta.abs().amax(1).sum(1) <= rho + 1e-14)
      assert torch.linalg.svdvals(delta)[:, 1:].abs().max() < 1e-12
      lower = rho * expected + (delta * (z - j)).sum((1, 2))
      assert torch.all(lower <= rho * torch.linalg.matrix_norm(z, ord=1) + 1e-12)
      assert torch.count_nonzero(delta[0]) == 0
      assert right[1].argmax().item() == 0
