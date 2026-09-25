import torch
import robust_kernels.lowrank_factors as factors
import robust_kernels.rkhs_projection as projection
import tests.reference as reference


def _state(centers, alpha, beta):
  result = {"alpha": alpha.clone(), "factors": factors.initialize(centers, alpha.shape[1])}
  basis = torch.eye(alpha.shape[1], dtype=centers.dtype, device=centers.device)
  for i in range(len(centers)):
    for c in range(alpha.shape[1]):
      factors.add(result["factors"], i, beta[i, :, c], basis[c])
  return result


def _coefficients(state):
  n, outputs = state["alpha"].shape
  beta = torch.stack([factors.column(state["factors"], 0, n, c)
                      for c in range(outputs)], dim=-1)
  return torch.cat((state["alpha"], beta.reshape(-1, outputs))), beta


def test_rkhs_projection_preserves_values_and_matches_dense_metric():
  x = torch.tensor([[-0.8, 0.1], [0.35, -0.9], [1.0, 0.7]], dtype=torch.float64)
  generator = torch.Generator().manual_seed(38)
  alpha = torch.randn(3, 2, generator=generator, dtype=x.dtype)
  beta = torch.randn(3, 2, 2, generator=generator, dtype=x.dtype)
  target = torch.cat((alpha, beta.reshape(6, 2)))
  u, singular, vh = torch.linalg.svd(beta, full_matrices=False)
  svd_beta = (u[:, :, :1] * singular[:, None, :1]) @ vh[:, :1, :]
  for kernel in ("gaussian", "matern52"):
    gram = reference.dense_blocks(x, x, kernel, 1.1)
    db = (svd_beta - beta).reshape(6, 2)
    da = -torch.linalg.solve(gram[:3, :3], gram[:3, 3:] @ db)
    initial = torch.cat((da, db))
    initial_error = (initial * (gram @ initial)).sum().item()
    state = _state(x, alpha, beta)
    info = projection.project(x, state, max_steps=20, kernel=kernel, length_scale=1.1,
                              solve_rtol=1e-12, solve_atol=1e-14, query_tile=2, center_tile=2)
    actual, compressed = _coefficients(state)
    residual = actual - target
    expected_error = (residual * (gram @ residual)).sum().item()
    assert abs(info["error_squared"] - expected_error) < 2e-9
    assert abs(info["initial_error_squared"] - initial_error) < 2e-9
    assert expected_error <= initial_error + 2e-9
    torch.testing.assert_close(gram[:3] @ actual, gram[:3] @ target, atol=2e-9, rtol=0)
    assert info["max_value_error"] < 2e-9
    assert not state["factors"]["pending"]
    assert torch.linalg.svdvals(compressed)[:, 1:].abs().max() < 1e-12


def test_rkhs_projection_leaves_representable_targets_unchanged():
  generator = torch.Generator().manual_seed(92)
  x = torch.randn(3, 2, generator=generator, dtype=torch.float64)
  for outputs in (1, 2):
    alpha = torch.randn(3, outputs, generator=generator, dtype=x.dtype)
    left = torch.randn(3, 2, generator=generator, dtype=x.dtype)
    right = torch.randn(3, outputs, generator=generator, dtype=x.dtype)
    beta = left[:, :, None] * right[:, None, :]
    state = _state(x, alpha, beta)
    target, _ = _coefficients(state)
    info = projection.project(x, state, solve_rtol=1e-12, solve_atol=1e-14)
    actual, _ = _coefficients(state)
    torch.testing.assert_close(actual, target, atol=2e-9, rtol=2e-9)
    assert abs(info["error_squared"]) < 1e-14
    assert not state["factors"]["pending"]


def test_one_center_projection_attains_known_global_optimum():
  x = torch.zeros(1, 2, dtype=torch.float64)
  alpha = torch.tensor([[0.4, -0.2]], dtype=x.dtype)
  beta = torch.tensor([[[3.0, 0.0], [0.0, 1.0]]], dtype=x.dtype)
  state = _state(x, alpha, beta)
  info = projection.project(x, state, length_scale=1.7,
                            solve_rtol=1e-12, solve_atol=1e-14)
  _, compressed = _coefficients(state)
  torch.testing.assert_close(compressed, torch.tensor([[[3.0, 0.0], [0.0, 0.0]]],
                                                     dtype=x.dtype), atol=1e-12, rtol=0)
  torch.testing.assert_close(state["alpha"], alpha, atol=1e-12, rtol=0)
  assert abs(info["error_squared"] - 1 / 1.7 ** 2) < 1e-12


def test_coupled_projection_can_activate_a_zero_block():
  x = torch.tensor([[-0.8, 0.1], [0.35, -0.9], [1.0, 0.7]], dtype=torch.float64)
  beta = torch.tensor([[[0., 0.], [0., 0.]], [[2., 1.], [1., 2.]],
                       [[1., -1.], [2., 1.]]], dtype=x.dtype)
  state = _state(x, torch.zeros(3, 2, dtype=x.dtype), beta)
  info = projection.project(x, state, max_steps=20, solve_rtol=1e-12, solve_atol=1e-14)
  _, compressed = _coefficients(state)
  assert compressed[0].norm().item() > 1e-3
  assert info["error_squared"] < info["initial_error_squared"] - 1e-3
