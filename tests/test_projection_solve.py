import torch
import robust_kernels as rk
import robust_kernels.eigenpro_solve as eigenpro
import robust_kernels.eigenpro_iteration as iteration
import robust_kernels.lowrank_factors as factors
import tests.reference as reference


def kernel_matrix(x, kernel, length_scale):
  return torch.stack([torch.stack([
    reference.scalar_kernel(a, b, kernel, length_scale) for b in x]) for a in x])


def test_eigenpro_solve_matches_dense_and_reuses_preconditioner():
  generator = torch.Generator().manual_seed(751)
  x = torch.randn(8, 3, dtype=torch.float64, generator=generator)
  rhs = torch.randn(8, 3, dtype=x.dtype, generator=generator)
  rhs[:, 1] = 0
  for kernel in ("gaussian", "matern52"):
    options = dict(kernel=kernel, length_scale=.9, query_tile=3, center_tile=3)
    workspace = eigenpro.prepare(x, options, samples=8, rank=7, batch_size=8, seed=9)
    vector_pointer = workspace["vectors"].data_ptr()
    extended_pointer = workspace["extended"].data_ptr()
    solution, info = eigenpro.solve(workspace, rhs, rtol=1e-10, atol=1e-12)
    matrix = kernel_matrix(x, kernel, .9)
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, rhs),
                               atol=2e-10, rtol=2e-10)
    assert info["method"] == "eigenpro2" and info["epochs"] > 0
    assert torch.all((matrix @ solution - rhs).norm(dim=0) <=
                     1e-12 + 1e-10 * rhs.norm(dim=0))
    elapsed = workspace["total_epochs"]
    repeated, info = eigenpro.solve(workspace, rhs, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(repeated, solution)
    assert info["epochs"] == 0 and workspace["total_epochs"] == elapsed
    next_rhs = rhs + .03 * torch.randn(rhs.shape, dtype=x.dtype, generator=generator)
    changed, _ = eigenpro.solve(workspace, next_rhs, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(changed, torch.linalg.solve(matrix, next_rhs),
                               atol=2e-10, rtol=2e-10)
    zero, info = eigenpro.solve(workspace, torch.zeros_like(rhs), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(zero, torch.zeros_like(rhs), atol=0, rtol=0)
    assert info["epochs"] == 0 and workspace["solve_calls"] == 4
    assert vector_pointer == workspace["vectors"].data_ptr()
    assert extended_pointer == workspace["extended"].data_ptr()


def test_eigenpro_full_batch_epoch_matches_independent_spectral_update():
  generator = torch.Generator().manual_seed(17)
  x = torch.randn(8, 3, dtype=torch.float64, generator=generator)
  rhs = torch.randn(8, 2, dtype=x.dtype, generator=generator)
  options = dict(kernel="matern52", length_scale=1.1, query_tile=3, center_tile=3)
  workspace = eigenpro.prepare(x, options, samples=8, rank=3, batch_size=8, seed=8)
  matrix = kernel_matrix(x, "matern52", 1.1)
  indices = workspace["indices"]
  sampled = matrix[indices][:, indices]
  values, vectors = torch.linalg.eigh(sampled)
  values, vectors = values.flip(0), vectors.flip(1)
  next_value = values[3]
  beta = (sampled.diag() - (vectors[:, :3].square() *
                            (values[:3] - next_value)).sum(1)).max()
  preconditioner = vectors[:, :3] * ((1 - next_value / values[:3]) / values[:3]).sqrt()
  critical = int(beta * 8 / next_value) + 1
  rate = 1 / beta if 8 < critical else 2 / (beta + 7 * next_value / 8)
  expected = rate * rhs
  expected[indices] -= rate * preconditioner @ (
    preconditioner.T @ matrix[:, indices].T @ rhs)
  actual = torch.zeros_like(rhs)
  iteration.epoch(workspace, actual, rhs)
  torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


def test_failed_eigenpro_projection_preserves_model_state():
  generator = torch.Generator().manual_seed(55)
  x = torch.randn(8, 3, dtype=torch.float64, generator=generator)
  state = rk.initialize_multiclass(x, 3)
  state["alpha"].copy_(torch.randn(8, 3, dtype=x.dtype, generator=generator))
  state["iterations"], state["last_projection"] = 9, 3
  for i in range(8):
    for c in range(3):
      left = torch.randn(3, dtype=x.dtype, generator=generator)
      factors.add(state["factors"], i, left, torch.eye(3, dtype=x.dtype)[c])
  old_alpha, old_factors = state["alpha"].clone(), factors.clone(state["factors"])
  try:
    rk.project(x, state, solve_rtol=1e-12, solve_atol=1e-14,
               solve_max_epochs=1, eigenpro_rank=0, eigenpro_batch_size=2)
  except RuntimeError as error:
    assert "EigenPro2 failed its residual tolerance" in str(error)
  else:
    raise AssertionError("An underconverged EigenPro2 solve must fail projection")
  torch.testing.assert_close(state["alpha"], old_alpha, atol=0, rtol=0)
  assert state["iterations"] == 9 and state["last_projection"] == 3
  assert "projection" not in state and state["factors"]["scale"] == old_factors["scale"]
  for key in ("u", "v"):
    torch.testing.assert_close(state["factors"][key], old_factors[key], atol=0, rtol=0)
  assert state["factors"]["pending"].keys() == old_factors["pending"].keys()
  for i, block in state["factors"]["pending"].items():
    for key, value in block.items():
      torch.testing.assert_close(value, old_factors["pending"][i][key], atol=0, rtol=0)
