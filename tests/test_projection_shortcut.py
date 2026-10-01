from unittest.mock import patch
import torch
import robust_kernels.lowrank_factors as factors_ops
import robust_kernels.rkhs_projection as projection
import robust_kernels.multioutput_evaluate as evaluation


def coefficients(factors):
  values = factors["u"][:, :, None] * factors["v"][:, None, :]
  for index, item in factors["pending"].items():
    values[index] = item["left"] @ item["core"] @ item["right"].T
  return values * factors["scale"]


def random_state(device, dtype, shapes):
  generator = torch.Generator(device=device).manual_seed(156)
  rand = lambda *size: torch.randn(*size, generator=generator, device=device, dtype=dtype)
  centers = rand(len(shapes) + 2, 5)
  factors = factors_ops.initialize(centers, 4)
  factors["u"].copy_(rand(len(centers), 5))
  factors["v"].copy_(rand(len(centers), 4))
  for index, (a, b) in enumerate(shapes):
    factors["pending"][index] = dict(
      left=torch.linalg.qr(rand(5, a), mode="reduced")[0], core=rand(a, b),
      right=torch.linalg.qr(rand(4, b), mode="reduced")[0])
  return centers, factors


def test_batched_rank_one_matches_dense_svd_for_mixed_shapes_and_scaling():
  devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
  for device in devices:
    dtype = torch.float64 if device == "cpu" else torch.float32
    _, factors = random_state(device, dtype, [(2, 2), (3, 1), (1, 3), (0, 0),
                                               (3, 4), (2, 2), (2, 3)])
    for scale in (1., -.37, 0.):
      factors["scale"] = scale
      dense = coefficients(factors)
      u, singular, vh = torch.linalg.svd(dense, full_matrices=False)
      expected = (u[:, :, :1] * singular[:, None, :1]) @ vh[:, :1, :]
      compressed = factors_ops.rank_one(factors)
      torch.testing.assert_close(coefficients(compressed), expected, atol=3e-5, rtol=3e-5)
      assert not compressed["pending"] and compressed["scale"] == 1.


def test_batched_rank_one_handles_more_than_one_scratch_chunk():
  count = 4100
  centers = torch.zeros(count, 2, dtype=torch.float64)
  factors = factors_ops.initialize(centers, 2)
  basis = torch.eye(2, dtype=centers.dtype)
  strengths = torch.linspace(2., 4., count, dtype=centers.dtype)
  for i in range(count):
    factors["pending"][i] = dict(left=basis, right=basis,
      core=torch.diag(torch.stack((strengths[i], strengths[i] / 3))))
  factors["scale"] = -.5
  compressed = factors_ops.rank_one(factors)
  expected = torch.zeros(count, 2, 2, dtype=centers.dtype)
  expected[:, 0, 0] = -.5 * strengths
  torch.testing.assert_close(coefficients(compressed), expected, atol=1e-12, rtol=1e-12)


def test_feasible_projection_shortcut_preserves_function_without_kernel_solve():
  devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
  for device in devices:
    dtype = torch.float64 if device == "cpu" else torch.float32
    centers, template = random_state(device, dtype, [(1, 1), (2, 1), (1, 3), (0, 0)])
    alpha = centers.new_ones((len(centers), 4))
    for scale in (1., -.25, 0.):
      factors = factors_ops.clone(template)
      factors["scale"] = scale
      expected = coefficients(factors)
      state = dict(alpha=alpha, factors=factors, iterations=17, epochs=3,
                   last_projection=0, last_projection_epoch=0)
      with patch.object(projection.eigenpro, "prepare", side_effect=AssertionError("unnecessary solve")):
        with patch.object(evaluation, "kernel_eval_and_grad",
                          side_effect=AssertionError("unnecessary kernel evaluation")):
          info = projection.project(centers, state)
      torch.testing.assert_close(coefficients(state["factors"]), expected, atol=3e-5, rtol=3e-5)
      assert state["alpha"] is alpha
      assert state["last_projection"] == 17 and state["last_projection_epoch"] == 3
      assert info["reason"] == "already_rank_one" and info["converged"]
      assert info["kernel_solve_calls"] == 0 and info["error_squared"] == 0.
      assert not state["factors"]["pending"]
