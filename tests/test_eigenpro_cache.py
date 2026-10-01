"""Scalar Gram caching preserves EigenPro's update and residual equations."""
import unittest
from unittest import mock
import torch
import robust_kernels._radial as radial
import robust_kernels.eigenpro_setup as setup
import robust_kernels.eigenpro_iteration as iteration
import robust_kernels.eigenpro_solve as solver
import robust_kernels.kernel_matmul as products


def _compare_epochs(device):
  generator = torch.Generator(device=device).manual_seed(229)
  x = torch.randn(11, 4, dtype=torch.float64, device=device, generator=generator)
  rhs = torch.randn(11, 10, dtype=x.dtype, device=device, generator=generator)
  initial = torch.randn(rhs.shape, dtype=x.dtype, device=device, generator=generator)
  for kernel in ("gaussian", "matern52"):
    options = dict(kernel=kernel, length_scale=.9, query_tile=3, center_tile=4)
    settings = dict(samples=8, rank=3, batch_size=4, seed=16)
    matfree = setup.prepare(x, options, **settings)
    dense = setup.prepare(x, options, storage="dense", **settings)
    assert matfree["gram"] is None and matfree["storage"] == "matfree"
    assert dense["gram"].shape == (11, 11) and dense["gram"].is_contiguous()
    for key in ("indices", "vectors", "extended", "beta", "next_value"):
      torch.testing.assert_close(dense[key], matfree[key], atol=0, rtol=0)
    assert torch.equal(dense["generator"].get_state(), matfree["generator"].get_state())
    torch.testing.assert_close(dense["gram"] @ initial, products.apply(x, initial, **options),
                               atol=2e-12, rtol=2e-12)
    expected, actual = initial.clone(), initial.clone()
    iteration.epoch(matfree, expected, rhs)
    with mock.patch.object(iteration.evaluation, "kernel_eval_and_grad",
                            side_effect=AssertionError("cached epoch recomputed kernels")):
      iteration.epoch(dense, actual, rhs)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    assert torch.equal(dense["generator"].get_state(), matfree["generator"].get_state())


def test_eigenpro_dense_epoch_matches_matfree_and_preserves_spectral_setup():
  _compare_epochs("cpu")


def test_eigenpro_dense_small_gpu_epoch_matches_matfree():
  # GPU 0 and 1 can be occupied by the comparison; use only the spare device.
  if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
    raise unittest.SkipTest("a third CUDA device is required for the spare-GPU check")
  _compare_epochs("cuda:2")


def test_eigenpro_dense_solve_uses_cache_for_true_residuals_and_warm_starts():
  generator = torch.Generator().manual_seed(227)
  x = torch.randn(8, 3, dtype=torch.float64, generator=generator)
  rhs = torch.randn(8, 10, dtype=x.dtype, generator=generator)
  rhs[:, 4] = 0
  for kernel in ("gaussian", "matern52"):
    options = dict(kernel=kernel, length_scale=.8, query_tile=3, center_tile=3)
    settings = dict(samples=8, rank=7, batch_size=8, seed=13)
    matfree = setup.prepare(x, options, **settings)
    dense = setup.prepare(x, options, storage="dense", **settings)
    cache_pointer = dense["gram"].data_ptr()
    expected, _ = solver.solve(matfree, rhs, rtol=1e-10, atol=1e-12)
    with (mock.patch.object(solver.kernel_operator, "apply",
                             side_effect=AssertionError("cached residual recomputed kernels")),
          mock.patch.object(iteration.evaluation, "kernel_eval_and_grad",
                             side_effect=AssertionError("cached epoch recomputed kernels"))):
      actual, info = solver.solve(dense, rhs, rtol=1e-10, atol=1e-12)
      repeated, repeated_info = solver.solve(dense, rhs, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(actual, torch.linalg.solve(dense["gram"], rhs),
                               atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
    norms = (dense["gram"] @ actual - rhs).norm(dim=0)
    assert torch.all(norms <= 1e-12 + 1e-10 * rhs.norm(dim=0))
    assert info["storage"] == "dense" and repeated_info["epochs"] == 0
    assert dense["gram"].data_ptr() == cache_pointer


def test_eigenpro_dense_matrix_is_built_with_bounded_kernel_tiles():
  generator = torch.Generator().manual_seed(530)
  x = torch.randn(11, 3, dtype=torch.float64, generator=generator)
  options = dict(kernel="matern52", length_scale=.7, query_tile=3, center_tile=4)
  with mock.patch.object(setup.radial, "radial_terms", wraps=radial.radial_terms) as evaluate:
    actual = setup._gram_matrix(x, options)
  assert len(evaluate.call_args_list) == 12
  assert all(len(call.args[0]) <= 3 and len(call.args[1]) <= 4
             for call in evaluate.call_args_list)
  expected = radial.radial_terms(x, x, options["kernel"], options["length_scale"])[0]
  torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
  try:
    setup.prepare(x, options, storage="unsupported")
  except ValueError as error:
    assert "storage" in str(error)
  else:
    raise AssertionError("unsupported storage must be rejected")
