from unittest.mock import patch
import torch
import robust_kernels.lowrank_factors as factors
import robust_kernels.rkhs_projection as projection
import robust_kernels.projection_search as search
import tests.reference as reference


def test_accepted_projection_trials_reuse_kernel_solve_and_derivative_pass():
  x = torch.tensor([[-.8, .1], [.35, -.9], [1., .7]], dtype=torch.float64)
  beta = torch.tensor([[[2., .3], [.5, 1.]], [[2., 1.], [1., 2.]],
                       [[1., -1.], [2., 1.]]], dtype=x.dtype)
  alpha = torch.tensor([[.2, -.1], [.3, .5], [-.2, .4]], dtype=x.dtype)
  for kernel in ("gaussian", "matern52"):
    state = dict(alpha=alpha.clone(), factors=factors.initialize(x, 2))
    basis = torch.eye(2, dtype=x.dtype)
    for i in range(len(x)):
      for c in range(2):
        factors.add(state["factors"], i, beta[i, :, c], basis[c])
    events = []
    with patch.object(projection.residual, "measure", wraps=projection.residual.measure) as measure:
      info = projection.project(x, state, kernel=kernel, max_steps=3,
        tolerance=0., step_size=1e-3, solve_rtol=1e-12, solve_atol=1e-14,
        query_tile=2, center_tile=2, progress=events.append)
    assert [event["phase"] for event in events] == ["projection_started", "projection_initial",
      "projection_iteration", "projection_iteration", "projection_iteration"]
    assert [event["iteration"] for event in events[2:]] == [1, 2, 3]
    assert events[-1]["error_squared"] == info["error_squared"]
    assert all(event["elapsed_seconds"] >= 0 for event in events)
    assert "progress" not in state
    assert info["iterations"] == 3
    assert measure.call_count == 1 + info["iterations"]
    assert info["kernel_solve_calls"] == measure.call_count
    assert all(call.kwargs.get("gradient", True) for call in measure.call_args_list)
    compressed = torch.stack([factors.column(state["factors"], 0, len(x), c)
                              for c in range(2)], dim=-1)
    delta = torch.cat((state["alpha"] - alpha, (compressed - beta).reshape(-1, 2)))
    matrix = reference.dense_blocks(x, x, kernel, 1.)
    expected = (delta * (matrix @ delta)).sum().item()
    assert abs(info["error_squared"] - expected) < 2e-9
    assert info["error_squared"] < info["initial_error_squared"]
    assert (matrix[:len(x)] @ delta).abs().max() < 2e-9


def test_failed_projection_trial_returns_no_cached_measurement():
  value = torch.ones(1, 1, dtype=torch.float64)
  candidate = dict(u=value.clone(), v=value.clone(), scale=1., pending={})
  current = (1., value, value, value, {})
  measured = (2., value, value, value, {})
  with patch.object(search.residual, "measure", return_value=measured) as measure:
    proposed, rate, accepted_measurement = search.trial(
      None, None, candidate, None, {}, {}, {}, current, 1.)
  assert proposed is None and accepted_measurement is None
  assert measure.call_count == 20 and rate == 2. ** -20
  torch.testing.assert_close(candidate["u"], value, atol=0, rtol=0)
  torch.testing.assert_close(candidate["v"], value, atol=0, rtol=0)
