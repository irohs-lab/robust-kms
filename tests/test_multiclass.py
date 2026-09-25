import io
import torch
import robust_kernels as rk
import robust_kernels.lowrank_factors as factors
import tests.reference as reference


def coefficients(state):
  n, outputs = state["alpha"].shape
  beta = torch.stack([factors.column(state["factors"], 0, n, c)
                      for c in range(outputs)], dim=-1)
  return state["alpha"].clone(), beta


def test_multiclass_step_matches_dense_subgradient():
  generator = torch.Generator().manual_seed(112)
  x = torch.randn(5, 3, generator=generator, dtype=torch.float64)
  y, batch = torch.tensor([0, 2, 1, 2, 0]), torch.tensor([1, 4])
  for kernel in ("gaussian", "matern52"):
    for rho, lam, eta in ((.17, .8, .03), (0., .8, .03), (.17, 2., .5)):
      state = rk.initialize_multiclass(x, 3)
      state["alpha"].copy_(torch.randn(5, 3, generator=generator, dtype=x.dtype))
      state["factors"]["u"].copy_(torch.randn(5, 3, generator=generator, dtype=x.dtype))
      state["factors"]["v"].copy_(torch.randn(5, 3, generator=generator, dtype=x.dtype))
      factors.add(state["factors"], 4, x[0], torch.tensor([.3, -.5, .2], dtype=x.dtype))
      factors.scale(state["factors"], .87)
      alpha, beta = coefficients(state)
      matrix = reference.dense_blocks(x[batch], x, kernel, 1.2)
      evaluated = matrix @ torch.cat((alpha, beta.reshape(-1, 3)))
      values, jacobian = evaluated[:2], evaluated[2:].reshape(2, 3, 3)
      residual = values.softmax(1) - torch.nn.functional.one_hot(y[batch], 3)
      penalty = jacobian.abs().sum(1)
      update = torch.zeros_like(jacobian)
      for row in range(2):
        winner = int(penalty[row].argmax())
        update[row, :, winner] = rho * jacobian[row, :, winner].sign()
      rate = eta * len(x) / len(batch)
      expected_alpha, expected_beta = (1 - eta * lam) * alpha, (1 - eta * lam) * beta
      expected_alpha[batch] -= rate * residual
      expected_beta[batch] -= rate * update
      event = rk.step_multiclass(x, y, state, batch, rho=rho, lam=lam, eta=eta,
                                 kernel=kernel, length_scale=1.2, center_tile=2)
      actual_alpha, actual_beta = coefficients(state)
      torch.testing.assert_close(actual_alpha, expected_alpha, atol=2e-11, rtol=2e-11)
      torch.testing.assert_close(actual_beta, expected_beta, atol=2e-11, rtol=2e-11)
      assert state["iterations"] == 1 and state["last_projection"] == 0
      assert abs(event["batch_jacobian_penalty"] - penalty.max(1).values.mean().item()) < 2e-11


def test_multiclass_tied_max_columns_and_zero_jacobian():
  x = torch.zeros(1, 2, dtype=torch.float64)
  y, batch = torch.tensor([2]), torch.tensor([0])
  state = rk.initialize_multiclass(x, 3)
  state["factors"]["u"][0] = torch.tensor([1., -2.])
  state["factors"]["v"][0] = torch.tensor([1., -1., 0.])
  _, expected = coefficients(state)
  expected[0, :, 0] -= .02 * .3 * torch.tensor([1., -1.], dtype=x.dtype)
  rk.step_multiclass(x, y, state, batch, rho=.3, lam=0., eta=.02)
  torch.testing.assert_close(coefficients(state)[1], expected, atol=1e-12, rtol=1e-12)
  state = rk.initialize_multiclass(x, 3)
  rk.step_multiclass(x, y, state, batch, rho=.3, lam=0., eta=.02)
  assert not state["factors"]["pending"]
  torch.testing.assert_close(coefficients(state)[1], torch.zeros_like(expected))


def test_multiclass_fit_delays_projection_and_checkpoint_roundtrip():
  x = torch.tensor([[-.8, .1], [.35, -.9], [1., .7], [-.2, .9]], dtype=torch.float64)
  y, events, snapshots = torch.tensor([0, 1, 2, 1]), [], []
  def callback(state, event):
    events.append((event["iteration"], "projection" in event))
    if "projection" in event:
      assert not state["factors"]["pending"]
      assert event["projection"]["max_value_error"] < 2e-9
      assert state["last_projection"] == event["iteration"]
    if event["iteration"] == 4:
      checkpoint = io.BytesIO()
      torch.save(state, checkpoint)
      checkpoint.seek(0)
      snapshots.append(torch.load(checkpoint, weights_only=True))
      for actual, expected in zip(coefficients(snapshots[-1]), coefficients(state)):
        torch.testing.assert_close(actual, expected)
  state = rk.fit_multiclass(x, y, outputs=3, rho=.2, lam=.5, eta=.03,
    steps=7, batch_size=3, project_every=3, seed=7, callback=callback,
    projection_options=dict(max_steps=3, solve_rtol=1e-10, solve_atol=1e-12))
  assert [iteration for iteration, projected in events if projected] == [3, 6, 7]
  assert [iteration for iteration, _ in events] == [1, 2, 3, 4, 5, 6, 7, 7]
  assert state["iterations"] == state["last_projection"] == 7
  assert state["alpha"].shape == (4, 3)
  assert len(snapshots) == 1 and snapshots[0]["factors"]["pending"]
  restored = snapshots[0]
  rk.step_multiclass(x, y, restored, torch.tensor([0, 3]), rho=.2, lam=.5, eta=.02)
  assert restored["iterations"] == 5
  rk.project(x, restored, max_steps=3, solve_rtol=1e-10, solve_atol=1e-12)
  assert restored["last_projection"] == 5 and not restored["factors"]["pending"]
