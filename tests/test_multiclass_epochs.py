import io
from unittest.mock import patch
import torch
import robust_kernels as rk
import robust_kernels.lowrank_factors as factors
import robust_kernels.multiclass_step as updates
import tests.reference as reference
from tests.test_multiclass import coefficients


def _data():
  x = torch.tensor([[-.8, .1, .4], [.35, -.9, .6], [1., .7, -.2],
                    [-.2, .9, .8], [.6, -.3, -.7]], dtype=torch.float64)
  return x, torch.tensor([0, 1, 2, 1, 0])


def _rank_one_state(x, outputs=3):
  generator = torch.Generator().manual_seed(18)
  state = rk.initialize_multiclass(x, outputs)
  state["alpha"].copy_(torch.randn(len(x), outputs, generator=generator, dtype=x.dtype))
  state["factors"]["u"].copy_(torch.randn(x.shape, generator=generator, dtype=x.dtype))
  state["factors"]["v"].copy_(torch.randn(len(x), outputs, generator=generator, dtype=x.dtype))
  return state


def _fast_project(centers, state, **options):
  state["factors"] = factors.rank_one(state["factors"])
  state["last_projection"] = state["iterations"]
  return {"max_value_error": 0.}


def test_multiclass_epochs_match_dense_updates_and_visit_every_sample_once():
  x, y = _data()
  state = _rank_one_state(x)
  alpha, beta = coefficients(state)
  gram = reference.dense_blocks(x, x, "gaussian", 1.2)
  seen, rates, events = [], [], []
  original_step, original_randperm = updates.step, torch.randperm
  generator = torch.Generator().manual_seed(7)
  expected_orders = [torch.randperm(len(x), generator=generator) for _ in range(2)]
  def record_step(centers, labels, state, batch, **options):
    seen.append(batch.clone())
    rates.append(options["eta"])
    return original_step(centers, labels, state, batch, **options)
  def check_dense(state, event):
    nonlocal alpha, beta
    batch, rate = seen[-1], rates[-1]
    count = len(batch)
    evaluated = gram @ torch.cat((alpha, beta.reshape(-1, 3)))
    values = evaluated[:len(x)][batch]
    jacobian = evaluated[len(x):].reshape(len(x), 3, 3)[batch]
    residual = values.softmax(1) - torch.nn.functional.one_hot(y[batch], 3)
    penalty = torch.zeros_like(jacobian)
    for row in range(count):
      winner = int(jacobian[row].abs().sum(0).argmax())
      penalty[row, :, winner] = .2 * jacobian[row, :, winner].sign()
    alpha, beta = (1 - rate * .5) * alpha, (1 - rate * .5) * beta
    alpha[batch] -= rate * len(x) / count * residual
    beta[batch] -= rate * len(x) / count * penalty
    for actual, expected in zip(coefficients(state), (alpha, beta)):
      torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
    assert "projection" not in event
    assert state["epochs"] == event["epoch"] - (not event["epoch_end"])
    events.append(event)
  with patch("robust_kernels.multiclass_fit.updates.step", side_effect=record_step), \
       patch("robust_kernels.multiclass_fit.torch.randperm", wraps=original_randperm) as permutations, \
       patch("robust_kernels.multiclass_fit.projection.project") as projector:
    rk.fit_multiclass(x, y, outputs=3, rho=.2, lam=.5, eta=.03, epochs=2,
      batch_size=2, decay=.6, project_every=3, seed=7, state=state,
      callback=check_dense, final_projection=False, length_scale=1.2)
    assert permutations.call_count == 2
    projector.assert_not_called()
  assert [len(batch) for batch in seen] == [2, 2, 1, 2, 2, 1]
  for epoch in range(2):
    actual = torch.cat(seen[3 * epoch:3 * (epoch + 1)])
    assert torch.equal(actual.sort().values, torch.arange(len(x)))
    assert torch.equal(actual, expected_orders[epoch])
  assert rates == [.03 / (iteration + 1) ** .6 for iteration in range(6)]
  assert [(e["epoch"], e["batch_in_epoch"], e["epoch_end"]) for e in events] == [
    (epoch, batch, batch == 3) for epoch in (1, 2) for batch in (1, 2, 3)]


def test_multiclass_projection_interval_counts_completed_epochs():
  x, y = _data()
  for options, expected in (({}, [1, 2, 3, 4, 5]),
      ({"project_every": 2}, [2, 4, 5]),
      ({"project_every": 2, "final_projection": False}, [2, 4])):
    projections, events = [], []
    def project(centers, state, **kwargs):
      projections.append(state["epochs"])
      assert state["iterations"] == 3 * state["epochs"]
      return _fast_project(centers, state, **kwargs)
    with patch("robust_kernels.multiclass_fit.projection.project", side_effect=project):
      state = rk.fit_multiclass(x, y, outputs=3, rho=.1, epochs=5,
        batch_size=2, callback=lambda state, event: events.append(event), **options)
    assert projections == expected
    assert state["epochs"] == 5 and state["iterations"] == 15
    assert state["last_projection_epoch"] == expected[-1]
    assert state["last_projection"] == 3 * expected[-1]
    for event in events:
      if "projection" in event:
        assert event["epoch_end"]
        assert event["batch_in_epoch"] in (3, None)
    assert len(events) == 15 + (options == {"project_every": 2})


def test_multiclass_epoch_boundary_resume_preserves_sampling_and_projection_schedule():
  x, y = _data()
  options = dict(outputs=3, rho=.2, lam=.5, eta=.03, batch_size=2,
                 project_every=2, seed=21, final_projection=False)
  with patch("robust_kernels.multiclass_fit.projection.project", side_effect=_fast_project):
    uninterrupted = rk.fit_multiclass(x, y, epochs=3, **options)
    split = rk.fit_multiclass(x, y, epochs=1, **options)
    checkpoint = io.BytesIO()
    torch.save(split, checkpoint)
    checkpoint.seek(0)
    split = torch.load(checkpoint, weights_only=True)
    split = rk.fit_multiclass(x, y, epochs=2, state=split, **options)
  for actual, expected in zip(coefficients(split), coefficients(uninterrupted)):
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
  assert split["epochs"] == 3 and split["last_projection_epoch"] == 2
  assert split["iterations"] == 9 and split["last_projection"] == 6
  assert torch.equal(split["fit_generator_state"], uninterrupted["fit_generator_state"])


def test_multiclass_delayed_projection_bounds_each_blocks_rank():
  x, y = _data()
  x = torch.cat((x, x[:, :1] ** 2), dim=1)
  state, inspected = _rank_one_state(x, 4), []
  def project(centers, state, **options):
    beta = coefficients(state)[1]
    assert torch.all(torch.linalg.matrix_rank(beta, atol=1e-10, rtol=0) <= 3)
    inspected.append(state["epochs"])
    return _fast_project(centers, state, **options)
  def callback(state, event):
    ranks = torch.linalg.matrix_rank(coefficients(state)[1], atol=1e-10, rtol=0)
    epochs_since_projection = event["epoch"] - state["last_projection_epoch"]
    assert torch.all(ranks <= epochs_since_projection + 1)
  with patch("robust_kernels.multiclass_fit.projection.project", side_effect=project):
    rk.fit_multiclass(x, y, outputs=4, rho=.2, eta=.03, epochs=4,
      batch_size=2, project_every=2, state=state, callback=callback)
  assert inspected == [2, 4]
  first_epoch = rk.fit_multiclass(x, y, outputs=4, rho=.2, eta=.03,
    epochs=1, batch_size=2, project_every=2, final_projection=False)
  beta = coefficients(first_epoch)[1]
  assert beta.norm() > 0
  assert torch.all(torch.linalg.matrix_rank(beta, atol=1e-10, rtol=0) <= 1)


def test_multiclass_zero_epochs_and_rejects_old_step_schedule():
  x, y = _data()
  with patch("robust_kernels.multiclass_fit.projection.project") as projector:
    state = rk.fit_multiclass(x, y, outputs=3, rho=.2, epochs=0, batch_size=2)
    projector.assert_not_called()
  assert state["epochs"] == state["iterations"] == state["last_projection_epoch"] == 0
  for options in ({"steps": 2}, {"epochs": -1}, {"epochs": 1.5},
                  {"epochs": True}, {"project_every": 0}, {"project_every": 1.5}):
    try:
      rk.fit_multiclass(x, y, outputs=3, rho=.2, batch_size=2, **options)
    except (TypeError, ValueError):
      pass
    else:
      raise AssertionError(f"invalid epoch schedule accepted: {options}")
