"""Reproducibility and per-example aggregation for multiple PGD restarts."""
import math
import torch
import torch.nn.functional as functional
from experiments import multiclass_metrics as metrics


def _problem():
  x = torch.tensor([[.02, .05], [.21, .3], [.38, .6], [.48, .82],
                    [.62, .48], [.81, .94], [.99, .01]], dtype=torch.float64)
  labels = torch.tensor([2, 0, 2, 0, 1, 2, 1])
  return x, labels


def _predict(x):
  a, b = x.unbind(1)
  return torch.stack((torch.sin(15 * a) + .3 * b,
    torch.cos(12 * a) - .2 * b, .8 * torch.sin(13 * b) + .2 * a), 1)


OPTIONS = dict(epsilon=.12, steps=2, step_size=.025, batch_size=3, seed=17)


def test_pgd_restart_default_compatibility():
  x, labels = _problem()
  random_state = torch.random.get_rng_state().clone()
  default = metrics.pgd(_predict, x, labels, **OPTIONS)
  explicit = metrics.pgd(_predict, x, labels, restarts=1, **OPTIONS)
  assert default == explicit
  assert torch.equal(torch.random.get_rng_state(), random_state)
  # Recorded before extending the original single-restart implementation.
  expected = dict(accuracy=1 / 7, attack="pgd-ce-linf", clean_accuracy=6 / 7,
    cross_entropy=1.3326324219612962, epsilon=.12, max_linf=.12,
    per_class_accuracy=[.5, 0., 0.], restarts=1, robust_accuracy=1 / 7,
    samples=7, seed=17, step_size=.025, steps=2)
  assert default.keys() == expected.keys()
  for key in expected:
    if key == "cross_entropy":
      assert math.isclose(default[key], expected[key], rel_tol=0, abs_tol=1e-12)
    else:
      assert default[key] == expected[key]
  detailed = metrics.pgd(_predict, x, labels, return_per_example=True, **OPTIONS)
  assert all(detailed[key] == value for key, value in default.items())


def test_pgd_restarts_match_independent_seed_union_and_maximum():
  x, labels = _problem()
  independent = [metrics.pgd(_predict, x, labels, return_per_example=True,
    **(OPTIONS | {"seed": OPTIONS["seed"] + restart})) for restart in range(2)]
  combined = metrics.pgd(_predict, x, labels, restarts=2,
    return_per_example=True, **OPTIONS)
  correct = [all(values) for values in zip(*(r["robust_correct"] for r in independent))]
  worst = [max(values) for values in zip(*(r["worst_cross_entropy"] for r in independent))]
  assert independent[0]["robust_correct"] != independent[1]["robust_correct"]
  assert combined["robust_correct"] == correct
  assert combined["worst_cross_entropy"] == worst
  assert combined["clean_correct"] == independent[0]["clean_correct"] == independent[1]["clean_correct"]
  assert all(type(value) is bool for value in combined["robust_correct"])
  assert all(type(value) is float for value in combined["worst_cross_entropy"])
  assert combined["robust_accuracy"] == sum(correct) / len(x)
  assert combined["robust_accuracy"] < independent[0]["robust_accuracy"]
  assert math.isclose(combined["cross_entropy"], sum(worst) / len(x), abs_tol=1e-12)
  assert combined["cross_entropy"] >= max(r["cross_entropy"] for r in independent)
  assert combined["max_linf"] == max(r["max_linf"] for r in independent)
  assert combined["restarts"] == 2
  for class_id, accuracy in enumerate(combined["per_class_accuracy"]):
    members = [i for i, label in enumerate(labels.tolist()) if label == class_id]
    assert accuracy == sum(correct[i] for i in members) / len(members)


def test_pgd_restarts_every_candidate_respects_radius_and_pixel_bounds():
  x, labels = _problem()
  original = x.clone()
  visited = []

  def recording_predict(inputs):
    visited.append(inputs.detach().clone())
    return _predict(inputs)

  options = OPTIONS | {"steps": 4, "step_size": .2}
  result = metrics.pgd(recording_predict, x, labels, restarts=3, **options)
  offset = 0
  for clean in x.split(options["batch_size"]):
    torch.testing.assert_close(visited[offset], clean, rtol=0, atol=0)
    count = 1 + 3 * (options["steps"] + 1)
    for candidate in visited[offset:offset + count]:
      assert candidate.min() >= 0 and candidate.max() <= 1
      assert (candidate - clean).abs().max() <= options["epsilon"] + 1e-15
    offset += count
  assert offset == len(visited)
  assert result["max_linf"] <= options["epsilon"] + 1e-15
  torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_pgd_restarts_argument_validation():
  x, labels = _problem()
  for invalid in (0, -1, 1., True, False, None, "2"):
    try:
      metrics.pgd(_predict, x, labels, restarts=invalid, **OPTIONS)
    except ValueError as error:
      assert "restarts" in str(error)
    else:
      raise AssertionError(f"Invalid restart count accepted: {invalid!r}")


def test_pgd_restarts_zero_epsilon_equals_clean_per_example():
  x, labels = _problem()
  clean = metrics.evaluate(_predict, x, labels, batch_size=OPTIONS["batch_size"])
  attacked = metrics.pgd(_predict, x, labels, restarts=3,
    return_per_example=True, **(OPTIONS | {"epsilon": 0.}))
  logits = _predict(x)
  expected_correct = (logits.argmax(1) == labels).tolist()
  expected_losses = functional.cross_entropy(logits, labels, reduction="none").tolist()
  assert attacked["robust_correct"] == attacked["clean_correct"] == expected_correct
  assert attacked["worst_cross_entropy"] == expected_losses
  assert attacked["accuracy"] == attacked["clean_accuracy"] == clean["accuracy"]
  assert attacked["per_class_accuracy"] == clean["per_class_accuracy"]
  assert math.isclose(attacked["cross_entropy"], clean["cross_entropy"], abs_tol=1e-12)
  assert attacked["max_linf"] == 0.
