import importlib.util
import math
import unittest
import torch
from torch.nn import functional as F
import experiments.multiclass_baseline as baseline


def gaussian(x, z):
  return torch.exp(-torch.cdist(x, z).square() / 2)


def require_klr():
  if importlib.util.find_spec("klr") is None:
    raise unittest.SkipTest("KLR checkout must be available on PYTHONPATH")


def test_cross_entropy_step_changes_only_sampled_rows_without_batch_scaling():
  require_klr()
  generator = torch.Generator().manual_seed(704)
  x = torch.randn(12, 3, dtype=torch.float64, generator=generator)
  y = torch.arange(len(x)) % 10
  state = baseline.initialize(x, y, kernel=gaussian, batch_size=5,
                              seed=13, kernel_batch_size=4, eta=.013)
  weights = torch.randn(12, 10, dtype=x.dtype, generator=generator) / 5
  state["model"].weights.copy_(weights)
  matrix = gaussian(x, x)
  rows = torch.tensor([0, 4, 5, 7, 10])
  logits = matrix[rows] @ weights
  ce_gradient = logits.softmax(1) - F.one_hot(y[rows], 10)
  torch.testing.assert_close(state["derivative"](logits, y[rows]), ce_gradient)
  assert not torch.allclose(ce_gradient, logits - F.one_hot(y[rows], 10))
  expected = weights.clone()
  expected[rows] -= .013 * ce_gradient
  report = baseline.step(state, rows)
  torch.testing.assert_close(state["model"].weights, expected, atol=1e-12, rtol=1e-12)
  untouched = torch.tensor([1, 2, 3, 6, 8, 9, 11])
  torch.testing.assert_close(state["model"].weights[untouched], weights[untouched],
                             atol=0, rtol=0)
  assert math.isclose(report["cross_entropy"], F.cross_entropy(logits, y[rows]).item(),
                      rel_tol=1e-12, abs_tol=1e-12)
  assert report["eta"] == .013 and state["iterations"] == 1
  torch.testing.assert_close(state["model"](x), matrix @ expected)
  logits = matrix[[9]] @ expected
  expected[9] -= .007 * (logits.softmax(1)[0] - F.one_hot(y[9], 10))
  baseline.step(state, torch.tensor([9]), eta=.007)
  torch.testing.assert_close(state["model"].weights, expected, atol=1e-12, rtol=1e-12)
  assert state["eta"] == .013


def test_epoch_uses_same_eta_for_tail_and_preserves_global_rng():
  require_klr()
  generator = torch.Generator().manual_seed(891)
  x = torch.randn(7, 2, dtype=torch.float64, generator=generator)
  y = torch.arange(len(x)) % 3
  before = torch.random.get_rng_state()
  options = dict(kernel=gaussian, outputs=3, batch_size=3,
                 seed=41, kernel_batch_size=3, eta=.021)
  state = baseline.initialize(x, y, **options)
  reference = baseline.initialize(x, y, storage="matfree", **options)
  order = torch.randperm(len(x), generator=torch.Generator().manual_seed(41))
  expected = torch.zeros_like(state["model"].weights)
  matrix = gaussian(x, x)
  for rows in order.split(3):
    logits = matrix[rows] @ expected
    expected[rows] -= .021 * (logits.softmax(1) - F.one_hot(y[rows], 3))
    baseline.step(reference, rows)
  events = baseline.epoch(state)
  torch.testing.assert_close(state["model"].weights, expected)
  torch.testing.assert_close(state["model"].weights, reference["model"].weights)
  assert len(events) == 3 and state["epochs"] == 1 and state["iterations"] == 3
  assert all(event["eta"] == .021 for event in events)
  assert torch.equal(before, torch.random.get_rng_state())
