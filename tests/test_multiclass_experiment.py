from unittest import mock
import torch
import torch.nn.functional as functional
import experiments.multiclass_data as data
import experiments.multiclass_metrics as metrics
import robust_kernels.lowrank_factors as factors_ops
import tests.reference as reference


def test_multiclass_experiment_split():
  labels = torch.arange(10).repeat_interleave(20)
  state = torch.random.get_rng_state().clone()
  first = data.split_indices(labels)
  second = data.split_indices(labels)
  assert torch.equal(state, torch.random.get_rng_state())
  for key in ("train", "validation"):
    assert first[key].device.type == "cpu"
    assert torch.equal(first[key], second[key])
  assert len(first["train"]) == 180 and len(first["validation"]) == 20
  all_indices = torch.cat((first["train"], first["validation"]))
  assert torch.equal(all_indices.sort().values, torch.arange(200))
  assert torch.equal(torch.bincount(labels[first["validation"]]),
                     torch.full((10,), 2))
  pilot = data.split_indices(labels, train_per_class=3)
  assert torch.equal(pilot["validation"], first["validation"])
  assert len(pilot["train"]) == 30
  assert torch.isin(pilot["train"], first["train"]).all()
  subset = data.stratified_indices(labels, 4)
  assert torch.equal(torch.bincount(labels[subset]), torch.full((10,), 4))
  assert torch.equal(subset, data.stratified_indices(labels, 4))


def test_multiclass_experiment_bandwidth():
  x = torch.tensor([[0., 0.], [3., 0.], [0., 4.], [3., 4.]])
  assert data.bandwidth(x, samples=4) == 4.
  assert data.bandwidth(x, samples=3) == data.bandwidth(x, samples=3)


def test_multiclass_logits_input_gradients():
  generator = torch.Generator().manual_seed(231)
  centers = torch.randn(4, 2, generator=generator, dtype=torch.float64)
  queries = torch.randn(3, 2, generator=generator, dtype=torch.float64)
  alpha = torch.randn(4, 3, generator=generator, dtype=torch.float64)
  factors = factors_ops.initialize(centers, 3)
  factors["u"].normal_(generator=generator)
  factors["v"].normal_(generator=generator)
  factors_ops.add(factors, 2, centers.new_tensor([.3, -.2]),
                  centers.new_tensor([.1, -.4, .7]))
  beta = factors["u"][:, :, None] * factors["v"][:, None, :]
  beta[2].add_(centers.new_tensor([.3, -.2])[:, None]
               * centers.new_tensor([.1, -.4, .7])[None, :])
  coefficients = torch.cat((alpha, beta.reshape(-1, 3)))
  weights = torch.randn(3, 3, generator=generator, dtype=torch.float64)
  for kernel in ("gaussian", "matern52"):
    expected = reference.dense_blocks(queries, centers, kernel, 1.3) @ coefficients
    images = queries.reshape(3, 1, 1, 2).clone().requires_grad_()
    logits = metrics.multiclass_logits(images, centers, alpha, factors,
                                       kernel=kernel, length_scale=1.3)
    torch.testing.assert_close(logits, expected[:3])
    gradient, = torch.autograd.grad((logits * weights).sum(), images)
    expected_gradient = torch.einsum("mdc,mc->md", expected[3:].reshape(3, 2, 3),
                                     weights)
    torch.testing.assert_close(gradient.flatten(1), expected_gradient)
  with mock.patch.object(metrics, "multioutput_eval_and_grad",
                          wraps=metrics.multioutput_eval_and_grad) as evaluator:
    with torch.no_grad():
      metrics.multiclass_logits(queries, centers, alpha, factors)
    assert evaluator.call_args.kwargs["gradients"] is False
    metrics.multiclass_logits(queries, centers, alpha, None)
    assert evaluator.call_args.kwargs["gradients"] is False


def test_multiclass_experiment_metrics_and_attack():
  x = torch.tensor([[.1, .3], [.3, .7], [.9, .2], [.7, .8]])
  weights = torch.tensor([[-2., 0., 2.], [0., 0., 0.]])
  bias = torch.tensor([1., .2, -1.])
  predict = lambda values: values @ weights + bias
  labels = torch.tensor([0, 0, 2, 1])
  clean = metrics.evaluate(predict, x, labels, batch_size=2)
  expected = functional.cross_entropy(predict(x), labels).item()
  assert abs(clean["cross_entropy"] - expected) < 1e-6
  assert clean["accuracy"] == .75
  assert clean["per_class_accuracy"] == [1., 0., 1.]
  first = metrics.pgd(predict, x, labels, epsilon=.2, steps=4,
                       step_size=.1, batch_size=2)
  second = metrics.pgd(predict, x, labels, epsilon=.2, steps=4,
                        step_size=.1, batch_size=2)
  assert first == second
  assert first["max_linf"] <= .2 + 1e-6
  assert first["robust_accuracy"] <= first["clean_accuracy"] == clean["accuracy"]
  assert first["cross_entropy"] >= clean["cross_entropy"] - 1e-6
  zero = metrics.pgd(predict, x, labels, epsilon=0., steps=1, batch_size=2)
  assert zero["robust_accuracy"] == clean["accuracy"]
