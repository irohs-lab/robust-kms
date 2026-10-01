"""Resume regressions using tiny synthetic ten-class inputs and real checkpoints."""
import contextlib
import io
import json
import pathlib
import tempfile
from types import SimpleNamespace
from unittest import mock
import torch
import experiments.fashion_multiclass as runner


def _resume_without_training(model_name, *, early_stopped):
  x = torch.eye(10)
  y = torch.arange(10)
  data = dict(train_x=x, train_y=y, val_x=.9 * x + .01, val_y=y,
              test_x=.8 * x + .02, test_y=y)
  split = dict(train=torch.arange(10), validation=torch.arange(10, 20))
  completed_epoch, best_epoch, patience = 5, 2, 3
  config = dict(seed=42, data_root="synthetic", train_per_class=1,
    kernel="matern52", length_scale=1., query_tile=4, center_tile=4,
    epochs=20, patience=patience, batch_size=4, baseline_eta=.01)
  options = {key: config[key] for key in ("kernel", "length_scale", "query_tile", "center_tile")}
  best_alpha, latest_alpha = 4 * torch.eye(10), -4 * torch.eye(10)
  best_predict = lambda values: runner.metrics.multiclass_logits(
    values, x, best_alpha, None, **options)
  expected = runner.metrics.evaluate(best_predict, data["test_x"], y)
  validation = runner.metrics.evaluate(best_predict, data["val_x"], y)
  assert expected["accuracy"] == 1.
  latest_predict = lambda values: runner.metrics.multiclass_logits(
    values, x, latest_alpha, None, **options)
  assert runner.metrics.evaluate(latest_predict, data["test_x"], y)["accuracy"] == 0.
  generator = torch.Generator().manual_seed(42)
  torch.randperm(len(x), generator=generator)
  generator_state = generator.get_state()
  initialized = []

  def initialize_baseline(samples, labels, **kwargs):
    # Keep runner checkpoint/metric I/O real without depending on optional KLR.
    model = SimpleNamespace(weights=torch.zeros(len(samples), 10))
    model.operator = lambda queries, **unused: kwargs["kernel"](queries, samples)
    state = dict(model=model, generator=torch.Generator().manual_seed(0),
                  iterations=0, epochs=0)
    initialized.append(state)
    return state

  with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    output = root / model_name
    output.mkdir()
    (root / "config.json").write_text(json.dumps(config))
    torch.save(split, root / "split_indices.pt")
    selected = dict(alpha=best_alpha, factors=None, epoch=best_epoch,
                     validation=validation, options=options, config=config)
    torch.save(selected, output / "model.pt")
    checkpoint = dict(epoch=completed_epoch, training_seconds=12.5,
      selection=dict(best_loss=validation["cross_entropy"], best_epoch=best_epoch,
                     stale=patience if early_stopped else 0))
    if model_name == "robust":
      state = runner.rk.initialize_multiclass(x, 10)
      state.update(alpha=latest_alpha, epochs=completed_epoch, iterations=15,
                    last_projection=15, last_projection_epoch=completed_epoch,
                    fit_generator_state=generator_state)
      checkpoint["state"] = state
    else:
      checkpoint.update(alpha=latest_alpha, iterations=15, generator_state=generator_state)
    torch.save(checkpoint, output / "latest.pt")
    original_latest = (output / "latest.pt").read_bytes()
    original_best = (output / "model.pt").read_bytes()
    command = ["fashion_multiclass", "--output", directory, "--model", model_name,
               "--device", "cpu", "--resume", "--no-attack"]
    if not early_stopped:
      command += ["--train-epochs", str(completed_epoch)]
    with (mock.patch("sys.argv", command),
          mock.patch.object(runner.data_io, "load", return_value=(data, split)),
          mock.patch.object(runner.baseline, "initialize", side_effect=initialize_baseline),
          mock.patch.object(runner.baseline, "epoch", side_effect=AssertionError("unexpected training")) as baseline_epoch,
          mock.patch.object(runner.rk, "fit_multiclass", side_effect=AssertionError("unexpected training")) as robust_fit,
          mock.patch.object(runner.metrics, "evaluate", wraps=runner.metrics.evaluate) as evaluate,
          mock.patch.object(runner.metrics, "pgd", side_effect=AssertionError("unexpected attack")),
          contextlib.redirect_stdout(io.StringIO())):
      runner.main()
    baseline_epoch.assert_not_called()
    robust_fit.assert_not_called()
    assert evaluate.call_count == 1
    assert evaluate.call_args.args[1] is data["test_x"]
    if model_name == "baseline":
      torch.testing.assert_close(initialized[0]["model"].weights, latest_alpha)
      assert torch.equal(initialized[0]["generator"].get_state(), generator_state)
      assert initialized[0]["epochs"] == completed_epoch
    result = json.loads((output / "result.json").read_text())
    assert result["selected_epoch"] == best_epoch
    assert result["epochs_run"] == completed_epoch
    assert result["training_seconds"] == 12.5
    assert result["test"] == expected
    assert result["validation"] == validation and result["attack"] is None
    assert (output / "latest.pt").read_bytes() == original_latest
    assert (output / "model.pt").read_bytes() == original_best
    events = [json.loads(line)["event"]
              for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert events == ["started", "evaluating", "complete"]


def test_resuming_completed_early_stop_evaluates_best_without_more_training():
  for model in ("baseline", "robust"):
    _resume_without_training(model, early_stopped=True)


def test_resuming_at_epoch_limit_evaluates_best_without_undefined_state():
  for model in ("baseline", "robust"):
    _resume_without_training(model, early_stopped=False)
