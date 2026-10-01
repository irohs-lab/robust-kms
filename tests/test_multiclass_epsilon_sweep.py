"""CPU end-to-end sweep checks with real derivative-kernel checkpoints."""
import contextlib
import functools
import io
import json
import math
from pathlib import Path
import tempfile
from unittest import mock
import numpy as np
import torch
import torch.nn.functional as functional
import robust_kernels as rk
from experiments import multiclass_epsilon_sweep as sweep


def _fixture(root, budgets=(0, 16, 64)):
  centers = torch.eye(3)
  inputs = torch.tensor([[.95, .03, .02], [.02, .97, .01], [.03, .04, .92],
    [.52, .48, .02], [.07, .4, .58], [.02, .53, .51], [.33, .33, .34]])
  labels = torch.tensor([0, 1, 2, 0, 2, 1, 0])
  data = dict(train_x=centers, train_y=torch.arange(3),
              test_x=inputs, test_y=labels)
  split = dict(train=torch.tensor([0, 3, 6]), validation=torch.tensor([1, 4, 7]))
  config = dict(seed=42, train_per_class=1, train_samples=3, data_root="synthetic",
    kernel="matern52", length_scale=.7, query_tile=2, center_tile=2)
  options = {key: config[key] for key in ("kernel", "length_scale", "query_tile", "center_tile")}
  state = rk.initialize_multiclass(centers, 3)
  state["factors"]["u"].copy_(torch.tensor([[.1, -.1, .03], [-.02, .1, .02], [.02, -.04, .1]]))
  state["factors"]["v"].copy_(.2 * torch.eye(3))
  checkpoint = dict(alpha=2 * torch.eye(3), factors=state["factors"],
                    epoch=2, config=config, options=options)
  checkpoint_path, config_path, split_path = root / "model.pt", root / "config.json", root / "split.pt"
  torch.save(checkpoint, checkpoint_path)
  config_path.write_text(json.dumps(config))
  torch.save(split, split_path)
  indices = [5, 0, 6, 2, 4]
  manifest = dict(models=[dict(id="tiny", checkpoint=str(checkpoint_path),
    checkpoint_sha256=sweep.digest(checkpoint_path), selected_epoch=2)],
    reference_config=str(config_path), reference_config_sha256=sweep.digest(config_path),
    split_indices=str(split_path), split_indices_sha256=sweep.digest(split_path),
    test_indices=indices, test_labels=labels[indices].tolist(), samples=len(indices),
    budgets_pixels=list(budgets), denominator=255, steps=3, restarts=2,
    seed=17, batch_size=2, step_size_rule="epsilon/4")
  (root / "manifest.json").write_text(json.dumps(manifest))
  predict = functools.partial(sweep.metrics.multiclass_logits, centers=centers,
    alpha=checkpoint["alpha"], factors=checkpoint["factors"], **options)
  return data, split, manifest, checkpoint, predict


def _run(root, data, split):
  with (mock.patch.object(sweep.data_io, "load", return_value=(data, split)),
        contextlib.redirect_stdout(io.StringIO())):
    return sweep.run(root, "tiny", "cpu")


def _raises_runtime(function, message):
  try:
    function()
  except RuntimeError as error:
    assert message in str(error), str(error)
  else:
    raise AssertionError("Changed evaluation identity was accepted")


def test_epsilon_sweep_saved_outcomes_reproduce_aggregates_and_clean():
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    data, split, manifest, _, predict = _fixture(root)
    result = _run(root, data, split)
    assert result["complete"] and len(result["results"]) == 3
    assert json.loads((root / "results/tiny.json").read_text()) == result
    indices = manifest["test_indices"]
    inputs, labels = data["test_x"][indices], data["test_y"][indices]
    clean = sweep.metrics.evaluate(predict, inputs, labels, batch_size=2)
    logits = predict(inputs)
    expected_correct = (logits.argmax(1) == labels).numpy()
    expected_losses = functional.cross_entropy(logits, labels, reduction="none").numpy()
    assert 0 < clean["accuracy"] < 1
    for row in result["results"]:
      with np.load(root / "results" / row["per_example_file"]) as saved:
        assert set(saved.files) == {"test_indices", "labels", "robust_correct", "clean_correct", "worst_cross_entropy"}
        np.testing.assert_array_equal(saved["test_indices"], indices)
        np.testing.assert_array_equal(saved["labels"], labels.numpy())
        np.testing.assert_array_equal(saved["clean_correct"], expected_correct)
        assert saved["robust_correct"].dtype == np.bool_
        assert np.all(saved["robust_correct"] <= saved["clean_correct"])
        assert row["robust_accuracy"] == float(saved["robust_correct"].mean())
        assert row["clean_accuracy"] == float(saved["clean_correct"].mean())
        assert math.isclose(row["cross_entropy"], float(saved["worst_cross_entropy"].astype(np.float64).mean()), abs_tol=2e-7)
        assert row["max_linf"] <= row["epsilon"] + 1e-6
        if row["epsilon_pixels"] == 0:
          assert row["attack"] == "clean" and row["steps"] == row["restarts"] == 0
          assert row["robust_accuracy"] == clean["accuracy"]
          assert row["per_class_accuracy"] == clean["per_class_accuracy"]
          assert math.isclose(row["cross_entropy"], clean["cross_entropy"], abs_tol=2e-7)
          np.testing.assert_array_equal(saved["robust_correct"], expected_correct)
          np.testing.assert_allclose(saved["worst_cross_entropy"], expected_losses, rtol=0, atol=1e-7)
        else:
          assert row["restarts"] == 2 and row["steps"] == 3
          assert row["step_size"] == row["epsilon"] / 4


def test_epsilon_sweep_resumes_only_unfinished_budgets():
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    data, split, _, _, _ = _fixture(root)
    original_pgd = sweep.metrics.pgd

    def interrupted(*args, **kwargs):
      if kwargs["epsilon"] == 64 / 255:
        raise RuntimeError("simulated interruption")
      return original_pgd(*args, **kwargs)

    with mock.patch.object(sweep.metrics, "pgd", side_effect=interrupted):
      _raises_runtime(lambda: _run(root, data, split), "simulated interruption")
    partial = json.loads((root / "results/tiny.json").read_text())
    assert not partial["complete"]
    assert [row["epsilon_pixels"] for row in partial["results"]] == [0, 16]
    artifacts = {row["per_example_file"]: (root / "results" / row["per_example_file"]).read_bytes()
                 for row in partial["results"]}
    with (mock.patch.object(sweep.metrics, "pgd", wraps=original_pgd) as attack,
          mock.patch.object(sweep, "_clean", side_effect=AssertionError("clean budget rerun"))):
      completed = _run(root, data, split)
    assert completed["complete"] and attack.call_count == 1
    assert attack.call_args.kwargs["epsilon"] == 64 / 255
    assert completed["results"][:2] == partial["results"]
    assert all((root / "results" / name).read_bytes() == value for name, value in artifacts.items())
    saved = (root / "results/tiny.json").read_bytes()
    with (mock.patch.object(sweep.metrics, "pgd", side_effect=AssertionError("completed attack rerun")),
          mock.patch.object(sweep, "_clean", side_effect=AssertionError("completed clean rerun"))):
      assert _run(root, data, split) == completed
    assert (root / "results/tiny.json").read_bytes() == saved


def test_epsilon_sweep_rejects_changed_checkpoint_even_with_rehashed_manifest():
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    data, split, manifest, checkpoint, _ = _fixture(root, budgets=(0,))
    _run(root, data, split)
    checkpoint["alpha"][0, 0] += 1
    torch.save(checkpoint, root / "model.pt")
    _raises_runtime(lambda: _run(root, data, split), "checkpoint changed")
    manifest["models"][0]["checkpoint_sha256"] = sweep.digest(root / "model.pt")
    (root / "manifest.json").write_text(json.dumps(manifest))
    _raises_runtime(lambda: _run(root, data, split), "another model/protocol")


def test_epsilon_sweep_rejects_changed_protocol_or_shared_image_order():
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    data, split, manifest, _, _ = _fixture(root, budgets=(0, 16))
    _run(root, data, split)
    variants = [{"steps": 4}, {"restarts": 3}, {"seed": 18},
      {"budgets_pixels": [0, 16, 32]},
      {"test_indices": list(reversed(manifest["test_indices"])),
       "test_labels": list(reversed(manifest["test_labels"]))}]
    with (mock.patch.object(sweep.metrics, "pgd", side_effect=AssertionError("changed protocol attacked")),
          mock.patch.object(sweep, "_clean", side_effect=AssertionError("changed protocol evaluated"))):
      for change in variants:
        (root / "manifest.json").write_text(json.dumps(manifest | change))
        _raises_runtime(lambda: _run(root, data, split), "another model/protocol")


def test_epsilon_sweep_rejects_changed_split_and_paired_loader_mutation():
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    data, split, manifest, _, _ = _fixture(root, budgets=(0,))
    _run(root, data, split)
    altered = {key: value.clone() for key, value in split.items()}
    altered["train"][0] = 2
    _raises_runtime(lambda: _run(root, data, altered), "split")
    torch.save(altered, root / "split.pt")
    _raises_runtime(lambda: _run(root, data, altered), "split")
    # Updating the manifest hash cannot silently reuse an existing evaluation.
    manifest["split_indices_sha256"] = sweep.digest(root / "split.pt")
    (root / "manifest.json").write_text(json.dumps(manifest))
    _raises_runtime(lambda: _run(root, data, altered), "another model/protocol")
