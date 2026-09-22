import contextlib
import io
import json
import pathlib
import tempfile
import torch
import experiments.train_fashion as training


def test_training_restores_best_validation_model():
  x = torch.tensor([[-1.], [-0.8], [0.8], [1.]], dtype=torch.float64)
  y = torch.tensor([0, 0, 1, 1])
  validation = (torch.tensor([[-0.9], [0.9]], dtype=torch.float64), torch.tensor([0, 1]))
  with tempfile.TemporaryDirectory() as output:
    config = dict(output=output, epochs=20, eval_every=1, patience=2, seed=42,
      batch_size=2, rho=0.03, lam=0., step_size=0.02, decay=0.6)
    with contextlib.redirect_stdout(io.StringIO()):
      model = training.run((x, y, x, y), config, dict(kernel="matern52"), validation)
    records = [json.loads(line) for line in
               (pathlib.Path(output) / "metrics.jsonl").read_text().splitlines()]
    epochs = [r for r in records if r["event"] == "training"]
    best = max(epochs, key=lambda r: r["validation_accuracy"])
    assert model["metrics"]["epoch"] == best["epoch"]
    assert model["iterations"] == best["step"]
    assert any(r["event"] == "early_stopped" for r in records)
    assert records[-1]["event"] == "selected"
