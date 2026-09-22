import pathlib
import time
import torch
import robust_kernels as rk
import experiments.early_stopping as stopping
import experiments.report as report
import experiments.save_model as checkpoints
import experiments.training_metrics as metrics
import experiments.train_epoch as epochs


def run(data, config, options, validation=None):
  x, y, _, _ = data
  state = rk.initialize(x)
  generator = torch.Generator(device=x.device).manual_seed(config["seed"])
  selection = dict(best_accuracy=-1., best_epoch=0, stale=0)
  started = time.monotonic()
  for epoch in range(config["epochs"] + 1):
    if epoch:
      epochs.run(x, y, state, config, options, generator)
    if epoch % config["eval_every"] and epoch != config["epochs"]:
      continue
    record = metrics.evaluate(data, state, config, options, epoch, started, validation)
    report.write(config["output"], "training", **record)
    if validation is None:
      continue
    improved, stop = stopping.update(selection, record["validation_accuracy"],
                                      epoch, config["patience"])
    if improved:
      checkpoints.save(x, state, config, options, record)
    if stop:
      report.write(config["output"], "early_stopped", epoch=epoch, **selection)
      break
  if validation is None:
    checkpoints.save(x, state, config, options, record)
  model = torch.load(pathlib.Path(config["output"]) / "model.pt",
                     map_location=x.device, weights_only=True)
  report.write(config["output"], "selected",
               criterion="validation_accuracy" if validation is not None else "fixed_budget",
               **model["metrics"])
  return model
