import functools
import torch
import experiments.baseline_kernel as kernels
import experiments.baseline_metrics as metrics
import experiments.eigenpro_epoch as eigenpro
import experiments.report as report


def run(x, labels, matrices, config):
  kernel = functools.partial(kernels.matrix, length_scale=config["length_scale"])
  state = eigenpro.initialize(kernel, x)
  alpha = torch.zeros(len(x), device=x.device)
  targets = 2 * labels[0].float() - 1
  best, stale, selected = -1., 0, None
  for epoch in range(1, config["epochs"] + 1):
    eigenpro.run(alpha, matrices[0], targets, state)
    scores = metrics.measure([matrix @ alpha for matrix in matrices], labels)
    report.write(config["output"], "epoch", epoch=epoch, **scores)
    if scores["validation_accuracy"] > best:
      best, stale = scores["validation_accuracy"], 0
      selected = dict(alpha=alpha.clone(), bias=0., epoch=epoch, **scores)
    else:
      stale += 1
    if stale >= config["patience"]:
      break
  selected["stopped_epoch"] = epoch
  selected["preconditioner"] = dict(samples=1024, rank=100, batch_size=128)
  return selected
