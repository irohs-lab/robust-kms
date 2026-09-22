import pathlib
import time
import torch
import robust_kernels as rk
import experiments.report as report
import experiments.score as score
import experiments.train_epoch as epochs


def run(data, config, options):
  x, y, test_x, test_y = data
  state = rk.initialize(x)
  generator = torch.Generator(device=x.device).manual_seed(config["seed"])
  started = time.monotonic()
  for epoch in range(config["epochs"] + 1):
    if epoch:
      epochs.run(x, y, state, config, options, generator)
    if epoch % config["eval_every"] and epoch != config["epochs"]:
      continue
    alpha, beta = state["average_alpha"], state["average_beta"]
    train = score.evaluate(x, y, x, alpha, beta, options, training=True)
    test = score.evaluate(test_x, test_y, x, alpha, beta, options)
    objective = (len(x) * train["loss"] + config["rho"] * train["gradient_l1"]
                 + config["lam"] / 2 * train["norm_squared"])
    report.write(config["output"], "training", epoch=epoch, step=state["iterations"],
                 train_accuracy=train["accuracy"], test_accuracy=test["accuracy"],
                 train_logistic_loss=train["loss"], test_logistic_loss=test["loss"],
                 objective=objective, elapsed_seconds=time.monotonic() - started)
  model = dict(centers=x, alpha=state["average_alpha"], beta=state["average_beta"],
               options=options, config=config, iterations=state["iterations"])
  path = pathlib.Path(config["output"]) / "model.pt"
  temporary = path.with_suffix(".tmp")
  torch.save(model, temporary)
  temporary.replace(path)
  return model
