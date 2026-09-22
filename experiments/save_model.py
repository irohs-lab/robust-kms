import pathlib
import torch


def save(x, state, config, options, record):
  model = dict(centers=x, alpha=state["average_alpha"], beta=state["average_beta"],
    options=options, config=config, iterations=state["iterations"], metrics=record)
  path = pathlib.Path(config["output"]) / "model.pt"
  temporary = path.with_suffix(".tmp")
  torch.save(model, temporary)
  temporary.replace(path)
