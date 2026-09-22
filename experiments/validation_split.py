import pathlib
import torch


def split(data, config):
  fraction = config.get("validation_fraction", 0.)
  if fraction == 0:
    config["validation_samples"] = 0
    return data, None
  if not 0 < fraction < 1:
    raise ValueError("validation_fraction must be in [0, 1)")
  x, y, test_x, test_y = data
  generator = torch.Generator(device=x.device).manual_seed(config["seed"])
  train_indices, validation_indices = [], []
  for label in (0, 1):
    indices = torch.where(y == label)[0]
    indices = indices[torch.randperm(len(indices), device=x.device, generator=generator)]
    count = round(fraction * len(indices))
    if not 0 < count < len(indices):
      raise ValueError("each class needs training and validation samples")
    validation_indices.append(indices[:count])
    train_indices.append(indices[count:])
  train = torch.cat(train_indices).sort().values
  validation = torch.cat(validation_indices).sort().values
  torch.save(dict(train=train.cpu(), validation=validation.cpu()),
             pathlib.Path(config["output"]) / "split_indices.pt")
  config["validation_samples"] = len(validation)
  return (x[train], y[train], test_x, test_y), (x[validation], y[validation])
