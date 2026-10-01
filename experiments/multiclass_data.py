"""Shared, reproducible ten-class FashionMNIST inputs for both estimators."""

import math
import torch


def _labels_cpu(labels):
  if labels.ndim != 1 or not len(labels):
    raise ValueError("labels must be a nonempty vector")
  labels = labels.detach().cpu()
  if labels.dtype != torch.long or labels.min() < 0:
    raise ValueError("labels must be nonnegative int64 class indices")
  return labels


def stratified_indices(labels, per_class, *, seed=42):
  """Return original CPU indices with exactly per_class examples per class."""
  labels = _labels_cpu(labels)
  if type(per_class) is not int or per_class < 1:
    raise ValueError("per_class must be a positive integer")
  generator = torch.Generator(device="cpu").manual_seed(seed)
  selected = []
  for label in labels.unique(sorted=True):
    indices = torch.where(labels == label)[0]
    if per_class > len(indices):
      raise ValueError("per_class exceeds the available examples in a class")
    order = torch.randperm(len(indices), generator=generator)
    selected.append(indices[order[:per_class]])
  return torch.cat(selected).sort().values


def split_indices(labels, *, seed=42, validation_fraction=.1,
                  train_per_class=None):
  """Stratify the official training set; indices retain its original order."""
  labels = _labels_cpu(labels)
  if not 0 < validation_fraction < 1:
    raise ValueError("validation_fraction must be between zero and one")
  if train_per_class is not None and (
      type(train_per_class) is not int or train_per_class < 1):
    raise ValueError("train_per_class must be a positive integer or None")
  generator = torch.Generator(device="cpu").manual_seed(seed)
  train, validation = [], []
  for label in labels.unique(sorted=True):
    indices = torch.where(labels == label)[0]
    indices = indices[torch.randperm(len(indices), generator=generator)]
    count = round(validation_fraction * len(indices))
    if not 0 < count < len(indices):
      raise ValueError("every class needs training and validation examples")
    validation.append(indices[:count])
    remaining = indices[count:]
    if train_per_class is not None:
      if train_per_class > len(remaining):
        raise ValueError("train_per_class exceeds the available training examples")
      remaining = remaining[:train_per_class]
    train.append(remaining)
  return {"train": torch.cat(train).sort().values,
          "validation": torch.cat(validation).sort().values}


def load(root, device, *, seed=42, validation_fraction=.1,
         train_per_class=None):
  """Load raw [0,1] pixels, all ten classes, and an untouched official test set.

  The returned split uses CPU indices into the official 60,000-example training
  set. Both estimators must share it. Optional train_per_class is for pilots;
  by default every example outside the validation split is used for training.
  Data must already be present under root; this function never downloads it.
  """
  from torchvision.datasets import FashionMNIST

  training = FashionMNIST(root, train=True, download=False)
  testing = FashionMNIST(root, train=False, download=False)
  expected = torch.arange(10)
  for dataset in (training, testing):
    if not torch.equal(dataset.targets.unique(sorted=True), expected):
      raise ValueError("FashionMNIST must contain all ten classes labelled 0..9")
  split = split_indices(training.targets, seed=seed,
                        validation_fraction=validation_fraction,
                        train_per_class=train_per_class)

  def pixels(values):
    return values.flatten(1).to(device=device, dtype=torch.float32).div_(255)

  data = {}
  for name, indices in (("train", split["train"]),
                        ("val", split["validation"])):
    data[name + "_x"] = pixels(training.data[indices])
    data[name + "_y"] = training.targets[indices].to(device=device)
  data["test_x"] = pixels(testing.data)
  data["test_y"] = testing.targets.to(device=device)
  return data, split


@torch.no_grad()
def bandwidth(train_x, *, seed=42, samples=1024):
  """Median pairwise Euclidean distance using only a training subsample."""
  if train_x.ndim != 2 or len(train_x) < 2:
    raise ValueError("bandwidth needs at least two training examples")
  if type(samples) is not int or samples < 2:
    raise ValueError("samples must be an integer of at least two")
  generator = torch.Generator(device="cpu").manual_seed(seed)
  indices = torch.randperm(len(train_x), generator=generator)[:samples]
  values = train_x[indices.to(train_x.device)].contiguous()
  result = torch.pdist(values, p=2).median().item()
  if not math.isfinite(result) or result <= 0:
    raise ValueError("median training distance must be finite and positive")
  return result
