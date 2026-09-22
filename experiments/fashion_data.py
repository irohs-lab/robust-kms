import torch
import torchvision


def load(root, device):
  """Use raw [0,1] pixels and the official 0-vs-1 train/test splits."""
  splits = []
  for train in (True, False):
    dataset = torchvision.datasets.FashionMNIST(root, train=train, download=False)
    selected = (dataset.targets == 0) | (dataset.targets == 1)
    x = dataset.data[selected].to(device=device, dtype=torch.float32).flatten(1)
    x.div_(255)
    y = dataset.targets[selected].to(device=device)
    splits.extend((x, y))
  return tuple(splits)
