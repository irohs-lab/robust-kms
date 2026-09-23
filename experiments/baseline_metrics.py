def measure(predictions, labels):
  names = ("train", "validation", "test")
  return {name + "_accuracy": ((values > 0) == y.bool()).float().mean().item()
    for name, values, y in zip(names, predictions, labels)}
