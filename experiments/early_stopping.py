import math


def update(selection, accuracy, epoch, patience):
  """Use only clean validation accuracy; retain the earliest checkpoint on ties."""
  if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
    raise ValueError("validation accuracy must be finite and in [0, 1]")
  improved = accuracy > selection["best_accuracy"]
  if improved:
    selection.update(best_accuracy=accuracy, best_epoch=epoch, stale=0)
  elif epoch > 0:
    selection["stale"] += 1
  return improved, selection["stale"] >= patience
