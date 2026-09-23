import hashlib
import json
import pathlib
import shutil


def collect(name, checkpoint, reference, output):
  source = pathlib.Path("runs/fashion01-budget-sweep-" + name)
  status = json.loads((source / "status.json").read_text())
  if status["event"] != "complete":
    raise RuntimeError(name + " sweep is not complete")
  manifest = json.loads((source / "manifest.json").read_text())
  digest = hashlib.sha256(pathlib.Path(checkpoint).read_bytes()).hexdigest()
  if manifest["sha256"] != digest:
    raise RuntimeError(name + " checkpoint changed")
  target = output / name
  target.mkdir(exist_ok=True)
  results = json.loads((source / "results.json").read_text())
  if {row["budget_pixels"] for row in results} != {1, 2, 4, 12, 16}:
    raise RuntimeError(name + " has missing budgets")
  accuracies = {0: reference["clean_test_accuracy"],
                8: reference["robust_test_accuracy"]}
  for row in results:
    budget = row["budget_pixels"]
    if row["samples"] != 2000 or row["clean_test_accuracy"] != accuracies[0]:
      raise RuntimeError(name + " test set or clean predictions changed")
    if abs(row["epsilon"] - budget / 255) > 1e-12:
      raise RuntimeError(name + " epsilon mismatch")
    if row["max_linf"] > budget / 255 + 1e-6:
      raise RuntimeError(name + " perturbation exceeds budget")
    accuracies[budget] = row["robust_test_accuracy"]
    directory = "eps" + str(budget) + "-255"
    (target / directory).mkdir(exist_ok=True)
    for filename in ("config.json", "autoattack.log", "status.json"):
      shutil.copyfile(source / directory / filename, target / directory / filename)
  for filename in ("manifest.json", "results.json"):
    shutil.copyfile(source / filename, target / filename)
  return accuracies
