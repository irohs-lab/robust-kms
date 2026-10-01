"""Evaluate fixed validation-selected checkpoints at several pixel-space radii."""
import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import experiments.multiclass_data as data_io
import experiments.multiclass_metrics as metrics


def digest(path):
  checksum = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024**2), b""):
      checksum.update(block)
  return checksum.hexdigest()


def _json(path, value):
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  temporary.replace(path)


def _sync(device):
  if torch.device(device).type == "cuda":
    torch.cuda.synchronize(device)


@torch.no_grad()
def _clean(predict, inputs, labels, batch_size, seed):
  correct, losses = [], []
  for start in range(0, len(inputs), batch_size):
    y = labels[start:start + batch_size]
    logits = predict(inputs[start:start + batch_size])
    metrics._check_logits(logits, y)
    correct.extend((logits.argmax(1) == y).cpu().tolist())
    losses.extend(torch.nn.functional.cross_entropy(logits, y, reduction="none").cpu().tolist())
  counts = torch.bincount(labels, minlength=logits.shape[1])
  good = torch.bincount(labels[torch.tensor(correct, device=labels.device)], minlength=logits.shape[1])
  result = metrics._summary(sum(correct), sum(losses), counts.tolist(), good.tolist(), len(labels))
  result.update(robust_accuracy=result["accuracy"], clean_accuracy=result["accuracy"],
    epsilon=0., steps=0, restarts=0, step_size=0., seed=seed, max_linf=0., attack="clean",
    robust_correct=correct, clean_correct=correct, worst_cross_entropy=losses)
  return result


def run(root, model_id, device):
  root = Path(root)
  manifest = json.loads((root / "manifest.json").read_text())
  budgets = manifest["budgets_pixels"]
  if (not budgets or any(isinstance(v, bool) or not isinstance(v, (int, float))
      or not math.isfinite(v) or v < 0 for v in budgets)
      or len(set(budgets)) != len(budgets)):
    raise ValueError("budgets must be distinct finite nonnegative numbers")
  denominator = manifest["denominator"]
  if not isinstance(denominator, (int, float)) or not math.isfinite(denominator) or denominator <= 0:
    raise ValueError("denominator must be finite and positive")
  for key in ("steps", "restarts", "batch_size", "samples"):
    if type(manifest[key]) is not int or manifest[key] < 1:
      raise ValueError(f"{key} must be a positive integer")
  if manifest["step_size_rule"] != "epsilon/4":
    raise ValueError("only the declared epsilon/4 step rule is supported")
  matches = [entry for entry in manifest["models"] if entry["id"] == model_id]
  if len(matches) != 1:
    raise ValueError("model id must identify exactly one manifest entry")
  model = matches[0]
  for path_key in ("split_indices", "reference_config"):
    if digest(manifest[path_key]) != manifest[path_key + "_sha256"]:
      raise RuntimeError(f"{path_key} changed after the evaluation manifest was frozen")
  checkpoint_path = Path(model["checkpoint"])
  if digest(checkpoint_path) != model["checkpoint_sha256"]:
    raise RuntimeError("checkpoint changed after the evaluation manifest was frozen")
  torch.set_num_threads(1)
  torch.backends.cuda.matmul.allow_tf32 = False
  checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
  if checkpoint["epoch"] != model["selected_epoch"]:
    raise RuntimeError("selected checkpoint epoch differs from the manifest")
  config = checkpoint["config"]
  shared = json.loads(Path(manifest["reference_config"]).read_text())
  for key in ("seed", "train_per_class", "train_samples", "kernel", "length_scale"):
    if config[key] != shared[key]:
      raise RuntimeError(f"{model_id} differs from the shared data/kernel setting: {key}")
  data, split = data_io.load(config["data_root"], device, seed=config["seed"],
                            train_per_class=config["train_per_class"])
  saved_split = torch.load(manifest["split_indices"], weights_only=True)
  if any(not torch.equal(split[key], saved_split[key]) for key in split):
    raise RuntimeError("dataset split does not match the evaluation reference")
  indices = torch.tensor(manifest["test_indices"], dtype=torch.long)
  if (indices.ndim != 1 or len(indices) != manifest["samples"]
      or len(indices.unique()) != len(indices) or len(indices) == 0
      or indices.min() < 0 or indices.max() >= len(data["test_x"])):
    raise ValueError("invalid or duplicate test indices")
  inputs, labels = data["test_x"][indices], data["test_y"][indices]
  if labels.cpu().tolist() != manifest["test_labels"]:
    raise RuntimeError("test labels differ from the frozen evaluation manifest")
  predict = functools.partial(metrics.multiclass_logits, centers=data["train_x"],
    alpha=checkpoint["alpha"], factors=checkpoint["factors"], **checkpoint["options"])
  protocol = {key: manifest[key] for key in ("budgets_pixels", "denominator", "steps",
    "restarts", "seed", "samples", "batch_size", "step_size_rule")}
  protocol["test_indices_sha256"] = hashlib.sha256(indices.numpy().tobytes()).hexdigest()
  protocol["test_labels_sha256"] = hashlib.sha256(labels.cpu().numpy().tobytes()).hexdigest()
  protocol["split_indices_sha256"] = manifest["split_indices_sha256"]
  protocol["reference_config_sha256"] = manifest["reference_config_sha256"]
  output = root / "results"
  output.mkdir(parents=True, exist_ok=True)
  result_path = output / (model_id + ".json")
  record = dict(model=model, protocol=protocol, results=[], complete=False)
  if result_path.exists():
    record = json.loads(result_path.read_text())
    if record["model"] != model or record["protocol"] != protocol:
      raise RuntimeError("existing evaluation belongs to another model/protocol")
  seen = {row["epsilon_pixels"] for row in record["results"]}
  for pixels in manifest["budgets_pixels"]:
    if pixels in seen:
      continue
    epsilon = pixels / manifest["denominator"]
    _sync(device)
    started = time.monotonic()
    if pixels == 0:
      result = _clean(predict, inputs, labels, manifest["batch_size"], manifest["seed"])
    else:
      result = metrics.pgd(predict, inputs, labels, epsilon=epsilon,
        steps=manifest["steps"], step_size=epsilon / 4,
        batch_size=manifest["batch_size"], seed=manifest["seed"],
        restarts=manifest["restarts"], return_per_example=True)
    _sync(device)
    details = {key: result.pop(key) for key in
               ("robust_correct", "clean_correct", "worst_cross_entropy")}
    example_path = output / f"{model_id}_eps_{pixels:g}.npz"
    with example_path.with_suffix(".tmp").open("wb") as stream:
      np.savez_compressed(stream, test_indices=indices.numpy(), labels=labels.cpu().numpy(),
        robust_correct=np.asarray(details["robust_correct"], dtype=bool),
        clean_correct=np.asarray(details["clean_correct"], dtype=bool),
        worst_cross_entropy=np.asarray(details["worst_cross_entropy"], dtype=np.float32))
    example_path.with_suffix(".tmp").replace(example_path)
    result.update(epsilon_pixels=pixels, evaluation_seconds=time.monotonic() - started,
                  device=device, per_example_file=example_path.name)
    record["results"].append(result)
    seen.add(pixels)
    record["results"].sort(key=lambda row: row["epsilon_pixels"])
    record["complete"] = {row["epsilon_pixels"] for row in record["results"]} == set(manifest["budgets_pixels"])
    _json(result_path, record)
    print(json.dumps(dict(model=model_id, **result)), flush=True)
  return record


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--runs", type=Path, required=True)
  parser.add_argument("--model", required=True)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  run(args.runs, args.model, args.device)


if __name__ == "__main__":
  main()
