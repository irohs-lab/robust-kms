"""Shared FashionMNIST comparison: projected RKHS surrogate versus plain KLR."""
import argparse
import functools
import json
import math
import pathlib
import subprocess
import time
import traceback
import torch
import robust_kernels as rk
import experiments.baseline_kernel as kernels
import experiments.multiclass_baseline as baseline
import experiments.multiclass_data as data_io
import experiments.multiclass_metrics as metrics
import experiments.report as reporting


def _json(path, value):
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  temporary.replace(path)


def _save(path, value):
  temporary = path.with_suffix(path.suffix + ".tmp")
  torch.save(value, temporary)
  temporary.replace(path)


def _synchronize(device):
  if torch.device(device).type == "cuda":
    torch.cuda.synchronize(device)


def _scores(logits, labels):
  correct = logits.argmax(1) == labels
  counts = torch.bincount(labels, minlength=logits.shape[1])
  good = torch.bincount(labels[correct], minlength=logits.shape[1])
  return dict(accuracy=correct.float().mean().item(),
    cross_entropy=torch.nn.functional.cross_entropy(logits, labels).item(),
    per_class_accuracy=(good / counts.clamp_min(1)).tolist(), samples=len(labels))


def prepare(args):
  root = pathlib.Path(args.output)
  root.mkdir(parents=True, exist_ok=True)
  if (root / "config.json").exists():
    raise FileExistsError("shared run configuration already exists; choose a new output")
  data, split = data_io.load(args.data_root, args.device, seed=args.seed,
    train_per_class=args.train_per_class)
  x = data["train_x"]
  bandwidth = data_io.bandwidth(x, seed=args.seed)
  config = dict(dataset="FashionMNIST", classes=list(range(10)), seed=args.seed,
    data_root=args.data_root, train_per_class=args.train_per_class,
    validation_fraction=.1, train_samples=len(x), validation_samples=len(data["val_x"]),
    test_samples=len(data["test_x"]), kernel="matern52", length_scale=bandwidth,
    rho=args.rho, epsilon=args.epsilon, lam=0., epochs=args.epochs,
    patience=args.patience, batch_size=args.batch_size, project_every=1,
    robust_eta=args.robust_eta if args.robust_eta is not None else 2 / len(x),
    robust_decay=args.robust_decay, baseline_eta=args.baseline_eta,
    query_tile=args.query_tile, center_tile=args.center_tile,
    projection_steps=args.projection_steps, eigenpro_storage=args.eigenpro_storage,
    eigenpro_samples=args.eigenpro_samples, eigenpro_rank=args.eigenpro_rank,
    solve_rtol=args.solve_rtol,
    solve_atol=args.solve_atol, solve_max_epochs=args.solve_max_epochs,
    attack_per_class=args.attack_per_class, attack_steps=args.attack_steps,
    attack_step_size=2 / 255, attack_batch_size=64,
    selection="lowest validation cross-entropy after complete projected epochs",
    baseline_update="alpha[batch] -= eta * (softmax(logits) - one_hot(labels)); no preconditioner",
    penalty="rho * max_class sum_input abs(J[class,input])",
    code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
  patch = subprocess.check_output(["git", "diff", "HEAD", "--", "robust_kernels", "experiments"], text=True)
  config["code_dirty"] = bool(patch)
  if patch:
    (root / "source.patch").write_text(patch)
    config["source_patch"] = "source.patch"
  _save(root / "split_indices.pt", split)
  _save(root / "attack_indices.pt", data_io.stratified_indices(
    data["test_y"], args.attack_per_class, seed=args.seed))
  _json(root / "config.json", config)
  print(json.dumps(config), flush=True)


def train(args):
  root = pathlib.Path(args.output)
  config = json.loads((root / "config.json").read_text())
  output = root / args.model
  output.mkdir(parents=True, exist_ok=True)
  if (output / "latest.pt").exists() and not args.resume:
    raise FileExistsError("training checkpoint already exists; pass --resume")
  device = args.device
  torch.manual_seed(config["seed"])
  torch.backends.cuda.matmul.allow_tf32 = False
  data, split = data_io.load(config["data_root"], device, seed=config["seed"],
    train_per_class=config["train_per_class"])
  saved_split = torch.load(root / "split_indices.pt", weights_only=True)
  if any(not torch.equal(split[key], saved_split[key]) for key in split):
    raise RuntimeError("data split does not match the shared run")
  x, y = data["train_x"], data["train_y"]
  options = {key: config[key] for key in ("kernel", "length_scale", "query_tile", "center_tile")}
  train_indices = data_io.stratified_indices(y, min(100, len(x) // 10), seed=config["seed"])
  train_x, train_y = x[train_indices], y[train_indices]
  started = time.monotonic()
  selection = dict(best_loss=math.inf, best_epoch=0, stale=0)
  training_seconds, epoch_start, state = 0., 0, None
  checkpoint = None
  if args.resume:
    checkpoint = torch.load(output / "latest.pt", map_location=device, weights_only=True)
    selection, training_seconds = checkpoint["selection"], checkpoint["training_seconds"]
    epoch_start = checkpoint["epoch"]
  if args.model == "robust":
    state = checkpoint["state"] if checkpoint else rk.initialize_multiclass(x, 10)
    val_operator, train_operator = None, None
  else:
    kernel = functools.partial(kernels.matrix, length_scale=config["length_scale"])
    state = baseline.initialize(x, y, kernel=kernel, outputs=10,
      batch_size=config["batch_size"], eta=config["baseline_eta"], seed=config["seed"])
    if checkpoint:
      state["model"].weights.copy_(checkpoint["alpha"])
      state["iterations"], state["epochs"] = checkpoint["iterations"], checkpoint["epoch"]
      state["generator"].set_state(checkpoint["generator_state"].cpu())
    val_operator = state["model"].operator(data["val_x"], storage="dense", batch_size=2048)
    train_operator = state["model"].operator(train_x, storage="dense", batch_size=2048)
  del checkpoint
  reporting.write(output, "started", model=args.model, device=device, config=config,
                  resume_epoch=epoch_start, setup_seconds=time.monotonic() - started)
  max_epochs = args.train_epochs if args.train_epochs is not None else config["epochs"]
  if selection["stale"] >= config["patience"]:
    max_epochs = epoch_start
  epoch = epoch_start
  alpha, factors, predict, saved = None, None, None, None
  for epoch in range(epoch_start + 1, max_epochs + 1):
    projection = None
    _synchronize(device)
    tick = time.monotonic()
    if args.model == "robust":
      def callback(current, event):
        if event.get("epoch_end") or current["iterations"] % 100 == 0:
          reporting.write(output, "batch", **event)
      def projection_progress(event):
        details = dict(event)
        reporting.write(output, details.pop("phase"), epoch=epoch, **details)
      try:
        state = rk.fit_multiclass(x, y, outputs=10, rho=config["rho"], lam=config["lam"],
          eta=config["robust_eta"], decay=config["robust_decay"], epochs=1,
          batch_size=config["batch_size"], project_every=config["project_every"],
          final_projection=False, seed=config["seed"], state=state, callback=callback,
          projection_options=dict(max_steps=config["projection_steps"],
            progress=projection_progress,
            eigenpro_storage=config.get("eigenpro_storage", "matfree"),
            eigenpro_samples=config.get("eigenpro_samples", 1024),
            eigenpro_rank=config.get("eigenpro_rank", 100),
            solve_rtol=config["solve_rtol"], solve_atol=config["solve_atol"],
            solve_max_epochs=config["solve_max_epochs"]), **options)
      except Exception:
        _save(output / "failed_state.pt", dict(state=state, epoch=state["epochs"]))
        raise
      alpha, factors = state["alpha"], state["factors"]
      projection = state.get("projection")
    else:
      baseline.epoch(state)
      alpha, factors = state["model"].weights, None
    _synchronize(device)
    epoch_seconds = time.monotonic() - tick
    training_seconds += epoch_seconds
    predict = functools.partial(metrics.multiclass_logits, centers=x, alpha=alpha,
                                 factors=factors, **options)
    if args.model == "baseline":
      validation = _scores(val_operator @ alpha, data["val_y"])
      training = _scores(train_operator @ alpha, train_y)
    else:
      validation = metrics.evaluate(predict, data["val_x"], data["val_y"])
      training = metrics.evaluate(predict, train_x, train_y)
    if not math.isfinite(validation["cross_entropy"]):
      raise RuntimeError("nonfinite validation cross-entropy")
    record = dict(epoch=epoch, validation=validation, train_subset=training,
      training_seconds=training_seconds, epoch_training_seconds=epoch_seconds,
      elapsed_seconds=time.monotonic() - started, projection=projection)
    reporting.write(output, "epoch", **record)
    if validation["cross_entropy"] < selection["best_loss"]:
      selection.update(best_loss=validation["cross_entropy"], best_epoch=epoch, stale=0)
      _save(output / "model.pt", dict(alpha=alpha, factors=factors, epoch=epoch,
        validation=validation, options=options, config=config))
    else:
      selection["stale"] += 1
    saved = dict(epoch=epoch, selection=selection, training_seconds=training_seconds)
    if args.model == "robust":
      saved["state"] = state
    else:
      saved.update(alpha=alpha, iterations=state["iterations"],
                   generator_state=state["generator"].get_state())
    _save(output / "latest.pt", saved)
    if selection["stale"] >= config["patience"]:
      reporting.write(output, "early_stopped", epoch=epoch, **selection)
      break
  if args.train_only:
    reporting.write(output, "training_paused", epoch=state["epochs"], **selection)
    return
  del predict, alpha, factors, state, saved, val_operator, train_operator
  if torch.device(device).type == "cuda":
    torch.cuda.empty_cache()
  selected = torch.load(output / "model.pt", map_location=device, weights_only=True)
  predict = functools.partial(metrics.multiclass_logits, centers=x,
    alpha=selected["alpha"], factors=selected["factors"], **options)
  reporting.write(output, "evaluating", selected_epoch=selected["epoch"])
  test = metrics.evaluate(predict, data["test_x"], data["test_y"])
  attack = None
  if not args.no_attack:
    indices = torch.load(root / "attack_indices.pt", weights_only=True)
    reporting.write(output, "attacking", samples=len(indices), steps=config["attack_steps"])
    attack = metrics.pgd(predict, data["test_x"][indices], data["test_y"][indices],
      epsilon=config["epsilon"], steps=config["attack_steps"],
      step_size=config["attack_step_size"], batch_size=config["attack_batch_size"],
      seed=config["seed"])
  result = dict(model=args.model, selected_epoch=selected["epoch"],
    epochs_run=epoch, epoch_limit=max_epochs, validation=selected["validation"], test=test, attack=attack,
    training_seconds=training_seconds, config=config)
  _json(output / "result.json", result)
  reporting.write(output, "complete", **result)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", required=True)
  parser.add_argument("--prepare", action="store_true")
  parser.add_argument("--model", choices=("robust", "baseline"))
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--data-root", default="/janaki/common/Datasets/rahulky")
  parser.add_argument("--epochs", type=int, default=200)
  parser.add_argument("--train-epochs", type=int)
  parser.add_argument("--patience", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--baseline-eta", type=float, default=.01)
  parser.add_argument("--robust-eta", type=float)
  parser.add_argument("--robust-decay", type=float, default=.6)
  parser.add_argument("--rho", type=float, default=8 / 255)
  parser.add_argument("--epsilon", type=float, default=8 / 255)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--query-tile", type=int, default=256)
  parser.add_argument("--center-tile", type=int, default=4096)
  parser.add_argument("--projection-steps", type=int, default=3)
  parser.add_argument("--eigenpro-storage", choices=("matfree", "dense"), default="dense")
  parser.add_argument("--eigenpro-samples", type=int, default=1024)
  parser.add_argument("--eigenpro-rank", type=int, default=100)
  parser.add_argument("--solve-rtol", type=float, default=1e-3)
  parser.add_argument("--solve-atol", type=float, default=1e-6)
  parser.add_argument("--solve-max-epochs", type=int, default=100)
  parser.add_argument("--attack-per-class", type=int, default=100)
  parser.add_argument("--attack-steps", type=int, default=20)
  parser.add_argument("--train-per-class", type=int)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--train-only", action="store_true")
  parser.add_argument("--no-attack", action="store_true")
  args = parser.parse_args()
  if not args.prepare and args.model is None:
    parser.error("choose --prepare or --model")
  try:
    prepare(args) if args.prepare else train(args)
  except Exception:
    if not args.prepare and args.model is not None:
      output = pathlib.Path(args.output) / args.model
      output.mkdir(parents=True, exist_ok=True)
      reporting.write(output, "failed", traceback=traceback.format_exc())
    raise


if __name__ == "__main__":
  main()
