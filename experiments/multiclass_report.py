"""Render a transparent comparison from shared multiclass run artifacts."""

import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = {"robust": "RKHS adversarial training", "baseline": "Base kernel cross-entropy"}
FIELDS = (
  "model", "status", "stopping_reason", "selected_epoch", "epochs_run", "training_seconds",
  "validation_samples", "validation_accuracy", "validation_cross_entropy",
  "test_samples", "test_accuracy", "test_cross_entropy",
  "attack_samples", "attack_clean_accuracy", "attack_robust_accuracy",
  "attack_cross_entropy", "attack", "epsilon", "steps", "step_size",
  "restarts", "attack_seed", "max_linf")


def _read_json(path, warnings):
  if not path.exists():
    return None
  try:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
      raise ValueError("expected a JSON object")
    return value
  except (ValueError, OSError) as error:
    warnings.append(f"Could not read {path}: {error}")
    return None


def _history(path, warnings, events=None):
  rows = []
  if not path.exists():
    return rows
  for number, line in enumerate(path.read_text().splitlines(), 1):
    if not line.strip():
      continue
    try:
      row = json.loads(line)
    except ValueError:
      warnings.append(f"Skipped incomplete or invalid JSON at {path}:{number}.")
      continue
    if events is not None and isinstance(row, dict):
      events.append(row)
    if (isinstance(row, dict) and row.get("event", "epoch") == "epoch"
        and isinstance(row.get("epoch"), (int, float))):
      rows.append(row)
  return sorted(rows, key=lambda row: row["epoch"])


def _stopping_reason(result, config, events):
  epochs = result.get("epochs_run")
  if any(event.get("event") == "early_stopped" and event.get("epoch") == epochs
         for event in events):
    return "patience exhausted (logged)"
  cap = result.get("epoch_limit", config.get("epochs", config.get("max_epochs")))
  if isinstance(epochs, (int, float)) and isinstance(cap, (int, float)) and epochs >= cap:
    return "epoch cap reached"
  patience, selected = config.get("patience"), result.get("selected_epoch")
  if (isinstance(epochs, (int, float)) and isinstance(selected, (int, float))
      and isinstance(patience, (int, float)) and epochs - selected >= patience):
    return "patience exhausted (inferred from selected/last epochs)"
  return "ended before configured cap; stopping reason not logged"


def _flat(name, result, history, config=None, events=()):
  complete = (result is not None and isinstance(result.get("validation"), dict)
              and isinstance(result.get("test"), dict)
              and result.get("selected_epoch") is not None)
  status = "complete" if complete else ("incomplete" if history or result else "not started")
  row = dict.fromkeys(FIELDS)
  row.update(model=name, status=status)
  if not complete:
    row["epochs_run"] = history[-1]["epoch"] if history else None
    return row
  row["stopping_reason"] = _stopping_reason(result, (config or {}) | (result.get("config") or {}), events)
  for key in ("selected_epoch", "epochs_run", "training_seconds"):
    row[key] = result.get(key)
  for split in ("validation", "test"):
    for key in ("samples", "accuracy", "cross_entropy"):
      row[split + "_" + key] = result[split].get(key)
  attack = result.get("attack")
  if isinstance(attack, dict):
    for key in ("samples", "clean_accuracy", "cross_entropy"):
      row["attack_" + key] = attack.get(key)
    row["attack_robust_accuracy"] = attack.get("robust_accuracy", attack.get("accuracy"))
    row["attack_seed"] = attack.get("seed")
    for key in ("attack", "epsilon", "steps", "step_size", "restarts", "max_linf"):
      row[key] = attack.get(key)
  return row


def _number(value, digits=4, percent=False):
  if not isinstance(value, (int, float)) or not math.isfinite(value):
    return "—"
  return f"{100 * value:.2f}%" if percent else f"{value:.{digits}f}"


def _integer(value):
  return str(value) if isinstance(value, (int, float)) else "—"


def _table(headers, rows):
  return ["| " + " | ".join(headers) + " |",
          "| " + " | ".join("---" for _ in headers) + " |"] + [
    "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
    for row in rows]


def _curves(histories, output):
  figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
  colors = {"robust": "#0072B2", "baseline": "#D55E00"}
  for axis, metric, ylabel in zip(axes, ("accuracy", "cross_entropy"),
                                  ("Accuracy (%)", "Cross-entropy")):
    plotted = False
    for name, rows in histories.items():
      for split, style, suffix in (("validation", "-", "validation"),
                                   ("train_subset", "--", "training subset")):
        available = [row for row in rows if isinstance(row.get(split), dict)
                     and isinstance(row[split].get(metric), (int, float))
                     and math.isfinite(row[split][metric])]
        if not available:
          continue
        plotted = True
        scale = 100 if metric == "accuracy" else 1
        axis.plot([row["epoch"] for row in available],
                  [scale * row[split][metric] for row in available], style,
                  color=colors[name], linewidth=1.7,
                  label=f"{MODELS[name]}: {suffix}")
    if plotted:
      axis.legend(fontsize=8)
    else:
      axis.text(.5, .5, "No epoch metrics available", ha="center", va="center",
                 transform=axis.transAxes)
    axis.set(xlabel="Epoch", ylabel=ylabel)
    axis.grid(alpha=.25)
  figure.savefig(output / "learning_curves.png", dpi=180)
  plt.close(figure)



def _scientific(value):
  return f"{value:.6g}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"


def _projection_summary(history):
  records = [(row["epoch"], row["projection"]) for row in history
             if isinstance(row.get("projection"), dict)]
  if not records:
    return ["## Projection diagnostics", "", "No projection diagnostics have been recorded.", ""]
  reasons = Counter(report.get("reason", "unreported") for _, report in records)
  lines = ["## Projection diagnostics", "",
           f"Diagnostics cover {len(records)} recorded projected epochs. "
           + "Termination reasons: " + ", ".join(f"`{reason}`: {count}"
                                                   for reason, count in sorted(reasons.items())) + ".", "",
           "`max_steps` means the factor-step budget was reached; `stationary` "
           "reports the solver's first-order stopping test. Neither establishes "
           "a globally optimal projection. `already_rank_one` means compression "
           "was unnecessary for that target.", "", "Latest recorded projection:", ""]
  epoch, latest = records[-1]
  solve = latest.get("kernel_solve", {})
  fields = (("Epoch", epoch), ("Termination reason", latest.get("reason", "—")),
            ("Factor steps", latest.get("iterations")),
            ("Initial squared RKHS error", latest.get("initial_error_squared")),
            ("Final squared RKHS error", latest.get("error_squared")),
            ("Maximum training-logit correction error", latest.get("max_value_error")),
            ("Final solve maximum relative residual", solve.get("max_relative_residual")),
            ("Final solve maximum scaled residual", solve.get("max_scaled_residual")),
            ("Full-center EigenPro diagonal bound", solve.get("beta")),
            ("EigenPro solve calls", latest.get("kernel_solve_calls")),
            ("EigenPro epochs across all solves", latest.get("kernel_solve_epochs")),
            ("Projection seconds", latest.get("elapsed_seconds")))
  lines += _table(("Diagnostic", "Value"), [
    (key, value if isinstance(value, str) else _scientific(value)) for key, value in fields]) + [""]
  aggregate = []
  for label, key, reduction in (
      ("Largest initial squared RKHS error", "initial_error_squared", max),
      ("Largest final squared RKHS error", "error_squared", max),
      ("Largest training-logit correction error", "max_value_error", max),
      ("Total EigenPro solve calls", "kernel_solve_calls", sum),
      ("Total EigenPro solver epochs", "kernel_solve_epochs", sum),
      ("Total projection seconds", "elapsed_seconds", sum)):
    values = [report[key] for _, report in records
              if isinstance(report.get(key), (int, float)) and math.isfinite(report[key])]
    aggregate.append((label, _scientific(reduction(values)) if values else "—"))
  lines += ["Across the recorded projection epochs:", ""]
  lines += _table(("Diagnostic", "Value"), aggregate) + ["",
    "Squared RKHS errors are the numerical values reported by the coupled "
    "projection objective. The logit error measures how accurately the value "
    "coefficient correction preserves training logits. EigenPro solver epochs "
    "count passes inside the linear solves, separately from training epochs.", ""]
  return lines


def _summary(runs, config, results, histories, rows, warnings):
  lines = ["# FashionMNIST: ten-class kernel comparison", "",
           f"Run artifacts: `{runs.resolve()}`.", ""]
  incomplete = [MODELS[row["model"]] for row in rows if row["status"] != "complete"]
  if incomplete:
    lines += ["**Comparison incomplete.** Final test results are unavailable for "
              + ", ".join(incomplete) + ". Missing values below are not estimates.", ""]
  lines += ["Both models use raw image pixels in [0, 1], all ten FashionMNIST "
            "classes, a shared stratified training/validation split, and the "
            "official test set. Bandwidth is estimated from training data only.", "",
            "The robust model uses the implemented RKHS stochastic primal/dual "
            "updates with rank-one derivative coefficients restored at projection "
            "boundaries. Its regularizer is the maximum classwise input-gradient "
            "L1 norm. The baseline minimizes multiclass cross-entropy using the "
            "KLR model and plain kernel SGD, without EigenPro preconditioning.", "",
            "The coupled RKHS projection is approximate: it starts with rank-one "
            "block approximations and takes at most "
            + str(config.get("projection_steps", "the configured number of"))
            + " factor refinement steps per projection. No global projection "
            "optimality is claimed. EigenPro is used only for the robust model's "
            "projection linear solves. The optional `dense` EigenPro storage "
            "mode caches the scalar training kernel matrix; it does not change "
            "the projection objective or add preconditioning to the baseline.", ""]
  settings = []
  keys = ("dataset", "classes", "outputs", "seed", "train_samples", "validation_samples",
          "test_samples", "train_per_class", "validation_fraction", "kernel",
          "length_scale", "bandwidth", "lam", "lambda", "rho", "epsilon",
          "eta", "robust_eta", "robust_decay", "baseline_eta", "batch_size", "project_every",
          "projection_steps", "solve_rtol", "solve_atol", "solve_max_epochs", "eigenpro_storage",
          "eigenpro_samples", "eigenpro_rank",
          "max_epochs", "epochs", "patience", "min_delta", "selection_metric", "selection",
          "code_commit", "baseline_update", "penalty", "attack_per_class",
          "attack_samples", "attack_epsilon", "attack_steps", "attack_step_size")
  for key in keys:
    if key in config:
      settings.append((key, json.dumps(config[key], ensure_ascii=False)))
  if settings:
    lines += ["## Shared run configuration", ""]
    lines += _table(("Setting", "Value"), settings) + [""]
  lines += ["## Selected checkpoints", "",
            "Final metrics below come only from saved result files. Checkpoint "
            "selection uses validation cross-entropy; test performance is reported "
            "after selection. Training time is the accumulated training time "
            "recorded by the runner, not total wall time.", ""]
  lines += _table(("Model", "Status", "Training stop", "Selected epoch", "Epochs run", "Training seconds"), [
    (MODELS[row["model"]], row["status"], row["stopping_reason"] or "—", _integer(row["selected_epoch"]),
     _integer(row["epochs_run"]), _number(row["training_seconds"], 1)) for row in rows])
  lines += ["", "Reaching the epoch cap is not evidence of optimizer convergence. "
            "The selected epoch remains the best validation checkpoint among "
            "the epochs actually run.", ""]
  lines += _projection_summary(histories["robust"])
  lines += ["## Clean evaluation", ""]
  lines += _table(("Model", "Validation n", "Validation accuracy", "Validation CE",
                    "Test n", "Test accuracy", "Test CE"), [
    (MODELS[row["model"]], _integer(row["validation_samples"]),
     _number(row["validation_accuracy"], percent=True), _number(row["validation_cross_entropy"]),
     _integer(row["test_samples"]), _number(row["test_accuracy"], percent=True),
     _number(row["test_cross_entropy"])) for row in rows])
  lines += ["", "## Adversarial evaluation", "",
            "PGD cross-entropy attacks use a fixed, stratified test subset shared "
            "by both models. Reported robust accuracy counts a sample as correct "
            "only if the clean image, random start, and every attack iterate "
            "remain correctly classified. It is empirical accuracy under this "
            "attack, not a certificate or an AutoAttack result. Attack CE is the "
            "mean largest CE encountered per sample.", ""]
  lines += _table(("Model", "Attack n", "Subset clean accuracy", "PGD accuracy", "Attack CE"), [
    (MODELS[row["model"]], _integer(row["attack_samples"]),
     _number(row["attack_clean_accuracy"], percent=True),
     _number(row["attack_robust_accuracy"], percent=True),
     _number(row["attack_cross_entropy"])) for row in rows])
  lines += [""]
  lines += _table(("Model", "Attack", "Linf budget", "Steps", "Step size", "Restarts",
                    "Seed", "Largest observed Linf"), [
    (MODELS[row["model"]], row["attack"] or "not evaluated", _number(row["epsilon"], 6),
     _integer(row["steps"]), _number(row["step_size"], 6), _integer(row["restarts"]),
     _integer(row["attack_seed"]), _number(row["max_linf"], 6)) for row in rows])
  lines += ["", "A single split and seed do not measure run-to-run variability. "
            "These aggregate artifacts do not support paired bootstrap intervals "
            "or significance tests; none are claimed.", ""]
  class_rows = []
  for index in range(10):
    values = [str(index)]
    for name in MODELS:
      result = results[name] or {}
      for split in ("test", "attack"):
        section = result.get(split)
        accuracies = section.get("per_class_accuracy", []) if isinstance(section, dict) else []
        values.append(_number(accuracies[index], percent=True) if index < len(accuracies) else "—")
    class_rows.append(values)
  if any(value != "—" for row in class_rows for value in row[1:]):
    lines += ["## Per-class accuracy", ""]
    lines += _table(("Class", "Robust model: clean", "Robust model: PGD",
                      "Base kernel: clean", "Base kernel: PGD"), class_rows) + [""]
  lines += ["## Learning curves", "", "![Training-subset and validation learning curves](learning_curves.png)",
            "", "Dashed lines use the fixed training evaluation subset, not the "
            "entire training set. No test metrics are used in these curves.", "",
            "Machine-readable aggregate results: [comparison.csv](comparison.csv).", ""]
  reference = _read_json(runs / "baseline_200_reference" / "result.json", warnings)
  if reference is not None and isinstance(reference.get("test"), dict):
    attack = reference.get("attack") or {}
    lines += ["## Longer baseline reference", "",
      "This separately retained baseline uses more epochs than the first comparison. "
      "It shares the split, kernel, learning rate, and attack subset. Its performance "
      "does not establish a difference under matched training budgets.", ""]
    lines += _table(("Model", "Epochs run", "Selected epoch", "Clean test accuracy",
                     "Test CE", "PGD accuracy", "Training seconds"), [
      ("Base kernel cross-entropy", reference.get("epochs_run", "—"),
       reference.get("selected_epoch", "—"),
       _number(reference["test"].get("accuracy"), percent=True),
       _number(reference["test"].get("cross_entropy")),
       _number(attack.get("robust_accuracy"), percent=True),
       _number(reference.get("training_seconds"), 1))]) + [""]
  if warnings:
    lines += ["## Input warnings", ""] + ["- " + warning for warning in warnings] + [""]
  return "\n".join(lines)


def build(runs, output):
  """Write a summary, CSV, and plot; never substitute metrics for missing runs."""
  runs, output = Path(runs), Path(output)
  output.mkdir(parents=True, exist_ok=True)
  warnings = []
  config = _read_json(runs / "config.json", warnings) or {}
  results, histories, rows = {}, {}, []
  for name in MODELS:
    results[name] = _read_json(runs / name / "result.json", warnings)
    events = []
    histories[name] = _history(runs / name / "metrics.jsonl", warnings, events)
    rows.append(_flat(name, results[name], histories[name], config, events))
  with (output / "comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
  _curves(histories, output)
  (output / "summary.md").write_text(
    _summary(runs, config, results, histories, rows, warnings))
  return rows


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--runs", type=Path, required=True,
                      help="Shared root containing config.json and model directories")
  parser.add_argument("--output", type=Path, required=True,
                      help="Directory for summary.md, comparison.csv, and learning_curves.png")
  arguments = parser.parse_args()
  rows = build(arguments.runs, arguments.output)
  print(json.dumps({"output": str(arguments.output.resolve()),
                    "models": {row["model"]: row["status"] for row in rows}}, indent=2))


if __name__ == "__main__":
  main()
