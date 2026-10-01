"""Render validation-selected learning-rate trials without mixing training budgets."""
import argparse
import csv
import json
import math
from itertools import cycle
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.multiclass_report import _history, _number, _read_json, _table


def _finite(value):
  return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric(record, key):
  value = record.get(key) if isinstance(record, dict) else None
  return value if _finite(value) else None


def _valid_scores(record):
  return (isinstance(record, dict) and _finite(record.get("cross_entropy"))
    and record["cross_entropy"] >= 0 and _finite(record.get("accuracy"))
    and 0 <= record["accuracy"] <= 1)


def build(runs, output):
  runs, output = Path(runs), Path(output)
  warnings = []
  manifest = _read_json(runs / "manifest.json", warnings)
  if manifest is None:
    raise ValueError("Missing or invalid trial manifest. " + " ".join(warnings))
  reference = Path(manifest["reference"])
  if not reference.is_absolute():
    reference = runs / reference
  output.mkdir(parents=True, exist_ok=True)
  entries = [dict(name="robust_reference", effective_step_full_batch=.01,
                 path=reference / "robust", label="Robust: step 0.01"),
             dict(name="baseline", effective_step_full_batch=.01,
                 path=reference / "baseline", label="Base CE: step 0.01")]
  for candidate in manifest["candidates"]:
    path = Path(candidate["output"])
    if not path.is_absolute():
      path = runs / path
    entries.append(dict(candidate, path=path / "robust",
                        label=f"Robust: step {candidate['effective_step_full_batch']:g}"))
  histories, flat = {}, []
  selection_path = runs / "selection.json"
  selection = _read_json(selection_path, warnings)
  pilot_selection = _read_json(runs / "pilot_selection.json", warnings)
  extensions = manifest.get("extension_candidates", [])
  if selection and not isinstance(selection.get("name"), str):
    warnings.append("Ignored selection without a candidate name.")
    selection = None
  for entry in entries:
    events = []
    recorded = _history(entry["path"] / "metrics.jsonl", warnings, events)
    valid = {}
    for item in recorded:
      if not _finite(item.get("epoch")) or not _valid_scores(item.get("validation")):
        warnings.append(f"Skipped invalid validation record in {entry['name']} "
                        f"at epoch {item.get('epoch')}.")
        continue
      # A resumed run can log an epoch again; retain its latest complete record.
      valid[item["epoch"]] = item
    history = [valid[epoch] for epoch in sorted(valid)]
    histories[entry["name"]] = history
    pilot = next((row for row in history if row["epoch"] == manifest["pilot_epochs"]), None)
    result_path = entry["path"] / "result.json"
    result = _read_json(result_path, warnings) or {}
    if result and (not _valid_scores(result.get("test"))
                   or not _finite(result.get("selected_epoch"))):
      warnings.append(f"Ignored incomplete or invalid test result for {entry['name']}.")
      result = {}
    best = min(history, key=lambda row: row["validation"]["cross_entropy"]) if history else None
    projected = [item for item in history if isinstance(item.get("projection"), dict)]
    projection = [item["projection"] for item in projected]
    latest_projection = projection[-1] if projection else {}
    errors = [p["max_value_error"] for p in projection if _metric(p, "max_value_error") is not None]
    status = str(events[-1].get("event") or "unknown") if events else "not_started"
    result_epochs = _metric(result, "epochs_run")
    stale = bool(result and history and result_epochs is not None
                 and result_epochs < history[-1]["epoch"])
    if stale:
      warnings.append(f"{entry['name']} test results cover training through epoch "
        f"{result_epochs}; the history now reaches epoch {history[-1]['epoch']}.")
    if result and result_epochs is None:
      warnings.append(f"{entry['name']} test result has no valid epochs_run metadata; "
                      "its freshness cannot be established.")
    if result and status in ("complete", "not_started") and not stale:
      status = "evaluated"
    residual_errors = [p["error_squared"] for p in projection if _metric(p, "error_squared") is not None]
    ratios = [p["error_squared"] / p["initial_error_squared"] for p in projection
              if _metric(p, "error_squared") is not None
              and _metric(p, "initial_error_squared") is not None
              and p["initial_error_squared"] > 0]
    row = dict(name=entry["name"], status=status, effective_step_full_batch=entry["effective_step_full_batch"],
      pilot_epoch=manifest["pilot_epochs"], pilot_validation_ce=None,
      pilot_validation_accuracy=None, epochs_run=history[-1]["epoch"] if history else 0,
      best_validation_epoch=best["epoch"] if best else None,
      best_validation_ce=best["validation"]["cross_entropy"] if best else None,
      best_validation_accuracy=best["validation"]["accuracy"] if best else None,
      selected_for_extension=(entry["name"] in extensions if extensions else
        bool(selection and selection.get("name") == entry["name"])),
      final_validation_selection=bool(selection and selection.get("name") == entry["name"]),
      max_projection_logit_error=max(errors) if errors else None,
      max_projection_error_squared=max(residual_errors) if residual_errors else None,
      max_projection_remaining_fraction=max(ratios) if ratios else None,
      latest_projection_epoch=projected[-1]["epoch"] if projected else None,
      latest_projection_initial_distance_squared=_metric(latest_projection, "initial_error_squared"),
      latest_projection_final_distance_squared=_metric(latest_projection, "error_squared"),
      latest_projection_reason=latest_projection.get("reason"),
      latest_projection_factor_steps=_metric(latest_projection, "iterations"),
      latest_solve_relative_residual=_metric(latest_projection.get("kernel_solve"), "max_relative_residual"),
      latest_solve_scaled_residual=_metric(latest_projection.get("kernel_solve"), "max_scaled_residual"),
      training_seconds=_metric(history[-1], "training_seconds") if history else None,
      selected_test_epoch=result.get("selected_epoch"), result_training_epochs=result_epochs,
      test_result_from_earlier_training=stale,
      test_accuracy=_metric(result.get("test"), "accuracy"),
      test_ce=_metric(result.get("test"), "cross_entropy"),
      pgd_accuracy=_metric(result.get("attack"), "robust_accuracy"))
    if pilot:
      row.update(pilot_validation_ce=pilot["validation"]["cross_entropy"],
                 pilot_validation_accuracy=pilot["validation"]["accuracy"])
    flat.append(row)
  with (output / "comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=flat[0].keys())
    writer.writeheader()
    writer.writerows(flat)
  figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
  colors = ("#777777", "#000000", "#0072B2", "#009E73", "#D55E00")
  for entry, color in zip(entries, cycle(colors)):
    history = histories[entry["name"]]
    for axis, key, scale in zip(axes, ("cross_entropy", "accuracy"), (1, 100)):
      axis.plot([row["epoch"] for row in history],
                [scale * row["validation"][key] for row in history],
                "--" if entry["name"] == "baseline" else "-o",
                color=color, markersize=3, label=entry["label"])
  for axis, ylabel in zip(axes, ("Validation cross-entropy", "Validation accuracy (%)")):
    axis.axvline(manifest["pilot_epochs"], color="gray", alpha=.4, linestyle=":")
    axis.set(xlabel="Epoch", ylabel=ylabel)
    axis.grid(alpha=.2)
    axis.legend(fontsize=8)
  figure.savefig(output / "learning_rates.png", dpi=180)
  plt.close(figure)
  figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
  for entry, color in zip(entries, cycle(colors)):
    history = [row for row in histories[entry["name"]]
               if _metric(row, "training_seconds") is not None and row["training_seconds"] > 0]
    axis.plot([row["training_seconds"] for row in history],
              [row["validation"]["cross_entropy"] for row in history],
              "--" if entry["name"] == "baseline" else "-o", color=color,
              markersize=3, label=entry["label"])
  axis.set(xscale="log", xlabel="Recorded training time (seconds; log scale)",
           ylabel="Validation cross-entropy")
  axis.grid(alpha=.2)
  axis.legend(fontsize=8)
  figure.savefig(output / "learning_rates_time.png", dpi=180)
  plt.close(figure)
  lines = ["# FashionMNIST: robust learning-rate comparison", "",
    "Same ten-class split, Matérn-5/2 kernel, seed 42, rho = 8/255, lambda = 0, "
    "minibatch 128, constant step, and projection after every epoch as the reference run. "
    "The baseline uses plain kernel cross-entropy updates without preconditioning. "
    "Robust projection uses dense scalar-kernel storage and EigenPro2 with relative "
    "solve tolerance 1e-3 and at most three factor refinement steps.", "",
    "Step means the effective coefficient step on a full minibatch: "
    "`raw_eta * 54000 / 128`. The final minibatch has 112 examples, so its step "
    "is 128/112 times this value. Larger rates change only the robust primal step.", "",
    "## Equal-budget pilot", "",
    f"The protocol screens candidates after {manifest['pilot_epochs']} complete projected epochs "
    "using validation cross-entropy, then extends the chosen candidates to "
    f"{manifest['final_epochs']} epochs and retains its best validation checkpoint. "
    "Test metrics are not used to choose the rate. Recorded completion is shown below.", ""]
  lines += _table(("Model", "Step", "Pilot validation CE", "Pilot validation accuracy", "Selected for extension"), [
    (row["name"], row["effective_step_full_batch"], _number(row["pilot_validation_ce"]),
     _number(row["pilot_validation_accuracy"], percent=True), "yes" if row["selected_for_extension"] else "—")
    for row in flat]) + [""]
  if pilot_selection:
    lines += ["Pilot winner: `" + pilot_selection["name"] + "`.", ""]
  if manifest.get("protocol_amendment"):
    lines += ["Protocol amendment: " + manifest["protocol_amendment"], ""]
  if selection:
    lines += ["Final validation selection: `" + selection["name"] + "`. " + str(selection.get("reason", "")), ""]
  else:
    lines += ["The final validation selection has not yet been recorded.", ""]
  lines += ["## Available checkpoints", "",
    "Blank test entries mean no final test result is available. "
    "Candidates stopped at the pilot have a smaller training budget. Best validation "
    "metrics use all recorded epochs; test metrics belong to the explicitly listed "
    "test checkpoint. Earlier evaluation artifacts remain labeled and generate a warning.", ""]
  lines += _table(("Model", "Status", "Epochs run", "Best validation epoch", "Best validation CE",
                   "Test checkpoint epoch", "Test accuracy", "Test CE", "PGD accuracy"), [
    (row["name"], row["status"] + ("; earlier evaluation" if row["test_result_from_earlier_training"] else ""),
     row["epochs_run"], row["best_validation_epoch"] if row["best_validation_epoch"] is not None else "—",
     _number(row["best_validation_ce"]),
     row["selected_test_epoch"] if row["selected_test_epoch"] is not None else "—",
     _number(row["test_accuracy"], percent=True), _number(row["test_ce"]),
     _number(row["pgd_accuracy"], percent=True)) for row in flat]) + ["",
    "Clean test accuracy uses 10,000 examples. PGD uses the same fixed, stratified "
    "1,000-example subset as the reference: Linf 8/255, 20 steps of size 2/255, "
    "one random start. This is empirical attack accuracy from one training seed.", "",
    "## Projection diagnostics", ""]
  lines += _table(("Model", "Largest final squared RKHS distance", "Largest remaining distance fraction",
                   "Largest training-logit correction error", "Training seconds"), [
    (row["name"], _number(row["max_projection_error_squared"]),
     _number(row["max_projection_remaining_fraction"]),
     _number(row["max_projection_logit_error"], 6), _number(row["training_seconds"], 1))
    for row in flat]) + ["",
    "The remaining distance fraction is the final squared RKHS distance divided by "
    "the initial squared distance for each projection that needed compression; the table "
    "shows the largest observed fraction. A fixed three-step refinement budget "
    "compares the implemented algorithms, not exact projected subgradient descent. "
    "The projection is nonconvex and approximate. Small kernel-solve residuals "
    "do not establish a globally optimal rank-one projection. Squared RKHS distance "
    "measures displacement from the unprojected target, not a constraint violation "
    "or a certified optimality gap; even an exact nearest projection can have positive "
    "distance. Rank-one feasibility holds after projection independently of the "
    "refinement stopping reason. The first epoch "
    "already has rank-one blocks; later epochs exercise the compression step. "
    "Training times come from different GPU models and are not hardware-matched speed comparisons.", ""]
  lines += ["Latest completed projection per model:", ""]
  lines += _table(("Model", "Epoch", "Initial squared distance", "Final squared distance",
                   "Factor steps", "Termination", "Solve relative residual", "Solve scaled residual"), [
    (row["name"], row["latest_projection_epoch"] if row["latest_projection_epoch"] is not None else "—",
     _number(row["latest_projection_initial_distance_squared"]),
     _number(row["latest_projection_final_distance_squared"]),
     row["latest_projection_factor_steps"] if row["latest_projection_factor_steps"] is not None else "—",
     row["latest_projection_reason"] or "—", _number(row["latest_solve_relative_residual"], 6),
     _number(row["latest_solve_scaled_residual"], 6)) for row in flat]) + ["",
    "The scaled solve residual divides each output residual norm by its absolute-plus-relative "
    "stopping threshold; at most one means that threshold was met. No solve is needed "
    "for the already-rank-one shortcut.", "",
    "![Validation learning curves](learning_rates.png)", "",
    "![Validation CE versus recorded training time](learning_rates_time.png)", "",
    "The time axis is logarithmic. Runs used different GPU assignments (recorded in "
    "manifest.json), so this shows observed cost rather than a hardware-controlled "
    "throughput benchmark. Training seconds include projection work but exclude "
    "initial model/kernel-cache setup, validation, checkpoint I/O and final test evaluation.", "",
    "[Machine-readable comparison](comparison.csv)", ""]
  objectives = []
  for entry in entries:
    if entry["name"] == "baseline":
      continue
    measured = _read_json(entry["path"].parent / "pilot_objective.json", warnings)
    if measured and measured.get("epoch") == manifest["pilot_epochs"]:
      objectives.append((entry["name"], measured.get("samples", "—"),
        _number(measured.get("cross_entropy")),
        _number(measured.get("jacobian_penalty_mean")),
        _number(measured.get("mean_surrogate_objective"))))
  if objectives:
    lines += ["## Training surrogate at the pilot boundary", "",
      "Measured after projection on the same fixed stratified training subset. "
      "The mean surrogate is mean CE + rho times mean(max_class sum_input abs(J)). "
      "Lambda is zero. These diagnostics were not used to select the learning rate.", ""]
    lines += _table(("Model", "Samples", "Mean CE", "Mean Jacobian penalty", "Mean surrogate"),
                    objectives) + [""]
  final_objectives = []
  for entry in entries:
    if entry["name"] == "baseline":
      continue
    measured = _read_json(entry["path"].parent / "final_objective.json", warnings)
    if measured:
      final_objectives.append((entry["name"], measured.get("epoch", "—"),
        _number(measured.get("cross_entropy")),
        _number(measured.get("jacobian_penalty_mean")),
        _number(measured.get("mean_surrogate_objective"))))
  if final_objectives:
    lines += ["## Training surrogate at each run's best validation checkpoint", "",
      "The same fixed 1,000-example training subset is used. These diagnostics "
      "do not use test data and do not change the validation-CE selection rule.", ""]
    lines += _table(("Model", "Checkpoint epoch", "Mean CE", "Mean Jacobian penalty", "Mean surrogate"),
                    final_objectives) + [""]
  sensitivity = _read_json(runs / "projection_sensitivity.json", warnings)
  if sensitivity:
    lines += ["## Projection refinement sensitivity", "",
      "One replay of epoch four at effective step 0.30 resumed the same saved "
      "epoch-three state on the same GPU. Only the factor-refinement budget changed "
      "from three to ten. All logged preprojection minibatch metrics, the initial "
      "projection distance and the first three refinement iterates matched exactly. "
      "This run was excluded from learning-rate selection and had no test evaluation.", ""]
    sensitivity_rows = []
    for name in ("original", "projection_steps10"):
      item = sensitivity[name]
      projection = item["projection"]
      sensitivity_rows.append((projection["iterations"],
        _number(projection["error_squared"]),
        _number(item["validation"]["cross_entropy"], 7),
        _number(item["validation"]["accuracy"], percent=True),
        _number(projection["elapsed_seconds"], 1), projection["reason"]))
    lines += _table(("Factor steps", "Final squared RKHS distance", "Validation CE",
      "Validation accuracy", "Projection seconds", "Termination"), sensitivity_rows) + ["",
      "Additional refinement reduced distance but had negligible immediate impact "
      "on clean validation performance in this replay. This single-epoch check "
      "does not establish the effect of using ten steps throughout training.", "",
      "[Complete sensitivity measurements](projection_sensitivity.json)", ""]
  storage = _read_json(runs / "storage_benchmark.json", warnings)
  if storage:
    memory, timing = storage["memory"], storage["timings"]
    lines += ["## Scalar-kernel storage benchmark", "",
      "The KLR new_api packed operator was benchmarked on a synthetic symmetric "
      "54,000 by 54,000 float32 matrix with ten output columns. This measures "
      "operator arithmetic, not FashionMNIST training or radial-kernel construction. "
      "Device: " + storage["environment"]["device_name"] + ".", ""]
    lines += _table(("Storage", "Matrix GiB", "128-row product (ms)", "Full product (ms)"), [
      (name, _number(memory[name + "_matrix_storage_bytes"] / 2**30, 3),
       _number(timing[name + "_rows"]["median_seconds"] * 1000, 3),
       _number(timing[name + "_full"]["median_seconds"] * 1000, 3))
      for name in ("dense", "packed")]) + ["",
      "The packed constructor temporarily holds both dense and packed storage "
      "(16.30 GiB for the matrices alone). Both products passed analytic-reference "
      "and dense-versus-packed checks. The tested packed matrix-right-hand-side "
      "path loops over diagonal rows and rectangular tiles in Python. Dense storage "
      "fits each available GPU and was retained for the training trials. "
      "Derivative contractions remain factored and tiled.", "",
      "[Full benchmark measurements and provenance](storage_benchmark.json)", ""]
  if warnings:
    lines += ["Input warnings:", ""] + ["- " + item for item in warnings] + [""]
  (output / "summary.md").write_text("\n".join(lines))
  return flat


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--runs", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  print(json.dumps(build(args.runs, args.output), indent=2))


if __name__ == "__main__":
  main()
