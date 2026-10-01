"""Render saved multiclass PGD sweeps without filling missing measurements."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIELDS = (
  "model_id", "label", "variant", "model_status", "point_status",
  "selected_epoch", "training_epochs", "checkpoint", "checkpoint_sha256",
  "epsilon_pixels", "epsilon", "attack", "robust_accuracy", "clean_accuracy",
  "cross_entropy", "samples", "steps", "restarts", "step_size", "seed",
  "max_linf", "per_class_accuracy")


def _finite(value):
  return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _read(path, warnings):
  if not path.exists():
    return None
  try:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
      raise ValueError("expected a JSON object")
    return value
  except (OSError, ValueError) as error:
    warnings.append(f"Could not read {path}: {error}")
    return None


def _number(value, percent=False):
  if not _finite(value):
    return "—"
  return f"{100 * value:.2f}%" if percent else f"{value:g}"


def _table(headers, rows):
  def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")
  return ["| " + " | ".join(map(cell, headers)) + " |",
          "| " + " | ".join("---" for _ in headers) + " |"] + [
    "| " + " | ".join(map(cell, row)) + " |" for row in rows]


def _budget(point, denominator):
  if _finite(point.get("epsilon_pixels")):
    return point["epsilon_pixels"]
  if _finite(point.get("epsilon")):
    return point["epsilon"] * denominator
  return None


def _same_budget(left, right):
  return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _collect(runs, manifest, warnings):
  models = manifest.get("models", [])
  if not isinstance(models, list) or any(not isinstance(model, dict) for model in models):
    raise ValueError("manifest.models must be a list of objects")
  denominator = manifest.get("denominator", 255)
  if not _finite(denominator) or denominator <= 0:
    raise ValueError("manifest.denominator must be a positive finite number")
  budgets = manifest.get("budgets_pixels", [])
  if not isinstance(budgets, list) or any(not _finite(value) or value < 0 for value in budgets):
    raise ValueError("manifest.budgets_pixels must contain nonnegative finite numbers")
  budgets = sorted(set(budgets))
  identifiers = [model.get("id") for model in models]
  if any(not isinstance(value, str) or not value or Path(value).name != value
         or value in (".", "..") for value in identifiers):
    raise ValueError("every model must have a nonempty filename-safe id")
  if len(set(identifiers)) != len(identifiers):
    raise ValueError("model ids must be unique")
  rows, statuses = [], []
  for model in models:
    name = model["id"]
    result = _read(runs / "results" / (name + ".json"), warnings)
    points = result.get("results", []) if result is not None else []
    if not isinstance(points, list):
      warnings.append(f"{name}: results is not a list; no points were read.")
      points = []
    recorded = []
    for index, point in enumerate(points):
      if not isinstance(point, dict):
        warnings.append(f"{name}: skipped non-object result {index}.")
        continue
      budget = _budget(point, denominator)
      if budget is None or budget < 0:
        warnings.append(f"{name}: skipped result {index} with an invalid epsilon.")
        continue
      if any(_same_budget(budget, prior[0]) for prior in recorded):
        warnings.append(f"{name}: duplicate epsilon {budget:g}; all recorded points are preserved in CSV.")
      recorded.append((budget, point))
    missing = [budget for budget in budgets
               if not any(_same_budget(budget, actual) and _finite(point.get("robust_accuracy"))
                          for actual, point in recorded)]
    complete = result is not None and result.get("complete") is True and not missing
    status = "complete" if complete else ("partial" if result is not None else "missing")
    if result is not None and result.get("complete") is True and missing:
      warnings.append(f"{name}: marked complete but lacks valid accuracy for budgets "
                      + ", ".join(_number(value) for value in missing) + ".")
    if result is not None:
      saved_model = result.get("model", {})
      if isinstance(saved_model, dict):
        for key in ("id", "checkpoint_sha256", "selected_epoch", "training_epochs"):
          if key in model and key in saved_model and model[key] != saved_model[key]:
            warnings.append(f"{name}: result model.{key} differs from the manifest; "
                            "CSV metadata follows the result file.")
      else:
        saved_model = {}
      protocol = result.get("protocol", {})
      if isinstance(protocol, dict):
        for key in ("denominator", "steps", "restarts", "seed", "samples", "step_size_rule"):
          if key in protocol and key in manifest and protocol[key] != manifest[key]:
            warnings.append(f"{name}: recorded protocol {key}={protocol[key]!r} differs "
                            f"from manifest {manifest[key]!r}; compare with care.")
    else:
      saved_model = {}
    metadata = model | saved_model
    base = dict(model_id=name, label=metadata.get("label", name),
                variant=metadata.get("variant"), model_status=status,
                **{key: metadata.get(key) for key in
                   ("selected_epoch", "training_epochs", "checkpoint", "checkpoint_sha256")})
    statuses.append(base | {"missing_budgets": missing, "recorded_points": len(recorded)})
    for budget, point in sorted(recorded, key=lambda pair: pair[0]):
      row = dict.fromkeys(FIELDS)
      row.update(base)
      row.update({key: point.get(key) for key in FIELDS if key in point})
      # Identity and provenance always come from model metadata, never a point.
      row.update(base)
      row["epsilon_pixels"] = budget
      row["point_status"] = "measured" if _finite(point.get("robust_accuracy")) else "incomplete"
      if not _finite(point.get("robust_accuracy")):
        warnings.append(f"{name}: epsilon {budget:g} has no finite robust_accuracy.")
      if isinstance(row["per_class_accuracy"], (list, dict)):
        row["per_class_accuracy"] = json.dumps(row["per_class_accuracy"], separators=(",", ":"))
      rows.append(row)
    for budget in budgets:
      if any(_same_budget(budget, actual) for actual, _ in recorded):
        continue
      row = dict.fromkeys(FIELDS)
      row.update(base, epsilon_pixels=budget, point_status="missing")
      rows.append(row)
  return rows, statuses, budgets, denominator


def _curves(rows, statuses, budgets, denominator, output):
  figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
  plotted = False
  for index, status in enumerate(statuses):
    available = sorted((row for row in rows if row["model_id"] == status["model_id"]
                        and row["point_status"] == "measured"),
                       key=lambda row: row["epsilon_pixels"])
    if not available:
      continue
    plotted = True
    color = plt.get_cmap("tab10")(index % 10)
    label = (f"{status['label']} (trained {_number(status['training_epochs'])} ep, "
             f"selected {_number(status['selected_epoch'])})")
    # Plot the raw measurements; finite PGD estimates need not be monotone.
    axis.plot([row["epsilon_pixels"] for row in available],
              [100 * row["robust_accuracy"] for row in available],
              marker="o", markersize=4, linewidth=1.4, color=color, label=label)
    clean = [row for row in available if _same_budget(row["epsilon_pixels"], 0)]
    if clean:
      axis.scatter([0] * len(clean), [100 * row["robust_accuracy"] for row in clean],
                   marker="s", s=35, color=color, zorder=4)
  if plotted:
    axis.legend(fontsize=8, loc="upper right")
  else:
    axis.text(.5, .5, "No attack measurements available", ha="center", va="center",
              transform=axis.transAxes)
  all_budgets = sorted(set(budgets + [row["epsilon_pixels"] for row in rows]))
  if all_budgets:
    axis.set_xticks(all_budgets, ["0 (clean)" if value == 0 else _number(value)
                                for value in all_budgets])
  axis.set(xlabel=f"L∞ perturbation budget (pixel units; ε = value/{denominator:g})",
           ylabel="Empirical accuracy under PGD (%)", ylim=(0, 100),
           title="FashionMNIST: saved checkpoints across perturbation budgets")
  axis.grid(alpha=.25)
  figure.savefig(output / "robustness_curves.png", dpi=180)
  plt.close(figure)


def render(runs, output):
  runs, output = Path(runs), Path(output)
  warnings = []
  manifest = _read(runs / "manifest.json", warnings)
  if manifest is None:
    raise ValueError("A readable manifest.json is required. " + " ".join(warnings))
  rows, statuses, budgets, denominator = _collect(runs, manifest, warnings)
  output.mkdir(parents=True, exist_ok=True)
  with (output / "comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
  _curves(rows, statuses, budgets, denominator, output)
  lines = ["# FashionMNIST: PGD across perturbation budgets", "",
           f"Run artifacts: `{runs.resolve()}`.", "",
           "These are empirical attacks on saved checkpoints. Training budgets and "
           "selected epochs are listed explicitly; checkpoints are not selected by "
           "attack performance. The evaluation does not certify robustness and does "
           "not constitute AutoAttack or an exhaustive attack search.", ""]
  if any(status["model_status"] != "complete" for status in statuses):
    lines += ["**Evaluation incomplete.** Available measurements are retained; "
              "missing points are shown as — and are not estimated.", ""]
  lines += ["## Protocol", ""]
  protocol = [("L∞ budgets (pixel units)", ", ".join(_number(value) for value in budgets)),
              ("Pixel denominator", _number(denominator)),
              ("Samples", _number(manifest.get("samples"))),
              ("PGD steps", _number(manifest.get("steps"))),
              ("Random restarts", _number(manifest.get("restarts"))),
              ("Attack seed", _number(manifest.get("seed"))),
              ("Step-size rule", manifest.get("step_size_rule", "—"))]
  lines += _table(("Setting", "Value"), protocol) + ["",
            "The ε=0 point is clean accuracy on the attack sample. Reported values "
            "cover the recorded sample and seed only, without uncertainty estimates "
            "across training or attack seeds. Finite PGD runs can produce nonmonotonic "
            "accuracy across budgets. The figure shows raw measured points joined "
            "for readability; no monotonic correction or missing-point estimates "
            "are applied. Per-point settings and checkpoint hashes are in comparison.csv.", "",
            "## Checkpoints", ""]
  lines += _table(("Model", "Variant", "Status", "Training epochs", "Selected epoch", "Points recorded"), [
    (status["label"], status["variant"] or "—", status["model_status"],
     _number(status["training_epochs"]), _number(status["selected_epoch"]), status["recorded_points"])
    for status in statuses]) + ["", "## Empirical accuracy", ""]
  display_budgets = sorted(set(budgets + [row["epsilon_pixels"] for row in rows]))
  table_rows = []
  for status in statuses:
    values = [status["label"]]
    for budget in display_budgets:
      points = [row for row in rows if row["model_id"] == status["model_id"]
                and _same_budget(row["epsilon_pixels"], budget)
                and row["point_status"] == "measured"]
      values.append(" / ".join(_number(row["robust_accuracy"], percent=True) for row in points)
                    if points else "—")
    table_rows.append(values)
  headers = ["Model"] + ["0 (clean)" if value == 0 else f"{value:g}/{denominator:g}"
                         for value in display_budgets]
  lines += _table(headers, table_rows) + ["", "![Raw PGD accuracy by budget](robustness_curves.png)", ""]
  if warnings:
    lines += ["## Data warnings", ""] + ["- " + warning for warning in warnings] + [""]
  (output / "summary.md").write_text("\n".join(lines))
  return {"models": len(statuses), "complete": sum(status["model_status"] == "complete"
                                                   for status in statuses),
          "measured_points": sum(row["point_status"] == "measured" for row in rows),
          "warnings": warnings, "output": str(output.resolve())}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--runs", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  print(json.dumps(render(args.runs, args.output), indent=2))


if __name__ == "__main__":
  main()
