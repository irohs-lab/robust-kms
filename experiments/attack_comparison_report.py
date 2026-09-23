import csv
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experiments.collect_attack_sweep as sweeps


def main():
  output = pathlib.Path("reports/fashion01-all-attack-budgets")
  output.mkdir(parents=True, exist_ok=True)
  read = lambda path: json.loads(pathlib.Path(path).read_text())
  previous = read("reports/fashion01-budget-sweep/results.json")
  table = {"Primal-dual, lambda=1": {
    row["budget_pixels"]: row["robust_test_accuracy"] for row in previous}}
  baseline = read("reports/fashion01-baselines/results.json")
  for name, label, checkpoint, reference in (
    ("lambda0", "Primal-dual, lambda=0", "fashion01-matern52-lambda0-val",
      read("reports/fashion01-matern52-lambda0-val/result.json")),
    ("eigenpro2", "EigenPro2, early stopping", "fashion01-baseline-eigenpro2",
      baseline["eigenpro2"]["result"]),
    ("svm", "Kernel SVM, lambda=0.1", "fashion01-baseline-svm",
      baseline["svm"]["result"])):
    table[label] = sweeps.collect(name, "runs/" + checkpoint + "/model.pt",
                                  reference, output)
  budgets = [0, 1, 2, 4, 8, 12, 16]
  with (output / "accuracy.csv").open("w") as handle:
    writer = csv.writer(handle)
    writer.writerow(["model"] + ["epsilon=" + str(b) + "/255" for b in budgets])
    writer.writerows([name] + [100 * row[b] for b in budgets]
                     for name, row in table.items())
  lines = ["# Fashion-MNIST 0 vs 1: all attack budgets", "",
    "Test accuracy (%) on all 2,000 official test images. Epsilon=0 is clean.",
    "Same fixed checkpoints and binary-compatible AutoAttack (APGD-CE, FAB, Square), seed 42.",
    "APGD-CE/FAB: 100 iterations, 5 restarts; Square: 5,000 queries.", "",
    "| Model | " + " | ".join("eps=" + str(b) + "/255" for b in budgets) + " |",
    "|---|" + "---:|" * len(budgets)]
  for name, row in table.items():
    lines.append("| " + name + " | " + " | ".join(f"{100 * row[b]:.2f}" for b in budgets) + " |")
    plt.plot(budgets, [100 * row[b] for b in budgets], ".-", label=name)
  lines += ["", "Lambda=1 used 12,000 training samples and length scale 8.90477.",
    "The other models share 10,800 training / 1,200 validation samples and length scale 9.05994.",
    "No retraining or test-based model selection. Existing 8/255 results are reused.",
    "Individual attack runs are empirical evaluations, not robustness certificates.", "",
    "![Accuracy vs attack budget](accuracy.png)"]
  (output / "summary.md").write_text("\n".join(lines) + "\n")
  (output / "results.json").write_text(json.dumps(table, indent=2) + "\n")
  plt.xlabel("Pixel-space Linf budget (epsilon × 255)")
  plt.ylabel("Test accuracy (%)")
  plt.xticks(budgets)
  plt.legend()
  plt.grid(alpha=.25)
  plt.tight_layout()
  plt.savefig(output / "accuracy.png", dpi=180)


if __name__ == "__main__":
  main()
