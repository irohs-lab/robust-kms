import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(histories, output):
  figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
  for axis, (name, rows) in zip(axes, histories.items()):
    key = "epoch" if name == "eigenpro2" else "lam"
    rows = sorted(rows, key=lambda row: row[key])
    for split in ("train", "validation", "test"):
      axis.plot([row[key] for row in rows],
        [100 * row[split + "_accuracy"] for row in rows], ".-", label=split)
    axis.set(xlabel="Epoch" if key == "epoch" else "SVM lambda",
      ylabel="Clean accuracy (%)", title=name)
    if key == "lam":
      axis.set_xscale("log")
    axis.grid(alpha=.25)
    axis.legend()
  figure.savefig(output / "accuracy.png", dpi=180)
  plt.close(figure)
