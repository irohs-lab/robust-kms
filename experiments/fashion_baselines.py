import functools
import json
import pathlib
import subprocess
import traceback
import torch
import experiments.attack_model as attacks
import experiments.baseline_kernel as kernels
import experiments.fashion_data as data_loader
import experiments.report as report
import experiments.train_eigenpro as eigenpro
import experiments.train_svm as svm


def main():
  source = pathlib.Path("runs/fashion01-matern52-lambda0-val")
  original = json.loads((source / "config.json").read_text())
  torch.manual_seed(original["seed"])
  torch.backends.cuda.matmul.allow_tf32 = False
  x, y, test_x, test_y = data_loader.load(original["data_root"], original["device"])
  split = torch.load(source / "split_indices.pt", weights_only=True)
  val_x, val_y = x[split["validation"]], y[split["validation"]]
  x, y = x[split["train"]], y[split["train"]]
  kernel = functools.partial(kernels.matrix, length_scale=original["length_scale"])
  matrices = [kernel(points, x) for points in (x, val_x, test_x)]
  for name, trainer in (("eigenpro2", eigenpro), ("svm", svm)):
    output = pathlib.Path("runs/fashion01-baseline-" + name)
    output.mkdir(parents=True, exist_ok=False)
    config = original | dict(output=str(output), baseline=name, rho=0.,
      lam=None, code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
      text=True).strip(), svm_backend="sklearn-libsvm", explicit_ridge=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.save(split, output / "split_indices.pt")
    try:
      selected = trainer.run(x, (y, val_y, test_y), matrices, config)
      alpha = selected.pop("alpha")
      support = alpha != 0
      options = {key: config[key] for key in (
        "kernel", "length_scale", "query_tile", "center_tile")}
      model = dict(centers=x[support], alpha=alpha[support],
        beta=torch.zeros_like(x[support]), bias=selected["bias"],
        options=options, config=config, selection=selected)
      torch.save(model, output / "model.pt")
      (output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
      report.write(output, "selected", **selected)
      attacks.run(model, test_x, test_y, config)
    except Exception:
      report.write(output, "failed", traceback=traceback.format_exc())
      raise


if __name__ == "__main__":
  main()
