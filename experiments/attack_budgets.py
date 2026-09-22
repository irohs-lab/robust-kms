import hashlib
import json
import pathlib
import subprocess
import traceback
import torch
import experiments.attack_model as attacks
import experiments.budget_config as configuration
import experiments.fashion_data as data_loader
import experiments.report as report


def main():
  args = configuration.parse()
  output = pathlib.Path(args.output)
  output.mkdir(parents=True, exist_ok=False)
  try:
    checkpoint = pathlib.Path(args.checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = torch.load(checkpoint, map_location=args.device, weights_only=True)
    config = model["config"]
    torch.backends.cuda.matmul.allow_tf32 = False
    _, _, test_x, test_y = data_loader.load(config["data_root"], args.device)
    manifest = dict(checkpoint=str(checkpoint.resolve()), sha256=digest,
      training_epsilon=config["epsilon"], rho=config["rho"], budgets=args.budgets,
      seed=config["seed"], device=args.device, test_samples=len(test_x),
      code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           text=True).strip(),
      autoattack_commit=config["autoattack_commit"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report.write(output, "started", **manifest)
    results = []
    for budget in args.budgets:
      directory = output / ("eps" + str(budget) + "-255")
      directory.mkdir()
      current = config | dict(output=str(directory), epsilon=budget / 255,
        training_epsilon=config["epsilon"], device=args.device,
        attack_batch_size=args.attack_batch_size, checkpoint_sha256=digest)
      (directory / "config.json").write_text(json.dumps(current, indent=2) + "\n")
      attacks.run(model, test_x, test_y, current)
      result = json.loads((directory / "status.json").read_text())
      results.append(dict(budget_pixels=budget, **result))
      report.write(output, "budget_complete", budget_pixels=budget, result=result)
      (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    report.write(output, "complete", results=results)
  except Exception:
    report.write(output, "failed", traceback=traceback.format_exc())
    raise


if __name__ == "__main__":
  main()
