import json
import pathlib
import subprocess
import traceback
import torch
import experiments.fashion_config as configuration
import experiments.fashion_data as data_loader
import experiments.train_fashion as training
import experiments.attack_model as attacks
import experiments.report as report


def main():
  config = vars(configuration.parse())
  output = pathlib.Path(config["output"])
  output.mkdir(parents=True, exist_ok=True)
  try:
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    data = data_loader.load(config["data_root"], config["device"])
    x = data[0]
    config.update(epsilon=8 / 255, rho=8 / 255, classes=[0, 1],
                  train_samples=len(x), test_samples=len(data[2]), kernel="matern52")
    if config["length_scale"] is None:
      sample = x[torch.randperm(len(x), device=x.device)[:1024]]
      config["length_scale"] = torch.pdist(sample).median().item()
    if config["step_size"] is None:
      config["step_size"] = 1 / (config["lam"] + len(x) / 4)
    config["code_commit"] = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], text=True).strip()
    config["autoattack_commit"] = subprocess.check_output(
      ["git", "-C", ".deps/auto-attack", "rev-parse", "HEAD"], text=True).strip()
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report.write(output, "started", config=config)
    options = {key: config[key] for key in (
      "kernel", "length_scale", "query_tile", "center_tile")}
    model = training.run(data, config, options)
    attacks.run(model, data[2], data[3], config)
  except Exception:
    report.write(output, "failed", traceback=traceback.format_exc())
    raise


if __name__ == "__main__":
  main()
