import csv
import hashlib
import json
import pathlib
import shutil
import experiments.baseline_plots as plots


def main():
  output = pathlib.Path("reports/fashion01-baselines")
  output.mkdir(parents=True, exist_ok=True)
  histories, summaries = {}, {}
  for name in ("eigenpro2", "svm"):
    source = pathlib.Path("runs/fashion01-baseline-" + name)
    target = output / name
    target.mkdir(exist_ok=True)
    for filename in ("config.json", "selection.json", "autoattack.log"):
      shutil.copyfile(source / filename, target / filename)
    result = json.loads((source / "status.json").read_text())
    if result["event"] != "complete":
      raise RuntimeError(name + " is not complete")
    (target / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = [json.loads(line) for line in (source / "metrics.jsonl").read_text().splitlines()]
    histories[name] = [row for row in rows if row["event"] in ("epoch", "candidate")]
    with (target / "training_metrics.csv").open("w") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(histories[name][0]))
      writer.writeheader()
      writer.writerows(histories[name])
    selection = json.loads((source / "selection.json").read_text())
    summaries[name] = dict(selection=selection, result=result,
      checkpoint=str((source / "model.pt").resolve()),
      sha256=hashlib.sha256((source / "model.pt").read_bytes()).hexdigest())
  (output / "results.json").write_text(json.dumps(summaries, indent=2) + "\n")
  plots.save(histories, output)


if __name__ == "__main__":
  main()
