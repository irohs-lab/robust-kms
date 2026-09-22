import datetime
import json
import os
import pathlib
import subprocess
import sys
import experiments.fashion_config as configuration
import experiments.report as report


def main():
  args = configuration.parse()
  root = pathlib.Path(__file__).resolve().parents[1]
  output = pathlib.Path(args.output).resolve()
  output.mkdir(parents=True, exist_ok=False)
  environment = os.environ.copy()
  environment["PYTHONPATH"] = os.pathsep.join((str(root / ".deps/auto-attack"), str(root)))
  command = ["flock", "/tmp/robust-kernels-gpu0.lock", sys.executable, "-u",
             "-m", "experiments.fashion_binary", *sys.argv[1:]]
  report.write(output, "queued", command=command, device=args.device)
  with (output / "console.log").open("w") as stream:
    process = subprocess.Popen(command, cwd=root, env=environment,
      stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
  launch = dict(pid=process.pid, command=command,
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
  print(json.dumps(launch), flush=True)


if __name__ == "__main__":
  main()
