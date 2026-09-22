import datetime
import json
import pathlib


def write(output, event, **fields):
  output = pathlib.Path(output)
  record = dict(event=event, utc=datetime.datetime.now(
    datetime.timezone.utc).isoformat(), **fields)
  line = json.dumps(record, allow_nan=False)
  with (output / "metrics.jsonl").open("a") as stream:
    stream.write(line + "\n")
  temporary = output / "status.tmp"
  temporary.write_text(line + "\n")
  temporary.replace(output / "status.json")
  print(line, flush=True)
