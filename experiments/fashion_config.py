import argparse


def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-root", default="/janaki/common/Datasets/rahulky")
  parser.add_argument("--output", required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--eval-every", type=int, default=1)
  parser.add_argument("--lam", type=float, default=1.)
  parser.add_argument("--step-size", type=float, default=None)
  parser.add_argument("--decay", type=float, default=0.6)
  parser.add_argument("--length-scale", type=float, default=None)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--attack-batch-size", type=int, default=128)
  parser.add_argument("--center-tile", type=int, default=4096)
  parser.add_argument("--query-tile", type=int, default=128)
  parser.add_argument("--validation-fraction", type=float, default=0.)
  parser.add_argument("--patience", type=int, default=10)
  return parser.parse_args()
