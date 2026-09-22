import argparse


def parse():
  parser = argparse.ArgumentParser(description="Attack one fixed Fashion-MNIST model.")
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 4, 12, 16])
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--attack-batch-size", type=int, default=128)
  args = parser.parse_args()
  if any(e <= 0 or e > 255 for e in args.budgets):
    parser.error("budgets must be integers in 1..255, interpreted as epsilon/255")
  if len(set(args.budgets)) != len(args.budgets):
    parser.error("budgets must be unique")
  return args
