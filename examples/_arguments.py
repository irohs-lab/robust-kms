import argparse


def parse():
  parser = argparse.ArgumentParser(description="Train a scalar robust RKHS model.")
  parser.add_argument("--data", required=True, help="torch file with x and y tensors")
  parser.add_argument("--output", default="model.pt")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--kernel", choices=("gaussian", "matern52"), default="gaussian")
  parser.add_argument("--length-scale", type=float, default=None)
  parser.add_argument("--rho", type=float, required=True, help="L * epsilon")
  parser.add_argument("--lam", type=float, required=True)
  parser.add_argument("--step-size", type=float, required=True)
  parser.add_argument("--decay", type=float, default=0.75)
  parser.add_argument("--iterations", type=int, default=1000)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--query-tile", type=int, default=128)
  parser.add_argument("--center-tile", type=int, default=1024)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--no-average", action="store_true")
  parser.add_argument("--evaluate", action="store_true", help="full objective pass")
  return parser.parse_args()
