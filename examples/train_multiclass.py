import argparse
import json
import torch
import robust_kernels as rk


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--epochs", type=int, default=3)
  parser.add_argument("--project-every", type=int, default=1,
                      help="number of epochs between RKHS projections")
  parser.add_argument("--batch-size", type=int, default=12)
  args = parser.parse_args()
  torch.manual_seed(7)
  x = torch.randn(48, 4, device=args.device, dtype=torch.float64)
  teacher = torch.randn(4, 3, device=args.device, dtype=x.dtype)
  labels = (x @ teacher).argmax(1)
  options = dict(kernel="matern52", length_scale=2., query_tile=16, center_tile=24)
  state = rk.fit_multiclass(x, labels, outputs=3, rho=2 * 8 / 255,
    lam=.1, eta=.01, epochs=args.epochs, batch_size=args.batch_size,
    project_every=args.project_every,
    projection_options=dict(max_steps=5, solve_rtol=1e-8, solve_atol=1e-10),
    callback=lambda state, event: print(json.dumps(event)), **options)
  values, _ = rk.multioutput_eval_and_grad(
    x, x, state["alpha"], state["factors"], gradients=False, **options)
  print(json.dumps(dict(train_accuracy=(values.argmax(1) == labels).double().mean().item(),
                       pending_blocks=len(state["factors"]["pending"]),
                       epochs=state["epochs"], iterations=state["iterations"])))


if __name__ == "__main__":
  main()
