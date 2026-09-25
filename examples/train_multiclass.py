import argparse
import json
import torch
import robust_kernels as rk


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()
  torch.manual_seed(7)
  x = torch.randn(48, 4, device=args.device, dtype=torch.float64)
  teacher = torch.randn(4, 3, device=args.device, dtype=x.dtype)
  labels = (x @ teacher).argmax(1)
  options = dict(kernel="matern52", length_scale=2., query_tile=16, center_tile=24)
  state = rk.fit_multiclass(x, labels, outputs=3, rho=2 * 8 / 255,
    lam=.1, eta=.01, steps=12, batch_size=12, project_every=4,
    projection_options=dict(max_steps=5, solve_rtol=1e-8, solve_atol=1e-10),
    callback=lambda state, event: print(json.dumps(event)), **options)
  values, _ = rk.multioutput_eval_and_grad(
    x, x, state["alpha"], state["factors"], gradients=False, **options)
  print(json.dumps(dict(train_accuracy=(values.argmax(1) == labels).double().mean().item(),
                       pending_blocks=len(state["factors"]["pending"]))))


if __name__ == "__main__":
  main()
