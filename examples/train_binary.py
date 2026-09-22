import json
import math
import pathlib
import torch
import robust_kernels as rk
import examples._arguments as arguments


def main():
  args = arguments.parse()
  data = torch.load(args.data, map_location="cpu", weights_only=True)
  x = data["x"].to(device=args.device, dtype=torch.float32)
  y = data["y"].to(device=args.device, dtype=torch.float32)
  scale = math.sqrt(x.shape[1]) if args.length_scale is None else args.length_scale
  options = dict(kernel=args.kernel, length_scale=scale,
                 query_tile=args.query_tile, center_tile=args.center_tile)
  state = rk.fit(x, y, rho=args.rho, lam=args.lam, step_size=args.step_size,
                 decay=args.decay, iterations=args.iterations,
                 batch_size=args.batch_size, seed=args.seed,
                 average=not args.no_average, **options)
  prefix = "" if args.no_average else "average_"
  alpha, beta = state[prefix + "alpha"], state[prefix + "beta"]
  if args.evaluate:
    scores = rk.objective(x, y, alpha, beta, rho=args.rho, lam=args.lam, **options)
    print(json.dumps({key: value.item() for key, value in scores.items()}))
  model = dict(centers=x, alpha=alpha, beta=beta, kernel=args.kernel,
               length_scale=scale, rho=args.rho, lam=args.lam,
               iterations=args.iterations, averaged=not args.no_average)
  pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
  torch.save(model, args.output)
  print("Saved " + args.output)


if __name__ == "__main__":
  main()
