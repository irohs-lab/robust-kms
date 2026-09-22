import torch
import robust_kernels as rk


def run(x, y, state, config, options, generator):
  """An epoch is ceil(n/b) fresh draws, not a reshuffled partition."""
  steps = (len(x) + config["batch_size"] - 1) // config["batch_size"]
  for _ in range(steps):
    batch = torch.randperm(len(x), device=x.device, generator=generator)
    batch = batch[:config["batch_size"]]
    eta = config["step_size"] / (state["iterations"] + 1) ** config["decay"]
    rk.step(x, y, state, batch, rho=config["rho"], lam=config["lam"], eta=eta,
            **options)
