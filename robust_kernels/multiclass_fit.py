import math
import torch
import robust_kernels.multiclass_state as initialization
import robust_kernels.multiclass_step as updates
import robust_kernels.rkhs_projection as projection


def fit(centers, labels, *, outputs, rho, lam=1., eta=1e-3, steps=1000,
        batch_size=128, decay=.6, project_every=100, projection_options=None,
        seed=0, state=None, callback=None, final_projection=True, **kernel_options):
  """Sample fresh minibatches; periodically call the separate RKHS projector.

  No dense iterate averaging is maintained. The interval is a heuristic.
  callback(state, event) receives batch metrics and projection diagnostics.
  """
  if type(steps) is not int or steps < 0:
    raise ValueError("steps must be a nonnegative integer")
  if type(batch_size) is not int or not 1 <= batch_size <= len(centers):
    raise ValueError("batch_size must lie in 1..n")
  if type(project_every) is not int or project_every < 1:
    raise ValueError("project_every must be a positive integer")
  if not math.isfinite(decay) or decay < 0:
    raise ValueError("decay must be finite and nonnegative")
  if state is None:
    state = initialization.initialize(centers, outputs)
  elif state["alpha"].shape != (len(centers), outputs):
    raise ValueError("state shape does not match centers and outputs")
  generator = torch.Generator(device=centers.device).manual_seed(seed)
  options = dict(projection_options or {}) | kernel_options
  for _ in range(steps):
    batch = torch.randperm(len(centers), generator=generator,
                           device=centers.device)[:batch_size]
    event = updates.step(centers, labels, state, batch, rho=rho, lam=lam,
      eta=eta / (state["iterations"] + 1) ** decay, **kernel_options)
    if state["iterations"] - state["last_projection"] >= project_every:
      event["projection"] = projection.project(centers, state, **options)
    if callback is not None:
      callback(state, event)
  if final_projection and state["iterations"] != state["last_projection"]:
    report = projection.project(centers, state, **options)
    if callback is not None:
      callback(state, dict(iteration=state["iterations"], projection=report))
  return state
