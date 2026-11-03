import math
import torch
import robust_kernels.multiclass_state as initialization
import robust_kernels.multiclass_step as updates
import robust_kernels.rkhs_projection as projection


def fit(centers, labels, *, outputs, rho, lam=1., eta=1e-3, epochs=10,
        batch_size=128, decay=.6, project_every=1, projection_options=None,
        seed=0, state=None, callback=None, final_projection=True, **kernel_options):
  """Shuffle once per epoch and project in the RKHS every project_every epochs.

  Each epoch visits every sample once, including the final smaller minibatch.
  States saved at epoch boundaries resume the schedule and shuffle sequence;
  seed is used only when the state has no saved generator state.
  callback(state, event) receives batch metrics and projection diagnostics.
  final_projection also compresses an unfinished interval at the end of a call.
  """
  if "steps" in kernel_options:
    raise TypeError("fit_multiclass no longer accepts steps; use epochs instead "
                    "(project_every is also measured in epochs)")
  if type(epochs) is not int or epochs < 0:
    raise ValueError("epochs must be a nonnegative integer")
  if type(batch_size) is not int or not 1 <= batch_size <= len(centers):
    raise ValueError("batch_size must lie in 1..n")
  if type(project_every) is not int or project_every < 1:
    raise ValueError("project_every must be a positive integer")
  if not math.isfinite(decay) or decay < 0:
    raise ValueError("decay must be finite and nonnegative")
  if any(not math.isfinite(value) or value < 0 for value in (rho, lam, eta)):
    raise ValueError("rho and lam must be nonnegative; eta must be positive")
  if eta == 0 or eta * lam > 1:
    raise ValueError("require eta > 0 and eta*lam <= 1")
  if state is None:
    state = initialization.initialize(centers, outputs)
  elif state["alpha"].shape != (len(centers), outputs):
    raise ValueError("state shape does not match centers and outputs")
  generator = torch.Generator(device=centers.device).manual_seed(seed)
  if "fit_generator_state" in state:
    generator.set_state(state["fit_generator_state"].cpu())
  state.setdefault("epochs", 0)
  state.setdefault("last_projection_epoch", 0)
  options = dict(projection_options or {}) | kernel_options
  for _ in range(epochs):
    epoch = state["epochs"] + 1
    order = torch.randperm(len(centers), generator=generator,
                           device=centers.device)
    for batch_number, start in enumerate(range(0, len(centers), batch_size), 1):
      batch = order[start:start + batch_size]
      event = updates.step(centers, labels, state, batch, rho=rho, lam=lam,
        eta=eta / (state["iterations"] + 1) ** decay, **kernel_options)
      epoch_end = start + batch_size >= len(centers)
      event.update(epoch=epoch, batch_in_epoch=batch_number, epoch_end=epoch_end)
      if epoch_end:
        state["epochs"] = epoch
        state["fit_generator_state"] = generator.get_state()
        if epoch - state["last_projection_epoch"] >= project_every:
          event["projection"] = projection.project(centers, state, **options)
          state["last_projection_epoch"] = epoch
      if callback is not None:
        callback(state, event)
  if final_projection and state["iterations"] != state["last_projection"]:
    report = projection.project(centers, state, **options)
    state["last_projection_epoch"] = state["epochs"]
    if callback is not None:
      callback(state, dict(iteration=state["iterations"], epoch=state["epochs"],
        batch_in_epoch=None, epoch_end=True, projection=report))
  return state
