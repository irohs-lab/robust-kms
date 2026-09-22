import torch
import robust_kernels._average as averaging
import robust_kernels._step_inputs as inputs
import robust_kernels.evaluate as evaluation


@torch.no_grad()
def step(centers, labels, state, batch, *, rho, lam, eta, kernel="gaussian",
         length_scale=1.0, query_tile=128, center_tile=1024):
  """Mutate state using simultaneous updates from the same old-state batch.

  Labels are binary {0, 1}. The objective is summed, not averaged.
  Non-sampled dual blocks remain unchanged. Returns the updated state.
  """
  inputs.validate(centers, labels, state, batch, rho, lam, eta)
  queries = centers.index_select(0, batch)
  values, gradients = evaluation.kernel_eval_and_grad(
    queries, centers, state["alpha"], state["beta"], kernel=kernel,
    length_scale=length_scale, query_tile=query_tile, center_tile=center_tile)
  targets = labels.index_select(0, batch).to(values.dtype)
  if not torch.all((targets == 0) | (targets == 1)):
    raise ValueError("logistic labels must be 0 or 1")
  residual = torch.sigmoid(values) - targets
  old_delta = state["delta"].index_select(0, batch)
  averaging.update(state, eta)
  scale = eta * len(centers) / batch.numel()
  shrink = 1 - eta * lam
  state["alpha"].mul_(shrink).index_add_(0, batch, residual, alpha=-scale)
  state["beta"].mul_(shrink).index_add_(0, batch, old_delta, alpha=-scale)
  old_delta.add_(gradients, alpha=scale).clamp_(-rho, rho)
  state["delta"].index_copy_(0, batch, old_delta)
  state["iterations"] += 1
  return state
