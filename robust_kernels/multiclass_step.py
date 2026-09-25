import torch
import robust_kernels.lowrank_factors as factors
import robust_kernels.multiclass_inputs as inputs
import robust_kernels.multioutput_evaluate as evaluation


@torch.no_grad()
def step(centers, labels, state, batch, *, rho, lam, eta, kernel="gaussian",
         length_scale=1., query_tile=128, center_tile=1024):
  """One summed-loss RKHS subgradient step; projection is a separate operation."""
  inputs.validate(centers, labels, state, batch, rho, lam, eta)
  values, jacobian = evaluation.kernel_eval_and_grad(
    centers[batch], centers, state["alpha"], state["factors"],
    kernel=kernel, length_scale=length_scale, query_tile=query_tile,
    center_tile=center_tile)
  target = labels[batch].long()
  residual = values.softmax(1)
  residual[torch.arange(len(batch), device=batch.device), target] -= 1
  columns = jacobian.abs().sum(1)
  winners = columns.argmax(1)
  directions = jacobian.gather(
    2, winners[:, None, None].expand(-1, centers.shape[1], 1)).squeeze(2).sign()
  basis = torch.nn.functional.one_hot(winners, values.shape[1]).to(values.dtype)
  rate, shrink = eta * len(centers) / len(batch), 1 - eta * lam
  state["alpha"].mul_(shrink).index_add_(0, batch, residual, alpha=-rate)
  factors.scale(state["factors"], shrink)
  if rho:
    for row, index in enumerate(batch.tolist()):
      factors.add(state["factors"], index, -rate * rho * directions[row], basis[row])
  state["iterations"] += 1
  return dict(iteration=state["iterations"],
    batch_cross_entropy=torch.nn.functional.cross_entropy(values, target).item(),
    batch_accuracy=(values.argmax(1) == target).float().mean().item(),
    batch_jacobian_penalty=columns.max(1).values.mean().item())
