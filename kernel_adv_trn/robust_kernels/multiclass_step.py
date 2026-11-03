import torch
import robust_kernels.lowrank_factors as factors
import robust_kernels.multiclass_inputs as inputs
import robust_kernels.multioutput_evaluate as evaluation
import robust_kernels.multiclass_dual as dual


@torch.no_grad()
def step(centers, labels, state, batch, *, rho, lam, eta, kernel="gaussian",
         length_scale=1., query_tile=128, center_tile=1024):
  """Refresh the exact dual maximizer, then take one RKHS primal step.

  The rank-one dual blocks maximize their pairing with the old Jacobians;
  they are computed on demand rather than retained for dual ascent.
  Projection is a separate operation. The objective is a summed loss.
  """
  inputs.validate(centers, labels, state, batch, rho, lam, eta)
  values, jacobian = evaluation.kernel_eval_and_grad(
    centers[batch], centers, state["alpha"], state["factors"],
    kernel=kernel, length_scale=length_scale, query_tile=query_tile,
    center_tile=center_tile)
  target = labels[batch].long()
  residual = values.softmax(1)
  residual[torch.arange(len(batch), device=batch.device), target] -= 1
  delta_left, delta_right, penalty = dual.maximize(jacobian, rho)
  rate, shrink = eta * len(centers) / len(batch), 1 - eta * lam
  state["alpha"].mul_(shrink).index_add_(0, batch, residual, alpha=-rate)
  factors.scale(state["factors"], shrink)
  if rho:
    for row, index in enumerate(batch.tolist()):
      factors.add(state["factors"], index, -rate * delta_left[row], delta_right[row])
  state["iterations"] += 1
  return dict(iteration=state["iterations"],
    batch_cross_entropy=torch.nn.functional.cross_entropy(values, target).item(),
    batch_accuracy=(values.argmax(1) == target).float().mean().item(),
    batch_jacobian_penalty=penalty.mean().item())
