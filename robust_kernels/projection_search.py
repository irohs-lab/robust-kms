import robust_kernels.projection_residual as residual


def trial(centers, target, candidate, values, options, solver, workspace, current, step_size):
  loss, du, dv, _, _ = current
  squared = du.square().sum().item() + dv.square().sum().item()
  rate = step_size
  for _ in range(20):
    proposal = dict(u=candidate["u"] - rate * du,
      v=candidate["v"] - rate * dv, scale=1., pending={})
    measured = residual.measure(centers, target, proposal, values,
                                options, solver, workspace, gradient=False)
    if measured[0] <= loss - 1e-4 * rate * squared:
      return proposal, rate
    rate *= .5
  return None, rate
