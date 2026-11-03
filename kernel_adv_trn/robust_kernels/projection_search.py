import robust_kernels.projection_residual as residual


def trial(centers, target, candidate, values, options, solver, workspace, current, step_size):
  """Return an accepted proposal and its reusable objective/gradient measurement."""
  loss, du, dv, _, _ = current
  squared = du.square().sum().item() + dv.square().sum().item()
  rate = step_size
  for attempt in range(20):
    progress = workspace.get("progress")
    if progress is not None:
      progress(dict(phase="projection_line_search", attempt=attempt + 1, step_size=rate))
    proposal = dict(u=candidate["u"] - rate * du,
      v=candidate["v"] - rate * dv, scale=1., pending={})
    measured = residual.measure(centers, target, proposal, values,
                                options, solver, workspace, gradient=True)
    if measured[0] <= loss - 1e-4 * rate * squared:
      return proposal, rate, measured
    rate *= .5
  return None, rate, None
