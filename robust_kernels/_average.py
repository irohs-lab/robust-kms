def update(state, eta):
  """Average the pre-update primal iterates with weights eta."""
  if "average_alpha" not in state:
    return
  state["weight_sum"] += eta
  rate = eta / state["weight_sum"]
  state["average_alpha"].lerp_(state["alpha"], rate)
  state["average_beta"].lerp_(state["beta"], rate)
