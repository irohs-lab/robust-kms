import torch


@torch.no_grad()
def maximize(jacobian, rho):
  """Return a factored exact dual maximizer and the unscaled norm penalty.

  jacobian has shape (batch, d, c), the transpose of a (c, d) Jacobian.
  The dual set is sum_c ||delta[:, c]||_infinity <= rho. Each returned
  block is left[:, None] * right[None, :], with at most one nonzero column.
  At ties choose the first maximizing column, and use sign(0) = 0.
  """
  columns = jacobian.abs().sum(1)
  winners = columns.argmax(1)
  left = jacobian.gather(
    2, winners[:, None, None].expand(-1, jacobian.shape[1], 1)
  ).squeeze(2).sign() * rho
  right = torch.nn.functional.one_hot(winners, jacobian.shape[2]).to(jacobian.dtype)
  return left, right, columns.max(1).values
