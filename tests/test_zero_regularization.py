import torch
import robust_kernels as rk


def test_zero_regularization_does_not_shrink_unsampled_coefficients():
  x = torch.tensor([[-1.], [0.], [1.]], dtype=torch.float64)
  y = torch.tensor([0, 0, 1])
  state = rk.initialize(x)
  state["alpha"].copy_(torch.tensor([0.2, 0.3, 0.4]))
  state["beta"].fill_(0.1)
  state["delta"].fill_(0.03)
  batch = torch.tensor([1])
  before_alpha, before_beta = state["alpha"].clone(), state["beta"].clone()
  values, gradients = rk.kernel_eval_and_grad(x[batch], x, before_alpha, before_beta)
  rk.step(x, y, state, batch, rho=0.1, lam=0., eta=0.02)
  torch.testing.assert_close(state["alpha"][[0, 2]], before_alpha[[0, 2]])
  torch.testing.assert_close(state["beta"][[0, 2]], before_beta[[0, 2]])
  torch.testing.assert_close(state["alpha"][1], before_alpha[1] - 0.06 * torch.sigmoid(values[0]))
  torch.testing.assert_close(state["beta"][1], before_beta[1] - 0.06 * 0.03)
  rk.fit(x, y, rho=0.1, lam=0., step_size=0.02, iterations=2, batch_size=2)
