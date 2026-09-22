import torch
import robust_kernels as rk
import tests.reference as reference


def test_step_matches_dense_simultaneous_update():
  generator = torch.Generator().manual_seed(19)
  for kernel in ("gaussian", "matern52"):
    x = torch.randn(5, 2, generator=generator, dtype=torch.float64)
    y, batch = torch.tensor([0, 1, 0, 1, 1]), torch.tensor([1, 4])
    state = rk.initialize(x)
    for key in ("alpha", "beta", "delta"):
      state[key].copy_(torch.randn(state[key].shape, generator=generator))
    state["delta"].clamp_(-0.15, 0.15)
    old = {key: value.clone() for key, value in state.items()
           if isinstance(value, torch.Tensor)}
    matrix = reference.dense_blocks(x[batch], x, kernel, 1.2)
    evaluated = matrix @ torch.cat((old["alpha"], old["beta"].flatten()))
    scale, shrink = 0.02 * 5 / 2, 1 - 0.02 * 0.7
    alpha, beta = shrink * old["alpha"], shrink * old["beta"]
    alpha[batch] -= scale * (torch.sigmoid(evaluated[:2]) - y[batch])
    beta[batch] -= scale * old["delta"][batch]
    delta = old["delta"].clone()
    delta[batch] = (delta[batch] + scale * evaluated[2:].reshape(2, 2)).clamp(
      -0.15, 0.15)
    rk.step(x, y, state, batch, rho=0.15, lam=0.7, eta=0.02,
            kernel=kernel, length_scale=1.2, center_tile=2)
    for key, expected in (("alpha", alpha), ("beta", beta), ("delta", delta)):
      torch.testing.assert_close(state[key], expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(state["average_beta"], old["beta"])
    rk.step(x, y, state, batch, rho=0.15, lam=0.7, eta=0.01,
            kernel=kernel, length_scale=1.2)
    torch.testing.assert_close(state["average_beta"], (2 * old["beta"] + beta) / 3)
