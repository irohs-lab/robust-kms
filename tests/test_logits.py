import torch
import robust_kernels as rk
import robust_kernels.logits as logits


def test_attack_logits_have_exact_input_gradient():
  generator = torch.Generator().manual_seed(29)
  x = torch.randn(5, 4, generator=generator, dtype=torch.float64)
  a = torch.randn(5, generator=generator, dtype=torch.float64)
  b = torch.randn(5, 4, generator=generator, dtype=torch.float64)
  inputs = torch.randn(2, 1, 2, 2, generator=generator, dtype=torch.float64)
  inputs.requires_grad_()
  for kernel in ("gaussian", "matern52"):
    fn = lambda z: logits.binary_logits(z, x, a, b, bias=.37, kernel=kernel)
    assert torch.autograd.gradcheck(fn, (inputs,), atol=1e-5, rtol=1e-4)
    result = fn(inputs)
    actual, = torch.autograd.grad(result[:, 1].sum(), inputs)
    values, expected = rk.kernel_eval_and_grad(inputs.detach().flatten(1), x, a, b,
                                               kernel=kernel)
    torch.testing.assert_close(result[:, 1], values + .37)
    torch.testing.assert_close(actual.flatten(1), expected)
    torch.testing.assert_close(result[:, 0], torch.zeros_like(values))
