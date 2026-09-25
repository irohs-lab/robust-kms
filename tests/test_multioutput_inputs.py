import unittest
import torch
import tests.reference as reference
import robust_kernels.multioutput_evaluate as evaluate


def test_multioutput_zero_derivatives_and_inputs():
  centers = torch.tensor([[0.1, 0.2], [-0.2, 0.4]], dtype=torch.float64)
  alpha = torch.tensor([[0.3, -0.6], [0.8, 0.2]], dtype=torch.float64)
  values, jacobian = evaluate.kernel_eval_and_grad(centers, centers, alpha)
  matrix = reference.dense_blocks(centers, centers, "gaussian", 1.0)
  expected = matrix[:, :len(centers)] @ alpha
  torch.testing.assert_close(values, expected[:2])
  torch.testing.assert_close(jacobian, expected[2:].reshape(2, 2, 2))
  values, jacobian = evaluate.kernel_eval_and_grad(centers[:0], centers, alpha)
  assert values.shape == (0, 2) and jacobian.shape == (0, 2, 2)
  values, jacobian = evaluate.kernel_eval_and_grad(centers, centers[:0], alpha[:0])
  assert values.count_nonzero() == 0 and jacobian.count_nonzero() == 0
  cases = [
    {"alpha": alpha[:, 0]}, {"alpha": alpha.float()},
    {"kernel": "wrong"}, {"length_scale": 0}, {"query_tile": 0},
    {"gradients": 1}, {"factors": {}},
  ]
  for case in cases:
    arguments = {"queries": centers, "centers": centers, "alpha": alpha, **case}
    with unittest.TestCase().assertRaises(ValueError):
      evaluate.kernel_eval_and_grad(**arguments)
