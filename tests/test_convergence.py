import torch
import robust_kernels as rk
import tests.reference as reference


def test_averaged_solution_has_small_independent_duality_gap():
  x = torch.tensor([[-1.0], [0.1], [1.2]], dtype=torch.float64)
  y = torch.tensor([0., 0., 1.], dtype=torch.float64)
  for kernel in ("gaussian", "matern52"):
    state = rk.fit(x, y, rho=0.1, lam=1., step_size=0.15, iterations=6000,
                    batch_size=2, decay=0.6, kernel=kernel)
    matrix = reference.dense_blocks(x, x, kernel, 1.)
    theta = torch.cat((state["average_alpha"], state["average_beta"].flatten()))
    evaluated = matrix @ theta
    probability = torch.sigmoid(evaluated[:3])
    dual_coeff = torch.cat((probability - y, state["delta"].flatten()))
    entropy = (probability * torch.log(probability)
               + (1 - probability) * torch.log1p(-probability)).sum()
    dual_bound = -0.5 * (dual_coeff @ matrix @ dual_coeff) - entropy
    primal = rk.objective(x, y, state["average_alpha"], state["average_beta"],
                          rho=0.1, lam=1., kernel=kernel)["total"]
    assert 0 <= primal - dual_bound < 0.004
    assert primal < 3 * torch.log(torch.tensor(2.))
