import torch
import robust_kernels as rk


def test_seeded_fit_and_averaging():
  x = torch.tensor([[-1.], [0.2], [0.8]], dtype=torch.float64)
  y = torch.tensor([False, True, True])
  options = dict(rho=0.1, lam=0.5, step_size=0.02, batch_size=2,
                 iterations=20, seed=91, decay=0.6)
  fitted = rk.fit(x, y, **options)
  state = rk.initialize(x)
  generator = torch.Generator().manual_seed(91)
  for m in range(20):
    batch = torch.randperm(3, generator=generator)[:2]
    rk.step(x, y, state, batch, rho=0.1, lam=0.5, eta=0.02 / (m + 1) ** 0.6)
  for key in ("alpha", "beta", "delta", "average_alpha", "average_beta"):
    torch.testing.assert_close(fitted[key], state[key], rtol=0, atol=0)
  lean = rk.fit(x, y, average=False, **options)
  assert "average_beta" not in lean
  torch.testing.assert_close(lean["beta"], fitted["beta"], rtol=0, atol=0)
  assert fitted["delta"].abs().max() <= 0.1


def test_invalid_hyperparameters_and_labels():
  x, y = torch.zeros(3, 2), torch.zeros(3)
  options = dict(rho=0.1, lam=0.5, step_size=0.02, batch_size=2, iterations=1)
  for replacement in (dict(decay=0.5), dict(step_size=3), dict(lam=0),
                      dict(rho=-1), dict(batch_size=4), dict(kernel="rff")):
    try:
      rk.fit(x, y, **(options | replacement))
    except ValueError:
      continue
    raise AssertionError(f"accepted invalid options: {replacement}")
  try:
    rk.fit(x, y - 1, **options)
  except ValueError:
    return
  raise AssertionError("accepted labels outside {0, 1}")
