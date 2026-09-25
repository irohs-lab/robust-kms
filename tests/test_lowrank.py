import torch
import robust_kernels.lowrank_factors as factors


def dense(state):
  return torch.stack([factors.column(state, 0, len(state["u"]), c)
                      for c in range(state["v"].shape[1])], dim=-1)


def test_incremental_qr_outer_updates_and_scaling():
  devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
  for device in devices:
    generator = torch.Generator(device=device).manual_seed(32)
    centers = torch.zeros(4, 9, dtype=torch.float64, device=device)
    state = factors.initialize(centers, 3)
    expected = centers.new_zeros((4, 9, 3))
    for step in range(36):
      index = step % 4
      left = torch.randn(9, dtype=centers.dtype, device=device, generator=generator)
      right = torch.randn(3, dtype=centers.dtype, device=device, generator=generator)
      if step % 3 == 0:
        right = torch.nn.functional.one_hot(torch.tensor(step % 3, device=device), 3).to(centers)
      factors.scale(state, 0.91)
      factors.add(state, index, left, right)
      expected = 0.91 * expected
      expected[index] += left[:, None] * right[None, :]
      torch.testing.assert_close(dense(state), expected, atol=1e-11, rtol=1e-11)
    for item in state["pending"].values():
      for name in ("left", "right"):
        basis = item[name]
        assert basis.shape[1] <= 3
        torch.testing.assert_close(basis.T @ basis, torch.eye(basis.shape[1], device=device,
                                   dtype=centers.dtype), atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(factors.column(state, 1, 3, 2), expected[1:3, :, 2])


def test_lowrank_dependent_updates_reset_and_compress():
  centers = torch.zeros(3, 6, dtype=torch.float64)
  state = factors.initialize(centers, 3)
  state["u"][0] = torch.arange(6, dtype=centers.dtype)
  state["v"][0] = torch.tensor([0., 2., 0.])
  expected = dense(state)
  left = torch.arange(6, dtype=centers.dtype).flip(0)
  right = torch.tensor([0., 1., 0.], dtype=centers.dtype)
  for _ in range(8):
    factors.add(state, 0, left, right)
    expected[0] += left[:, None] * right[None, :]
  factors.add(state, 0, left, torch.tensor([1., 0., 0.], dtype=centers.dtype))
  expected[0] += left[:, None] * torch.tensor([1., 0., 0.], dtype=centers.dtype)[None, :]
  factors.scale(state, -0.25)
  expected *= -0.25
  torch.testing.assert_close(dense(state), expected, atol=1e-11, rtol=1e-11)
  compressed = factors.rank_one(state)
  assert compressed["scale"] == 1.0 and not compressed["pending"]
  error = expected.new_zeros(())
  for i in range(len(centers)):
    u, singular, vh = torch.linalg.svd(expected[i], full_matrices=False)
    projected = singular[0] * u[:, :1] @ vh[:1]
    torch.testing.assert_close(dense(compressed)[i], projected, atol=1e-11, rtol=1e-11)
    error += singular[1:].square().sum()
  torch.testing.assert_close(factors.frobenius_rank_error(state), error.sqrt())
  copied = factors.clone(state)
  factors.scale(state, 0)
  assert not state["pending"]
  torch.testing.assert_close(dense(state), torch.zeros_like(expected))
  torch.testing.assert_close(dense(copied), expected, atol=1e-11, rtol=1e-11)
  factors.add(state, 1, left, right)
  torch.testing.assert_close(dense(state)[1], left[:, None] * right[None, :])


def test_lowrank_tiny_scale_and_float32():
  generator = torch.Generator().manual_seed(42)
  centers = torch.zeros(2, 12)
  state = factors.initialize(centers, 4)
  expected = centers.new_zeros((2, 12, 4))
  for _ in range(16):
    left = torch.randn(12, generator=generator)
    right = torch.randn(4, generator=generator)
    factors.add(state, 1, left, right)
    expected[1] += left[:, None] * right[None, :]
  factors.scale(state, 1e-12)
  expected *= 1e-12
  left = torch.randn(12, generator=generator)
  right = torch.randn(4, generator=generator)
  factors.add(state, 1, left, right)
  expected[1] += left[:, None] * right[None, :]
  torch.testing.assert_close(dense(state), expected, atol=2e-6, rtol=2e-6)
