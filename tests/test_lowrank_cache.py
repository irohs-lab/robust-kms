import torch
import tests.reference as reference
import robust_kernels.lowrank_cache as cache
import robust_kernels.lowrank_factors as factors_ops
import robust_kernels.multioutput_evaluate as evaluate


def make_case(device, dtype):
  generator = torch.Generator(device=device).manual_seed(714)
  rand = lambda *shape: torch.randn(*shape, device=device, dtype=dtype,
                                   generator=generator)
  centers = rand(8, 5)
  factors = factors_ops.initialize(centers, 4)
  factors["u"].copy_(rand(8, 5))
  factors["v"].copy_(rand(8, 4))
  factors["u"][6].zero_()
  for index, left_rank, right_rank in [(7, 2, 2), (0, 2, 1), (3, 1, 2),
                                      (2, 0, 0), (5, 3, 4)]:
    factors["pending"][index] = dict(left=rand(5, left_rank),
      core=rand(left_rank, right_rank), right=rand(4, right_rank))
  return centers, rand(8, 4), factors


def dense_coefficients(factors):
  expected = factors["u"][:, :, None] * factors["v"][:, None, :]
  for index, item in factors["pending"].items():
    expected[index] = item["left"] @ item["core"] @ item["right"].T
  return factors["scale"] * expected


def test_lowrank_packed_columns_mixed_ranks_tiles_and_scaling():
  devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
  for device in devices:
    dtype = torch.float64 if device == "cpu" else torch.float32
    centers, _, factors = make_case(device, dtype)
    for scalar in (1., -.7, 0.):
      factors["scale"] = scalar
      expected = dense_coefficients(factors)
      for tile_size in (1, 3, 20):
        packed = cache.prepare(factors, tile_size)
        for start in range(0, len(centers), tile_size):
          stop = min(start + tile_size, len(centers))
          for output in range(4):
            actual = factors_ops.column(factors, start, stop, output,
                                        prepared=packed[start // tile_size])
            torch.testing.assert_close(actual, expected[start:stop, :, output],
                                       atol=2e-5, rtol=2e-5)
    assert set(factors) == {"u", "v", "scale", "pending"}


def test_lowrank_packed_evaluation_matches_dense_kernel_derivatives():
  cpu_centers, cpu_alpha, cpu_factors = make_case("cpu", torch.float64)
  cpu_factors["scale"] = -.7
  queries = cpu_centers[[0, 4, 7]] + .12
  coefficients = torch.cat((cpu_alpha, dense_coefficients(cpu_factors).reshape(-1, 4)))
  devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
  for kernel in ("gaussian", "matern52"):
    expected = reference.dense_blocks(queries, cpu_centers, kernel, 1.3) @ coefficients
    for device in devices:
      dtype = torch.float64 if device == "cpu" else torch.float32
      convert = lambda value: value.to(device=device, dtype=dtype)
      centers, alpha = convert(cpu_centers), convert(cpu_alpha)
      factors = dict(u=convert(cpu_factors["u"]), v=convert(cpu_factors["v"]),
        scale=cpu_factors["scale"], pending={index: {key: convert(value)
          for key, value in item.items()} for index, item in cpu_factors["pending"].items()})
      options = dict(kernel=kernel, length_scale=1.3, query_tile=1, center_tile=3)
      packed = cache.prepare(factors, 3)
      for supplied in (None, packed):
        values, jacobian = evaluate.kernel_eval_and_grad(
          convert(queries), centers, alpha, factors, factor_cache=supplied, **options)
        torch.testing.assert_close(values, convert(expected[:3]), atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(jacobian, convert(expected[3:].reshape(3, 5, 4)),
                                   atol=3e-5, rtol=3e-5)
      values_only, absent = evaluate.kernel_eval_and_grad(
        convert(queries), centers, alpha, factors, gradients=False,
        factor_cache=packed, **options)
      assert absent is None
      torch.testing.assert_close(values_only, values)


def test_lowrank_evaluation_rebuilds_cache_after_factor_mutation():
  centers, alpha, factors = make_case("cpu", torch.float64)
  queries = centers[:2] + .2
  before, _ = evaluate.kernel_eval_and_grad(queries, centers, alpha, factors,
                                            gradients=False, center_tile=3)
  factors_ops.scale(factors, 0.)
  left = torch.arange(5, dtype=centers.dtype)
  right = torch.tensor([1., -1., .5, 0.], dtype=centers.dtype)
  factors_ops.add(factors, 1, left, right)
  after, _ = evaluate.kernel_eval_and_grad(queries, centers, alpha, factors,
                                           gradients=False, center_tile=3)
  coefficients = torch.cat((alpha, dense_coefficients(factors).reshape(-1, 4)))
  expected = reference.dense_blocks(queries, centers, "gaussian", 1.) @ coefficients
  torch.testing.assert_close(after, expected[:2], atol=1e-11, rtol=1e-11)
  assert not torch.allclose(before, after)
  assert set(factors) == {"u", "v", "scale", "pending"}
