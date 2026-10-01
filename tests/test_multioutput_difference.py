import torch
import tests.reference as reference
import robust_kernels.lowrank_factors as factors_ops
import robust_kernels.lowrank_cache as cache
import robust_kernels.multioutput_evaluate as evaluation


def make_factors(generator, centers):
  rand = lambda *size: torch.randn(*size, generator=generator, dtype=centers.dtype)
  factors = factors_ops.initialize(centers, 4)
  factors["u"].copy_(rand(len(centers), 3))
  factors["v"].copy_(rand(len(centers), 4))
  for index, a, b in ((0, 2, 1), (2, 1, 2), (3, 2, 3)):
    factors["pending"][index] = dict(left=rand(3, a), core=rand(a, b), right=rand(4, b))
  return factors


def dense(factors):
  values = factors["u"][:, :, None] * factors["v"][:, None, :]
  for index, item in factors["pending"].items():
    values[index] = item["left"] @ item["core"] @ item["right"].T
  return values * factors["scale"]


def move(factors, convert):
  return dict(u=convert(factors["u"]), v=convert(factors["v"]), scale=factors["scale"],
    pending={i: {key: convert(value) for key, value in item.items()}
             for i, item in factors["pending"].items()})


def test_multioutput_difference_matches_independent_dense_derivatives():
  generator = torch.Generator().manual_seed(537)
  centers = torch.randn(5, 3, generator=generator, dtype=torch.float64)
  queries = centers[:3] + .1
  alpha = torch.randn(5, 4, generator=generator, dtype=centers.dtype)
  candidate, target = [make_factors(generator, centers) for _ in range(2)]
  candidate["scale"], target["scale"] = -.6, .8
  beta = dense(candidate) - dense(target)
  devices = ["cpu"]
  if torch.cuda.is_available():
    devices.append("cuda:2" if torch.cuda.device_count() > 2 else "cuda:0")
  for kernel in ("gaussian", "matern52"):
    matrix = reference.dense_blocks(queries, centers, kernel, 1.2)
    expected = matrix @ torch.cat((alpha, beta.reshape(-1, 4)))
    for device in devices:
      dtype = torch.float64 if device == "cpu" else torch.float32
      convert = lambda value: value.to(device=device, dtype=dtype)
      left, right = move(candidate, convert), move(target, convert)
      original_left, original_right = dense(left), dense(right)
      options = dict(kernel=kernel, length_scale=1.2, query_tile=1, center_tile=2)
      for reuse in (False, True):
        extra = (dict(factor_cache=cache.prepare(left, 2),
                      subtract_factor_cache=cache.prepare(right, 2)) if reuse else {})
        values, jacobian = evaluation.kernel_eval_and_grad(convert(queries),
          convert(centers), convert(alpha), left, subtract_factors=right, **options, **extra)
        torch.testing.assert_close(values, convert(expected[:3]), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(jacobian, convert(expected[3:].reshape(3, 3, 4)),
                                   atol=2e-5, rtol=2e-5)
        values_only, missing = evaluation.kernel_eval_and_grad(convert(queries),
          convert(centers), convert(alpha), left, subtract_factors=right,
          gradients=False, **options, **extra)
        assert missing is None
        torch.testing.assert_close(values_only, values)
      torch.testing.assert_close(dense(left), original_left, atol=0, rtol=0)
      torch.testing.assert_close(dense(right), original_right, atol=0, rtol=0)


def test_multioutput_difference_handles_missing_and_identical_positive_factors():
  generator = torch.Generator().manual_seed(285)
  centers = torch.randn(5, 3, generator=generator, dtype=torch.float64)
  queries = centers[:2] + .2
  alpha = torch.randn(5, 4, generator=generator, dtype=centers.dtype)
  factors = make_factors(generator, centers)
  for scale in (-.5, 0., 1.):
    factors["scale"] = scale
    for kernel in ("gaussian", "matern52"):
      options = dict(kernel=kernel, length_scale=1.3, query_tile=1, center_tile=2)
      actual = evaluation.kernel_eval_and_grad(queries, centers, alpha,
                                               subtract_factors=factors, **options)
      matrix = reference.dense_blocks(queries, centers, kernel, 1.3)
      expected = matrix @ torch.cat((alpha, -dense(factors).reshape(-1, 4)))
      torch.testing.assert_close(actual[0], expected[:2], atol=1e-11, rtol=1e-11)
      torch.testing.assert_close(actual[1], expected[2:].reshape(2, 3, 4),
                                 atol=1e-11, rtol=1e-11)
      cancelled = evaluation.kernel_eval_and_grad(queries, centers, alpha, factors,
                                                  subtract_factors=factors, **options)
      base = evaluation.kernel_eval_and_grad(queries, centers, alpha, **options)
      for result, reference_result in zip(cancelled, base):
        torch.testing.assert_close(result, reference_result, atol=0, rtol=0)
