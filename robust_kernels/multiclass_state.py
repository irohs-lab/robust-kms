import robust_kernels.lowrank_factors as factors


def initialize(centers, outputs):
  """Keep full value coefficients and factored derivative coefficients."""
  derivative = factors.initialize(centers, outputs)
  if outputs < 2:
    raise ValueError("multiclass training requires at least two outputs")
  return dict(alpha=centers.new_zeros((len(centers), outputs)),
    factors=derivative, iterations=0, last_projection=0)
