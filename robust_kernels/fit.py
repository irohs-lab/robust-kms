import torch
import robust_kernels._fit_inputs as inputs
import robust_kernels.state as initialization
import robust_kernels.updates as updates


@torch.no_grad()
def fit(centers, labels, *, rho, lam, step_size, iterations=1000,
        batch_size=128, decay=0.75, kernel="gaussian", length_scale=1.0,
        query_tile=128, center_tile=1024, seed=0, average=True):
  """Fit with eta_m = step_size / (m+1)**decay and uniform fresh batches.

  Returns last-iterate alpha/beta/delta and, by default, weighted-average
  alpha/beta. Averaging costs an additional (n, d) buffer plus n scalars.
  Data stays on its supplied device; pass CUDA:0 tensors for GPU 0.
  """
  inputs.validate(centers, labels, iterations, batch_size, step_size, decay)
  state = initialization.initialize(centers, average=average)
  generator = torch.Generator(device=centers.device).manual_seed(seed)
  for m in range(iterations):
    batch = torch.randperm(len(centers), generator=generator,
                           device=centers.device)[:batch_size]
    updates.step(centers, labels, state, batch, rho=rho, lam=lam,
                 eta=step_size / (m + 1) ** decay, kernel=kernel,
                 length_scale=length_scale, query_tile=query_tile,
                 center_tile=center_tile)
  return state
