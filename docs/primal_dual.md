# Scalar stochastic primal-dual RKHS solver

The solver minimizes the **summed** objective
`sum_i logistic(f(x_i), y_i) + rho * sum_ik |partial_k f(x_i)| + lam/2 * ||f||_H^2`.
Labels are 0 or 1, `rho = L * epsilon >= 0`, and `lam >= 0`.
The averaged-iterate convergence guarantee discussed here assumes `lam > 0`;
`lam=0` is also supported for finite-budget or early-stopped experiments.
It implements simultaneous stochastic descent/ascent, not Chambolle–Pock,
Condat–Vu, or the earlier variance-reduced predictor/corrector proposal.

## Usage

Run from the repository root with PyTorch and torchkernels installed, or install
the library with `python -m pip install -e .`. No feature approximation is used.

```python
import torch
import robust_kernels as rk

# x: (n, d), y: (n,), binary labels. GPU 0 is selected explicitly.
x = x.to(device="cuda:0", dtype=torch.float32)
y = y.to(device="cuda:0", dtype=torch.float32)
state = rk.fit(
  x, y, rho=0.01, lam=1.0, step_size=1e-5,
  iterations=1000, batch_size=64, decay=0.75,
  kernel="gaussian", length_scale=x.shape[1] ** 0.5,
  query_tile=64, center_tile=1024, seed=0,
)
values, gradients = rk.kernel_eval_and_grad(
  x[:64], x, state["average_alpha"], state["average_beta"],
  kernel="gaussian", length_scale=x.shape[1] ** 0.5,
)
probabilities = torch.sigmoid(values)
```

The numerical settings above illustrate the API; they are not tuned for a dataset.
Use `kernel="matern52"` for the exact Matern-5/2 kernel. Gaussian uses
`exp(-r^2/(2*length_scale^2))`; Matern uses `(1+s+s^2/3)*exp(-s)`,
`s=sqrt(5)*r/length_scale`. Both use the installed torchkernels squared-distance
primitive with analytic value and derivative contractions. Coincident-point
limits are finite; no differentiation through a square root at zero is needed.

A runnable trainer accepts a torch file containing `{"x": x, "y": y}`:

```bash
python -m examples.train_binary --data binary.pt --output model.pt \
  --device cuda:0 --rho 0.01 --lam 1 --step-size 1e-5 \
  --kernel matern52 --batch-size 64 --iterations 1000
```

The saved model contains centers, selected alpha/beta coefficients, and kernel
parameters. Load with `torch.load("model.pt", map_location="cuda:0",
weights_only=True)` and pass its centers, coefficients, kernel and length scale
to `rk.kernel_eval_and_grad`. This API returns analytic input gradients and
deliberately does not retain an autograd graph.

## Exact iteration

Represent `f = E* alpha + D* beta`. For a fresh uniform subset B, size b, first
evaluate `F_B, J_B = kernel_eval_and_grad(X_B, X, alpha, beta)`. All quantities
on the right below refer to the **old state**:

```text
eta = step_size / (m + 1)^decay
r_B = sigmoid(F_B) - y_B
alpha+ = (1 - eta*lam) alpha - eta*(n/b) scatter_B(r_B)
beta+  = (1 - eta*lam) beta  - eta*(n/b) scatter_B(delta_B)
delta_B+ = clip(delta_B + eta*(n/b)*J_B, -rho, rho)
delta_outside_B+ = delta_outside_B
```

The same batch is used for both updates, with n/b weighting for the summed
objective. The implementation saves the old sampled delta before clipping.
`fit` draws a fresh subset without replacement at each iteration; it is not
random reshuffling over epochs. The low-level `rk.initialize` and `rk.step`
functions expose state and user-supplied batches for custom loops. Start with
feasible dual blocks and sample batches independently of the current state.

`step_size * lam <= 1` and `0.5 < decay <= 1` enforce the discussed diminishing
step conditions. These are asymptotic conditions, not a finite-budget accuracy
guarantee. Averaging accumulates the **pre-update** functions f^0,...,f^(T-1)
with weights eta_0,...,eta_(T-1); after a single step its result is still zero.
The last iterate is available as `alpha, beta`; the averaged coefficients are
`average_alpha, average_beta`.

## Memory and diagnostics

Persistent tensors are X, y, alpha, beta and delta. Weighted averaging adds one
(n, d) beta buffer and n alpha scalars. Set `average=False` (CLI:
`--no-average`) to save that memory, but then the averaged-iterate guarantee
does not describe the returned last iterate.

Evaluation tiles queries and centers. It never forms full K/G/H matrices,
pairwise (b, n, d) differences, or d-by-d Hessian blocks. Workspace is
O(query_tile*center_tile + (query_tile+center_tile)*d), plus output arrays.
A training step evaluates only b queries but still visits every center:
O(b*n*d) arithmetic, plus O(n*d) coefficient shrinkage and averaging.
The full dual and derivative coefficients remain O(n*d).

`rk.objective(x, y, alpha, beta, rho=..., lam=..., ...)` streams exact loss,
gradient l1 penalty, squared RKHS norm, and total objective. It includes the
value/derivative cross term through
`||f||_H^2 = alpha.T @ F + sum(beta * J)`. This diagnostic costs a full
O(n^2*d) pass, so it is not called by the training loop. CLI `--evaluate`
opts into it after training.

All operations follow the input device and dtype. Float32 and float64 are
supported. Tile sizes bound workspace; exact formulas still incur ordinary
floating-point roundoff. The library does not change global TF32 settings.

## Verification

```bash
python -m tests
```

Tests use the standard-library unittest runner (no pytest dependency). They
compare rectangular tiled products to independently differentiated dense
kernels, verify coincidence limits and mixed-derivative signs, check the exact
simultaneous minibatch update and averaging, and evaluate a small-problem
primal-dual certificate. CUDA tests run on GPU 0 when available.

A four-step smoke check on GPU 0 (RTX 5000 Ada, 32 GB), with n=50,000,
batch size 64, float32, TF32 disabled, center tiles of 1024, and averaging
retained, used 0.763 GiB peak allocated CUDA memory at d=1,000 and 3.023 GiB
at d=4,000 for both kernels. These include data and state allocations, not
CUDA context or allocator-reserved memory. These are execution/memory checks,
not full-size convergence benchmarks. A separate three-sample float64 test
reaches an independently evaluated primal-dual gap below 0.004 for both
kernels after 6,000 steps.
