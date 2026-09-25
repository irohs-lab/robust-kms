# Multiclass learning with delayed RKHS projection

The product RKHS model has unrestricted alpha (n,C) and derivative blocks
B_i = u_i v_i^T after projection. Between projections, the blocks accumulate
additional rank-one terms. Gaussian and Matérn-5/2 kernels are supported.

The training objective is summed cross-entropy plus
rho * sum_i max_c ||grad f_c(x_i)||_1 + lambda/2 * ||f||_H^2.
For the first-order cross-entropy bound, use rho=2*epsilon. This is the convex
Jacobian-norm surrogate, not the nonlinear adversarial-training objective.

## Separate update and projection calls

```python
import robust_kernels as rk

options = dict(kernel="matern52", length_scale=length_scale)
state = rk.initialize_multiclass(x, outputs=10)

for step, batch in enumerate(batches, start=1):
  metrics = rk.step_multiclass(
    x, labels, state, batch, rho=2*epsilon, lam=lam, eta=eta, **options)
  if step % 100 == 0:
    diagnostics = rk.project(x, state, max_steps=20, **options)

# Compress any remaining temporary updates before saving the model.
diagnostics = rk.project(x, state, max_steps=20, **options)
```

step_multiclass never projects. fit_multiclass provides a convenience loop:
project_every=100 by default, plus a final projection for leftover steps.
Set final_projection=False if the caller will handle final compression.
A callback receives minibatch loss/accuracy and projection diagnostics.
There is no dense coefficient averaging. Save state using torch.save.

## Updating the factors

A fresh maximizing column j_i gives
R_i = rho sign(grad f_j_i(x_i)) e_j_i^T.
This is an exact subgradient of the max-column norm and avoids a stored
(n,d,C) dual. Alpha uses the softmax-minus-one-hot residual. Beta receives
-R_i times eta*n/b, with global RKHS shrinkage applied to both coefficients.

state["factors"] contains contiguous u(n,d), v(n,C), a scalar scale, and a
pending dictionary for visited centers. A pending block replaces the base
block and stores Q_left, core, Q_right, representing
scale * Q_left @ core @ Q_right.T.

Each outer-product update uses incremental thin QR on both sides, with
two reorthogonalization passes. Residuals below 8*machine_epsilon times
the update norm are treated as numerically dependent. QR accumulation
does not deliberately truncate nonzero directions. A small-core SVD can
remove redundant basis dimensions. It bounds the basis widths by min(d,C).
The full n*d*C coefficient array is never formed.

## RKHS projection

For the target coefficients (alpha_tilde,B_tilde), alpha is eliminated:
alpha(B) = alpha_tilde - K^-1 G (B-B_tilde).
The factor optimization minimizes
0.5 * tr[(B-B_tilde)^T (H-G^T K^-1 G) (B-B_tilde)]
subject to rank(B_i)<=1.

Independent small-core SVDs supply an initializer only. A coupled factor
gradient optimization with Armijo backtracking improves its RKHS error.
Zero blocks can acquire nonzero coefficients through coupling. The solver
is nonconvex; it returns an approximate projection, not a certified global
minimum. Diagnostics distinguish stationarity, iteration limit, and failed
line search, and report initial/final squared RKHS error. It also reports projection wall time,
pending-block count, maximum QR width, and allocated pending-buffer bytes.

K solves use matrix-free EigenPro2 without a hidden ridge/jitter. One
Nyström preconditioner is cached for the whole projection (including line
search), with warm starts between right-hand sides. The default uses up to
1,024 samples, rank 100, minibatch 128, and 100 solve epochs; sizes are capped
for small datasets. Configure eigenpro_samples, eigenpro_rank,
eigenpro_batch_size, solve_max_epochs, solve_rtol, and solve_atol.
The full true linear residual is checked every epoch. An unsuccessful solve
raises before the input model is modified. Increase the solve budget or
preconditioner quality when needed. Float64 can help stringent tolerances;
loosening tolerance also loosens training-logit preservation.

The alpha correction preserves f(X) up to numerical solve/evaluation error.
max_value_error reports the observed maximum change over all training
logits. No K, G, H, Schur matrix, or n*d*C residual is stored; derivative
residuals are evaluated in query tiles. Alpha remains an n*C array.

## Projection frequency and cost

Start with project_every=100; this is a configurable heuristic, not an
optimality claim. At CIFAR-10 n=50,000 and b=128, it is about one quarter of
a sampling epoch. Uncompressed updates cost at most approximately
T*b*(d+C) values before QR redundancy reductions, in addition to the base.

Increase the interval when projection time dominates and the temporary
buffer fits comfortably. Shorten it when buffer memory or the measured
compression error becomes large. Monitor validation accuracy after
projection, since preserving training logits does not preserve all test
predictions. Thin QR makes buffering and initialization cheaper; it does
not remove the coupled RKHS optimization or its kernel solves.

Base float32 coefficient storage for CIFAR-10 is n*(d+2*C)*4 bytes:
618.4 MB for d=3072,C=10. This excludes the training data, pending factors,
inner-optimization copies/gradients, and tiled workspace. The implementation
has been checked on small CPU/GPU cases; full CIFAR-10 projection throughput
has not been benchmarked.

Run the executable smoke example with:
```bash
python -m examples.train_multiclass --device cpu
python -m examples.train_multiclass --device cuda:0
python -m tests
```
