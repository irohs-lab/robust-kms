# Multiclass RKHS subgradient training with epoch-based projection

The product RKHS model has unrestricted alpha (n,c) and derivative blocks
B_i = theta_i gamma_i^T, with theta_i in R^d and gamma_i in R^c after projection.
Between projections, derivative blocks accumulate additional rank-one terms.
Gaussian and Matérn-5/2 kernels are supported.

With the mathematical Jacobian J_i = grad f(x_i) of shape (c,d), the summed
objective is

```text
sum_i [cross_entropy(f(x_i), y_i) + rho * ||J_i.T||_(1->1)]
    + lambda/2 * ||f||_H^2.
```

The induced norm is max_a sum_j |J_i[a,j]|, not an entrywise matrix l1 norm.
The evaluator stores transposed Jacobians as (batch,d,c). Use rho=epsilon
for a penalty of weight epsilon. For the first-order cross-entropy upper
bound for l-infinity perturbations of radius epsilon, use rho=2*epsilon.
This is a convex Jacobian-norm surrogate, not the nonlinear adversarial loss.

## Training and projection schedule

```python
import robust_kernels as rk

options = dict(kernel="matern52", length_scale=length_scale)
state = rk.fit_multiclass(
    x, labels, outputs=10, rho=epsilon, lam=lam, eta=eta,
    epochs=10, batch_size=128, project_every=1,
    projection_options=dict(max_steps=20), **options)
```

`epochs` is the number of additional complete data passes in this call.
Each epoch draws one fresh permutation and visits every center exactly once,
including a smaller final minibatch. `project_every` counts epochs and defaults
to **1**. For example, `epochs=5, project_every=2` projects after epochs 2 and 4,
then once at epoch 5 because `final_projection=True` by default. Set
`final_projection=False` to leave an unfinished interval uncompressed.
Scheduled projections always occur at epoch boundaries.

This replaces the previous `steps` budget and minibatch-based projection
interval. Passing `steps` now raises an error rather than silently changing
its meaning. The scalar binary trainer keeps its existing API.

A callback receives `callback(state, event)` after each minibatch. Events
include `iteration`, `epoch`, `batch_in_epoch`, `epoch_end`, minibatch loss,
accuracy, and the unscaled Jacobian penalty. The last batch's event includes
`projection` when a scheduled projection occurs. A final leftover projection
emits a separate event with `batch_in_epoch=None` and no batch metrics.
Epochs and iterations are absolute counters in the state; batch numbers are
one-based within the current epoch.

Save the state with `torch.save` at an epoch boundary. Passing it back as
`state=...` continues the iteration decay, epoch/projection counters, and saved
shuffle generator state. `seed` initializes only a state without saved shuffle
state. Mid-epoch continuation is not supported. For a split run that should
match an uninterrupted run, use `final_projection=False` on intermediate calls
so they do not introduce extra projections. Keep the data order, device,
batch size, and optimizer/kernel settings the same.

## Primal step and exact dual update

For a batch I of size b, evaluate the old logits and Jacobians. The dual set
for each transposed Jacobian is

```text
D_rho = {delta in R^(d,c): sum_a ||delta[:,a]||_infinity <= rho}.
```

Refresh the dual by exact maximization:

```text
a_i = argmax_a sum_j |J_i[a,j]|
q_i = sign(J_i[a_i,:])
delta_i = rho * q_i e_(a_i)^T
```

This is a rank-at-most-one maximizer of <delta_i,J_i.T>. Ties use the first
maximizing class and sign(0)=0. The implementation computes its two factors
on demand. It does not maintain a persistent dual-ascent iterate; such an
iterate generally becomes higher rank and is a different algorithm.

Using all old-state quantities, the RKHS primal update is

```text
eta_t = eta / (iterations + 1)^decay
r_i = softmax(f(x_i)) - one_hot(y_i)
kappa = 1 - eta_t * lambda
alpha_i <- kappa * alpha_i - eta_t * (n/b) * 1_(i in I) * r_i
B_i     <- kappa * B_i     - eta_t * (n/b) * 1_(i in I) * delta_i
```

The n/b factor applies to the actual batch size, including the final short
batch, because the objective is summed. The low-level `step_multiclass` never
projects; `project` remains callable separately. No dense iterate averaging
is maintained. Random reshuffling does not give a conditionally unbiased
subgradient at each minibatch, and no global convergence guarantee is claimed
for the nonconvex projected method.

## Rank and storage between projections

Starting with rank-at-most-one blocks, each visit adds one outer product.
After T complete epochs without projection, each block has rank at most
min(d,c,T+1). With the default T=1 it has rank at most two. Starting from zero,
the first epoch has rank at most one. A rank-one dual update does not by itself
keep a nonzero primal block rank one.

`state["factors"]` contains contiguous u(n,d), v(n,c), a scalar scale, and a
pending dictionary for visited centers. A pending block replaces the base
block and stores Q_left, core, Q_right, representing
scale * Q_left @ core @ Q_right.T.

Each outer-product update uses incremental thin QR on both sides, with
two reorthogonalization passes. Residuals below 8*machine_epsilon times
the update norm are treated as numerically dependent. QR accumulation
does not deliberately truncate nonzero directions. A small-core SVD can
remove redundant basis dimensions. It bounds the basis widths by min(d,c).
Pending skinny factors are packed in batches once per evaluation and reused
across classes and query tiles. The full n*d*c coefficient array is never formed. At T=1, independent rank-one
initialization uses cores of at most 2 by 2, up to roundoff in QR accumulation.

## Coupled RKHS projection

For target coefficients (alpha_tilde,B_tilde), alpha is eliminated:

```text
alpha(B) = alpha_tilde - K^-1 G (B-B_tilde)
S = H-G^T K^-1 G
minimize 0.5 * tr[(B-B_tilde)^T S (B-B_tilde)]
subject to rank(B_i) <= 1 for every center i.
```

Independent small-core SVDs supply an initializer only. The implementation
retains the coupled factor-gradient solver with Armijo backtracking to improve
its RKHS error. Zero blocks can acquire nonzero coefficients through coupling.
This is an approximate projection: the nonconvex solver does not certify a
global minimum. Projection is not replaced with Euclidean coefficient
truncation. Diagnostics distinguish stationarity, iteration limit, and failed
line search, and report initial/final squared RKHS error, projection wall time,
pending-block count, maximum QR width, and pending-buffer bytes.

K solves use EigenPro2 without a hidden ridge/jitter. The default is
matrix-free; `eigenpro_storage="dense"` optionally caches the scalar K matrix
for repeated solves when n*n floating-point values fit in memory. The cache
is temporary projection workspace and is not saved in model checkpoints. One
Nyström preconditioner is cached for the whole projection, including line
search, with warm starts between right-hand sides. The default uses up to
1,024 samples, rank 100, minibatch 128, and 100 solve epochs; sizes are capped
for small datasets. Configure `eigenpro_samples`, `eigenpro_rank`,
`eigenpro_batch_size`, `solve_max_epochs`, `solve_rtol`, and `solve_atol`.
The full true linear residual is checked every epoch. An unsuccessful solve
raises before the projector changes the model. Increase the solve budget or
preconditioner quality when needed. The step-size diagonal bound is checked
on all training centers, including those outside the spectral sample.
Float64 can help stringent tolerances or large, numerically delicate spectral
setups. The FashionMNIST comparison uses `solve_rtol=1e-3` and records both
relative and scaled true-residual diagnostics.
Loosening tolerance also loosens training-logit preservation.
Accepted line-search measurements (including factor gradients) are reused.
An optional `progress(event)` callback reports initialization and accepted
projection steps; callbacks are never stored in model checkpoints.

The alpha correction preserves f(X) up to numerical solve/evaluation error.
`max_value_error` reports the observed maximum change over all training logits.
Already rank-one targets are consolidated without kernel solves. These
reports have reason `already_rank_one` and `max_value_error=None`, because no
additional training-logit pass is performed.
No G, H, Schur matrix, or full n*d*c residual is stored; derivative residuals
are evaluated in query tiles. The default matrix-free mode also avoids K.
Candidate-minus-target coefficients are contracted together so each residual
Jacobian needs only one kernel pass. Alpha remains an n*c array. State records
`last_projection` in iterations and `last_projection_epoch` in completed epochs.

Increase `project_every` when projection time dominates and the temporary
buffer fits comfortably. Shorten it when buffer memory or compression error
becomes large. Monitor validation accuracy after projection: preserving
training logits does not preserve all test predictions.

Base float32 coefficient storage is n*(d+2*c)*4 bytes, excluding training data,
pending factors, inner-optimization copies/gradients, and tiled workspace.
Full CIFAR-10 projection throughput has not been benchmarked.


For the 54,000-center FashionMNIST split, float32 scalar K occupies 10.86 GiB
when dense, or 5.43 GiB with one packed triangle. Three packed scalar radial
matrices would occupy 16.30 GiB before coefficients and workspace. This does
not make the full derivative matrices practical: dense G would require about
9.14 TB and dense H about 7.17 PB (decimal units). Keeping derivative
contractions factored and tiled remains necessary.

KLR `new_api` provides `kernel_operator(..., storage="packed")`, including
selected-row and matrix-right-hand-side products. At commit `04d56d69`, its
public constructor first builds a dense temporary, and its packed matrix
products use Python loops over tiles and diagonal rows. The memory saving
therefore does not imply a speed improvement for ten-class EigenPro solves.
The FashionMNIST runner retains its dense scalar cache, which fits on each
available GPU. Packed storage is not currently an `eigenpro_storage` option
in this repository. Reproduce the arithmetic and memory comparison with:

```bash
PYTHONPATH=/path/to/klr python -m experiments.multiclass_storage_benchmark \
  --size 54000 --outputs 10 --batch-size 128 --device cuda:2 \
  --output runs/packed-storage-benchmark.json
```

The benchmark uses a synthetic symmetric matrix to compare storage and
products; it does not measure FashionMNIST training or radial-kernel
construction speed. It reports construction peaks separately from retained
matrix storage and checks dense and packed products against an analytic
reference. Do not run the full-size construction on a GPU that lacks space
for its temporary dense matrix plus packed destination.

Run the executable smoke example and tests with:

```bash
python -m examples.train_multiclass --device cpu --epochs 3 --project-every 1
python -m examples.train_multiclass --device cuda:0 --epochs 3 --project-every 2
python -m tests
```
