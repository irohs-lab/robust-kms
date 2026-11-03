# Binary robust kernel training and evaluation

`train_robust.py` trains the objective already implemented in this repository:

```
sum_i BCEWithLogits(f(x_i), y_i)
+ rho * sum_i ||gradient_x f(x_i)||_1
+ lam / 2 * ||f||_RKHS^2
```

This is gradient-regularized kernel fitting. It does not generate adversarial
training examples. `rho` is a penalty coefficient; the validation and test
attack radii are separate parameters. No particular hyperparameter setting
is guaranteed to improve robustness.

The default `--optimizer subgradient` uses the exact dual maximizer
`rho * sign(gradient_x f)` in the coefficient update. This avoids waiting for
a tiny dual ascent step to reach the rho boundary. `--optimizer primal_dual`
uses the original simultaneous updates. Existing callers of `rk.fit` and
`rk.step` retain the original `primal_dual` default.

Use the existing environment with torch, torchvision, torchkernels and
AutoAttack installed. Run these commands from `kernel_adv_trn`.

## First run a small diagnostic

```bash
python train_robust.py \
  --output runs/new_smoke --dataset fmnist --classes 0 1 \
  --device cuda:0 --optimizer subgradient --rho 0.04 \
  --train-limit 512 --validation-samples 64 \
  --iterations 1000 --eval-every 250 --average-start 500 \
  --step-size 0.0001 --decay 0.6 --length-scale 28 \
  --validation-budget 8 --pgd-steps 20 --pgd-restarts 2 \
  --full-objective
```

Inspect `metrics.jsonl`: BCE, gradient L1 norm, beta norm, full objective,
clean validation accuracy, and validation PGD accuracy should be considered
together. `train_probe_bce_plus_penalty` excludes the RKHS norm term and is
only a probe diagnostic; `full_training_objective.total` is the exact summed
objective. Full objective evaluation is costly for large training sets.

## Controlled benign versus robust comparison

```bash
python train_robust.py \
  --output runs/new_benign --dataset fmnist --classes 0 1 \
  --optimizer subgradient --rho 0 --seed 0 \
  --step-size 0.0001 --decay 0.6 --iterations 20000 \
  --average-start 10000 --length-scale 28 --validation-budget 8

python train_robust.py \
  --output runs/new_rho004 --dataset fmnist --classes 0 1 \
  --optimizer subgradient --rho 0.04 --seed 0 \
  --step-size 0.0001 --decay 0.6 --iterations 20000 \
  --average-start 10000 --length-scale 28 --validation-budget 8
```

These are starting settings, not validated optimal settings. Tune step size,
length scale, rho, lam, and training duration using validation results. Do not
infer convergence merely from a completed iteration count. The loss is summed,
so changing the number of training samples changes the scale relative to lam
and affects a sensible step size. With fixed n, switching to a mean loss would
require rescaling lam and the learning rate to preserve the same optimization.

Training uses a deterministic stratified holdout of the official training set;
it never reads the test set. This split differs from the old `adv_trn.py` split.
Retrain both comparison models with the same new workflow and seed. Validation
PGD evaluates a fixed subset (default 256 samples); use
`--validation-samples 0` for the whole validation set. Validation PGD is for
model selection, not the final robustness claim.

Files written:

- `best.pt`: best validation PGD accuracy across evaluated iterations and variants;
  ties are broken by smaller validation BCE. Iteration zero is also evaluated.
- `last.pt`: last evaluated unaveraged coefficients.
- `average.pt`: latest weighted average of pre-update iterates after the warmup.
- `config.json`, `metrics.jsonl`, `best.json`: settings and diagnostics.

Each model contains its centers, coefficients, kernel options, dataset, class
order, and selection metrics. Output directories must be new to prevent mixing
runs. There is no resume mode; use a new output directory for another run.

## Evaluate with a binary AutoAttack suite

```bash
python attack_robust.py \
  --checkpoint runs/new_benign/best.pt --output runs/new_benign_aa \
  --budgets 0 4 8 16 32 --device cuda:0 --batch-size 64

python attack_robust.py \
  --checkpoint runs/new_rho004/best.pt --output runs/new_rho004_aa \
  --budgets 0 4 8 16 32 --device cuda:0 --batch-size 64
```

Budgets are in pixel units: `--budgets 8` means epsilon = 8/255 on raw [0,1]
images. All budgets evaluate the same checkpoint and test examples. Zero means
clean evaluation only. The default evaluates the full official test class pair;
`--test-limit 32` is useful for a smoke test and is recorded in the manifest.

The suite is `version="custom"` with APGD-CE, untargeted FAB, and Square, using
100 APGD/FAB iterations, five restarts, and 5,000 Square queries. Report it as a
**custom binary AutoAttack suite**, not standard AutoAttack. The standard
DLR-based attacks assume more than two logits. `--attacks apgd-ce` is available
for quick debugging, but is not the full custom suite.

Outputs include `manifest.json` (checkpoint hash, dataset/classes, sample count,
and attack settings), `results.json`, and each budget's attack log and adversarial
images. Perturbation norms, image bounds, and finite outputs are checked.

The new evaluator can also read old `adv_trn.py` checkpoints that contain
`config.dataset`, `config.classes`, and `config.data_root`. It loads exactly that
dataset/class pair and maps the first class to 0 and second to 1. Both new scripts
support fmnist, mnist, kmnist, qmnist, and emnist digits; Gaussian and Matérn-5/2
are the supported kernels. Dataset downloading is opt-in via `--download`.

## Python API

```python
import robust_kernels as rk

state = rk.fit(
    x_train, y_train,  # (n,d) raw pixels and binary labels; same device
    rho=0.04, lam=1.0, step_size=1e-4,
    iterations=20000, batch_size=64, decay=0.6,
    kernel="gaussian", length_scale=28.0,
    optimizer="subgradient",  # use "primal_dual" for the original algorithm
)
```

The basic `fit` API retains averaging from iteration zero; the training CLI adds
warmup averaging, validation, logging, and model selection. The analytic logits
wrapper supports first input derivatives for attacks, not autograd training of
coefficient tensors. Conventional PGD adversarial training would require a
separate differentiable training implementation.

## CPU verification

```bash
python tests/test_robust_workflow.py
```

These tests check the subgradient against finite differences of the objective,
original optimizer compatibility, rho sensitivity, analytic attack gradients,
validation splitting, and a PGD example with a known decision boundary.
