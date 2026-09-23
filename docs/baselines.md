# Fashion-MNIST kernel baselines

Run `PYTHONPATH=.deps/auto-attack:. python -m experiments.fashion_baselines` on GPU 0.
The runner reuses the saved lambda-zero experiment split (10,800 training,
1,200 validation, 2,000 official test images) and its train-only median
Matérn-5/2 length scale, 9.05993938446045. Pixels remain in [0,1].

EigenPro2 minimizes squared loss with targets -1/+1. Early stopping is its
regularizer: this is an early-stopped least-squares kernel machine, not the
closed-form positive-ridge solution. Its update matches torchkernels 0.1
`solvers/eigenpro2.py`; a deterministic one-epoch comparison matched exactly.
Use 1,024 preconditioner samples, rank 100, minibatch 128, seed 42,
maximum 200 epochs, and patience 10; retain the earliest highest validation
accuracy. The Gram matrix is cached for these baseline experiments.

SVM minimizes sum hinge loss + lambda/2 times squared RKHS norm, with an
unregularized intercept. Thus C=1/lambda. Search lambda=10^4,...,10^-6;
validation-accuracy ties select larger lambda. A boundary winner triggers
an error rather than silently accepting a truncated search. sklearn/libsvm
is used because the installed cuML/CuPy dependencies fail during import.
Exported signed dual coefficients and intercept are checked against the
solver decision function before attacks. GPU kernel matrices use float32;
libsvm receives the same matrices converted to float64.

Test accuracy is logged but never used for model selection. The saved
selected checkpoints are attacked on all 2,000 test images at 8/255 with
the same binary-compatible AutoAttack as the previous experiments:
APGD-CE (100 iterations, 5 restarts), FAB (100 iterations, 5 restarts), and
Square (5,000 queries), seed 42. DLR is excluded because there are two logits.
The attack adapter includes the SVM intercept and exact first derivatives.
