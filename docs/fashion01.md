# Fashion-MNIST 0 vs 1 experiment

The experiment uses the official 12,000 training / 2,000 test samples from
classes 0 (T-shirt/top) and 1 (trouser). Pixels are divided by 255 and stay in
[0,1]; there is no standardization or per-image normalization.

Training uses scalar logistic loss, exact Matern-5/2 kernels, lambda=1, and
rho=epsilon=8/255. Binary logistic loss is 1-Lipschitz in its scalar logit.
The input-gradient penalty is the specified first-order robustness surrogate,
not a certificate of adversarial robustness.

Defaults are 100 sampling epochs, batch size 128, eta_m=eta_0/(m+1)^0.6,
and eta_0=1/(lambda+n/4). An epoch means ceil(n/128) fresh uniform subset
draws; it is not a pass through a fixed permutation. The kernel length scale
is the median pairwise Euclidean distance among 1,024 seeded training
examples. These are fixed initial-experiment settings, not test-tuned choices.
The final attacked model is the eta-weighted average; test accuracy is
monitored but not used for model selection or early stopping.

The JSONL log reports full train and test accuracy after every epoch,
along with their logistic losses and the full training objective.
Accuracy is stored as a fraction. Checkpoints contain the centers and
value/derivative representer coefficients.

## Queue on GPU 0

The existing Fashion-MNIST cache defaults to
/janaki/common/Datasets/rahulky.
Alternatively pass --data-root with a torchvision FashionMNIST cache.

Fetch the official attack implementation into the ignored dependency directory:

```bash
git clone https://github.com/fra31/auto-attack.git .deps/auto-attack
python -m experiments.queue_fashion \
  --output runs/fashion01-matern52-eps8-255 --device cuda:0
```

Use the project's PyTorch Python environment. torchvision and numpy are
required in addition to the library dependencies. The queue launcher adds
the attack checkout to its child's PYTHONPATH. It uses a local flock on
/tmp/robust-kernels-gpu0.lock to serialize these experiment jobs, detaches
from the terminal, and records the PID in launch.json. This lock coordinates
these launchers only, not unrelated GPU jobs. The output directory must be new.

## AutoAttack

Training is followed automatically by attacks on all 2,000 test images,
under Linf epsilon=8/255 and valid pixel bounds [0,1]. The official AutoAttack
source is used, with version='custom' and:

- APGD-CE: 100 iterations, 5 restarts.
- Untargeted FAB: 100 iterations, 5 restarts.
- Square: 5,000 queries, 1 restart.

The DLR attacks require more than two classes and are omitted. This is
explicitly a **binary-compatible custom AutoAttack** evaluation, not the
standard four-component multiclass preset. The two logits [0,f(x)] reproduce
the scalar logistic model exactly, with no extra classes.

The attack adapter exposes the exact analytic input Jacobian to PyTorch
autograd without retaining the kernel tiles. It is tested against numerical
finite differences and supports first-order attacks, not higher derivatives.
A separate GPU smoke test exercises all three attack components.

Artifacts in the run directory:

- config.json: hyperparameters, dataset sizes, and code/AutoAttack commits.
- metrics.jsonl: per-epoch accuracy/loss and final clean/robust accuracy.
- console.log and autoattack.log: live progress.
- status.json: most recent event or an exception traceback.
- model.pt: final averaged model, saved before attacking.
- adversarial.pt: attacked test images and labels.

Final robust accuracy uses all test examples as the denominator, including
those already misclassified without perturbation. The runner verifies the
pixel bounds and maximum Linf perturbation of the returned examples.
