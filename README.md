# robust-kms
This code accompanies the submission [Adversarial Robustness of Kernel Machines: A Comprehensive Empirical Analysis](https://arxiv.org/abs/2611.07474) made to arXiv. Go to [cvx-formulation](https://github.com/irohs-lab/robust-kms/tree/cvx-formulation) branch for better understanding of adversarial KLR code.

# Kernel and Neural Network Robustness Experiments

Train image classifiers and evaluate their robustness with L2 and Linf
APGD-CE and SPSA attacks.

Supported model families:

- Kernel ridge regression (KRR), Gaussian and Laplacian.
- Kernel logistic regression (KLR), Gaussian and Laplacian.
- Precomputed-kernel SVC, Gaussian and Laplacian, binary and multiclass.
- Random Feature Classifiers (RFC), Gaussian and Laplace, with ridge or cross-entropy classifiers.
- FCNN, LeNet, and WideResNet.
- Gaussian gradient-regularized KLR, binary and multiclass (Adversarial KLR).
- PGD adversarially trained FCNN and LeNet.

## Installation

Run from the repository root. Create and activate the vision environment:

```bash
conda env create --file blackwell_ker_rfm.yml --name blackwell_ker_rfm
conda activate blackwell_ker_rfm
```

For CAFA dataset experiments, use the separate environment:

```bash
conda env create --file tab_attack.yml --name tab_attack
conda activate tab_attack
```

## Data and experiment settings

The grayscale examples below use Fashion-MNIST (`fmnist`). Related loaders
support KMNIST (`kmnist`), QMNIST (`qmnist`), and EMNIST (`emnist`). Support
for SVHN and CIFAR varies by model; see the limitations below.

Before running on another machine, configure dataset paths in `dataset.py`,
`rff_code/dataset.py`, and `rff_code/trainers/dataset.py`. The current loaders
use absolute paths. These three implementations differ and are required by
different entry points. Keep the repository layout intact.

Run every command from the repository root. `PYTHONPATH=.` allows scripts
under subdirectories to find root modules. Paths such as `path/to/model.pt`
must be replaced with real outputs from training.

Classification flags differ by script:

| Scripts | Multiclass | Binary |
| --- | --- | --- |
| KRR | `--mc` | `--class1 0 --class2 1` |
| Benign KLR | `--multiclass` | `--classes 0 1` |
| NN and RFC | `--MC` | Omit `--MC`; NN training pairs are configured in source |
| SVC training | `--MC` | Omit `--MC` to train all 45 digit-label pairs |
| SVC attacks | Omit `--CLASSES` | `--CLASSES 0 1` |

## Benign training

### Kernel ridge regression

Use `--version kernel` for ordinary KRR. `--version rfm` performs metric
learning and is a separate experiment.

```bash
python save_rfms.py --dataset fmnist --kernel gaussian --version kernel --L 1 --reg 0.001 --mc
python save_rfms.py --dataset fmnist --kernel laplacian --version kernel --L 1 --reg 0.001 --class1 0 --class2 1
```

Checkpoints are saved under `fmnist2/`. Omit `--metrics_only` so training
centers and coefficients are retained for attacks. `best_round_accuracy` records the best accuracy.

### Kernel logistic regression

```bash
python KernelLR.py --dataset fmnist --kernel gaussian --multiclass
python KernelLR.py --dataset fmnist --kernel laplacian --classes 0 1
```

Outputs are under `klr_sweep_fmnist/`. Supply multiple binary pairs sequentially,
for example `--classes 0 1 2 3`. Hyperparameter grids are configured in source.

### Kernel SVC

```bash
PYTHONPATH=. python rff_code/trainers/train_precomputed_svc.py --DATASET fmnist --MC --kernels gaussian laplacian --output-root svc_cv
```

This builds kernels in PyTorch and fits sklearn precomputed-kernel SVCs.
Keep each `*_best.pkl`, `*_results.json`, and `*_xtrain.pt` together: attacks
need the exact fit-time training rows. Omit `--MC` for pairwise training.

### Random Feature Classifiers (RFCs)

```bash
PYTHONPATH=. python rff_code/run_train.py --DATASET fmnist --MC --kernel gaussian --trn square --num_features 5000 --seed 42
PYTHONPATH=. python rff_code/run_train.py --DATASET fmnist --MC --kernel Laplace --trn ce --num_features 5000 --seed 42
```

`square` fits ridge weights; `ce` fits a linear classifier with cross entropy.
For RFC, spell the kernel `Laplace` or `laplace`, rather than `laplacian`.
Keep training summaries and checkpoints, including feature-map seeds and bias.

### FCNN, LeNet, and WRN

```bash
python best_nn.py --DATASET fmnist --MC --SEED 42
python best_nn.py --DATASET fmnist --MC --CNN --SEED 42
python best_nn.py --DATASET fmnist --MC --WRN --SEED 42
```

Default architecture is FCNN; `--CNN` selects LeNet and `--WRN` selects WRN.
Use one architecture flag at a time. Omitting `--MC` currently trains only
the source-configured pair `(5,8)`; change it before comparing another pair.
The selected checkpoint is `best_model.pt` inside the printed output directory.

## Robust and adversarial training

### Gaussian robust KLR

```bash
PYTHONPATH=. python kernel_adv_trn/adv_trn_binary.py --DATASET fmnist --CLASS_PAIRS 0 1 --KERNEL gaussian --RHO 0.04 --LAM 1 --OUTPUT kernel_adv_trn/runs/binary_models
PYTHONPATH=. python kernel_adv_trn/adv_trn_mc.py --DATASET fmnist --KERNEL gaussian --EPS 0.031372549 --RHO_MULT 1 --OUTPUT kernel_adv_trn/runs/multiclass.pt
```

For binary training, `RHO` is a gradient-penalty weight. The multiclass trainer
sets `rho = EPS * RHO_MULT`. These coefficients are not certified attack radii.
The multiclass `--LAM` accepts one value; binary `--LAM` accepts a list. `--RHO_MULT` is 1 for binary and 2 for multiclass KLR.

### FCNN and LeNet adversarial training

```bash
python adv_trn.py --DATASET fmnist --MC --MODEL fcnn --NORM Linf
python adv_trn.py --DATASET fmnist --MC --MODEL lenet --NORM L2
```

Both architectures accept `Linf` and `L2`. Current training budgets are fixed
in source: FCNN Linf `24/255`, LeNet Linf `6/255`, and L2 `1.0` for both.
Outputs use `best_model_1.pt`.

## Attack evaluation

Linf budget lists use pixel units: `8` means `8/255` on inputs in `[0,1]`.
L2 lists use raw input-space radii, such as `0.5` or `1.0`.
Train once, then evaluate each norm separately. Except for the dedicated
multiclass APGD evaluator, scripts may use fixed budget lists in source.

### KRR: prepare the attack summary

The KRR evaluator reads a JSON list of checkpoint paths. For the Gaussian
multiclass example above, create its summary:

```bash
python - <<'PY'
import json
from pathlib import Path

checkpoint = Path("fmnist2/kernel_results_gaussian_fmnist_L1_reg0.001_mc.pt")
assert checkpoint.is_file(), checkpoint
summary = Path("fmnist2/gaussian_kernel_mc/gaussian_kernel.json")
summary.parent.mkdir(parents=True, exist_ok=True)
summary.write_text(json.dumps({"results": [{"path": str(checkpoint.resolve())}]}))
PY
```

For binary, use `<dataset>2/<kernel>_kernel/<kernel>_kernel.json` and add
`"classes": [0, 1]` to each entry. Use a dictionary containing `results`,
rather than a bare JSON list. The optional tuning script currently has
inconsistent output paths; explicit summary preparation avoids that dependency.

```bash
python svhn_new.py --DATASET fmnist --MC --VERSION kernel --Kernel_Name gaussian --AA_norm Linf --PGD
python svhn_new.py --DATASET fmnist --MC --VERSION kernel --Kernel_Name gaussian --AA_norm L2 --BB_attack spsa
```

Both attacks accept either norm. Substitute `laplacian` and its matching
summary/checkpoint for the Laplacian model.

### Benign KLR

```bash
python attack_klr_benign.py --dataset fmnist --kernel gaussian --multiclass --norm Linf
python kernel_spsa_atk.py --dataset fmnist --kernel gaussian --multiclass --norm L2
```

The first command runs APGD-CE; the second runs SPSA. Each supports both norms
and kernels. For binary, replace `--multiclass` with `--classes 0 1`.

### Precomputed SVC

Use the same evaluator for either attack, kernel, norm, or class setting:

```bash
PYTHONPATH=. python rff_code/trainers/attack_precomputed_svc.py --DATASET fmnist --KERNEL gaussian --AA_norm Linf --ATTACK apgd-ce --MODEL_FILE path/to/model_best.pkl --RESULTS_JSON path/to/model_results.json --XTRAIN_FILE path/to/model_xtrain.pt --EPSILONS 0 4 8
PYTHONPATH=. python rff_code/trainers/attack_precomputed_svc.py --DATASET fmnist --KERNEL laplacian --AA_norm L2 --ATTACK spsa --MODEL_FILE path/to/model_best.pkl --RESULTS_JSON path/to/model_results.json --XTRAIN_FILE path/to/model_xtrain.pt --EPSILONS 0 0.5 1
```

Add `--CLASSES 0 1` for a binary checkpoint. Keep the sklearn agreement check
enabled. Optional `--OUTPUT_PATH` selects the result JSON; `--SAVE_ADV` also
saves attacked inputs. Default output filenames distinguish APGD and SPSA.

### FCNN, LeNet, and WRN

```bash
python attack_nn.py --DATASET fmnist --MC --MODEL_TYPE fcnn --CKPT_PATH path/to/best_model.pt --AA_norm Linf --PGD --OUTPUT_PATH results/fcnn_linf_apgd.json
python attack_nn.py --DATASET fmnist --MC --MODEL_TYPE lenet --CKPT_PATH path/to/best_model.pt --AA_norm L2 --BB_attack spsa --OUTPUT_PATH results/lenet_l2_spsa.json
```

Use `--MODEL_TYPE wrn` for WRN. Both attacks support both norms.
For adversarial FCNN/LeNet, select their `best_model_1.pt` checkpoint through
`--CKPT_PATH`. For binary evaluation omit `--MC`; checkpoint paths currently
need a `(class1, class2)` component because the evaluator parses the pair from
the path. Set an explicit output path to distinguish different training runs.

### Random features

```bash
PYTHONPATH=. python rff_code/attack_rff.py --DATASET fmnist --MC --MODEL_TYPE rff --RFF_KERNEL gaussian --TRN_TYPE square --NUM_FEATURES 5000 --SEED 42 --JSON_PATH path/to/training_summary.json --AA_norm Linf --PGD
PYTHONPATH=. python rff_code/attack_rff.py --DATASET fmnist --MC --MODEL_TYPE rff --RFF_KERNEL Laplace --TRN_TYPE ce --NUM_FEATURES 5000 --SEED 42 --JSON_PATH path/to/training_summary.json --AA_norm L2 --BB_attack spsa
```

Match the kernel, training type, feature count, seed, and summary to the actual
trained model. Both attacks support both norms. Supply `--JSON_PATH` explicitly.

### Gaussian robust KLR

Binary checkpoints use one evaluator for both attacks:

```bash
PYTHONPATH=. python kernel_adv_trn/attack_adv_klr.py --checkpoint path/to/fmnist_0_vs_1_best.pt --NORM Linf --ATTACK apgd-ce --EPSILONS 0 4 8
PYTHONPATH=. python kernel_adv_trn/attack_adv_klr.py --checkpoint path/to/fmnist_0_vs_1_best.pt --NORM L2 --ATTACK spsa --EPSILONS 0 0.5 1
```

Multiclass checkpoints have separate evaluators:

```bash
PYTHONPATH=. python kernel_adv_trn/mc_robust_attack.py --checkpoint kernel_adv_trn/runs/multiclass.pt --norm Linf --attacks apgd-ce --budgets 0 4 8 --output results/mc_apgd_linf
PYTHONPATH=. python kernel_adv_trn/klr_spsa_mc.py --checkpoint kernel_adv_trn/runs/multiclass.pt --norm L2 --budgets 0 0.5 1 --output results/mc_spsa_l2.json
```

Both multiclass evaluators accept either norm. Use a new APGD output directory
for each run. Binary and multiclass checkpoint formats are different.

### SPSA controls for SVC and binary robust KLR

The two updated evaluators accept:

| Option | Default | Meaning |
| --- | --- | --- |
| `--SPSA_ITERS` | 100 | Optimization iterations |
| `--SPSA_SAMPLES` | 32 | Random directions per iteration |
| `--SPSA_BATCH_SIZE` | 32 | Images processed together |
| `--SPSA_QUERY_BATCH_SIZE` | 128 | Maximum images per prediction call |
| `--SPSA_ETA` | 0.01 | Finite-difference radius, raw units |
| `--SPSA_LR` | epsilon/10 | Update size, raw units |

`--SEED` controls reproducibility. Increasing iterations or samples increases
query cost. These controls apply to the shared SPSA implementation; other
evaluators have their own settings and CLI spellings.

## Results and limitations

Preserve checkpoint metadata, preprocessing, class order, training splits,
and random-feature seeds alongside reported results. Accuracy scales differ:
the precomputed SVC evaluator reports percentages; binary robust KLR records
clean accuracy as a fraction and its legacy `pgd_acc` values as percentages.
For binary robust KLR, the JSON `pgd` list is retained for compatibility;
the `attack` field records whether the run used APGD or SPSA.


---

# For CaFA:-
```
cd cafa
conda env create -f tab_attack.yml
conda activate tab_attack

pip install pytictoc
pip install git+https://github.com/fra31/auto-attack
pip install -I git+https://github.com/parthe/torchkernels
```
You can run cafa attack on KRR, RFM or NN using this command, i.e, `python sweep_cafa.py`.
To change the dataset and model to be attacked, just change this snippet in the code, 
```
# datasets = ["adult","bank", "phishing"] # We can also evaluate on multiple datasets at once, in that case uncomment this line.
datasets = ["adult"] # Either we can evaluate over single dataset, in that case use this line with appropriate dataset name
# models = ["mlp"] # models can be ["mlp"],["rfm"],["kernel"]
models = ["rfm"] # Only laplacian for now.
# models = ["rfm"]
```
by uncommenting appropriate datasets and models.
*datasets* can take values in "adult", "bank" or "phishing".
While models can be "mlp" for FCNN, "rfm" for Recursive Feature Machines and "kernel" for Kernel Ridge Regressors, currently it only supports Laplacian Kernel.

Here, see used is 0. We can also, change the $\ell_\infty$ epsilons on which our attack is to be evaluated by having appropriate values for eps_values. 
```
# eps_values = [0.01, 0.03333333333333333 , 0.075]
eps_values = [0.075]
```
Also, in the code `base_out` is the directory to which the output will be saved. The consolidated results can be seen in `base_out/evaluations.json`.
This CaFA attack is taken from [CaFA](https://github.com/matanbt/attack-tabular), we have optimized it to evaluate our attack only on the compliant samples.

