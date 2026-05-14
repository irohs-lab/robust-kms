# robust-kms
This code accompanies the submission 'Adversarial Robustness of Kernel Machines: A Comprehensive Empirical Analysis' made to NeurIPS 2026.

This repo consists of multiple models, and attacks precisely Fully Connected Neural Networks defined as **FCNN** in `best_nn.py`, Convolutional Neural Networks defined as **LeNet** in `best_nn.py`, **Kernel Ridge Regressors** defined in `updated_rfm.py`, **Recursive Feature Machines** defined in `updated_rfm.py`, and **Random Feature Classifiers** defined in `rff_training.py`. We will be using **RFC** for Random Feature Classifiers, **KRR** for Kernel Ridge Regressors, **RFM** for Recursive Feature Machines, **FCNN** for Fully Connected Neural Networks, and **LeNet** for CNNs. Only difference between **Kernel Ridge Regressors** and **Recursive Feature Machines** is the value of *iteration_idx* which is $0$ for **KRR's** and some $0 \ge value \le 5$, this 5 is the maximum number of iterations we wish to do, that can be changed, but we are doing for 5 iterations. 

We have three different types of attacks, one is **PGD $\ell_\infty$**, another is autoattack's apgd-ce for $\ell_2$ norm, and one more is black box attack namely **Simultaneous Perturbation Stochastic Approximation** attack(defined in `spsa_attack.py`).

We also have cafa attack for tabular dataset everything related to it is inside **cafa** folder.

`dataset.py` contains the code for datasets used for training and attacks. You need to change the path at which the dataset needs to be stored, for that in `get_dataset_path()` function in `dataset.py` file,  change `dataset_dir` variable to a directory name of your choice. It is the directory from where the data would ne stored and fetched.

This repository provides implementations of:

- Kernel Ridge Regression (KRR)(Gaussian / Laplacian)
- Recursive Feature Machines (RFM)(Gaussian / Laplacian)
- FCNN / LeNet( Benign and Adversarially trained)
- Random Feature Classifier Training(Gaussian / Laplacian)


# Robustness of Kernel Methods, Recursive Feature Machines, Random Features, and Neural Networks

This repository provides:

---

## Learning Algorithms

### Kernel Methods

- Kernel Ridge Regression (KRR)
- Recursive Feature Machines (RFM)
- Random Feature Classifiers (RFC)
### Neural Networks

- Fully Connected Neural Networks (FCNN)
- LeNet CNNs

### Robust Training

- PGD adversarial training

### Robust Evaluation

- PGD ($\ell_\infty$)
- AutoAttack($\ell_2$)
- Simultaneous Perturbation Stochastic Approximation(SPSA)
---

# Repository Structure

In the below directory structure *mc* is for multiclass, *pairwise* is for binary problem,  *dataset* can take values fmnist, kmnist, qmnist. 
```bash
.
├── tune_hparams.py
├── save_rfms.py
├── updated_rfm.py
├── <dataset>2/ #

├── rff_training.py
├── rff_<kernel>_<dataset>_pairwise # pairwise is for binary problen.
├── rff_<kernel>_<dataset>_mc # 'mc' is for multiclass. 


├── best_nn.py
├── adv_trn.py
├── FCNN_nn_<dataset>_per_class
├── FCNN_nn_<dataset>_mc
├── FCNN_lenet_<dataset>_per_class
├── FCNN_lenet_<dataset>_mc

├── attack_nn.py
├── attack_adv_nn.py
├── attack_ker_rfm.py
├── spsa_attack.py

├── dataset.py
├── results/
```

---

# Installation

```bash
conda env create -f blackwell_ker_rfm.yml
conda activate blackwell_ker_rfm

pip install pytictoc
pip install git+https://github.com/fra31/auto-attack
pip install -I git+https://github.com/parthe/torchkernels
```

---

# Supported Datasets

- FashionMNIST(fmnist)
- KMNIST(kmnist)
- QMNIST(qmnist)

---

# Pipeline for KRR and RFM

---

# 1. Hyperparameter Tuning

The script `tune_hparams.py` performs a grid search over:

- Kernel bandwidth (`L`)
- Ridge regularization (`λ`)

and selects the best configuration based on validation accuracy.

---

## Binary Classification

Example:

```bash
python tune_hparams.py \
    --DATASET fmnist \
    --VERSION rfm
```

---

## Multiclass Classification

Example:

```bash
python tune_hparams.py \
    --DATASET fmnist \
    --VERSION rfm \
    --MC
```

*--VERSION* can be either 'kernel' for KRR, or 'rfm' for Recursive Feature Machines. Additionally, to sweep over different kernels, you need to uncomment and comment this snippet 
```
SWEEPS = {
    "gaussian": {
        "length_scales": [0.05, 0.125, 0.25, 0.5, 1, 2.0, 4.0, 6.0],
        # "length_scales": [0.05, 0.125],
        # "length_scales": [0.1],
        # "regularizers": [1e-2],
        "regularizers": [1e-4, 1e-3, 1e-2],
    }
    # }, 

    # "laplacian": {
    #     # "length_scales": [0.1, 0.05, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 6.0],
    #     "length_scales": [0.05, 0.125, 0.25, 0.5, 1, 2.0, 4.0, 6.0],
    #     # "length_scales": [1.0],
    #     "regularizers": [1e-4, 1e-3, 1e-2],
    #     # "regularizers": [1e-2],
    # }
    
}
```
in *tune_hparams.py*.

---

## Output

This generates:

### Metrics

```bash
f"./{args.dataset}2/{args.kernel}_{args.version}/metrics_*.json" # For binary.
f"./{args.dataset}2/{args.kernel}_{args.version}_mc/metrics_*.json" # For multi-class.
```

### Best configurations

```bash
tune_idx_*.txt
```

### Sweep summaries

```bash
./results/*.json
```

---

# 2. Training and Saving RFMs

The script `save_rfms.py` trains either:

- Kernel model (`version=kernel`)
- Recursive Feature Machine (`version=rfm`)

---
## Laplacian RFM

For binary, here class1 can take values in 0-9, and class2 in 1-9. 
```bash
python save_rfms.py \
    --kernel laplacian \
    --L 1.0 \
    --reg 0.001 \
    --dataset fmnist \
    --class1 0 \
    --class2 1 \
    --version rfm
```

For  multiclass, keep class1 to be 0, and class2 to be 1.
```bash
python save_rfms.py \
    --kernel laplacian \
    --L 1.0 \
    --reg 0.001 \
    --dataset fmnist \
    --class1 0 \
    --class2 1 \
    --version rfm \
    --mc
```

--kernel is either "laplacian" or "gaussian". 
--L is the length_scale value for the kernel or rfm to be trained.
--reg is the ridge regulariser value.
--dataset is in "fmnist","kmnist","qmnist".
--version is either "kernel" or "rfm".
--MC to be included for multiclass training.

---

## Metrics-only mode

For lightweight tuning:

```bash
python save_rfms.py ... --metrics_only
```

This saves:

```bash
f"./{args.dataset}2/{args.kernel}_{args.version}/metrics_*.json
```

instead of full tensor artifacts.

---

## Full Artifact

Training mode saves:

```bash
f"./{args.dataset}2/{args.kernel}_{args.version}/*.pt" # For binary.
f"./{args.dataset}2/{args.kernel}_{args.version}_mc/*.pt # For multiclass
```

---

# Example Workflow

---

## Tune hyperparameters

For binary or 1v1.
```bash
python tune_hparams.py \
    --DATASET fmnist \
    --VERSION rfm
```
For multiclass
```bash
python tune_hparams.py \
    --DATASET fmnist \
    --VERSION rfm \
    --MC
```

---

## Train single model

```bash
python save_rfms.py \
    --kernel gaussian \
    --L 1.0 \
    --reg 0.001 \
    --dataset fmnist \
    --class1 0 \
    --class2 1 \
    --version rfm
```
---

# Notes

- `VERSION=kernel` performs standard kernel regression.
- `VERSION=rfm` performs recursive feature learning.

---

# Pipeline for FCNN, and LeNet benign and adversarial training.

# 1. Clean Neural Network Training

Implemented in:

```bash
best_nn.py
```

Supports:

- FCNN
- LeNet

---

## FCNN

```bash
python best_nn.py \
    --DATASET fmnist
```

---

## LeNet

```bash
python best_nn.py \
    --DATASET fmnist \
    --CNN
```

---

## Multiclass

```bash
python best_nn.py \
    --DATASET fmnist \
    --MC
```

---

## Output

Saved artifacts:

```bash
best_model.pt
history_*.jsonl
runs_*.jsonl
runs*.csv
```

---

# 2. Adversarial Training

Implemented in:

```bash
adv_trn.py
```

Uses multi-step PGD adversarial training.

Supports:

- PGD $\ell_\infty$ attacks
- Autoattck $\ell_2$ attacks

---

## FCNN + PGD $\ell_\infty$

```bash
python adv_trn.py \
    --DATASET fmnist \
    --MODEL fcnn \
    --NORM Linf
```

---

## LeNet + $\ell_2$

```bash
python adv_trn.py \
    --DATASET fmnist \
    --MODEL lenet \
    --NORM L2
```

---

Training uses:

```math
\epsilon=6/255 \quad (\ell_\infty)
```

```math
\epsilon=1.0 \quad (\ell_2)
```

---

## Output

```bash
best_model.pt
history_*.jsonl
runs*.csv
```

---

# Pipeline for Random Feature Classifier training.

Implemented in:

```bash
rff_training.py
```

Supports:

- Gaussian RFF
- Laplacian RFF

Hyperparameter tuning over:

- λ (ridge)
- length scale

---

## Gaussian RFF

```bash
python rff_training.py \
    --DATASET fmnist \
    --kernel Gaussian
```

---

## Laplacian RFF

```bash
python rff_training.py \
    --DATASET fmnist \
    --kernel Laplace
```

---

## Output

```bash
*.json
*.pt
```

Containing:

- learned weights
- kernel parameters
- validation/test accuracy

---

# Attack Codes

# 1. Attack Clean Neural Networks / RFF

Implemented in:

```bash
attack_nn.py
```

Supports:

- FCNN
- LeNet
- RFF

---

## FCNN

```bash
python attack_nn.py
```

---

## LeNet

Change:

```python
model_type="lenet"
```

---

## RFF

Change:

```python
model_type="rff"
```

inside the code.

---

---

# 2. Attack Adversarially Trained Networks

Implemented in:

```bash
attack_adv_nn.py
```

Supports:

- adversarial FCNN
- adversarial LeNet

---

## Example

```bash
python attack_adv_nn.py
```

Evaluates:

- robust accuracy
- ε sweep

---

---

# 3. Attack Kernel / RFM Models

Implemented in:

```bash
attack_ker_rfm.py
```

Supports:

- Kernel Ridge Regression
- Recursive Feature Machines

Supported kernels:

- Gaussian
- Laplacian

---

## Example

```bash
python attack_ker_rfm.py \
    --DATASET qmnist \
    --AA_norm Linf \
    --VERSION rfm \
    --Kernel_Name laplacian
```

---

# 4. SPSA Evaluation

Implemented in:

```bash
spsa_attack.py
```

Provides:

- batched SPSA
---

## Example

```bash
python spsa_attack.py \
    --DATASET qmnist \
    --AA_norm Linf \
    --VERSION rfm \
    --Kernel_Name laplacian
```

---

# Output

All scripts save:

```bash
PGD_result_*.json
```

Containing:

- epsilon
- step size
- robust accuracy
- model metadata

---

# Epsilon Sweep

ℓ∞:

```math
\epsilon \in \{0,0.1,1,2,4,6,8,12,20\}/255
```

ℓ₂:

```math
\epsilon \in \{0,0.5,1,2,4,6,8\}
```

---

# Reproducibility

All experiments use:

- seed = 42
- deterministic subset sampling
- identical train/validation/test splits

---
