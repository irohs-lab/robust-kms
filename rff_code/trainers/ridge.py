import torch
import gc

from kernels import select_kernel
from tensor_utils import ensure_labels, ensure_onehot, augment
from metrics import accuracy_from_logits
from cv_utils import cross_validate_grid, merge_train_val

def solve_ridge_with_bias_auto(
    Phi_train,
    y_train_onehot,
    lamda,
    device,
):
    """
    Solves ridge regression with an unregularized bias:

        min_{W,b} ||Phi W + b - Y||^2 + lamda ||W||^2

    Bias is handled correctly by centering Phi and Y.

    Uses:
        primal if p <= N
        dual   if N < p

    unless prefer_dual is explicitly True/False.

    Parameters
    ----------
    Phi_train:
        [N, p] random feature matrix

    y_train_onehot:
        [N, C] one-hot labels

    lamda:
        ridge regularization strength

    Returns
    -------
    W:
        [p, C]

    b:
        [C]

    branch:
        "primal" or "dual"
    """

    N, p = Phi_train.shape
    dtype = Phi_train.dtype

    y_train_onehot = y_train_onehot.to(device=device, dtype=dtype)

    # Means
    phi_mean = Phi_train.mean(dim=0, keepdim=True)      # [1, p]
    y_mean = y_train_onehot.mean(dim=0, keepdim=True)   # [1, C]


    use_dual = N < p


    lamda = float(lamda)

    if not use_dual:
        print(f"Using PRIMAL branch: N={N} >= p={p}")
        # ------------------------------------------------------------
        # PRIMAL:
        #
        # W = (Phi_c^T Phi_c + lambda I)^(-1) Phi_c^T Y_c
        #
        # But compute Phi_c^T Phi_c without explicitly storing Phi_c:
        #
        # Phi_c^T Phi_c = Phi^T Phi - N mu_phi^T mu_phi
        # Phi_c^T Y_c   = Phi^T Y   - N mu_phi^T mu_y
        # ------------------------------------------------------------

        G = Phi_train.T @ Phi_train
        outer = phi_mean.T @ phi_mean
        G.sub_(outer, alpha=N)
        del outer

        B = Phi_train.T @ y_train_onehot
        B.sub_(phi_mean.T @ y_mean, alpha=N)

        # Regularize in-place: G += lamda * I  (touches only the diagonal)
        G.diagonal().add_(lamda)

        W = torch.linalg.solve(G, B)  # [p, C]

        branch = "primal"

        del G, B

    else:
        # ------------------------------------------------------------
        # DUAL:
        #
        # alpha = (Phi_c Phi_c^T + lambda I)^(-1) Y_c
        # W     = Phi_c^T alpha
        #
        # We build the centered Gram matrix without explicitly storing Phi_c:
        #
        # K_c = H K H
        # H = I - 11^T / N
        # K = Phi Phi^T
        # ------------------------------------------------------------
        print(f"Using DUAL branch: N={N} < p={p}")

        K = Phi_train @ Phi_train.T  # [N, N]

        row_mean = K.mean(dim=1, keepdim=True)  # [N, 1]
        col_mean = K.mean(dim=0, keepdim=True)  # [1, N]
        total_mean = K.mean()

        # In-place centering to avoid another full N x N allocation
        K.sub_(row_mean)
        K.sub_(col_mean)
        K.add_(total_mean)
        del row_mean, col_mean

        Yc = y_train_onehot - y_mean

        # reg = torch.eye(N, device=device, dtype=dtype)
        K.diagonal().add_(lamda)

        # alpha = torch.linalg.solve(K + lamda * reg, Yc)  # [N, C]
        alpha = torch.linalg.solve(K, Yc)  # [N, C]

        # W = Phi_c^T alpha
        #   = Phi^T alpha - mu_phi^T (1^T alpha)
        alpha_sum = alpha.sum(dim=0, keepdim=True)       # [1, C]
        W = Phi_train.T @ alpha - phi_mean.T @ alpha_sum # [p, C]

        branch = "dual"

        del K, Yc, alpha, alpha_sum

    b = (y_mean - phi_mean @ W).squeeze(0)

    return W, b, branch

def fit_rff_ridge(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    num_classes,
    num_features=1000,
    kernel_type="Laplace",
    lamdas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    length_scales=(0.1, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0),
    seed=42,
    device=torch.device("cuda"),
):
    y_train = ensure_onehot(y_train, num_classes).to(device).float()
    y_val = ensure_onehot(y_val, num_classes).to(device).float()
    y_test = ensure_onehot(y_test, num_classes).to(device).float()

    d = x_train.shape[1]
    p = num_features

    best = None
    all_results = []

    print("Using RFF ridge with automatic primal/dual branch")

    for length_scale in length_scales:
        phi = select_kernel(
            input_dim=d,
            num_features=p,
            kernel_type=kernel_type,
            length_scale=length_scale,
            seed=seed,
            device=device,
        )

        with torch.no_grad():
            Phi_train = phi(x_train.cpu()).to(device=device)
            Phi_val = phi(x_val.cpu()).to(device=device)
            Phi_test = phi(x_test.cpu()).to(device=device)

        N = Phi_train.shape[0]
        print(
            f"\nlength_scale={length_scale} "
            f"| N={N} "
            f"| p={p} "
            f"| auto branch={'dual' if N < p else 'primal'}"
        )

        for lamda in lamdas:
            with torch.no_grad():
                W, b, branch = solve_ridge_with_bias_auto(
                    Phi_train=Phi_train,
                    y_train_onehot=y_train,
                    lamda=float(lamda),
                    device=device,
                )

                logits_train = Phi_train @ W + b
                logits_val = Phi_val @ W + b
                logits_test = Phi_test @ W + b

                train_loss = torch.mean((logits_train - y_train) ** 2).item()
                val_loss = torch.mean((logits_val - y_val) ** 2).item()
                test_loss = torch.mean((logits_test - y_test) ** 2).item()

            train_acc = accuracy_from_logits(logits_train, y_train)
            val_acc = accuracy_from_logits(logits_val, y_val)
            test_acc = accuracy_from_logits(logits_test, y_test)

            result = {
                "lamda": float(lamda),
                "length_scale": float(length_scale),
                "branch": branch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "test_loss": float(test_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
            }

            all_results.append(result)

            print(
                f"length_scale={length_scale:<8} "
                f"lamda={lamda:<10} "
                f"branch={branch:<6} "
                f"train={train_acc*100:.2f}% "
                f"val={val_acc*100:.2f}% "
                f"test={test_acc*100:.2f}% "
                f"val_loss={val_loss:.4f}"
            )

            is_best = (
                best is None
                or val_acc > best["val_acc"]
                or (
                    val_acc == best["val_acc"]
                    and val_loss < best["val_loss"]
                )
            )

            if is_best:
                best = {
                    "lamda": float(lamda),
                    "length_scale": float(length_scale),
                    "branch": branch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "test_loss": float(test_loss),
                    "train_acc": float(train_acc),
                    "val_acc": float(val_acc),
                    "test_acc": float(test_acc),
                    "linear_state_dict": {
                        "weight": W.T.float().detach().cpu(),
                        "bias": b.float().detach().cpu(),
                    },
                    "phi_params": {
                        "input_dim": d,
                        "num_features": p,
                        "kernel_type": kernel_type,
                        "length_scale": float(length_scale),
                        "seed": seed,
                    },
                }

            del W, b, logits_train, logits_val, logits_test
            torch.cuda.empty_cache()

        del Phi_train, Phi_val, Phi_test
        torch.cuda.empty_cache()

    return best, all_results

def fit_rff_ridge_cv(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    num_classes,
    num_features=1000,
    kernel_type="Laplace",
    lamdas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    length_scales=(0.1, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0),
    n_splits=5,
    seed=42,
    device=torch.device("cuda"),
):
    """
    Cross-validation version of RFF ridge regression.

    Workflow:
        1. Merge train + val into development set.
        2. Run K-fold CV over length_scale x lamda.
        3. Pick best hyperparameters using mean CV validation accuracy.
        4. Refit final model on train + val.
        5. Evaluate on train, val, and test.
    """

    print("Using RFF Ridge with K-fold CV")

    y_train = ensure_labels(y_train).to(device)
    y_val = ensure_labels(y_val).to(device)
    y_test = ensure_labels(y_test).to(device)

    x_dev, y_dev = merge_train_val(x_train, y_train, x_val, y_val)

    d = x_dev.shape[1]
    p = num_features

    param_grid = {
        "length_scale": length_scales,
        "lamda": lamdas,
    }

    def fit_eval_fn(
        x_train_fold,
        y_train_fold,
        x_val_fold,
        y_val_fold,
        params,
        fold_id,
    ):
        length_scale = params["length_scale"]
        lamda = params["lamda"]

        phi = select_kernel(
            input_dim=d,
            num_features=p,
            kernel_type=kernel_type,
            length_scale=length_scale,
            seed=seed,
        )

        y_train_fold = ensure_labels(y_train_fold).to(device)
        y_val_fold = ensure_labels(y_val_fold).to(device)

        y_train_onehot = ensure_onehot(y_train_fold, num_classes).to(device).float()

        with torch.no_grad():
            Phi_train = phi(x_train_fold.cpu()).to(device=device)
            Phi_val = phi(x_val_fold.cpu()).to(device=device)

            W, b, branch = solve_ridge_with_bias_auto(
                Phi_train=Phi_train,
                y_train_onehot=y_train_onehot,
                lamda=float(lamda),
                device=device,
            )

            logits_train = Phi_train @ W + b
            logits_val = Phi_val @ W + b

            train_acc = accuracy_from_logits(logits_train, y_train_fold)
            val_acc = accuracy_from_logits(logits_val, y_val_fold)

            train_mse = torch.mean((logits_train - y_train_onehot) ** 2).item()

            y_val_onehot = ensure_onehot(y_val_fold, num_classes).to(device).float()
            val_mse = torch.mean((logits_val - y_val_onehot) ** 2).item()

        del Phi_train, Phi_val, W, b, logits_train, logits_val
        torch.cuda.empty_cache()

        return {
            "branch": branch,
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "train_loss": float(train_mse),
            "val_loss": float(val_mse),
        }

    best_cv, all_cv_results = cross_validate_grid(
        x=x_dev,
        y=y_dev,
        param_grid=param_grid,
        fit_eval_fn=fit_eval_fn,
        n_splits=n_splits,
        stratified=True,
        shuffle=True,
        seed=seed,
        primary_metric="val_acc",
        primary_mode="max",
        tie_breaker_metric="val_loss",
        tie_breaker_mode="min",
        verbose=True,
    )

    print("\nBest CV result:")
    print(best_cv)

    best_params = best_cv["params"]
    best_length_scale = best_params["length_scale"]
    best_lamda = best_params["lamda"]

    phi = select_kernel(
        input_dim=d,
        num_features=p,
        kernel_type=kernel_type,
        length_scale=best_length_scale,
        seed=seed,
        device=device,
    )

    y_dev_onehot = ensure_onehot(y_dev, num_classes).to(device).float()
    y_train_onehot = ensure_onehot(y_train, num_classes).to(device).float()
    y_val_onehot = ensure_onehot(y_val, num_classes).to(device).float()
    y_test_onehot = ensure_onehot(y_test, num_classes).to(device).float()

    with torch.no_grad():
        Phi_dev = phi(x_dev.cpu()).to(device=device)

        W, b, branch = solve_ridge_with_bias_auto(
            Phi_train=Phi_dev,
            y_train_onehot=y_dev_onehot,
            lamda=float(best_lamda),
            device=device,
        )

    
        del Phi_dev
        gc.collect() # Invokes the Python garbage collector to clear CPU/RAM references

        torch.cuda.empty_cache()



        Phi_train = phi(x_train.cpu()).to(device=device)

        logits_train = Phi_train @ W + b
        
        del Phi_train
        gc.collect()
        torch.cuda.empty_cache()

        Phi_val = phi(x_val.cpu()).to(device=device)
        
        logits_val = Phi_val @ W + b

        del Phi_val
        gc.collect()
        torch.cuda.empty_cache()

        Phi_test = phi(x_test.cpu()).to(device=device)
        
        logits_test = Phi_test @ W + b

        del Phi_test
        gc.collect()
        torch.cuda.empty_cache()

        train_acc = accuracy_from_logits(logits_train, y_train)
        val_acc = accuracy_from_logits(logits_val, y_val)
        test_acc = accuracy_from_logits(logits_test, y_test)

        train_loss = torch.mean((logits_train - y_train_onehot) ** 2).item()
        val_loss = torch.mean((logits_val - y_val_onehot) ** 2).item()
        test_loss = torch.mean((logits_test - y_test_onehot) ** 2).item()

    best = {
        "branch": branch,
        "lamda": float(best_lamda),
        "length_scale": float(best_length_scale),
        "n_splits": int(n_splits),

        "cv_train_acc_mean": float(best_cv["train_acc_mean"]),
        "cv_train_acc_std": float(best_cv["train_acc_std"]),
        "cv_val_acc_mean": float(best_cv["val_acc_mean"]),
        "cv_val_acc_std": float(best_cv["val_acc_std"]),

        "cv_train_loss_mean": float(best_cv["train_loss_mean"]),
        "cv_train_loss_std": float(best_cv["train_loss_std"]),
        "cv_val_loss_mean": float(best_cv["val_loss_mean"]),
        "cv_val_loss_std": float(best_cv["val_loss_std"]),

        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "test_loss": float(test_loss),

        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),

        "linear_state_dict": {
            "weight": W.T.float().detach().cpu(),  # [num_classes, p]
            "bias": b.float().detach().cpu(),      # [num_classes]
        },

        "phi_params": {
            "input_dim": d,
            "num_features": p,
            "kernel_type": kernel_type,
            "length_scale": float(best_length_scale),
            "seed": seed,
        },
    }

    print(
        f"\n[FINAL RIDGE REFIT] "
        f"lamda={best_lamda} "
        f"length_scale={best_length_scale} "
        f"train_acc={train_acc*100:.2f}% "
        f"val_acc={val_acc*100:.2f}% "
        f"test_acc={test_acc*100:.2f}%"
    )

    del logits_train, logits_val, logits_test
    torch.cuda.empty_cache()

    return best, all_cv_results