import torch
import cupy as cp
from cuml.linear_model import LogisticRegression

from kernels import select_kernel
from tensor_utils import ensure_labels
from metrics import accuracy_from_logits, ce_loss_from_proba, proba_to_valid_shape
from cv_utils import cross_validate_grid, merge_train_val

import rmm
from rmm.allocators.torch import rmm_torch_allocator
from rmm.allocators.cupy import rmm_cupy_allocator

rmm.reinitialize(pool_allocator=True, managed_memory=False)
cp.cuda.set_allocator(rmm_cupy_allocator)
torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
import gc

def fit_rff_cross_entropy_cuml(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    num_classes,
    num_features=1000,
    kernel_type="Laplace",
    Cs=(1e4, 1e3, 1e2, 1e1, 1e0, 1e-1),
    length_scales=(0.1, 0.3, 1.0, 3.0, 10.0),
    max_iter=1000,
    seed=42,
    device=torch.device("cuda"),
):
    print("Using CuML LogisticRegression on GPU for RFF training")

    y_train = ensure_labels(y_train).to(device)
    y_val = ensure_labels(y_val).to(device)
    y_test = ensure_labels(y_test).to(device)

    d = x_train.shape[1]
    print(x_train.shape[0])
    p = num_features

    best = None
    all_results = []

    print("Length scales:", length_scales)
    print("Cs:", Cs)
    for length_scale in length_scales:
        phi = select_kernel(
            input_dim=d,
            num_features=p,
            kernel_type=kernel_type,
            length_scale=length_scale,
            seed=seed,
            device=device,
        )
        print(f"length_scale: {length_scale}")

        with torch.no_grad():
            Phi_train = phi(x_train).contiguous()
            Phi_val = phi(x_val).contiguous()
            Phi_test = phi(x_test).contiguous()

        Phi_train_cp = cp.from_dlpack(Phi_train)
        Phi_val_cp = cp.from_dlpack(Phi_val)
        Phi_test_cp = cp.from_dlpack(Phi_test)

        y_train_cp = cp.from_dlpack(y_train.contiguous())

        for C in Cs:
            clf = LogisticRegression(
                C=float(C),
                max_iter=max_iter,
                solver="qn",
                verbose=0,
                fit_intercept=True,
                penalty_normalized=True,
                output_type="cupy",
            )

            clf.fit(Phi_train_cp, y_train_cp)

            print("C:", C)

            proba_train = torch.from_dlpack(clf.predict_proba(Phi_train_cp))
            proba_val = torch.from_dlpack(clf.predict_proba(Phi_val_cp))
            proba_test = torch.from_dlpack(clf.predict_proba(Phi_test_cp))

            proba_train = proba_to_valid_shape(proba_train, num_classes)
            proba_val = proba_to_valid_shape(proba_val, num_classes)
            proba_test = proba_to_valid_shape(proba_test, num_classes)

            train_acc = accuracy_from_logits(proba_train, y_train)
            val_acc = accuracy_from_logits(proba_val, y_val)
            test_acc = accuracy_from_logits(proba_test, y_test)

            train_loss = ce_loss_from_proba(proba_train, y_train)
            val_loss = ce_loss_from_proba(proba_val, y_val)
            test_loss = ce_loss_from_proba(proba_test, y_test)

            result = {
                "C": float(C),
                "length_scale": float(length_scale),
                "max_iter": int(max_iter),
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
                f"C={C:<8} "
                f"train_acc={train_acc*100:.2f}% "
                f"val_acc={val_acc*100:.2f}% "
                f"test_acc={test_acc*100:.2f}% "
                f"val_loss={val_loss:.4f}"
            )

            is_best = (
                best is None
                or val_acc > best["val_acc"]
                or (val_acc == best["val_acc"] and val_loss < best["val_loss"])
            )

            if is_best:
                W = torch.from_dlpack(cp.asarray(clf.coef_)).float()
                b = torch.from_dlpack(cp.asarray(clf.intercept_)).float()

                if num_classes == 2 and W.shape[0] == 1:
                    W0 = torch.zeros_like(W)
                    b0 = torch.zeros_like(b)

                    W = torch.cat([W0, W], dim=0)
                    b = torch.cat([b0, b], dim=0)

                best = {
                    "C": float(C),
                    "length_scale": float(length_scale),
                    "max_iter": int(max_iter),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "test_loss": float(test_loss),
                    "train_acc": float(train_acc),
                    "val_acc": float(val_acc),
                    "test_acc": float(test_acc),
                    "linear_state_dict": {
                        "weight": W.detach().cpu(),
                        "bias": b.detach().cpu(),
                    },
                    "phi_params": {
                        "input_dim": d,
                        "num_features": p,
                        "kernel_type": kernel_type,
                        "length_scale": float(length_scale),
                        "seed": seed,
                    },
                }

        del Phi_train, Phi_val, Phi_train_cp, Phi_val_cp, y_train_cp, proba_train, proba_val, clf, phi
        gc.collect()
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

    return best, all_results

def fit_rff_cross_entropy_cuml_cv(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    num_classes,
    num_features=1000,
    kernel_type="Laplace",
    Cs=(1e4, 1e3, 1e2, 1e1, 1e0, 1e-1),
    length_scales=(0.1, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0),
    max_iter=5000,
    n_splits=5,
    seed=42,
    device=torch.device("cuda"),
):
    print("Using CuML LogisticRegression with general CV module")

    y_train = ensure_labels(y_train).to(device)
    y_val = ensure_labels(y_val).to(device)
    y_test = ensure_labels(y_test).to(device)

    x_dev, y_dev = merge_train_val(x_train, y_train, x_val, y_val)

    d = x_dev.shape[1]
    p = num_features

    param_grid = {
        "length_scale": length_scales,
        "C": Cs,
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
        C = params["C"]

        phi = select_kernel(
            input_dim=d,
            num_features=p,
            kernel_type=kernel_type,
            length_scale=length_scale,
            seed=seed,
            device=device,
        )

        with torch.no_grad():
            Phi_train = phi(x_train_fold).contiguous()
            Phi_val = phi(x_val_fold).contiguous()

        Phi_train_cp = cp.from_dlpack(Phi_train)
        Phi_val_cp = cp.from_dlpack(Phi_val)

        y_train_cp = cp.from_dlpack(y_train_fold.contiguous())

        clf = LogisticRegression(
            C=float(C),
            max_iter=max_iter,
            solver="qn",
            verbose=0,
            fit_intercept=True,
            # fit_intercept=False,
            # penalty_normalized=False,
            output_type="cupy",
        )

        clf.fit(Phi_train_cp, y_train_cp)

        proba_train = torch.from_dlpack(clf.predict_proba(Phi_train_cp))
        proba_val = torch.from_dlpack(clf.predict_proba(Phi_val_cp))

        proba_train = proba_to_valid_shape(proba_train, num_classes)
        proba_val = proba_to_valid_shape(proba_val, num_classes)

        train_acc = accuracy_from_logits(proba_train, y_train_fold)
        val_acc = accuracy_from_logits(proba_val, y_val_fold)

        train_loss = ce_loss_from_proba(proba_train, y_train_fold)
        val_loss = ce_loss_from_proba(proba_val, y_val_fold)

        del Phi_train, Phi_val, Phi_train_cp, Phi_val_cp, y_train_cp, proba_train, proba_val, clf, phi
        gc.collect()
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

        return {
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        }

    best_cv, all_cv_results = cross_validate_grid(
        x=x_dev,
        y=y_dev,
        param_grid=param_grid,
        fit_eval_fn=fit_eval_fn,
        n_splits=n_splits,
        stratified=True,
        shuffle=True,
        seed=42,
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
    best_C = best_params["C"]

    phi = select_kernel(
        input_dim=d,
        num_features=p,
        kernel_type=kernel_type,
        length_scale=best_length_scale,
        seed=seed,
        device=device,
    )

    with torch.no_grad():
        Phi_dev = phi(x_dev).contiguous()
        Phi_train = phi(x_train).contiguous()
        Phi_val = phi(x_val).contiguous()
        Phi_test = phi(x_test).contiguous()

    Phi_dev_cp = cp.from_dlpack(Phi_dev)
    Phi_train_cp = cp.from_dlpack(Phi_train)
    Phi_val_cp = cp.from_dlpack(Phi_val)
    Phi_test_cp = cp.from_dlpack(Phi_test)

    y_dev_cp = cp.from_dlpack(y_dev.contiguous())

    final_clf = LogisticRegression(
        C=float(best_C),
        max_iter=max_iter,
        solver="qn",
        verbose=0,
        fit_intercept=True,
        penalty_normalized=False,
        output_type="cupy",
        lbfgs_memory_level=10,  # Increase memory level for L-BFGS
    )

    final_clf.fit(Phi_dev_cp, y_dev_cp)

    proba_train = torch.from_dlpack(final_clf.predict_proba(Phi_train_cp))
    proba_val = torch.from_dlpack(final_clf.predict_proba(Phi_val_cp))
    proba_test = torch.from_dlpack(final_clf.predict_proba(Phi_test_cp))

    proba_train = proba_to_valid_shape(proba_train, num_classes)
    proba_val = proba_to_valid_shape(proba_val, num_classes)
    proba_test = proba_to_valid_shape(proba_test, num_classes)

    train_acc = accuracy_from_logits(proba_train, y_train)
    val_acc = accuracy_from_logits(proba_val, y_val)
    test_acc = accuracy_from_logits(proba_test, y_test)

    train_loss = ce_loss_from_proba(proba_train, y_train)
    val_loss = ce_loss_from_proba(proba_val, y_val)
    test_loss = ce_loss_from_proba(proba_test, y_test)

    W = torch.from_dlpack(cp.asarray(final_clf.coef_)).float()
    b = torch.from_dlpack(cp.asarray(final_clf.intercept_)).float()

    if num_classes == 2 and W.shape[0] == 1:
        W0 = torch.zeros_like(W)
        b0 = torch.zeros_like(b)

        W = torch.cat([W0, W], dim=0)
        b = torch.cat([b0, b], dim=0)

    best = {
        "C": float(best_C),
        "length_scale": float(best_length_scale),
        "max_iter": int(max_iter),
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
            "weight": W.detach().cpu(),
            "bias": b.detach().cpu(),
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
        f"\n[FINAL REFIT] "
        f"C={best_C} "
        f"length_scale={best_length_scale} "
        f"train_acc={train_acc*100:.2f}% "
        f"val_acc={val_acc*100:.2f}% "
        f"test_acc={test_acc*100:.2f}%"
    )

    return best, all_cv_results

