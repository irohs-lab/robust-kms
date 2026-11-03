import torch
import torch.nn.functional as F

from kernels import select_kernel
from tensor_utils import ensure_labels
from metrics import accuracy_from_logits
from cv_utils import cross_validate_grid, merge_train_val

def fit_rff_cross_entropy(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    num_classes,
    num_features=1000,
    kernel_type="Laplace",
    weight_decays=(1e-4, 1e-3, 1e-2),
    learning_rates=(0.01, 0.03, 0.1),
    length_scales=(0.1, 0.3, 1.0, 3.0, 10.0),
    epochs=300,
    seed=42,
    device=torch.device("cpu"),
):
    y_train = ensure_labels(y_train).to(device)
    y_val = ensure_labels(y_val).to(device)
    y_test = ensure_labels(y_test).to(device)

    d = x_train.shape[1]
    p = num_features

    best = None
    all_results = []

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
            Phi_train = phi(x_train)
            Phi_val = phi(x_val)
            Phi_test = phi(x_test)

        for wd in weight_decays:
            for lr in learning_rates:
                torch.manual_seed(seed)

                clf = torch.nn.Linear(p, num_classes).to(device)

                optimizer = torch.optim.AdamW(
                    clf.parameters(),
                    lr=lr,
                    weight_decay=wd,
                )

                for epoch in range(epochs):
                    clf.train()

                    logits_train = clf(Phi_train)
                    loss = F.cross_entropy(logits_train, y_train)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                clf.eval()

                with torch.no_grad():
                    logits_train = clf(Phi_train)
                    logits_val = clf(Phi_val)
                    logits_test = clf(Phi_test)

                    train_loss = F.cross_entropy(logits_train, y_train).item()
                    val_loss = F.cross_entropy(logits_val, y_val).item()
                    test_loss = F.cross_entropy(logits_test, y_test).item()

                    train_acc = accuracy_from_logits(logits_train, y_train)
                    val_acc = accuracy_from_logits(logits_val, y_val)
                    test_acc = accuracy_from_logits(logits_test, y_test)

                result = {
                    "weight_decay": float(wd),
                    "length_scale": float(length_scale),
                    "lr": float(lr),
                    "epochs": int(epochs),
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
                    f"wd={wd:<8} "
                    f"lr={lr:<8} "
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
                    best = {
                        "weight_decay": float(wd),
                        "length_scale": float(length_scale),
                        "lr": float(lr),
                        "epochs": int(epochs),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "test_loss": float(test_loss),
                        "train_acc": float(train_acc),
                        "val_acc": float(val_acc),
                        "test_acc": float(test_acc),
                        "linear_state_dict": {
                            k: v.detach().cpu()
                            for k, v in clf.state_dict().items()
                        },
                        "phi_params": {
                            "input_dim": d,
                            "num_features": p,
                            "kernel_type": kernel_type,
                            "length_scale": float(length_scale),
                            "seed": seed,
                        },
                    }

        del Phi_train, Phi_val, Phi_test
        torch.cuda.empty_cache()

    return best, all_results

def fit_rff_cross_entropy_cv(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    num_classes,
    num_features=1000,
    kernel_type="Laplace",
    weight_decays=(1e-4, 1e-3, 1e-2),
    learning_rates=(0.01, 0.03, 0.1),
    length_scales=(0.1, 0.3, 1.0, 3.0, 10.0),
    epochs=300,
    n_splits=5,
    seed=42,
    device=torch.device("cpu"),
):

    print("Using optimizer algorithm for cross-entropy rff.")

    y_train = ensure_labels(y_train).to(device)
    y_val = ensure_labels(y_val).to(device)
    y_test = ensure_labels(y_test).to(device)

    x_dev, y_dev = merge_train_val(x_train, y_train, x_val, y_val)

    d = x_dev.shape[1]
    p = num_features

    param_grid = {
        "length_scale": length_scales,
        "weight_decay": weight_decays,
        "lr": learning_rates,
    }

    def train_classifier(Phi, labels, weight_decay, lr):
        torch.manual_seed(seed)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        clf = torch.nn.Linear(p, num_classes).to(device)

        optimizer = torch.optim.AdamW(
            clf.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        for _ in range(epochs):
            clf.train()

            logits = clf(Phi)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return clf

    def fit_eval_fn(
        x_train_fold,
        y_train_fold,
        x_val_fold,
        y_val_fold,
        params,
        fold_id,
    ):
        del fold_id

        phi = select_kernel(
            input_dim=d,
            num_features=p,
            kernel_type=kernel_type,
            length_scale=params["length_scale"],
            seed=seed,
            device=device,
        )

        with torch.no_grad():
            Phi_train = phi(x_train_fold)
            Phi_val = phi(x_val_fold)

        clf = train_classifier(
            Phi_train,
            y_train_fold,
            weight_decay=params["weight_decay"],
            lr=params["lr"],
        )

        clf.eval()

        with torch.no_grad():
            logits_train = clf(Phi_train)
            logits_val = clf(Phi_val)

            train_loss = F.cross_entropy(
                logits_train, y_train_fold
            ).item()

            val_loss = F.cross_entropy(
                logits_val, y_val_fold
            ).item()

            train_acc = accuracy_from_logits(
                logits_train, y_train_fold
            )

            val_acc = accuracy_from_logits(
                logits_val, y_val_fold
            )

        result = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        }

        del Phi_train, Phi_val, clf, phi

        if device.type == "cuda":
            torch.cuda.empty_cache()

        return result

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

    best_params = best_cv["params"]

    best_length_scale = best_params["length_scale"]
    best_weight_decay = best_params["weight_decay"]
    best_lr = best_params["lr"]

    phi = select_kernel(
        input_dim=d,
        num_features=p,
        kernel_type=kernel_type,
        length_scale=best_length_scale,
        seed=seed,
        device=device,
    )

    with torch.no_grad():
        Phi_dev = phi(x_dev)
        Phi_train = phi(x_train)
        Phi_val = phi(x_val)
        Phi_test = phi(x_test)

    final_clf = train_classifier(
        Phi_dev,
        y_dev,
        weight_decay=best_weight_decay,
        lr=best_lr,
    )

    final_clf.eval()

    with torch.no_grad():
        logits_train = final_clf(Phi_train)
        logits_val = final_clf(Phi_val)
        logits_test = final_clf(Phi_test)

        train_loss = F.cross_entropy(
            logits_train, y_train
        ).item()

        val_loss = F.cross_entropy(
            logits_val, y_val
        ).item()

        test_loss = F.cross_entropy(
            logits_test, y_test
        ).item()

        train_acc = accuracy_from_logits(
            logits_train, y_train
        )

        val_acc = accuracy_from_logits(
            logits_val, y_val
        )

        test_acc = accuracy_from_logits(
            logits_test, y_test
        )

    best = {
        "weight_decay": float(best_weight_decay),
        "length_scale": float(best_length_scale),
        "lr": float(best_lr),
        "epochs": int(epochs),
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
            key: value.detach().cpu()
            for key, value in final_clf.state_dict().items()
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
        f"length_scale={best_length_scale} "
        f"wd={best_weight_decay} "
        f"lr={best_lr} "
        f"train_acc={train_acc * 100:.2f}% "
        f"val_acc={val_acc * 100:.2f}% "
        f"test_acc={test_acc * 100:.2f}%"
    )

    return best, all_cv_results