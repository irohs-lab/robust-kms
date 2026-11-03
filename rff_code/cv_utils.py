import itertools
import numpy as np
import torch

try:
    from sklearn.model_selection import StratifiedKFold, KFold
except ImportError:
    StratifiedKFold = None
    KFold = None

def labels_to_numpy(y):
    """
    Convert labels to a 1D numpy array for sklearn fold splitting.

    Supports:
        y as torch tensor [N]
        y as one-hot torch tensor [N, C]
        y as numpy array [N]
        y as one-hot numpy array [N, C]
    """

    if torch.is_tensor(y):
        if y.ndim == 2:
            y = y.argmax(dim=1)
        return y.detach().cpu().numpy()

    y = np.asarray(y)

    if y.ndim == 2:
        y = y.argmax(axis=1)

    return y


def expand_param_grid(param_grid):
    """
    Example:
        param_grid = {
            "length_scale": [0.1, 0.3, 1.0],
            "C": [1e4, 1e3],
        }

    Yields:
        {"length_scale": 0.1, "C": 1e4}
        {"length_scale": 0.1, "C": 1e3}
        ...
    """

    keys = list(param_grid.keys())
    values = [list(param_grid[k]) for k in keys]

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def make_folds(
    y,
    n_splits=5,
    stratified=True,
    shuffle=True,
    seed=42,
    device=None,
):
    """
    Returns list of torch index tensors:
        [(train_idx, val_idx), ...]

    Use stratified=True for classification.
    """

    y_np = labels_to_numpy(y)
    n = len(y_np)

    if device is None:
        if torch.is_tensor(y):
            device = y.device
        else:
            device = torch.device("cpu")

    if stratified:
        if StratifiedKFold is None:
            raise ImportError(
                "sklearn is required for StratifiedKFold. "
                "Install scikit-learn or use stratified=False."
            )

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed,
        )

        splits = splitter.split(np.zeros(n), y_np)

    else:
        if KFold is None:
            raise ImportError(
                "sklearn is required for KFold. "
                "Install scikit-learn or use stratified=True if available."
            )

        splitter = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed,
        )

        splits = splitter.split(np.zeros(n))

    folds = []

    for train_idx_np, val_idx_np in splits:
        train_idx = torch.as_tensor(
            train_idx_np,
            dtype=torch.long,
            device=device,
        )

        val_idx = torch.as_tensor(
            val_idx_np,
            dtype=torch.long,
            device=device,
        )

        folds.append((train_idx, val_idx))

    return folds


def json_safe_value(x):
    """
    Convert numpy / torch scalar values into JSON-safe Python values.
    """

    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().tolist()

    if isinstance(x, np.generic):
        return x.item()

    return x


def json_safe_dict(d):
    return {k: json_safe_value(v) for k, v in d.items()}


def aggregate_fold_results(fold_results):
    """
    Input:
        [
            {"train_acc": 0.98, "val_acc": 0.94, "val_loss": 0.2},
            {"train_acc": 0.97, "val_acc": 0.95, "val_loss": 0.18},
            ...
        ]

    Output:
        {
            "train_acc_mean": ...,
            "train_acc_std": ...,
            "val_acc_mean": ...,
            "val_acc_std": ...,
            ...
        }
    """

    numeric_keys = set()

    for result in fold_results:
        for k, v in result.items():
            if k == "fold":
                continue

            if isinstance(v, (int, float, np.integer, np.floating)):
                numeric_keys.add(k)

    summary = {}

    for k in sorted(numeric_keys):
        vals = [float(r[k]) for r in fold_results if k in r]

        if len(vals) == 0:
            continue

        summary[f"{k}_mean"] = float(np.mean(vals))

        if len(vals) > 1:
            summary[f"{k}_std"] = float(np.std(vals, ddof=1))
        else:
            summary[f"{k}_std"] = 0.0

    return summary


def is_better_result(
    candidate,
    best,
    primary_metric="val_acc",
    primary_mode="max",
    tie_breaker_metric="val_loss",
    tie_breaker_mode="min",
):
    """
    Compares two aggregated CV results.

    Example:
        primary_metric="val_acc", primary_mode="max"
        tie_breaker_metric="val_loss", tie_breaker_mode="min"

    Means:
        Prefer higher mean validation accuracy.
        If equal, prefer lower mean validation loss.
    """

    if best is None:
        return True

    primary_key = f"{primary_metric}_mean"

    c_primary = candidate[primary_key]
    b_primary = best[primary_key]

    if primary_mode == "max":
        if c_primary > b_primary:
            return True
        if c_primary < b_primary:
            return False

    elif primary_mode == "min":
        if c_primary < b_primary:
            return True
        if c_primary > b_primary:
            return False

    else:
        raise ValueError(f"Unsupported primary_mode: {primary_mode}")

    if tie_breaker_metric is None:
        return False

    tie_key = f"{tie_breaker_metric}_mean"

    if tie_key not in candidate or tie_key not in best:
        return False

    c_tie = candidate[tie_key]
    b_tie = best[tie_key]

    if tie_breaker_mode == "min":
        return c_tie < b_tie

    if tie_breaker_mode == "max":
        return c_tie > b_tie

    raise ValueError(f"Unsupported tie_breaker_mode: {tie_breaker_mode}")


def cross_validate_grid(
    x,
    y,
    param_grid,
    fit_eval_fn,
    n_splits=5,
    stratified=True,
    shuffle=True,
    seed=42,
    primary_metric="val_acc",
    primary_mode="max",
    tie_breaker_metric="val_loss",
    tie_breaker_mode="min",
    verbose=True,
):
    """
    General grid-search cross-validation loop.

    Parameters
    ----------
    x:
        Full development input tensor. Usually train + val merged.

    y:
        Full development label tensor. Usually train + val merged.

    param_grid:
        Dict of hyperparameter lists.

        Example:
            {
                "length_scale": [0.1, 0.3, 1.0],
                "C": [1e4, 1e3, 1e2],
            }

    fit_eval_fn:
        A function written inside each trainer.

        Signature must be:

            fit_eval_fn(
                x_train_fold,
                y_train_fold,
                x_val_fold,
                y_val_fold,
                params,
                fold_id,
            ) -> dict

        It must return metrics such as:
            {
                "train_acc": ...,
                "val_acc": ...,
                "val_loss": ...,
            }

    Returns
    -------
    best_result:
        Best aggregated CV result.

    all_results:
        List of all aggregated CV results.
    """

    device = x.device if torch.is_tensor(x) else torch.device("cpu")

    folds = make_folds(
        y=y,
        n_splits=n_splits,
        stratified=stratified,
        shuffle=shuffle,
        seed=seed,
        device=device,
    )

    best_result = None
    all_results = []

    grid = list(expand_param_grid(param_grid))

    for grid_id, params in enumerate(grid, start=1):
        if verbose:
            print("\n" + "=" * 80)
            print(f"Grid {grid_id}/{len(grid)}")
            print("Params:", params)
            print("=" * 80)

        fold_results = []

        for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
            x_train_fold = x[train_idx]
            y_train_fold = y[train_idx]

            x_val_fold = x[val_idx]
            y_val_fold = y[val_idx]

            result = fit_eval_fn(
                x_train_fold=x_train_fold,
                y_train_fold=y_train_fold,
                x_val_fold=x_val_fold,
                y_val_fold=y_val_fold,
                params=params,
                fold_id=fold_id,
            )

            result = json_safe_dict(result)
            result["fold"] = int(fold_id)

            fold_results.append(result)

            if verbose:
                msg = f"  fold={fold_id}"

                for key in ["train_acc", "val_acc", "val_loss", "train_loss"]:
                    if key in result:
                        val = result[key]

                        if "acc" in key:
                            msg += f" {key}={val*100:.2f}%"
                        else:
                            msg += f" {key}={val:.4f}"

                print(msg)

        summary = aggregate_fold_results(fold_results)

        cv_result = {
            "params": json_safe_dict(params),
            "folds": fold_results,
            **json_safe_dict(params),
            **summary,
        }

        all_results.append(cv_result)

        if verbose:
            print("[CV SUMMARY]")

            for k, v in summary.items():
                if "acc" in k:
                    print(f"  {k}: {v*100:.2f}%")
                else:
                    print(f"  {k}: {v:.6f}")

        if is_better_result(
            candidate=cv_result,
            best=best_result,
            primary_metric=primary_metric,
            primary_mode=primary_mode,
            tie_breaker_metric=tie_breaker_metric,
            tie_breaker_mode=tie_breaker_mode,
        ):
            best_result = cv_result

            if verbose:
                print("[NEW BEST]")
                print("  params:", best_result["params"])

    return best_result, all_results


def merge_train_val(x_train, y_train, x_val, y_val):
    """
    For proper CV:

        x_dev = train + val
        y_dev = train + val

    Test set should not be used in CV.
    """

    x_dev = torch.cat([x_train, x_val], dim=0)
    y_dev = torch.cat([y_train, y_val], dim=0)

    return x_dev, y_dev