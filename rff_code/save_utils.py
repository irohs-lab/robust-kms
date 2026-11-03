import os
import json
import torch
import tempfile


def safe_torch_save(obj, path):
    dir_name = os.path.dirname(path) or "."

    with tempfile.NamedTemporaryFile(delete=False, dir=dir_name, suffix=".tmp") as f:
        tmp_path = f.name

    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
        print(f"Saved checkpoint: {path}")

    except Exception as e:
        os.remove(tmp_path)
        raise e


def get_output_dir(kernel_type, dataset_name, multiclass, training_type, seed):
    mode = "mc" if multiclass else "pairwise"

    base = f"./rff_{kernel_type}_{dataset_name}_{mode}"

    if training_type == "square":
        return base

    elif training_type == "square_cv":
        return os.path.join(base, "square_cv", f"seed_{seed}")

    elif training_type == "ce":
        return os.path.join(base, "ce")

    elif training_type == "ce_torch_cv":
        return os.path.join(base, "ce_torch_cv", f"seed_{seed}")

    elif training_type == "ce_cuml":
        return os.path.join(base, "ce_cuml",f"seed_{seed}")

    elif training_type == "ce_cuml_cv":
        return os.path.join(base, "ce_cuml_cv", f"seed_{seed}")

    elif training_type == "adv_ce":
        return os.path.join(base, "adv_ce")

    raise ValueError(f"Unsupported training_type: {training_type}")


def get_save_names(dataset_name, classes, multiclass, seed, num_features):
    if multiclass:
        save_name = f"{dataset_name}_mc_rff_results_sd{seed}_wb_{num_features}.json"
        save_ckpt = f"{dataset_name}_mc_rff_best_sd{seed}_wb_{num_features}.pt"
    else:
        save_name = f"{dataset_name}_{classes}_rff_results.json"
        save_ckpt = f"{dataset_name}_{classes}_rff_best.pt"

    return save_name, save_ckpt


def save_square_result(out_dir, save_name, save_ckpt, best, all_results, num_classes):
    import os
    import json

    os.makedirs(out_dir, exist_ok=True)

    best_json = {
        "lamda": best["lamda"],
        "length_scale": best["length_scale"],
        "branch": best.get("branch", None),
        "train_acc": best["train_acc"],
        "val_acc": best["val_acc"],
        "test_acc": best["test_acc"],
    }

    cv_keys = [
        "n_splits",
        "cv_train_acc_mean",
        "cv_train_acc_std",
        "cv_val_acc_mean",
        "cv_val_acc_std",
        "cv_train_loss_mean",
        "cv_train_loss_std",
        "cv_val_loss_mean",
        "cv_val_loss_std",
    ]

    for k in cv_keys:
        if k in best:
            best_json[k] = best[k]

    with open(os.path.join(out_dir, save_name), "w") as f:
        json.dump(
            {
                "best": best_json,
                "all_results": all_results,
            },
            f,
            indent=2,
        )

    safe_torch_save(
        {
            "linear_state_dict": best["linear_state_dict"],
            "phi_params": best["phi_params"],
            "lamda": best["lamda"],
            "length_scale": best["length_scale"],
            "branch": best.get("branch", None),
            "train_acc": best["train_acc"],
            "val_acc": best["val_acc"],
            "test_acc": best["test_acc"],
            "num_classes": num_classes,
        },
        os.path.join(out_dir, save_ckpt),
    )

def save_ce_result(
    out_dir,
    save_name,
    save_ckpt,
    best,
    all_results,
    num_classes,
):
    os.makedirs(out_dir, exist_ok=True)

    best_json = {
        "weight_decay": best["weight_decay"],
        "length_scale": best["length_scale"],
        "lr": best["lr"],
        "epochs": best["epochs"],
        "train_loss": best["train_loss"],
        "val_loss": best["val_loss"],
        "test_loss": best["test_loss"],
        "train_acc": best["train_acc"],
        "val_acc": best["val_acc"],
        "test_acc": best["test_acc"],
    }

    cv_keys = [
        "n_splits",
        "cv_train_acc_mean",
        "cv_train_acc_std",
        "cv_val_acc_mean",
        "cv_val_acc_std",
        "cv_train_loss_mean",
        "cv_train_loss_std",
        "cv_val_loss_mean",
        "cv_val_loss_std",
    ]

    for key in cv_keys:
        if key in best:
            best_json[key] = best[key]

    with open(os.path.join(out_dir, save_name), "w") as f:
        json.dump(
            {
                "best": best_json,
                "all_results": all_results,
            },
            f,
            indent=2,
        )

    checkpoint = {
        "linear_state_dict": best["linear_state_dict"],
        "phi_params": best["phi_params"],
        "weight_decay": best["weight_decay"],
        "length_scale": best["length_scale"],
        "lr": best["lr"],
        "epochs": best["epochs"],
        "train_loss": best["train_loss"],
        "val_loss": best["val_loss"],
        "test_loss": best["test_loss"],
        "train_acc": best["train_acc"],
        "val_acc": best["val_acc"],
        "test_acc": best["test_acc"],
        "num_classes": num_classes,
    }

    for key in cv_keys:
        if key in best:
            checkpoint[key] = best[key]

    safe_torch_save(
        checkpoint,
        os.path.join(out_dir, save_ckpt),
    )

def save_cuml_ce_result(out_dir, save_name, save_ckpt, best, all_results, num_classes):
    os.makedirs(out_dir, exist_ok=True)

    best_json = {
        "C": best["C"],
        "length_scale": best["length_scale"],
        "max_iter": best["max_iter"],
        "train_loss": best["train_loss"],
        "val_loss": best["val_loss"],
        "test_loss": best["test_loss"],
        "train_acc": best["train_acc"],
        "val_acc": best["val_acc"],
        "test_acc": best["test_acc"],
    }

    cv_keys = [
        "n_splits",
        "cv_train_acc_mean",
        "cv_train_acc_std",
        "cv_val_acc_mean",
        "cv_val_acc_std",
        "cv_train_loss_mean",
        "cv_train_loss_std",
        "cv_val_loss_mean",
        "cv_val_loss_std",
    ]

    for k in cv_keys:
        if k in best:
            best_json[k] = best[k]

    with open(os.path.join(out_dir, save_name), "w") as f:
        json.dump(
            {
                "best": best_json,
                "all_results": all_results,
            },
            f,
            indent=2,
        )

    ckpt = {
        "linear_state_dict": best["linear_state_dict"],
        "phi_params": best["phi_params"],
        "C": best["C"],
        "length_scale": best["length_scale"],
        "max_iter": best["max_iter"],
        "train_loss": best["train_loss"],
        "val_loss": best["val_loss"],
        "test_loss": best["test_loss"],
        "train_acc": best["train_acc"],
        "val_acc": best["val_acc"],
        "test_acc": best["test_acc"],
        "num_classes": num_classes,
    }

    for k in cv_keys:
        if k in best:
            ckpt[k] = best[k]

    safe_torch_save(
        ckpt,
        os.path.join(out_dir, save_ckpt),
    )