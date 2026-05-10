import os
import json
import argparse
import torch
import torch.nn.functional as F
import dataset
from torchkernels.feature_maps import LaplacianRFF, GaussianRFF


def ensure_onehot(y, num_classes):
    if y.ndim == 2:
        return y.float()
    return F.one_hot(y.long(), num_classes=num_classes).float()


def select_kernel(
    input_dim,
    num_features,
    kernel_type="Laplace",
    length_scale=1.0,
    seed=42,
    device=torch.device("cpu"),
):
    def normalize(x):
        return x / x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

    kernel_type_lower = kernel_type.lower()

    if kernel_type_lower == "laplace":
        phi_obj = LaplacianRFF(
            input_dim=input_dim,
            num_features=num_features,
            length_scale=length_scale,
            seed=seed,
            bias_term=False,
            device=device,
            dtype=torch.float32,
        )
    elif kernel_type_lower == "gaussian":
        phi_obj = GaussianRFF(
            input_dim=input_dim,
            num_features=num_features,
            length_scale=length_scale,
            seed=seed,
            bias_term=False,
            device=device,
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    def phi(x):
        return phi_obj(normalize(x))

    return phi


def loader_to_tensors(loader, device, flatten=True):
    xs, ys = [], []
    for x, y in loader:
        if flatten and x.ndim > 2:
            x = x.view(x.size(0), -1)
        xs.append(x)
        ys.append(y)

    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)
    return x, y


def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    true = y.argmax(dim=1) if y.ndim == 2 else y.long()
    return (pred == true).float().mean().item()


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
    lamdas=(1e-3, 1e-2, 1e-1, 1.0),
    length_scales=(0.1, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0 ,10.0),
    seed=42,
    device=torch.device("cpu"),
):
    y_train = ensure_onehot(y_train, num_classes)
    y_val = ensure_onehot(y_val, num_classes)
    y_test = ensure_onehot(y_test, num_classes)

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

        Phi_train = phi(x_train)
        Phi_val = phi(x_val)
        Phi_test = phi(x_test)

        G = Phi_train.T @ Phi_train
        B = Phi_train.T @ y_train
        I = torch.eye(p, device=device)

        for lamda in lamdas:
            U = torch.linalg.solve(G + lamda * I, B)

            logits_train = Phi_train @ U
            logits_val = Phi_val @ U
            logits_test = Phi_test @ U

            train_acc = accuracy_from_logits(logits_train, y_train)
            val_acc = accuracy_from_logits(logits_val, y_val)
            test_acc = accuracy_from_logits(logits_test, y_test)

            result = {
                "lamda": float(lamda),
                "length_scale": float(length_scale),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
            }
            all_results.append(result)

            print(
                f"length_scale={length_scale:<8} "
                f"lamda={lamda:<8} "
                f"train={train_acc*100:.2f}% "
                f"val={val_acc*100:.2f}% "
                f"test={test_acc*100:.2f}%"
            )

            if (
                best is None
                or val_acc > best["val_acc"]
                or (
                    val_acc == best["val_acc"]
                    and lamda < best["lamda"]
                )
            ):
                best = {
                    "lamda": float(lamda),
                    "length_scale": float(length_scale),
                    "train_acc": float(train_acc),
                    "val_acc": float(val_acc),
                    "test_acc": float(test_acc),
                    "U": U.detach().cpu(),
                    "phi_params": {
                        "input_dim": d,
                        "num_features": p,
                        "kernel_type": kernel_type,
                        "length_scale": float(length_scale),
                        "seed": seed,
                    },
                }

    return best, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--MC", action="store_true", help="Run multiclass instead of pairwise")
    parser.add_argument("--DATASET", type=str, default="kmnist")
    parser.add_argument("--kernel", type=str, default="Laplace")
    args = parser.parse_args()

    Dataset = args.DATASET.lower()
    kernel_type = args.kernel
    mc = args.MC

    if mc or Dataset in ["bank"]:
        print("Running multiclass...")
        classes1 = [0]
        classes2 = [1]
    else:
        print("Running per-class pairwise...")
        classes1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        classes2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lamdas=[1e-3, 1e-2, 1e-1, 1.0]
    length_scales=[0.1, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0 ,10.0]
    if mc == False:
        num_features = 1000
    else:
        num_features = 10000
    seed = 42

    out_dir = f"./rff_{kernel_type}_{Dataset}_{'mc' if mc else 'pairwise'}"
    os.makedirs(out_dir, exist_ok=True)

    for cl1 in classes1:
        for cl2 in range(cl1 + 1, len(classes2) + 1):
            classes = (cl1, cl2)

            if Dataset == "svhn":
                def load_dataset_svhn(classes):
                    return dataset.get_svhn(classes=classes)
                trainloader, valloader, testloader = load_dataset_svhn(classes=(cl1, cl2))
                print("I am SVHN")

            elif Dataset == "fmnist":
                def ld_fmnist():
                    return dataset.load_fmnist(
                        classes=(cl1, cl2),
                        which_mnist="fmnist",
                        multiclass=mc,
                    )
                trainloader, valloader, testloader = ld_fmnist()
                print("I am FMNIST")

            elif Dataset == "qmnist":
                def ld_fmnist():
                    return dataset.load_fmnist(
                        classes=(cl1, cl2),
                        which_mnist="qmnist",
                        multiclass=mc,
                    )
                trainloader, valloader, testloader = ld_fmnist()
                print("I am QMNIST")

            elif Dataset == "kmnist":
                def ld_fmnist():
                    return dataset.load_fmnist(
                        classes=(cl1, cl2),
                        which_mnist="kmnist",
                        multiclass=mc,
                    )
                trainloader, valloader, testloader = ld_fmnist()
                print("I am KMNIST")

            elif Dataset == "emnist":
                def ld_fmnist():
                    return dataset.load_fmnist(
                        classes=(cl1, cl2),
                        which_mnist="emnist",
                        multiclass=mc,
                    )
                trainloader, valloader, testloader = ld_fmnist()
                print("I am EMNIST")

            elif Dataset == "bank":
                def ld_bank():
                    return dataset.uci_bank()
                trainloader, testloader = ld_bank()
                valloader = testloader
                print("I am Bank")

            else:
                raise ValueError(f"Unsupported dataset: {Dataset}")

            for x, y in trainloader:
                print("Batch x shape:", x.shape)
                print("Batch y shape:", y.shape)
                break

            num_classes = 2 if not mc else 10
            print("Number of classes:", num_classes)

            x_train, y_train = loader_to_tensors(trainloader, device, flatten=True)
            x_val, y_val = loader_to_tensors(valloader, device, flatten=True)
            x_test, y_test = loader_to_tensors(testloader, device, flatten=True)

            best, all_results = fit_rff_ridge(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                num_classes=num_classes,
                num_features=num_features,
                kernel_type=kernel_type,
                lamdas=lamdas,
                length_scales=length_scales,
                seed=seed,
                device=device,
            )

            print("\nBest hyperparameters:")
            print(best)

            if mc:
                save_name = f"{Dataset}_mc_rff_results.json"
                save_ckpt = f"{Dataset}_mc_rff_best.pt"
            else:
                save_name = f"{Dataset}_{classes}_rff_results.json"
                save_ckpt = f"{Dataset}_{classes}_rff_best.pt"

            with open(os.path.join(out_dir, save_name), "w") as f:
                json.dump(
                    {
                        "best": {
                            "lamda": best["lamda"],
                            "length_scale": best["length_scale"],
                            "train_acc": best["train_acc"],
                            "val_acc": best["val_acc"],
                            "test_acc": best["test_acc"],
                        },
                        "all_results": all_results,
                    },
                    f,
                    indent=2,
                )

            torch.save(
                {
                    "U": best["U"],
                    "phi_params": best["phi_params"],
                    "train_acc": best["train_acc"],
                    "val_acc": best["val_acc"],
                    "test_acc": best["test_acc"],
                    "num_classes": num_classes,
                },
                os.path.join(out_dir, save_ckpt),
            )


# How To Load:-

# ckpt = torch.load(path)
# params = ckpt["phi_params"]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# phi = select_kernel(
#     input_dim=params["input_dim"],
#     num_features=params["num_features"],
#     kernel_type=params["kernel_type"],
#     length_scale=params["length_scale"],
#     seed=params["seed"],
#     device=device,
# )

# U = ckpt["U"].to(device)

# Example:
# logits = phi(x_test) @ U
