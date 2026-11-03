import torch
import torch.nn.functional as F

from kernels import select_kernel
from tensor_utils import ensure_labels, iterate_minibatches
from metrics import accuracy_from_logits
from attacks import pgd_attack_rfc

def evaluate_adv_rfc(
    x,
    y,
    phi,
    clf,
    adv_norm,
    adv_eps,
    adv_alpha,
    adv_steps,
    batch_size=256,
):
    clf.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0

    for xb, yb in iterate_minibatches(x, y, batch_size=batch_size, shuffle=False):
        x_adv = pgd_attack_rfc(
            x=xb,
            y=yb,
            phi=phi,
            clf=clf,
            eps=adv_eps,
            alpha=adv_alpha,
            steps=adv_steps,
            norm=adv_norm,
            clamp_min=0.0,
            clamp_max=1.0,
        )

        with torch.no_grad():
            logits_adv = clf(phi(x_adv))
            loss_adv = F.cross_entropy(logits_adv, yb)
            acc_adv = accuracy_from_logits(logits_adv, yb)

        bs = xb.size(0)
        total_loss += loss_adv.item() * bs
        total_acc += acc_adv * bs
        total_seen += bs

    return {
        "adv_loss": total_loss / total_seen,
        "adv_acc": total_acc / total_seen,
    }

def fit_rff_adv_cross_entropy(
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
    length_scales=(0.1, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0),
    epochs=50,
    adv_configs=(
        {"norm": "Linf", "eps": 6/255, "alpha": 6/255/10, "folder": "linf_6"},
        {"norm": "L2",   "eps": 1.0,   "alpha": 0.1,      "folder": "l2_1.0"},
    ),
    batch_size=256,
    adv_steps=20,
    seed=42,
    device=torch.device("cpu"),
):
    y_train = ensure_labels(y_train).to(device)
    y_val = ensure_labels(y_val).to(device)
    y_test = ensure_labels(y_test).to(device)

    d = x_train.shape[1]
    p = num_features

    for adv_cfg in adv_configs:
        adv_norm  = adv_cfg["norm"]
        adv_eps   = adv_cfg["eps"]
        adv_alpha = adv_cfg["alpha"]

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

            for wd in weight_decays:
                for lr in learning_rates:
                    torch.manual_seed(seed)

                    clf = torch.nn.Linear(p, num_classes).to(device)

                    optimizer = torch.optim.SGD(
                        clf.parameters(),
                        lr=lr,
                        weight_decay=wd,
                        momentum=0.9,
                    )

                    for epoch in range(epochs):
                        
                        total_loss = 0.0
                        total_seen = 0

                        for xb, yb in iterate_minibatches(
                            x_train,
                            y_train,
                            batch_size=batch_size,
                            shuffle=True,
                        ):
                            clf.eval()
                            x_adv = pgd_attack_rfc(
                                x=xb,
                                y=yb,
                                phi=phi,
                                clf=clf,
                                eps=adv_eps,
                                alpha=adv_alpha,
                                steps=adv_steps,
                                norm=adv_norm,
                                clamp_min=0.0,
                                clamp_max=1.0,
                            )

                            clf.train()

                            logits_adv = clf(phi(x_adv))
                            loss = F.cross_entropy(logits_adv, yb)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            total_loss += loss.item() * xb.size(0)
                            total_seen += xb.size(0)

                        if (epoch + 1) % 10 == 0:
                            print(
                                f"epoch={epoch+1}/{epochs} "
                                f"train_adv_loss={total_loss / total_seen:.4f}"
                            )

                    clf.eval()

                    with torch.no_grad():
                        logits_train = clf(phi(x_train))
                        logits_val = clf(phi(x_val))
                        logits_test = clf(phi(x_test))

                        train_loss = F.cross_entropy(logits_train, y_train).item()
                        val_loss = F.cross_entropy(logits_val, y_val).item()
                        test_loss = F.cross_entropy(logits_test, y_test).item()

                        train_acc = accuracy_from_logits(logits_train, y_train)
                        val_acc = accuracy_from_logits(logits_val, y_val)
                        test_acc = accuracy_from_logits(logits_test, y_test)

                    val_adv_metrics = evaluate_adv_rfc(
                                            x=x_val,
                                            y=y_val,
                                            phi=phi,
                                            clf=clf,
                                            adv_norm=adv_norm,
                                            adv_eps=adv_eps,
                                            adv_alpha=adv_alpha,
                                            adv_steps=adv_steps,
                                            batch_size=batch_size,
                                        )

                    val_adv_loss = val_adv_metrics["adv_loss"]
                    val_adv_acc = val_adv_metrics["adv_acc"]

                    result = {
                        "training": "adv_ce",
                        "adv_norm": adv_norm,
                        "weight_decay": float(wd),
                        "length_scale": float(length_scale),
                        "lr": float(lr),
                        "epochs": int(epochs),
                        "batch_size": int(batch_size),
                        "adv_eps": float(adv_eps),
                        "adv_alpha": float(adv_alpha),
                        "adv_steps": int(adv_steps),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "test_loss": float(test_loss),
                        "train_acc": float(train_acc),
                        "val_acc": float(val_acc),
                        "test_acc": float(test_acc),
                        "val_adv_loss": float(val_adv_loss),
                        "val_adv_acc": float(val_adv_acc),
                    }

                    all_results.append(result)

                    print(
                        f"[ADV-CE] length_scale={length_scale:<8} "
                        f"wd={wd:<8} "
                        f"lr={lr:<8} "
                        f"train_acc={train_acc*100:.2f}% "
                        f"val_acc={val_acc*100:.2f}% "
                        f"test_acc={test_acc*100:.2f}% "
                        f"val_adv_acc={val_adv_acc*100:.2f}% "
                        f"val_adv_loss={val_adv_loss:.4f}"
                    )

                    if (
                        best is None
                        or val_adv_acc > best["val_adv_acc"]
                        or (
                            val_adv_acc == best["val_adv_acc"]
                            and val_adv_loss < best["val_adv_loss"]
                        )
                    ):
                        best = {
                            "training": "adv_ce",
                            "adv_norm": adv_norm,
                            "weight_decay": float(wd),
                            "length_scale": float(length_scale),
                            "lr": float(lr),
                            "epochs": int(epochs),
                            "batch_size": int(batch_size),
                            "adv_eps": float(adv_eps),
                            "adv_alpha": float(adv_alpha),
                            "adv_steps": int(adv_steps),
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
                            "val_adv_loss": float(val_adv_loss),
                            "val_adv_acc": float(val_adv_acc),
                        }

        yield adv_cfg["folder"], best, all_results