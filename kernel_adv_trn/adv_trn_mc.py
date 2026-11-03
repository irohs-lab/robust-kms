"""Multiclass RKHS subgradient training with rk.fit_multiclass (epoch-based, projected)."""
import argparse
import sys
import time
from pathlib import Path

import torch
import dataset
import robust_kernels as rk

DATA_ROOT = "/janaki/common/Datasets/rahulky"

def eval_multiclass(x, centers, state, options):
    fn = getattr(rk, "multioutput_eval_and_grad", None)
    if fn is None:
        raise NotImplementedError(
            "Set the multiclass evaluator in eval_multiclass() (see dir(rk)).")
    return fn(x, centers, state["alpha"], state["factors"],
              gradients=False, **options)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--DATASET", type=str, default="kmnist")
    p.add_argument("--CLASSES", type=int, nargs="*", default=None,
                   help="Optional class subset; default = all classes")

    # penalty weight rho = RHO_MULT * EPS.  RHO_MULT=1: weight eps.
    # RHO_MULT=2: first-order CE bound for l_inf radius eps.
    p.add_argument("--EPS", type=lambda x: float(eval(x)), default=0.0)
    p.add_argument("--RHO_MULT", type=float, default=1.0)
    p.add_argument("--LAM", type=float, default=1.0,
                   help="Lambda sweep, e.g. --LAM 1e-4 1e-3 1e-2 1e-1 1")

    p.add_argument("--ETA", type=float, default=1e-2)
    p.add_argument("--DECAY", type=float, default=0.75)
    p.add_argument("--EPOCHS", type=int, default=10)
    p.add_argument("--BATCH_SIZE", type=int, default=128)
    p.add_argument("--PROJECT_EVERY", type=int, default=1)
    p.add_argument("--NO_FINAL_PROJECTION", action="store_true")
    p.add_argument("--PROJ_MAX_STEPS", type=int, default=100)
    p.add_argument("--PROJ_TOLERANCE", type=float, default=1e-3)
    p.add_argument("--SOLVE_RTOL", type=float, default=1e-3)
    p.add_argument("--SOLVE_ATOL", type=float, default=1e-8)
    p.add_argument("--SOLVE_MAX_EPOCHS", type=int, default=100)
    p.add_argument("--EIGENPRO_SAMPLES", type=int, default=1024)
    p.add_argument("--EIGENPRO_RANK", type=int, default=100)
    p.add_argument("--EIGENPRO_BATCH_SIZE", type=int, default=128)
    p.add_argument("--EIGENPRO_STORAGE", choices=("matfree", "dense"), default="matfree")

    p.add_argument("--KERNEL", type=str, default="gaussian")
    p.add_argument("--LENGTH_SCALE", type=float, default=1.0)
    p.add_argument("--QUERY_TILE", type=int, default=64)
    p.add_argument("--CENTER_TILE", type=int, default=1024)

    p.add_argument("--RESUME", type=str, default=None)
    p.add_argument("--SEED", type=int, default=0)
    p.add_argument("--DEVICE", type=str, default="cuda:0")
    p.add_argument("--OUTPUT", type=str, default="model_mc.pt")
    return p.parse_args()


def loader_to_tensor(loader, device):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    x = torch.cat(xs)
    y = torch.cat(ys)
    x = x.reshape(x.shape[0], -1)
    if y.ndim == 2:                                
        y = y.argmax(dim=1)
    return x.to(device, torch.float32), y.to(device).long().reshape(-1)


@torch.no_grad()
def evaluate_accuracy(loader, centers, state, options, class_ids):
    correct = total = 0
    for x, y in loader:
        x = x.reshape(x.shape[0], -1).to(centers.device, torch.float32)
        if y.ndim == 2:
            y = y.argmax(dim=1)
        y = torch.searchsorted(class_ids, y.to(centers.device).long().reshape(-1))
        values, _ = eval_multiclass(x, centers, state, options)
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite logits")
        correct += (values.argmax(dim=1) == y).sum().item()
        total += y.numel()
    if total == 0:
        raise ValueError("Empty loader")
    return correct / total, total


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("MULTICLASS RKHS TRAINING CONFIGURATION")
    print("=" * 60)

    for key, value in vars(args).items():
        print(f"{key:<25}: {value}")

    print("=" * 60 + "\n", flush=True)

    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    rho = args.RHO_MULT * args.EPS

    classes_arg = tuple(args.CLASSES) if args.CLASSES else None
    if args.DATASET.lower() == "svhn":
        train_loader, val_loader, test_loader = dataset.get_svhn(
            classes=classes_arg,
            multiclass=True,
        )
    else:
        train_loader, val_loader, test_loader = dataset.load_fmnist(
            classes=classes_arg,
            which_mnist=args.DATASET.lower(),
            multiclass=True,
        )
    x_train, y_raw = loader_to_tensor(train_loader, args.DEVICE)
    class_ids = torch.unique(y_raw).sort().values
    y_train = torch.searchsorted(class_ids, y_raw)  
    c = len(class_ids)

    if args.LENGTH_SCALE is None:
        args.LENGTH_SCALE = x_train.shape[1] ** 0.5

    print(f"x_train {tuple(x_train.shape)} | classes {class_ids.tolist()} | outputs {c}")
    print(f"eps {args.EPS} | rho {rho} | kernel {args.KERNEL} | "
          f"length scale {args.LENGTH_SCALE}")

    options = dict(kernel=args.KERNEL, length_scale=args.LENGTH_SCALE,
                   query_tile=args.QUERY_TILE, center_tile=args.CENTER_TILE)

    state = None
    if args.RESUME:
        state = torch.load(args.RESUME, map_location=args.DEVICE,
                           weights_only=False)["state"]
        print(f"Resuming from {args.RESUME}")

    projection_options = dict(
        max_steps=args.PROJ_MAX_STEPS, tolerance=args.PROJ_TOLERANCE,
        solve_rtol=args.SOLVE_RTOL, solve_atol=args.SOLVE_ATOL,
        solve_max_epochs=args.SOLVE_MAX_EPOCHS,
        eigenpro_samples=args.EIGENPRO_SAMPLES, eigenpro_rank=args.EIGENPRO_RANK,
        eigenpro_batch_size=args.EIGENPRO_BATCH_SIZE,
        eigenpro_storage=args.EIGENPRO_STORAGE,
    )
    print(f"projection settings: {projection_options}", flush=True)
    training_started = time.perf_counter()

    def projection_progress(event):
        details = " | ".join(f"{key}={value}" for key, value in event.items())
        print(f"projection | {details}", flush=True)

    def callback(state, event):
        if "projection" in event:
            report = event["projection"]
            print(
                f"projection done | epoch {event.get('epoch')} | "
                f"reason {report['reason']} | converged {report['converged']} | "
                f"steps {report['iterations']} | "
                f"error squared {report['error_squared']:.6g} | "
                f"solves {report['kernel_solve_calls']} | "
                f"solve epochs {report['kernel_solve_epochs']} | "
                f"seconds {report['elapsed_seconds']:.1f}", flush=True,
            )
        if event.get("epoch_end") and event.get("batch_in_epoch") is not None:
            print(
                f"epoch {event.get('epoch')} | "
                f"CE {event.get('batch_cross_entropy'):.6f} | "
                f"acc {event.get('batch_accuracy'):.4%} | "
                f"jac penalty {event.get('batch_jacobian_penalty'):.6f} | "
                f"training seconds {time.perf_counter() - training_started:.1f}",
                flush=True,
            )

    state = rk.fit_multiclass(
        x_train, y_train, outputs=c,
        rho=rho, lam=args.LAM, eta=args.ETA, decay=args.DECAY,
        epochs=args.EPOCHS, batch_size=args.BATCH_SIZE,
        project_every=args.PROJECT_EVERY,
        final_projection=not args.NO_FINAL_PROJECTION,
        projection_options=projection_options | dict(progress=projection_progress),
        state=state, seed=args.SEED, callback=callback,
        **options,
    )

    print("\n======== STATE DEBUG ========")
    print("alpha norm:", state["alpha"].norm().item())
    f = state.get("factors", {})
    if "u" in f:
        print("u norm:", f["u"].norm().item(), "| v norm:", f["v"].norm().item(),
              "| scale:", float(f["scale"]), "| pending:", len(f.get("pending", {})))
    print("last_projection:", state.get("last_projection"),
          "| epoch:", state.get("last_projection_epoch"))
    print("=============================\n")

    metrics = {}
    for split, loader in (("train", train_loader), ("val", val_loader),
                          ("test", test_loader)):
        acc, cnt = evaluate_accuracy(loader, x_train, state, options, class_ids)
        metrics[f"{split}_accuracy"], metrics[f"{split}_samples"] = acc, cnt
        print(f"  {split:5s} accuracy: {acc:.4%} ({cnt} samples)", flush=True)

    out = Path(args.OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "multiclass",
        "state": state,                    
        "centers": x_train.detach().cpu(),
        "class_ids": class_ids.cpu(),
        "outputs": c,
        "options": options,
        "config": {
            "dataset": args.DATASET,
            "classes": args.CLASSES,       
            "rho": rho, "lam": args.LAM,
            "epsilon": args.EPS,            
            "rho_mult": args.RHO_MULT,
            "eta": args.ETA, "decay": args.DECAY, "epochs": args.EPOCHS,
            "batch_size": args.BATCH_SIZE, "project_every": args.PROJECT_EVERY,
            "kernel": args.KERNEL, "length_scale": args.LENGTH_SCALE,
            "seed": args.SEED, "data_root": DATA_ROOT,
            "projection_options": projection_options,
        },
        "metrics": metrics,
    }, out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
