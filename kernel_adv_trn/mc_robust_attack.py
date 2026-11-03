"""AutoAttack (APGD-CE) sweep for a MULTICLASS checkpoint from train_multiclass.py.

Mirrors the reference attack script:
  * AutoAttack(version="standard"), then attacks_to_run overridden (default: apgd-ce only)
  * Linf (eps/255) or L2 (raw eps) threat model, same default eps lists, eps=0 = clean
  * whole test set attacked in one batch by default
  * accuracy reported in percent, JSON records {eps_float, norm, attack, pgd_acc}

Usage:
  python mc_robust_attack.py --checkpoint runs/mc_robust/model.pt --norm Linf --device cuda:0
  python mc_robust_attack.py --checkpoint runs/mc_robust/model.pt --norm L2 --budgets 0.5 1 2
Results go to <checkpoint folder>/attack_results unless --output is given.
"""
import argparse
import json
import pathlib
import sys
import traceback

import torch
import autoattack
import dataset
import robust_kernels as rk
import experiments.report as report

LINF_DEFAULT = [0, 0.1, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0, 40.0, 55.0]  # x/255
L2_DEFAULT = [0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]                                   # raw


def eval_multiclass(x, centers, state, options, gradients=False):
    out = rk.multioutput_eval_and_grad(
        x, centers, state["alpha"], state["factors"],
        gradients=gradients, **options)
    if isinstance(out, (tuple, list)):
        values = out[0]
        jac = out[1] if len(out) > 1 else None
    else:
        values, jac = out, None
    return values, jac


class _Logits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, centers, state, options):
        need_grad = ctx.needs_input_grad[0]
        with torch.no_grad():
            values, jac = eval_multiclass(x, centers, state, options,
                                          gradients=need_grad)
        if need_grad:
            if jac is None:
                raise RuntimeError("Evaluator returned no Jacobian with gradients=True")
            ctx.save_for_backward(jac)
        return values

    @staticmethod
    def backward(ctx, grad_out):
        (jac,) = ctx.saved_tensors                              # (b, d, c)
        return torch.einsum("bc,bdc->bd", grad_out, jac), None, None, None


def make_predict(model, device):
    centers = model["centers"].to(device)
    state = model["state"]
    options = model["options"]

    def predict(images):
        x = images.reshape(images.shape[0], -1)                
        return _Logits.apply(x, centers, state, options)
    return predict


def finite_difference_check(predict, images, labels, n=4, h=1e-3):
    """Compare autograd VJP of the CE loss to a central finite difference."""
    x = images[:n].clone().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(predict(x), labels[:n], reduction="sum")
    loss.backward()
    g = x.grad.reshape(n, -1)
    gen = torch.Generator(device=x.device).manual_seed(0)
    v = torch.randn(g.shape, device=x.device, generator=gen)
    v = v / v.norm(dim=1, keepdim=True)

    def f(sh):
        with torch.no_grad():
            xs = (images[:n].reshape(n, -1) + sh * v).reshape_as(images[:n])
            return torch.nn.functional.cross_entropy(
                predict(xs), labels[:n], reduction="none")
    fd = (f(h) - f(-h)) / (2 * h)
    an = (g * v).sum(1)
    print("VJP check (analytic vs finite diff):")
    for a, b in zip(an.tolist(), fd.tolist()):
        print(f"  {a:+.6e}  {b:+.6e}")
    return (an - fd).abs().max().item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default=None,
                   help="Default: <checkpoint folder>/attack_results (must not exist yet)")
    p.add_argument("--norm", default="Linf", choices=["Linf", "L2"])
    p.add_argument("--budgets", nargs="+", type=float, default=None,
                   help="Linf: eps in units of 1/255. L2: raw eps. "
                        "Default: same lists as the reference script")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attacks", nargs="+", default=["apgd-ce"])
    p.add_argument("--restarts", type=int, default=1,
                   help="APGD restarts (AutoAttack 'standard' uses 1)")
    p.add_argument("--iters", type=int, default=100, help="APGD iterations")
    p.add_argument("--seed", type=int, default=None, help="Default: seed in checkpoint config")
    p.add_argument("--skip_check", action="store_true")
    return p.parse_args()


def load_test(config, class_ids, device):
    classes = tuple(config["classes"]) if config.get("classes") else None
    _, _, test_loader = dataset.load_fmnist(
        classes=classes, which_mnist=config["dataset"], multiclass=True)
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x.reshape(x.shape[0], -1))
        ys.append(y.argmax(1) if y.ndim == 2 else y)
    x = torch.cat(xs).to(device, torch.float32)
    y = torch.cat(ys).to(device).long().reshape(-1)
    y = torch.searchsorted(class_ids.to(device), y)    
    return x, y


@torch.no_grad()
def accuracy(predict, images, labels, bs):
    correct = 0
    for s in range(0, len(images), bs):
        correct += (predict(images[s:s + bs]).argmax(1) == labels[s:s + bs]).sum().item()
    return 100.0 * correct / len(images)                 


def attack_one_eps(predict, images, labels, eps, args, seed, bs, log_dir, device):
    """Returns adversarial images for one epsilon (eps == 0 -> clean images)."""
    if eps == 0:
        return images.clone()
    adversary = autoattack.AutoAttack(
        predict, norm=args.norm, eps=eps, version="standard", seed=seed,
        device=device, log_path=str(log_dir / "autoattack.log"))
    adversary.attacks_to_run = list(args.attacks)               # as in the reference
    adversary.apgd.n_restarts = args.restarts
    adversary.apgd.n_iter = args.iters
    return adversary.run_standard_evaluation(images, labels, bs=bs)


def check_adversarial(adversarial, images, eps, norm):
    diff = (adversarial - images).reshape(len(images), -1)
    if norm == "Linf":
        size, ok = diff.abs().max().item(), diff.abs().max().item() <= eps + 1e-6
    else:
        size = diff.norm(dim=1).max().item()
        ok = size <= eps + 1e-4
    if not ok:
        raise RuntimeError(f"attack exceeded the {norm} radius: {size} > {eps}")
    if adversarial.min() < -1e-6 or adversarial.max() > 1 + 1e-6:
        raise RuntimeError("attack left the valid pixel range")
    return size


def main():
    args = parse_args()
    checkpoint = pathlib.Path(args.checkpoint)
    output = (pathlib.Path(args.output) if args.output
              else checkpoint.resolve().parent / "attack_results")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Results will be saved in {output}", flush=True)

    errors = []
    try:
        # state contains non-tensor python objects -> weights_only=False (trusted file only)
        model = torch.load(checkpoint, map_location=args.device, weights_only=False)
        if model.get("model_type") != "multiclass":
            raise ValueError("Not a multiclass checkpoint; use attack_binary.py")
        config = model["config"]
        torch.backends.cuda.matmul.allow_tf32 = False
        seed = config.get("seed", 0) if args.seed is None else args.seed

        x, y = load_test(config, model["class_ids"], args.device)
        images = x.reshape(-1, 1, 28, 28)
        predict = make_predict(model, args.device)
        bs = len(images)          # whole test set attacked in one go

        clean = accuracy(predict, images, y, min(bs, 1000))
        saved = model.get("metrics", {}).get("test_accuracy")
        print(f"Evaluation: {config['dataset']} classes {model['class_ids'].tolist()} "
              f"({len(x)} test samples)")
        print(f"Clean test accuracy before attack: {clean:.2f}%")
        if saved is not None:
            print(f"Saved training-run test accuracy:  {100 * saved:.2f}%")
            if abs(100 * saved - clean) > 0.1:
                print("WARNING: clean accuracy differs from training-time test accuracy")

        if not args.skip_check:
            err = finite_difference_check(predict, images, y)
            print(f"max |analytic - finite diff| = {err:.3e}")

        budgets = args.budgets or (LINF_DEFAULT if args.norm == "Linf" else L2_DEFAULT)
        attack_name = "+".join(args.attacks)

        manifest = dict(
            checkpoint=str(checkpoint.resolve()), training_epsilon=config["epsilon"],
            rho=config["rho"], norm=args.norm, budgets=budgets, attacks=args.attacks,
            restarts=args.restarts, iters=args.iters, seed=seed, device=args.device,
            test_samples=len(x), dataset=config["dataset"],
            classes=model["class_ids"].tolist(), clean_test_accuracy=clean,
            saved_test_accuracy=saved, attack_batch_size=bs,
        )
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report.write(output, "started", **manifest)

        out_file = output / (f"PGD_result_{args.norm}_{config['kernel']}_"
                             f"{config['dataset']}_mc.json")
        records = []
        for budget in budgets:
            eps = budget / 255.0 if args.norm == "Linf" else float(budget)
            directory = output / (f"eps{budget}-255" if args.norm == "Linf"
                                  else f"eps{budget}-l2")
            directory.mkdir()
            try:
                adversarial = attack_one_eps(predict, images, y, eps, args, seed,
                                             bs, directory, args.device)
                size = check_adversarial(adversarial, images, eps, args.norm)
                robust = accuracy(predict, adversarial, y, min(bs, 1000))
                print(f"[{args.norm} eps={budget}] robust accuracy: {robust:.2f}%",
                      flush=True)
                torch.save(dict(images=adversarial.cpu(), labels=y.cpu()),
                           directory / "adversarial.pt")
                records.append({
                    "eps_float": eps, "eps_budget": budget, "norm": args.norm,
                    "attack": attack_name, "pgd_acc": float(robust),
                    "clean_acc": float(clean), "max_perturbation": size,
                })
                report.write(output, "budget_complete", budget=budget,
                             robust_test_accuracy=robust)
            except Exception as e:
                errors.append({"budget": budget,
                               "error": f"{type(e).__name__}: {e}"})
                traceback.print_exc()

            out_obj = {
                "results": [{
                    "path": str(checkpoint.resolve()), "kernel": config["kernel"],
                    "classes": model["class_ids"].tolist(),
                    "bandwidth": config["length_scale"], "attack": attack_name,
                    "pgd": records,
                }],
                "errors": errors,
            }
            out_file.write_text(json.dumps(out_obj, indent=2) + "\n")

        report.write(output, "complete", results=records)
        print(f"[DONE] Saved results to: {out_file}")
        if errors:
            print(f"[WARN] {len(errors)} budgets failed; see 'errors' in the JSON.")

    except Exception:
        report.write(output, "failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()