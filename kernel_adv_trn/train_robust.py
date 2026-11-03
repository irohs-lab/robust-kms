"""Train the repository's gradient-regularized binary kernel objective."""
import argparse
import json
import math
from pathlib import Path

import torch
import robust_kernels as rk
from workflow import load_pair, split_train, subset, predictor, clean_metrics, pgd_accuracy


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["fmnist", "mnist", "kmnist", "qmnist", "emnist"], default="fmnist")
    p.add_argument("--classes", type=int, nargs=2, default=[0, 1])
    p.add_argument("--data-root", default="/janaki/common/Datasets/rahulky")
    p.add_argument("--download", action="store_true")
    p.add_argument("--output", type=Path, required=True, help="New experiment directory")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimizer", choices=["subgradient", "primal_dual"], default="subgradient")
    p.add_argument("--rho", type=float, default=0.04, help="Gradient penalty weight, NOT an attack radius")
    p.add_argument("--lam", type=float, default=1.)
    p.add_argument("--step-size", type=float, default=1e-4)
    p.add_argument("--decay", type=float, default=0.6)
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--kernel", choices=["gaussian", "matern52"], default="gaussian")
    p.add_argument("--length-scale", type=float, default=28.)
    p.add_argument("--query-tile", type=int, default=64)
    p.add_argument("--center-tile", type=int, default=1024)
    p.add_argument("--average-start", type=int, default=None, help="Start averaging after this many updates; default halfway")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--validation-budget", type=float, default=8., help="Linf pixel budget: 8 means 8/255")
    p.add_argument("--validation-samples", type=int, default=256, help="0 means all validation samples")
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--pgd-restarts", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--train-limit", type=int, default=0, help="0 means all training samples")
    p.add_argument("--diagnostic-samples", type=int, default=256)
    p.add_argument("--full-objective", action="store_true", help="Evaluate exact full training objective at each checkpoint (costly)")
    a = p.parse_args()
    if a.average_start is None:
        a.average_start = a.iterations // 2
    if not 0 <= a.average_start < a.iterations:
        p.error("average-start must be between 0 and iterations-1")
    if min(a.iterations, a.batch_size, a.eval_every, a.eval_batch_size, a.pgd_steps, a.pgd_restarts, a.query_tile, a.center_tile) <= 0:
        p.error("iteration counts and batch/tile sizes must be positive")
    if not all(math.isfinite(v) for v in [a.rho, a.lam, a.step_size, a.decay, a.length_scale, a.validation_budget]):
        p.error("hyperparameters must be finite")
    if a.rho < 0 or a.lam < 0 or a.step_size <= 0 or a.step_size * a.lam > 1:
        p.error("require rho, lam >= 0; step-size > 0; step-size * lam <= 1")
    if not .5 < a.decay <= 1 or a.length_scale <= 0 or not 0 <= a.validation_budget <= 255:
        p.error("require .5 < decay <= 1, length-scale > 0, and budget in [0,255]")
    return a


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    x, y = load_pair(a.data_root, a.dataset, a.classes, train=True, download=a.download)
    x, y, vx, vy = split_train(x, y, a.validation_fraction, a.seed)
    x, y = subset(x, y, a.train_limit, a.seed)
    vx, vy = subset(vx, vy, a.validation_samples, a.seed)
    dx, dy = subset(x, y, a.diagnostic_samples, a.seed)
    x, y, vx, vy, dx, dy = [t.to(a.device) for t in (x, y, vx, vy, dx, dy)]
    if a.batch_size > len(x):
        raise ValueError("batch-size exceeds training set size")
    a.output.mkdir(parents=True, exist_ok=False)
    config = vars(a).copy(); config["output"] = str(a.output)
    config.update(objective="summed_bce_plus_rho_input_gradient_l1_plus_rkhs_penalty",
                  training_samples=len(x), validation_samples_used=len(vx),
                  validation_epsilon=a.validation_budget / 255, split="stratified_train_holdout")
    (a.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    options = {k: getattr(a, k) for k in ("kernel", "length_scale", "query_tile", "center_tile")}
    state = rk.initialize(x, average=False)
    rng = torch.Generator(device=x.device).manual_seed(a.seed)
    best_key = (-1., -float("inf"))

    def model_for(alpha, beta):
        return dict(centers=x, alpha=alpha, beta=beta, bias=0., options=options)

    def save(name, model, iteration, variant, metrics):
        payload = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in model.items()}
        payload.update(config=config, iteration=iteration, variant=variant, validation=metrics)
        temporary = a.output / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(a.output / name)

    def evaluate(iteration):
        nonlocal best_key
        variants = {"last": (state["alpha"], state["beta"])}
        if state.get("weight_sum", 0) > 0:
            variants["average"] = (state["average_alpha"], state["average_beta"])
        for variant, (alpha, beta) in variants.items():
            model = model_for(alpha, beta); predict = predictor(model)
            clean = clean_metrics(predict, vx, vy, a.eval_batch_size)
            robust = pgd_accuracy(predict, vx, vy, epsilon=a.validation_budget / 255,
                                  steps=a.pgd_steps, restarts=a.pgd_restarts,
                                  batch_size=a.eval_batch_size, seed=a.seed)
            values, gradients = rk.kernel_eval_and_grad(dx, x, alpha, beta, **options)
            train = clean_metrics(predict, dx, dy, a.eval_batch_size)
            record = dict(iteration=iteration, variant=variant,
                          validation_accuracy=clean["accuracy"], validation_bce=clean["loss"],
                          validation_pgd_accuracy=robust, validation_epsilon=a.validation_budget / 255,
                          train_probe_accuracy=train["accuracy"], train_probe_bce=train["loss"],
                          train_probe_gradient_l1=gradients.abs().sum(1).mean().item(),
                          train_probe_mean_abs_logit=values.abs().mean().item(),
                          alpha_norm=alpha.norm().item(), beta_norm=beta.norm().item(),
                          delta_max=state["delta"].abs().max().item())
            record["train_probe_bce_plus_penalty"] = train["loss"] + a.rho * record["train_probe_gradient_l1"]
            if a.full_objective:
                result = rk.objective(x, y, alpha, beta, rho=a.rho, lam=a.lam, **options)
                record["full_training_objective"] = {k: v.item() for k, v in result.items()}
            line = json.dumps(record, allow_nan=False)
            print(line, flush=True)
            with (a.output / "metrics.jsonl").open("a") as stream:
                stream.write(line + "\n")
            # These names retain the latest evaluated last and averaged iterates.
            save(variant + ".pt", model, iteration, variant, record)
            key = (robust, -clean["loss"])
            if key > best_key:
                best_key = key
                save("best.pt", model, iteration, variant, record)
                (a.output / "best.json").write_text(line + "\n")

    print(f"Training {len(x)} examples; validation {len(vx)}; optimizer={a.optimizer}; rho={a.rho}", flush=True)
    evaluate(0)
    for m in range(a.iterations):
        if m == a.average_start:
            state.update(average_alpha=torch.zeros_like(state["alpha"]),
                         average_beta=torch.zeros_like(state["beta"]), weight_sum=0.)
        batch = torch.randperm(len(x), device=x.device, generator=rng)[:a.batch_size]
        eta = a.step_size / (m + 1) ** a.decay
        rk.step(x, y, state, batch, rho=a.rho, lam=a.lam, eta=eta,
                optimizer=a.optimizer, **options)
        if (m + 1) % a.eval_every == 0 or m + 1 == a.iterations:
            evaluate(m + 1)
    print(f"Selected checkpoint: {a.output / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
