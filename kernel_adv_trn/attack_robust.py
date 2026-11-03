"""Evaluate saved binary kernels with APGD-CE, untargeted FAB, and Square."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from workflow import load_pair, subset, predictor, clean_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New results directory")
    p.add_argument("--budgets", type=float, nargs="+", default=[4., 8., 16.], help="Pixel units: 8 means Linf epsilon=8/255")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--test-limit", type=int, default=0, help="0 evaluates the entire official test class pair")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default=None, help="Override checkpoint data root")
    p.add_argument("--download", action="store_true")
    p.add_argument("--attacks", nargs="+", choices=["apgd-ce", "fab", "square"], default=["apgd-ce", "fab", "square"])
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--square-queries", type=int, default=5000)
    a = p.parse_args()
    if any(not math.isfinite(b) or not 0 <= b <= 255 for b in a.budgets):
        p.error("budgets must be finite and in [0,255]")
    if len(set(a.budgets)) != len(a.budgets) or len(set(a.attacks)) != len(a.attacks):
        p.error("budgets and attacks must not contain duplicates")
    if min(a.batch_size, a.iterations, a.restarts, a.square_queries) <= 0:
        p.error("batch size and attack iteration counts must be positive")
    import autoattack

    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = torch.load(a.checkpoint, map_location=a.device, weights_only=True)
    config = model["config"]
    for required in ("dataset", "classes", "data_root"):
        if required not in config:
            raise ValueError(f"Checkpoint missing config.{required}; cannot safely infer evaluation data")
    x, y = load_pair(a.data_root or config["data_root"], config["dataset"],
                     config["classes"], train=False, download=a.download)
    x, y = subset(x, y, a.test_limit, a.seed)
    x, y = x.to(a.device), y.to(a.device)
    images = x.reshape(-1, 1, 28, 28)
    predict = predictor(model)
    clean = clean_metrics(predict, images, y, a.batch_size)
    a.output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with a.checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest = dict(checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=digest.hexdigest(),
                    dataset=config["dataset"], classes=config["classes"], rho=config["rho"],
                    data_root=a.data_root or config["data_root"], seed=a.seed, samples=len(x),
                    checkpoint_variant=model.get("variant", "legacy_average"),
                    checkpoint_iteration=model.get("iteration", config.get("iterations")),
                    clean_accuracy=clean["accuracy"], clean_bce=clean["loss"],
                    budgets_pixels=a.budgets, attack_version="custom-binary",
                    attacks=a.attacks, iterations=a.iterations, restarts=a.restarts,
                    square_queries=a.square_queries, device=a.device,
                    torch_version=str(torch.__version__), autoattack_source=str(Path(autoattack.__file__).resolve()))
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)
    results = []
    for budget in a.budgets:
        epsilon = budget / 255
        directory = a.output / f"eps{budget:g}-255"
        directory.mkdir()
        if epsilon == 0:
            adversarial = images.clone()
        else:
            adversary = autoattack.AutoAttack(predict, norm="Linf", eps=epsilon,
                version="custom", attacks_to_run=a.attacks, seed=a.seed, device=a.device,
                log_path=str(directory / "autoattack.log"))
            adversary.apgd.n_restarts = a.restarts
            adversary.apgd.n_iter = a.iterations
            adversary.fab.n_restarts = a.restarts
            adversary.fab.n_iter = a.iterations
            adversary.square.n_queries = a.square_queries
            adversarial = adversary.run_standard_evaluation(images, y, bs=a.batch_size)
        if not torch.isfinite(adversarial).all():
            raise FloatingPointError("Attack returned nonfinite inputs")
        maximum = (adversarial - images).abs().max().item()
        if maximum > epsilon + 1e-6 or adversarial.min() < -1e-6 or adversarial.max() > 1 + 1e-6:
            raise RuntimeError("Attack returned inputs outside the specified threat set")
        robust_correct = 0
        with torch.no_grad():
            for start in range(0, len(y), a.batch_size):
                rows = slice(start, start + a.batch_size)
                logits = predict(adversarial[rows])
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite adversarial predictions")
                clean_ok = predict(images[rows]).argmax(1) == y[rows]
                robust_correct += (clean_ok & (logits.argmax(1) == y[rows])).sum().item()
        result = dict(budget_pixels=budget, epsilon=epsilon, samples=len(y),
                      clean_accuracy=clean["accuracy"], robust_accuracy=robust_correct / len(y),
                      max_linf=maximum, attacks=a.attacks if epsilon else [], attack_version="custom-binary")
        torch.save(dict(images=adversarial.cpu(), labels=y.cpu()), directory / "adversarial.pt")
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        (a.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
