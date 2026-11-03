"""Batched, score-based SPSA for classifiers returning two or more logits."""
import math

import torch
import torch.nn.functional as F


def add_spsa_arguments(parser):
    parser.add_argument("--ATTACK", choices=["apgd-ce", "spsa"], default="apgd-ce")
    parser.add_argument("--SPSA_ITERS", type=int, default=100)
    parser.add_argument("--SPSA_SAMPLES", type=int, default=32,
                        help="Random directions per iteration; two queries per direction.")
    parser.add_argument("--SPSA_BATCH_SIZE", type=int, default=32)
    parser.add_argument("--SPSA_QUERY_BATCH_SIZE", type=int, default=128)
    parser.add_argument("--SPSA_ETA", type=float, default=0.01,
                        help="Finite-difference radius in raw input units.")
    parser.add_argument("--SPSA_LR", type=float, default=None,
                        help="Update size in raw input units; default epsilon/10.")


def spsa_options(args):
    return dict(iters=args.SPSA_ITERS, samples=args.SPSA_SAMPLES,
                batch_size=args.SPSA_BATCH_SIZE,
                query_batch_size=args.SPSA_QUERY_BATCH_SIZE,
                eta=args.SPSA_ETA, lr=args.SPSA_LR, seed=args.SEED)


@torch.no_grad()
def spsa_attack(predict, x, y, epsilon, norm, *, iters=100, samples=32,
                batch_size=32, query_batch_size=128, eta=0.01, lr=None, seed=42):
    """Maximize cross entropy without autograd, retaining successful iterates.

    Inputs must be floating-point images in [0,1]; labels index logit columns.
    Epsilon and step sizes are raw input-space units for both norms. Query
    tensors are bounded by query_batch_size, independently of sample count.
    """
    if norm not in ("Linf", "L2"):
        raise ValueError("norm must be Linf or L2")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if any(not isinstance(v, int) or v <= 0
           for v in (iters, samples, batch_size, query_batch_size)):
        raise ValueError("SPSA iteration, sample and batch counts must be positive integers")
    if not math.isfinite(eta) or eta <= 0 or (lr is not None and
            (not math.isfinite(lr) or lr <= 0)):
        raise ValueError("SPSA eta and learning rate must be finite and positive")
    if not x.is_floating_point() or x.ndim < 2 or len(x) == 0:
        raise ValueError("Expected a nonempty batch of floating-point inputs")
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("SPSA inputs must be finite and in [0,1]")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("Expected one class index per input")
    if epsilon == 0:
        return x.detach().clone()

    generator = torch.Generator(device=x.device).manual_seed(seed)
    step_size = epsilon / 10 if lr is None else lr
    result = torch.empty_like(x)

    def evaluate(inputs, labels):
        losses, predictions = [], []
        for start in range(0, len(inputs), query_batch_size):
            logits = predict(inputs[start:start + query_batch_size])
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite logits during SPSA")
            losses.append(F.cross_entropy(logits, labels[start:start + query_batch_size],
                                          reduction="none"))
            predictions.append(logits.argmax(1))
        return torch.cat(losses), torch.cat(predictions)

    for start in range(0, len(x), batch_size):
        original = x[start:start + batch_size].detach()
        labels = y[start:start + batch_size].to(device=x.device, dtype=torch.long)
        best = original.clone()
        best_loss, clean_pred = evaluate(best, labels)
        succeeded = clean_pred.ne(labels)

        def project(candidate):
            delta = candidate - original
            if norm == "Linf":
                delta = delta.clamp(-epsilon, epsilon)
            else:
                lengths = delta.flatten(1).norm(dim=1)
                scale = (epsilon / lengths.clamp_min(1e-12)).clamp(max=1)
                delta = delta * scale.reshape(-1, *([1] * (delta.ndim - 1)))
            return (original + delta).clamp(0, 1)

        adv = project(original + torch.empty_like(original).uniform_(
            -epsilon, epsilon, generator=generator))
        for iteration in range(iters + 1):
            loss, pred = evaluate(adv, labels)
            success = pred.ne(labels)
            improve = (success & ~succeeded) | (success.eq(succeeded) & (loss > best_loss))
            best[improve] = adv[improve]
            best_loss[improve] = loss[improve]
            succeeded |= success
            if iteration == iters:
                break
            grad = torch.zeros_like(adv)
            # Stream directions rather than allocating samples * batch * pixels.
            for _ in range(samples):
                direction = torch.empty_like(adv).bernoulli_(
                    0.5, generator=generator).mul_(2).sub_(1)
                plus, _ = evaluate((adv + eta * direction).clamp(0, 1), labels)
                minus, _ = evaluate((adv - eta * direction).clamp(0, 1), labels)
                slope = ((plus - minus) / (2 * eta)).reshape(
                    -1, *([1] * (adv.ndim - 1)))
                grad.add_(slope * direction / samples)
            if norm == "Linf":
                update = grad.sign()
            else:
                lengths = grad.flatten(1).norm(dim=1).clamp_min(1e-12)
                update = grad / lengths.reshape(-1, *([1] * (grad.ndim - 1)))
            adv = project(adv + step_size * update)
        result[start:start + len(original)] = best
    return result.detach()
