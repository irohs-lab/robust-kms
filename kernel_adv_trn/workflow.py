"""Shared binary image data and evaluation for the robust kernel CLI tools."""
import functools
import math

import torch
import torch.nn.functional as F
from robust_kernels.logits import binary_logits


def load_pair(root, dataset, classes, *, train, download=False):
    """Official splits, raw [0,1] pixels, class order defines labels 0 and 1."""
    import torchvision.datasets as datasets

    if len(classes) != 2 or classes[0] == classes[1]:
        raise ValueError("Provide two distinct classes")
    constructors = dict(fmnist=datasets.FashionMNIST, mnist=datasets.MNIST,
                        kmnist=datasets.KMNIST, qmnist=datasets.QMNIST,
                        emnist=datasets.EMNIST)
    kwargs = dict(root=root, download=download)
    if dataset == "qmnist":
        kwargs.update(what="train" if train else "test", compat=True)
    else:
        kwargs["train"] = train
    if dataset == "emnist":
        kwargs["split"] = "digits"
    source = constructors[dataset](**kwargs)
    labels = source.targets
    if labels.ndim == 2:
        labels = labels[:, 0]
    selected = (labels == classes[0]) | (labels == classes[1])
    x = source.data[selected]
    y = (labels[selected] == classes[1]).long()
    if not all((y == c).any() for c in (0, 1)):
        raise ValueError("Both requested classes must occur in the dataset")
    if dataset == "emnist":
        x = torch.rot90(x, 1, dims=(1, 2)).flip(2)
    return x.float().flatten(1).div_(255), y


def split_train(x, y, fraction, seed):
    """Stratified validation split; never uses the official test set."""
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = torch.Generator().manual_seed(seed)
    train, val = [], []
    for c in (0, 1):
        ids = (y == c).nonzero().flatten()
        ids = ids[torch.randperm(len(ids), generator=rng)]
        count = max(1, min(len(ids) - 1, round(len(ids) * fraction)))
        if len(ids) < 2:
            raise ValueError("Need at least two training samples per class")
        val.append(ids[:count]); train.append(ids[count:])
    def combine(parts):
        ids = torch.cat(parts)
        return ids[torch.randperm(len(ids), generator=rng)]
    tr, va = combine(train), combine(val)
    return x[tr], y[tr], x[va], y[va]


def subset(x, y, limit, seed):
    """Deterministic, approximately balanced subset (0 means all)."""
    if limit < 0 or limit == 1:
        raise ValueError("sample limit must be 0 or at least 2")
    if not limit or limit >= len(x):
        return x, y
    rng = torch.Generator().manual_seed(seed)
    groups = []
    for c in (0, 1):
        ids = (y == c).nonzero().flatten()
        ids = ids[torch.randperm(len(ids), generator=rng)]
        groups.append(ids)
    counts = [min(len(groups[0]), limit // 2), min(len(groups[1]), limit - limit // 2)]
    for c in (0, 1):
        counts[c] += min(limit - sum(counts), len(groups[c]) - counts[c])
    ids = torch.cat([g[:n] for g, n in zip(groups, counts)])
    ids = ids[torch.randperm(len(ids), generator=rng)]
    return x[ids], y[ids]


def predictor(model):
    return functools.partial(binary_logits, centers=model["centers"],
                             alpha=model["alpha"], beta=model["beta"],
                             bias=model.get("bias", 0.), **model["options"])


@torch.no_grad()
def clean_metrics(predict, x, y, batch_size):
    correct, loss = 0, 0.
    for start in range(0, len(x), batch_size):
        logits = predict(x[start:start + batch_size])
        labels = y[start:start + batch_size]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite predictions; reduce the training step size")
        correct += (logits.argmax(1) == labels).sum().item()
        loss += F.cross_entropy(logits, labels, reduction="sum").item()
    return dict(accuracy=correct / len(x), loss=loss / len(x))


def pgd_accuracy(predict, x, y, *, epsilon, steps, restarts, batch_size, seed):
    """Validation-only PGD; count any failure across steps/restarts as nonrobust."""
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    if min(steps, restarts, batch_size) <= 0:
        raise ValueError("PGD steps, restarts, and batch_size must be positive")
    rng = torch.Generator(device=x.device).manual_seed(seed)
    correct = 0
    for start in range(0, len(x), batch_size):
        clean, labels = x[start:start + batch_size], y[start:start + batch_size]
        with torch.no_grad():
            robust = predict(clean).argmax(1) == labels
        lower, upper = (clean - epsilon).clamp(0, 1), (clean + epsilon).clamp(0, 1)
        for _ in range(restarts):
            adv = clean + torch.empty_like(clean).uniform_(-epsilon, epsilon, generator=rng)
            adv = adv.clamp(0, 1)
            for k in range(steps + 1):
                adv = adv.detach().requires_grad_(True)
                logits = predict(adv)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite validation logits")
                robust &= logits.detach().argmax(1) == labels
                if k == steps or epsilon == 0:
                    break
                grad, = torch.autograd.grad(F.cross_entropy(logits, labels), adv)
                if not torch.isfinite(grad).all():
                    raise FloatingPointError("Nonfinite validation gradients")
                adv = (adv.detach() + (2.5 * epsilon / steps) * grad.sign()).clamp(lower, upper)
        correct += robust.sum().item()
    return correct / len(x)
