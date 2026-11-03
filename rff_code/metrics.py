import torch


def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    true = y.argmax(dim=1) if y.ndim == 2 else y.long()
    return (pred == true).float().mean().item()


def ce_loss_from_proba(proba, y, eps=1e-9):
    y = y.long()
    idx = torch.arange(y.numel(), device=y.device)
    return -proba[idx, y].clamp_min(eps).log().mean().item()


def proba_to_valid_shape(proba, num_classes):
    """
    Useful for cuML binary logistic regression.
    Sometimes binary predict_proba can behave differently from multiclass.
    This keeps output compatible with argmax.
    """

    if proba.ndim == 1:
        proba = torch.stack([1.0 - proba, proba], dim=1)

    if num_classes == 2 and proba.ndim == 2 and proba.shape[1] == 1:
        p1 = proba[:, 0]
        proba = torch.stack([1.0 - p1, p1], dim=1)

    return proba