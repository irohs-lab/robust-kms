import torch
import torch.nn.functional as F


def ensure_labels(y):
    if y.ndim == 2:
        return y.argmax(dim=1).long()
    return y.long()


def ensure_onehot(y, num_classes):
    if y.ndim == 2:
        return y.float()
    return F.one_hot(y.long(), num_classes=num_classes).float()


def iterate_minibatches(x, y, batch_size=256, shuffle=True):
    n = x.size(0)

    if shuffle:
        indices = torch.randperm(n, device=x.device)
    else:
        indices = torch.arange(n, device=x.device)

    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield x[idx], y[idx]


def batch_l2_norm(t):
    return t.view(t.size(0), -1).norm(p=2, dim=1).view(
        -1, *([1] * (t.ndim - 1))
    )


def augment(Phi):
    ones = torch.ones(Phi.shape[0], 1, device=Phi.device, dtype=Phi.dtype)
    return torch.cat([Phi, ones], dim=1)