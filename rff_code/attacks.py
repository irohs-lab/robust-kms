import torch
import torch.nn.functional as F

from tensor_utils import batch_l2_norm


def pgd_attack_rfc(
    x,
    y,
    phi,
    clf,
    eps,
    alpha,
    steps,
    norm="Linf",
    clamp_min=0.0,
    clamp_max=1.0,
    random_start=True,
):
    x_clean = x.detach()
    y = y.long()
    norm = norm.lower()

    if random_start:
        if norm == "linf":
            x_adv = x_clean + torch.empty_like(x_clean).uniform_(-eps, eps)
            x_adv = torch.max(torch.min(x_adv, x_clean + eps), x_clean - eps)
            x_adv = torch.clamp(x_adv, clamp_min, clamp_max).detach()

        elif norm == "l2":
            delta = torch.randn_like(x_clean)
            delta_norm = batch_l2_norm(delta).clamp_min(1e-12)
            delta = delta / delta_norm

            radius = torch.rand(
                x_clean.size(0),
                *([1] * (x_clean.ndim - 1)),
                device=x_clean.device,
                dtype=x_clean.dtype,
            ) * eps

            x_adv = x_clean + delta * radius
            x_adv = torch.clamp(x_adv, clamp_min, clamp_max).detach()

        else:
            raise ValueError(f"Unsupported norm: {norm}")

    else:
        x_adv = x_clean.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        logits = clf(phi(x_adv))
        loss = F.cross_entropy(logits, y)

        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            if norm == "linf":
                x_adv = x_adv + alpha * grad.sign()
                x_adv = torch.max(torch.min(x_adv, x_clean + eps), x_clean - eps)
                x_adv = torch.clamp(x_adv, clamp_min, clamp_max)

            elif norm == "l2":
                grad_norm = batch_l2_norm(grad).clamp_min(1e-12)
                x_adv = x_adv + alpha * grad / grad_norm

                delta = x_adv - x_clean
                delta_norm = batch_l2_norm(delta).clamp_min(1e-12)

                factor = torch.minimum(
                    torch.ones_like(delta_norm),
                    eps / delta_norm,
                )

                x_adv = x_clean + delta * factor
                x_adv = torch.clamp(x_adv, clamp_min, clamp_max)

        x_adv = x_adv.detach()

    return x_adv