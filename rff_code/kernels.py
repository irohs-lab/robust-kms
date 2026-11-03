import torch
from torchkernels.feature_maps import LaplacianRFF, GaussianRFF


def select_kernel(
    input_dim,
    num_features,
    kernel_type="Laplace",
    length_scale=1.0,
    seed=42,
    device=torch.device("cpu"),
):
    def normalize(x):
        return x / x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

    kernel_type_lower = kernel_type.lower()

    init_device = torch.device("cpu")

    if kernel_type_lower == "laplace":
        phi_obj = LaplacianRFF(
            input_dim=input_dim,
            num_features=num_features,
            length_scale=length_scale,
            seed=seed,
            bias_term=False,
            # device=init_device,
            device=device,
            dtype=torch.float32,
        )

    elif kernel_type_lower == "gaussian":
        phi_obj = GaussianRFF(
            input_dim=input_dim,
            num_features=num_features,
            length_scale=length_scale,
            seed=seed,
            bias_term=False,
            # device=init_device,
            device = device,
            dtype=torch.float32,
        )

    else:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    def phi(x):
        return phi_obj(normalize(x))

    return phi