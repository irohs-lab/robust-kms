import torch


def append(basis, vector):
  """Extend an orthonormal basis using twice-reorthogonalized Gram-Schmidt.

  Residuals at most 8 * machine epsilon * ||vector|| are treated as dependent.
  No nonzero singular directions are otherwise discarded.
  """
  coefficients = basis.T @ vector
  residual = vector - basis @ coefficients
  correction = basis.T @ residual
  residual = residual - basis @ correction
  coefficients = coefficients + correction
  norm = torch.linalg.vector_norm(residual)
  tolerance = 8 * torch.finfo(vector.dtype).eps * torch.linalg.vector_norm(vector)
  if basis.shape[1] == len(vector) or norm <= tolerance:
    return basis, coefficients
  return (torch.cat((basis, (residual / norm)[:, None]), dim=1),
          torch.cat((coefficients, norm.reshape(1))))
