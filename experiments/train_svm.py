import numpy
import sklearn.svm
import torch
import experiments.baseline_metrics as metrics
import experiments.report as report


def run(x, labels, matrices, config):
  gram = matrices[0].cpu().numpy().astype(numpy.float64)
  targets = labels[0].cpu().numpy()
  best, selected = -1., None
  # Descending lambda retains the strongest regularizer on validation ties.
  for exponent in range(4, -7, -1):
    lam = 10. ** exponent
    solver = sklearn.svm.SVC(C=1 / lam, kernel="precomputed", tol=1e-5)
    solver.fit(gram, targets)
    if solver.fit_status_:
      raise RuntimeError("SVM solver did not converge")
    alpha = torch.zeros(len(x), device=x.device)
    alpha[torch.as_tensor(solver.support_, device=x.device)] = torch.as_tensor(
      solver.dual_coef_[0], device=x.device, dtype=x.dtype)
    bias = float(solver.intercept_[0])
    predictions = [matrix @ alpha + bias for matrix in matrices]
    reference = torch.as_tensor(solver.decision_function(gram), device=x.device)
    error = (predictions[0] - reference).abs().max().item()
    if error > 2e-3:
      raise RuntimeError(f"SVM export mismatch: {error}")
    scores = metrics.measure(predictions, labels)
    report.write(config["output"], "candidate", lam=lam, C=1 / lam,
      support_vectors=len(solver.support_), export_max_error=error, **scores)
    if scores["validation_accuracy"] > best:
      best = scores["validation_accuracy"]
      selected = dict(alpha=alpha, bias=bias, lam=lam, C=1 / lam,
        support_vectors=len(solver.support_), export_max_error=error, **scores)
  if selected["lam"] in (1e4, 1e-6):
    raise RuntimeError("Selected SVM lambda is at the grid boundary; extend the grid")
  return selected
