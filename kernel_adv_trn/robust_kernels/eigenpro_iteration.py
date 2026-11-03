import torch
import time
import robust_kernels.multioutput_evaluate as evaluation


@torch.no_grad()
def epoch(workspace, solution, rhs):
  x, gram = workspace["centers"], workspace.get("gram")
  batches = torch.randperm(len(x), device=x.device,
    generator=workspace["generator"]).split(workspace["batch_size"])
  progress = workspace.get("progress")
  last_report = time.perf_counter()
  for batch_number, rows in enumerate(batches, 1):
    size = len(rows)
    rate = (1 / workspace["beta"] if size < workspace["critical"] else
      2 / (workspace["beta"] +
           (size - 1) * workspace["next_value"] / workspace["samples"]))
    if gram is None:
      prediction, _ = evaluation.kernel_eval_and_grad(
        x[rows], x, solution, gradients=False, **workspace["options"])
    else:
      prediction = gram[rows] @ solution
    residual = prediction - rhs[rows]
    solution[rows] -= rate * residual
    solution[workspace["indices"]] += rate * workspace["vectors"] @ (
      workspace["extended"][rows].T @ residual)
    if progress is not None and time.perf_counter() - last_report >= 30:
      progress(dict(phase="kernel_solve_batches", solve_call=workspace["solve_calls"],
        total_solve_epoch=workspace["total_epochs"] + 1,
        batch=batch_number, batches=len(batches)))
      last_report = time.perf_counter()
