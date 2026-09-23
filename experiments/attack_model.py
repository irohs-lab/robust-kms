import functools
import pathlib
import torch
import autoattack
import robust_kernels.logits as logits
import experiments.report as report


def run(model, test_x, test_y, config):
  output = pathlib.Path(config["output"])
  predict = functools.partial(logits.binary_logits, centers=model["centers"],
    alpha=model["alpha"], beta=model["beta"],
    bias=model.get("bias", 0.), **model["options"])
  images = test_x.reshape(-1, 1, 28, 28)
  adversary = autoattack.AutoAttack(
    predict, norm="Linf", eps=config["epsilon"], version="custom",
    attacks_to_run=["apgd-ce", "fab", "square"], seed=config["seed"],
    device=config["device"], log_path=str(output / "autoattack.log"))
  report.write(output, "attacking", samples=len(images), epsilon=config["epsilon"],
               attacks=adversary.attacks_to_run, version="custom",
               apgd_restarts=5, fab_restarts=5, iterations=100, square_queries=5000)
  adversarial = adversary.run_standard_evaluation(
    images, test_y, bs=config["attack_batch_size"])
  correct, clean_correct = 0, 0
  with torch.no_grad():
    for start in range(0, len(images), config["attack_batch_size"]):
      rows = slice(start, start + config["attack_batch_size"])
      correct += (predict(adversarial[rows]).argmax(1) == test_y[rows]).sum().item()
      clean_correct += (predict(images[rows]).argmax(1) == test_y[rows]).sum().item()
  maximum = (adversarial - images).abs().max().item()
  if maximum > config["epsilon"] + 1e-6:
    raise RuntimeError("attack exceeded the pixel-space Linf radius")
  if adversarial.min() < -1e-6 or adversarial.max() > 1 + 1e-6:
    raise RuntimeError("attack left the valid pixel range")
  torch.save(dict(images=adversarial.cpu(), labels=test_y.cpu()),
             output / "adversarial.pt")
  report.write(output, "complete", clean_test_accuracy=clean_correct / len(images),
               robust_test_accuracy=correct / len(images), max_linf=maximum,
               samples=len(images), epsilon=config["epsilon"],
               attack_version="custom-binary", attacks=adversary.attacks_to_run)
