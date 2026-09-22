import time
import experiments.score as score


def evaluate(data, state, config, options, epoch, started, validation):
  x, y, test_x, test_y = data
  alpha, beta = state["average_alpha"], state["average_beta"]
  train = score.evaluate(x, y, x, alpha, beta, options, training=True)
  test = score.evaluate(test_x, test_y, x, alpha, beta, options)
  objective = (len(x) * train["loss"] + config["rho"] * train["gradient_l1"]
               + config["lam"] / 2 * train["norm_squared"])
  record = dict(epoch=epoch, step=state["iterations"],
    train_accuracy=train["accuracy"], test_accuracy=test["accuracy"],
    train_logistic_loss=train["loss"], test_logistic_loss=test["loss"],
    objective=objective, elapsed_seconds=time.monotonic() - started)
  if validation is not None:
    val = score.evaluate(*validation, x, alpha, beta, options)
    record.update(validation_accuracy=val["accuracy"],
                  validation_logistic_loss=val["loss"])
  return record
