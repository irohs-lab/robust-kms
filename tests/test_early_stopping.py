import tempfile
import torch
import experiments.early_stopping as stopping
import experiments.validation_split as splitting


def test_early_stopping_retains_earliest_best_validation_checkpoint():
  selection = dict(best_accuracy=-1., best_epoch=0, stale=0)
  for epoch, accuracy in enumerate((0.5, 0.8, 0.8, 0.79, 0.81, 0.80, 0.81, 0.80)):
    improved, stop = stopping.update(selection, accuracy, epoch, patience=3)
    assert improved == (epoch in (0, 1, 4))
    assert stop == (epoch == 7)
  assert selection["best_epoch"] == 4
  assert selection["best_accuracy"] == 0.81


def test_validation_split_is_stratified_and_disjoint_from_training():
  x = torch.arange(40, dtype=torch.float64).reshape(20, 2)
  y = torch.arange(20) % 2
  test_x, test_y = x[:2].clone() + 1000, y[:2].clone()
  with tempfile.TemporaryDirectory() as output:
    config = dict(validation_fraction=0.2, seed=42, output=output)
    training, validation = splitting.split((x, y, test_x, test_y), config)
    repeated, repeated_validation = splitting.split((x, y, test_x, test_y), config)
    assert training[2] is test_x and training[3] is test_y
    assert len(training[0]) == 16 and len(validation[0]) == 4
    assert torch.bincount(validation[1]).tolist() == [2, 2]
    assert set(training[0][:, 0].tolist()).isdisjoint(validation[0][:, 0].tolist())
    torch.testing.assert_close(training[0], repeated[0])
    torch.testing.assert_close(validation[0], repeated_validation[0])
