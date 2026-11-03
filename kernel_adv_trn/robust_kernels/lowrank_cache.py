"""Per-evaluation batches of skinny pending factors, never dense coefficients."""

import torch


@torch.no_grad()
def prepare(factors, center_tile):
  """Group pending blocks once, sharing their contractions across query tiles.

  A group stores (indices, left, core @ right.T). Its storage is proportional
  to the pending ranks, not the product of input and output dimensions.
  The returned scratch must only be used while the factor state is unchanged.
  It is deliberately separate from that state and its saved checkpoints.
  """
  count = (len(factors["u"]) + center_tile - 1) // center_tile
  groups = [{} for _ in range(count)]
  for index, item in factors["pending"].items():
    tile, offset = divmod(index, center_tile)
    shape = tuple(item["core"].shape)
    groups[tile].setdefault(shape, []).append((offset, item))
  prepared = []
  for tile in groups:
    packed = []
    for blocks in tile.values():
      indices = torch.tensor([index for index, _ in blocks],
        dtype=torch.long, device=factors["u"].device)
      left = torch.stack([item["left"] for _, item in blocks])
      core = torch.stack([item["core"] for _, item in blocks])
      right = torch.stack([item["right"] for _, item in blocks])
      packed.append((indices, left, torch.bmm(core, right.transpose(1, 2))))
    prepared.append(packed)
  return prepared
