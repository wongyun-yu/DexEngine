"""Per-update PPO loss sums kept on the device (the one owner for PerSidePPO and KLSafePPO)."""

import torch


class LossSums:
    """One float64 accumulator per loss name, read with one sync after the update.

    Each ``add`` is ``acc += x.double()``: the same IEEE double addition, in the same order, as the Python float sum
    of ``x.item()`` it replaces (that sum started from the int 0, and 0 + x == 0.0 + x for every float x), so the
    logged means keep their bits. The update no longer waits for each minibatch's backward and optimizer step before
    it launches the next minibatch; the KL early-stop test stays the only per-minibatch sync."""

    def __init__(self, names, device):
        self._acc = {n: torch.zeros((), dtype=torch.float64, device=device) for n in names}

    def add(self, name: str, x: torch.Tensor) -> None:
        self._acc[name] += x.detach().double()

    def means(self, n: int) -> dict:
        vals = torch.stack(list(self._acc.values())).tolist()
        return {k: v / n for k, v in zip(self._acc, vals, strict=True)}
