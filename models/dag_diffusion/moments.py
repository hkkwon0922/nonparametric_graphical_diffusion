"""Numerically stable streaming moments for batched conditional covariance.

The DAG-ordering estimator accumulates posterior samples ``x_{0,S}`` in chunks:
retaining every sample would cost ``B x M x |S|`` floats with ``M`` in the
thousands.  :class:`BatchedWelford` keeps only first and second central moments
per anchor, using Chan et al.'s parallel (batch) update of Welford's algorithm.
"""
from __future__ import annotations

from typing import Optional

import torch

__all__ = ["BatchedWelford"]


class BatchedWelford:
    """Streaming mean / covariance for ``B`` independent anchors of dimension ``d``.

    Parameters
    ----------
    batch: ``B``, number of independent anchors tracked in parallel.
    dim: ``d``, dimension of each sample vector.
    device, dtype: accumulator placement.  ``float64`` is the default because
        the Tweedie formula divides the covariance by ``sigma^4``, which
        amplifies float32 rounding badly at small ``t``.

    Notes
    -----
    Sample layout for :meth:`update` is ``(B, m, d)``: ``m`` new samples for
    each of the ``B`` anchors.
    """

    def __init__(self, batch: int, dim: int, device=None, dtype=torch.float64):
        self.B = int(batch)
        self.d = int(dim)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.count = 0
        self.mean = torch.zeros((self.B, self.d), dtype=dtype, device=self.device)
        # M2 holds the sum of outer products of deviations (co-moment matrix).
        self.M2 = torch.zeros((self.B, self.d, self.d), dtype=dtype, device=self.device)

    def update(self, samples: torch.Tensor) -> "BatchedWelford":
        """Absorb ``(B, m, d)`` new samples.  Returns ``self`` for chaining."""
        if samples.ndim != 3:
            raise ValueError(f"samples must have shape (B, m, d); got {tuple(samples.shape)}")
        if samples.shape[0] != self.B or samples.shape[2] != self.d:
            raise ValueError(
                f"samples shape {tuple(samples.shape)} incompatible with "
                f"(B={self.B}, m, d={self.d})"
            )
        x = samples.to(device=self.device, dtype=self.dtype)
        m = x.shape[1]
        if m == 0:
            return self

        batch_mean = x.mean(dim=1)                      # (B, d)
        centred = x - batch_mean.unsqueeze(1)           # (B, m, d)
        batch_M2 = torch.einsum("bmd,bme->bde", centred, centred)

        if self.count == 0:
            self.mean = batch_mean
            self.M2 = batch_M2
            self.count = m
            return self

        n_a, n_b = self.count, m
        n_ab = n_a + n_b
        delta = batch_mean - self.mean                  # (B, d)
        self.mean = self.mean + delta * (n_b / n_ab)
        self.M2 = self.M2 + batch_M2 + torch.einsum(
            "bd,be->bde", delta, delta) * (n_a * n_b / n_ab)
        self.count = n_ab
        return self

    def covariance(self, unbiased: bool = True) -> torch.Tensor:
        """Return the ``(B, d, d)`` covariance.

        Uses the ``M - 1`` denominator by default.  Raises if fewer than two
        samples have been absorbed (an unbiased covariance is undefined).
        """
        if self.count < 2:
            raise ValueError(
                f"need at least 2 samples for a covariance estimate; got {self.count}"
            )
        denom = (self.count - 1) if unbiased else self.count
        cov = self.M2 / float(denom)
        # symmetrise to kill accumulated asymmetry from the einsum updates
        return 0.5 * (cov + cov.transpose(-1, -2))

    def means(self) -> torch.Tensor:
        """Return the ``(B, d)`` running mean."""
        return self.mean

    def variances(self, unbiased: bool = True) -> torch.Tensor:
        """Return the ``(B, d)`` diagonal of the covariance."""
        return torch.diagonal(self.covariance(unbiased=unbiased), dim1=-2, dim2=-1)
