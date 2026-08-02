"""Tweedie-identity Hessian of the noised marginal log-density.

For the DDPM forward process ``X_t = sqrt(alpha_bar_t) X_0 + sqrt(1 - alpha_bar_t) eps``,
the second-order Tweedie identity gives, for any index subset ``S``,

    H_S(x_S, t) = grad^2_{x_S} log p_{t,S}(x_S)
                = (alpha_bar_t / sigma2_t^2) * Cov(X_{0,S} | X_{t,S} = x_S)
                  - (1 / sigma2_t) * I_{|S|},

where

    sigma2_t = 1 - alpha_bar_t.

**Units note.** The ``sigma_t^4`` appearing in the usual write-up is
``sigma2_t ** 2`` in code, *not* ``sigma2_t ** 4``.  ``test_tweedie_hessian_gaussian``
pins this down against the closed-form Gaussian answer.

The sign of ``H`` is meaningful and is never absolute-valued here: the DAG leaf
criterion is the variance of the *signed* diagonal.
"""
from __future__ import annotations

from typing import Optional

import torch

from .moments import BatchedWelford

__all__ = [
    "tweedie_hessian_from_covariance",
    "estimate_conditional_covariance",
    "estimate_tweedie_hessian",
    "estimate_tweedie_hessian_diagonal",
    "alpha_bar_and_sigma2",
]

_MIN_SIGMA2 = 1e-12


def alpha_bar_and_sigma2(diffusion, t: int, device=None, dtype=torch.float64):
    """Return ``(alpha_bar_t, sigma2_t)`` as 0-dim tensors, with ``sigma2_t = 1 - alpha_bar_t``."""
    t_int = int(t)
    if t_int < 0 or t_int >= int(diffusion.timesteps):
        raise ValueError(f"t={t_int} out of range [0, {int(diffusion.timesteps) - 1}]")
    alpha_bar = torch.as_tensor(diffusion.alphas_bar[t_int], dtype=dtype, device=device)
    sigma2 = (1.0 - alpha_bar).clamp_min(_MIN_SIGMA2)
    return alpha_bar, sigma2


def tweedie_hessian_from_covariance(cov: torch.Tensor, alpha_bar, sigma2) -> torch.Tensor:
    """Apply the Tweedie formula to a covariance.

    Parameters
    ----------
    cov: (..., d, d) conditional covariance ``Cov(X_{0,S} | X_{t,S})``.
    alpha_bar, sigma2: scalars (``sigma2 = 1 - alpha_bar``).

    Returns
    -------
    (..., d, d) Hessian ``H = alpha_bar / sigma2**2 * cov - I / sigma2``.
    """
    if cov.ndim < 2 or cov.shape[-1] != cov.shape[-2]:
        raise ValueError(f"cov must be (..., d, d); got {tuple(cov.shape)}")
    alpha_bar = torch.as_tensor(alpha_bar, dtype=cov.dtype, device=cov.device)
    sigma2 = torch.as_tensor(sigma2, dtype=cov.dtype, device=cov.device).clamp_min(_MIN_SIGMA2)

    d = cov.shape[-1]
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    # sigma_t^4 in the maths == sigma2 ** 2 in code, because sigma2 = 1 - alpha_bar.
    return (alpha_bar / sigma2.pow(2)) * cov - eye / sigma2


def estimate_conditional_covariance(
    x0_S_samples: torch.Tensor, unbiased: bool = True
) -> torch.Tensor:
    """Covariance of ``(B, M, |S|)`` posterior samples -> ``(B, |S|, |S|)``.

    Uses the ``M - 1`` denominator by default and raises when ``M < 2``.
    Accumulation is done in float64.
    """
    if x0_S_samples.ndim != 3:
        raise ValueError(
            f"x0_S_samples must have shape (B, M, |S|); got {tuple(x0_S_samples.shape)}")
    B, M, d = x0_S_samples.shape
    if M < 2:
        raise ValueError(f"need M >= 2 posterior samples for a covariance estimate; got M={M}")

    acc = BatchedWelford(B, d, device=x0_S_samples.device, dtype=torch.float64)
    acc.update(x0_S_samples)
    return acc.covariance(unbiased=unbiased)


def estimate_tweedie_hessian(
    x0_S_samples: Optional[torch.Tensor],
    diffusion,
    t: int,
    accumulator: Optional[BatchedWelford] = None,
    unbiased: bool = True,
):
    """Full Tweedie Hessian for a batch of anchors.

    Provide **either** ``x0_S_samples`` of shape ``(B, M, |S|)`` **or** a
    pre-filled streaming ``accumulator``.

    Returns
    -------
    hessian: (B, |S|, |S|) float64
    cov:     (B, |S|, |S|) float64
    """
    if accumulator is not None:
        cov = accumulator.covariance(unbiased=unbiased)
    elif x0_S_samples is not None:
        cov = estimate_conditional_covariance(x0_S_samples, unbiased=unbiased)
    else:
        raise ValueError("provide either x0_S_samples or a filled accumulator")

    alpha_bar, sigma2 = alpha_bar_and_sigma2(diffusion, t, device=cov.device, dtype=cov.dtype)
    hessian = tweedie_hessian_from_covariance(cov, alpha_bar, sigma2)
    return hessian, cov


def estimate_tweedie_hessian_diagonal(
    x0_S_samples: Optional[torch.Tensor],
    diffusion,
    t: int,
    accumulator: Optional[BatchedWelford] = None,
    unbiased: bool = True,
) -> torch.Tensor:
    """Diagonal of the Tweedie Hessian -> ``(B, |S|)`` float64.

    Equivalent to ``diag(estimate_tweedie_hessian(...))`` but computed from the
    covariance diagonal only, so it never materialises a ``(B, |S|, |S|)``
    Hessian when only the diagonal is needed.
    """
    if accumulator is not None:
        var = accumulator.variances(unbiased=unbiased)
    elif x0_S_samples is not None:
        if x0_S_samples.ndim != 3:
            raise ValueError(
                f"x0_S_samples must have shape (B, M, |S|); got {tuple(x0_S_samples.shape)}")
        if x0_S_samples.shape[1] < 2:
            raise ValueError(
                f"need M >= 2 posterior samples; got M={x0_S_samples.shape[1]}")
        x = x0_S_samples.to(dtype=torch.float64)
        var = x.var(dim=1, unbiased=unbiased)
    else:
        raise ValueError("provide either x0_S_samples or a filled accumulator")

    alpha_bar, sigma2 = alpha_bar_and_sigma2(diffusion, t, device=var.device, dtype=var.dtype)
    return (alpha_bar / sigma2.pow(2)) * var - 1.0 / sigma2
