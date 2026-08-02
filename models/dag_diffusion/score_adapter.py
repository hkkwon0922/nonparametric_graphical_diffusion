"""Score adapter around a trained DDPM network + :class:`GaussianDiffusion`.

The repository's DDPM networks predict the *noise* ``eps_theta(x_t, t)`` (the
default ``model_mean_type == "eps"``).  Everything downstream in the DAG
ordering pipeline instead needs the **score** of the noised marginal,

    s_theta(x_t, t) ~= grad_{x_t} log p_t(x_t).

This module converts one into the other and validates shapes / timesteps so
that a mis-specified call fails loudly instead of silently producing garbage
Langevin chains.

Conversions
-----------
``model_mean_type == "eps"``::

    s_theta(x_t, t) = -eps_theta(x_t, t) / sqrt(1 - alpha_bar_t)

``model_mean_type == "x_0"``::

    s_theta(x_t, t) = (sqrt(alpha_bar_t) * x0_theta(x_t, t) - x_t) / (1 - alpha_bar_t)

Both follow from Tweedie's formula applied to
``x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps``.
"""
from __future__ import annotations

from typing import Union

import torch

__all__ = ["DDPMScoreAdapter", "AnalyticGaussianScoreAdapter"]

_MIN_SIGMA2 = 1e-12

TimestepLike = Union[int, torch.Tensor]


class DDPMScoreAdapter:
    """Expose ``score(x_t, t)`` for a trained epsilon- (or x_0-) prediction network.

    Parameters
    ----------
    model:
        Trained denoising network, callable as ``model(x_t, t_tensor)`` where
        ``x_t`` has shape ``(N, D)`` and ``t_tensor`` has shape ``(N,)`` and
        dtype ``int64``.
    diffusion:
        A :class:`models.ddpm.core.ddpm_torch.toy.GaussianDiffusion` instance
        (anything exposing ``alphas_bar`` and ``timesteps``).
    device:
        Device on which score evaluations are performed.  Defaults to the
        device of the model's first parameter.
    model_mean_type:
        Overrides ``diffusion.model_mean_type`` when given.  Only ``"eps"`` and
        ``"x_0"`` can be converted to a score.

    Notes
    -----
    The adapter runs the network under :func:`torch.inference_mode`; the
    fixed-time Langevin sampler only needs score *evaluations*, never
    derivatives of the network score, so autograd is never enabled here.
    """

    def __init__(self, model, diffusion, device=None, model_mean_type=None):
        self.model = model
        self.diffusion = diffusion

        if device is None:
            try:
                device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                device = torch.device("cpu")
        self.device = torch.device(device)

        mean_type = model_mean_type if model_mean_type is not None else getattr(
            diffusion, "model_mean_type", None)
        if mean_type not in ("eps", "x_0"):
            raise ValueError(
                "DDPMScoreAdapter supports model_mean_type in {'eps', 'x_0'}; "
                f"got {mean_type!r}. A 'mean' parameterisation would first have to be "
                "converted to an x_0 or eps prediction."
            )
        self.model_mean_type = mean_type

        self.timesteps = int(getattr(diffusion, "timesteps", len(diffusion.alphas_bar)))
        # Cache the schedule on the target device as float32, once.
        self.alphas_bar = torch.as_tensor(
            diffusion.alphas_bar, dtype=torch.float32, device=self.device)
        self.sqrt_alphas_bar = self.alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1.0 - self.alphas_bar).clamp_min(_MIN_SIGMA2).sqrt()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _as_timestep_tensor(self, t: TimestepLike, n: int) -> torch.Tensor:
        """Normalise ``t`` into an int64 tensor of shape ``(n,)`` on ``self.device``.

        Accepts a python ``int``, a 0-dim tensor, a length-1 tensor, or a
        length-``n`` tensor.
        """
        if isinstance(t, torch.Tensor):
            t_tensor = t.to(device=self.device, dtype=torch.int64).reshape(-1)
            if t_tensor.numel() == 1:
                t_tensor = t_tensor.expand(n).contiguous()
            elif t_tensor.numel() != n:
                raise ValueError(
                    f"timestep tensor has {t_tensor.numel()} entries but x_t has batch size {n}"
                )
        else:
            t_int = int(t)
            t_tensor = torch.full((n,), t_int, dtype=torch.int64, device=self.device)

        if bool((t_tensor < 0).any()) or bool((t_tensor >= self.timesteps).any()):
            raise ValueError(
                f"timestep(s) out of range [0, {self.timesteps - 1}]: "
                f"min={int(t_tensor.min())}, max={int(t_tensor.max())}"
            )
        return t_tensor

    def _validate_x(self, x_t: torch.Tensor) -> torch.Tensor:
        if not isinstance(x_t, torch.Tensor):
            raise TypeError(f"x_t must be a torch.Tensor, got {type(x_t)!r}")
        if x_t.ndim != 2:
            raise ValueError(f"x_t must have shape (N, D); got {tuple(x_t.shape)}")
        if not torch.isfinite(x_t).all():
            raise FloatingPointError("x_t contains non-finite values before score evaluation")
        return x_t.to(device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def score(self, x_t: torch.Tensor, t: TimestepLike) -> torch.Tensor:
        """Return ``grad_{x_t} log p_t(x_t)``.

        Parameters
        ----------
        x_t: (N, D) float tensor.
        t: scalar int, 0-dim/1-element tensor, or (N,) int tensor.

        Returns
        -------
        (N, D) float32 tensor on ``self.device``, same shape as ``x_t``.
        """
        x = self._validate_x(x_t)
        n = x.shape[0]
        t_tensor = self._as_timestep_tensor(t, n)

        out = self.model(x, t_tensor)
        if out.shape != x.shape:
            raise ValueError(
                f"network output shape {tuple(out.shape)} does not match x_t shape {tuple(x.shape)}"
            )

        if self.model_mean_type == "eps":
            sigma = self.sqrt_one_minus_alphas_bar.gather(0, t_tensor).unsqueeze(1)
            score = -out / sigma
        else:  # "x_0"
            sqrt_ab = self.sqrt_alphas_bar.gather(0, t_tensor).unsqueeze(1)
            sigma2 = (1.0 - self.alphas_bar.gather(0, t_tensor)).clamp_min(_MIN_SIGMA2).unsqueeze(1)
            score = (sqrt_ab * out - x) / sigma2

        if not torch.isfinite(score).all():
            raise FloatingPointError(
                "score evaluation produced non-finite values "
                f"(model_mean_type={self.model_mean_type}, t range "
                f"[{int(t_tensor.min())}, {int(t_tensor.max())}])"
            )
        return score

    __call__ = score


class AnalyticGaussianScoreAdapter:
    """Exact score of ``N(mu, Sigma)``, used for tests and diagnostics.

    Implements the same ``score(x, t)`` interface as :class:`DDPMScoreAdapter`
    but ignores ``t``: the target is a single fixed Gaussian.

    Parameters
    ----------
    mean: (D,) tensor.
    cov: (D, D) symmetric positive-definite tensor.
    """

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, device=None):
        device = torch.device(device) if device is not None else mean.device
        self.device = device
        self.mean = mean.to(device=device, dtype=torch.float64).reshape(-1)
        cov = cov.to(device=device, dtype=torch.float64)
        d = self.mean.numel()
        if cov.shape != (d, d):
            raise ValueError(f"cov must be ({d}, {d}); got {tuple(cov.shape)}")
        self.precision = torch.linalg.inv(cov)
        self.timesteps = 1

    @torch.inference_mode()
    def score(self, x_t: torch.Tensor, t: TimestepLike = 0) -> torch.Tensor:
        """Return ``-Sigma^{-1} (x - mu)`` with shape ``(N, D)``."""
        if x_t.ndim != 2:
            raise ValueError(f"x_t must have shape (N, D); got {tuple(x_t.shape)}")
        x = x_t.to(device=self.device, dtype=torch.float64)
        out = -(x - self.mean) @ self.precision.T
        return out.to(dtype=x_t.dtype)

    __call__ = score
