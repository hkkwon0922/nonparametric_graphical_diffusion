"""Fixed-time conditional Langevin sampling of ``X_{t,R} | X_{t,S} = x_{t,S}``.

Why this is needed
------------------
A trained DDPM only gives the *full-dimensional* noised score

    s_theta(x_t, t) ~= grad_{x_t} log p_t(x_t).

For a remaining set ``S`` and its complement ``R = {0..D-1} \\ S``, the
conditional score of the free block satisfies

    grad_{x_R} log p_t(x_R | x_S) = grad_{x_R} log p_t(x_R, x_S),

i.e. it is exactly the ``R`` rows of the full score with ``x_S`` held fixed.
No retraining or lower-dimensional model is required.

Sampler
-------
Overdamped unadjusted Langevin (ULA)::

    z_{k+1} = z_k + h * score_R(merge(x_S, z_k), t) + sqrt(2h) * xi_k,
    xi_k ~ N(0, I_{|R|}).

MALA is deliberately *not* implemented: ULA is the required baseline and a
Metropolis correction would need full-joint log-density evaluations that the
score network does not provide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

__all__ = [
    "LangevinConfig",
    "LangevinDiagnostics",
    "sample_conditional_langevin",
    "complement_indices",
]


def complement_indices(dim: int, condition_indices: Sequence[int]) -> list:
    """Return ``R = {0..dim-1} \\ S`` as a sorted list of original indices."""
    s = set(int(i) for i in condition_indices)
    if any(i < 0 or i >= dim for i in s):
        raise ValueError(f"condition_indices must lie in [0, {dim - 1}]; got {sorted(s)}")
    return [i for i in range(dim) if i not in s]


@dataclass
class LangevinConfig:
    """Hyper-parameters of the fixed-time conditional ULA sampler.

    Attributes
    ----------
    num_chains: independent chains run per conditioning anchor.
    burn_in: ULA steps discarded before any state is retained.
    num_samples: retained states **per chain**.
    thinning: ULA steps between two retained states.
    step_size: ULA step size ``h``.
    init: ``"normal"`` (``z_0 ~ N(0, I)``) or ``"forward_data"``
        (forward-diffuse reference ``x_0`` rows to time ``t`` and take their
        ``R`` coordinates).  ``"forward_data"`` is the recommended default when
        reference data are available because it starts the chain in the bulk of
        ``p_t``.
    score_norm_clip: optional per-row clipping of the score norm.  ``None``
        (default) disables clipping.
    chunk_size: maximum number of chain-rows evaluated per score call.
    max_abs_value: chains whose absolute value exceeds this are treated as
        exploded and raise an error.
    """

    num_chains: int = 4
    burn_in: int = 200
    num_samples: int = 10
    thinning: int = 10
    step_size: float = 1e-4
    init: str = "forward_data"
    score_norm_clip: Optional[float] = None
    chunk_size: int = 4096
    max_abs_value: float = 1e6

    def __post_init__(self):
        if self.init not in ("normal", "forward_data"):
            raise ValueError(f"init must be 'normal' or 'forward_data'; got {self.init!r}")
        if self.num_chains < 1:
            raise ValueError("num_chains must be >= 1")
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if self.thinning < 1:
            raise ValueError("thinning must be >= 1")
        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        if self.step_size <= 0:
            raise ValueError("step_size must be > 0")


@dataclass
class LangevinDiagnostics:
    """Summary statistics of one conditional Langevin run.

    All fields are plain python / numpy so the record can be JSON- or
    npz-serialised directly.
    """

    mean_score_norm: float = float("nan")
    max_score_norm: float = float("nan")
    mean_update_norm: float = float("nan")
    half_mean_shift: Optional[np.ndarray] = None   # (|R|,) 2nd-half minus 1st-half mean
    chain_mean: Optional[np.ndarray] = None        # (|R|,) mean over all retained states
    chain_var: Optional[np.ndarray] = None         # (|R|,) variance over all retained states
    per_chain_mean: Optional[np.ndarray] = None    # (num_chains, |R|)
    per_chain_var: Optional[np.ndarray] = None     # (num_chains, |R|)
    lag1_autocorr: Optional[np.ndarray] = None     # (|R|,)
    num_nonfinite: int = 0
    num_clipped_steps: int = 0
    num_retained: int = 0
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {}
        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                out[key] = val.tolist()
            elif isinstance(val, dict):
                out[key] = val
            else:
                out[key] = val
        return out


class LangevinDivergenceError(RuntimeError):
    """Raised when a Langevin chain produces non-finite or exploding states."""


def _merge(x_S: torch.Tensor, z_R: torch.Tensor, s_idx: torch.Tensor,
           r_idx: torch.Tensor, dim: int) -> torch.Tensor:
    """Scatter ``x_S`` and ``z_R`` back into a full ``(N, D)`` state."""
    n = x_S.shape[0]
    full = torch.empty((n, dim), dtype=x_S.dtype, device=x_S.device)
    full[:, s_idx] = x_S
    if r_idx.numel() > 0:
        full[:, r_idx] = z_R
    return full


def _score_R_chunked(score_adapter, x_S_rep, z, s_idx, r_idx, dim, t, chunk_size):
    """Evaluate the ``R`` block of the full score, in chunks over rows."""
    n = z.shape[0]
    if chunk_size is None or chunk_size >= n:
        full = _merge(x_S_rep, z, s_idx, r_idx, dim)
        return score_adapter.score(full, t)[:, r_idx]

    parts = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        full = _merge(x_S_rep[start:end], z[start:end], s_idx, r_idx, dim)
        parts.append(score_adapter.score(full, t)[:, r_idx])
    return torch.cat(parts, dim=0)


@torch.inference_mode()
def sample_conditional_langevin(
    score_adapter,
    x_t_S: torch.Tensor,
    condition_indices: Sequence[int],
    dim: int,
    t: int,
    config: LangevinConfig,
    diffusion=None,
    reference_x0: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None,
    collect_diagnostics: bool = True,
):
    """Sample ``X_{t,R} | X_{t,S} = x_{t,S}`` with fixed-time ULA.

    Parameters
    ----------
    score_adapter:
        Object exposing ``score(x, t) -> (N, D)`` (e.g.
        :class:`~models.dag_diffusion.score_adapter.DDPMScoreAdapter`).
    x_t_S: (B, |S|) tensor
        Conditioning values, one row per anchor.
    condition_indices:
        The set ``S`` in **original** ``D``-dimensional indexing.  Column ``j``
        of ``x_t_S`` corresponds to original index ``condition_indices[j]``.
    dim: int
        Full data dimension ``D``.
    t: int
        Diffusion timestep at which the conditional is taken.
    config: :class:`LangevinConfig`
    diffusion:
        Required when ``config.init == "forward_data"``; used for ``q_sample``.
    reference_x0: (N_ref, D) tensor, optional
        Reference ``x_0`` rows for ``"forward_data"`` initialisation.
    generator / seed:
        Explicit randomness.  ``generator`` takes precedence.
    collect_diagnostics:
        When ``False``, skips the (cheap but non-trivial) diagnostic reductions.

    Returns
    -------
    samples: (B, M, |R|) tensor
        ``M = config.num_chains * config.num_samples`` retained free-coordinate
        states per anchor.  When ``S`` is all of ``{0..D-1}`` (the first
        ordering stage) ``R`` is empty and the returned tensor has shape
        ``(B, 1, 0)`` — no Langevin simulation is performed.
    diagnostics: :class:`LangevinDiagnostics`

    Raises
    ------
    LangevinDivergenceError
        If any chain becomes non-finite or exceeds ``config.max_abs_value``.
    """
    if x_t_S.ndim != 2:
        raise ValueError(f"x_t_S must have shape (B, |S|); got {tuple(x_t_S.shape)}")

    s_list = [int(i) for i in condition_indices]
    if len(set(s_list)) != len(s_list):
        raise ValueError("condition_indices contains duplicates")
    if x_t_S.shape[1] != len(s_list):
        raise ValueError(
            f"x_t_S has {x_t_S.shape[1]} columns but condition_indices has {len(s_list)} entries"
        )
    r_list = complement_indices(dim, s_list)

    device = getattr(score_adapter, "device", x_t_S.device)
    x_S = x_t_S.to(device=device, dtype=torch.float32)
    B = x_S.shape[0]
    n_free = len(r_list)

    # --- edge case: S is the whole index set -> nothing to sample ----------
    if n_free == 0:
        empty = torch.zeros((B, 1, 0), dtype=torch.float32, device=device)
        return empty, LangevinDiagnostics(num_retained=1, extras={"skipped": True})

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) if seed is not None else 0)

    s_idx = torch.as_tensor(s_list, dtype=torch.long, device=device)
    r_idx = torch.as_tensor(r_list, dtype=torch.long, device=device)

    C, M_per_chain = config.num_chains, config.num_samples
    n_rows = B * C

    # Row layout: anchor-major, i.e. row (b * C + c).
    x_S_rep = x_S.repeat_interleave(C, dim=0).contiguous()
    x_S_fixed = x_S_rep.clone()  # kept for the "conditioning never moves" check

    # --- initialisation ---------------------------------------------------
    if config.init == "normal":
        z = torch.empty((n_rows, n_free), device=device, dtype=torch.float32).normal_(
            generator=generator)
    else:  # forward_data
        if diffusion is None or reference_x0 is None:
            raise ValueError(
                "init='forward_data' requires both `diffusion` and `reference_x0`"
            )
        ref = reference_x0.to(device=device, dtype=torch.float32)
        if ref.ndim != 2 or ref.shape[1] != dim:
            raise ValueError(f"reference_x0 must have shape (N_ref, {dim}); got {tuple(ref.shape)}")
        pick = torch.randint(0, ref.shape[0], (n_rows,), device=device, generator=generator)
        x0_init = ref[pick]
        t_vec = torch.full((n_rows,), int(t), dtype=torch.int64, device=device)
        noise = torch.empty_like(x0_init).normal_(generator=generator)
        x_t_init = diffusion.q_sample(x_0=x0_init, t=t_vec, noise=noise)
        z = x_t_init[:, r_idx].contiguous()

    total_steps = config.burn_in + M_per_chain * config.thinning
    h = float(config.step_size)
    sqrt_2h = float(np.sqrt(2.0 * h))

    retained = torch.empty((n_rows, M_per_chain, n_free), device=device, dtype=torch.float32)
    n_retained = 0

    score_norm_sum, score_norm_max, update_norm_sum, n_score_evals = 0.0, 0.0, 0.0, 0
    n_clipped = 0

    for step in range(total_steps):
        grad = _score_R_chunked(
            score_adapter, x_S_rep, z, s_idx, r_idx, dim, t, config.chunk_size)

        if not torch.isfinite(grad).all():
            raise LangevinDivergenceError(
                f"non-finite score at ULA step {step}/{total_steps} (t={t}, |S|={len(s_list)}, "
                f"h={h}, init={config.init}, chains={C})"
            )

        norms = grad.norm(dim=1)
        score_norm_sum += float(norms.sum())
        score_norm_max = max(score_norm_max, float(norms.max()))
        n_score_evals += norms.numel()

        if config.score_norm_clip is not None:
            clip = float(config.score_norm_clip)
            factor = (clip / norms.clamp_min(1e-12)).clamp(max=1.0)
            n_clipped += int((factor < 1.0).sum())
            grad = grad * factor.unsqueeze(1)

        xi = torch.empty_like(z).normal_(generator=generator)
        update = h * grad + sqrt_2h * xi
        update_norm_sum += float(update.norm(dim=1).sum())
        z = z + update

        if not torch.isfinite(z).all():
            raise LangevinDivergenceError(
                f"non-finite state at ULA step {step}/{total_steps} (t={t}, |S|={len(s_list)}, "
                f"h={h}, init={config.init}, chains={C}); reduce --langevin-step-size"
            )
        max_abs = float(z.abs().max())
        if max_abs > config.max_abs_value:
            raise LangevinDivergenceError(
                f"exploding chain at ULA step {step}/{total_steps}: max|z|={max_abs:.3e} > "
                f"{config.max_abs_value:.3e} (t={t}, |S|={len(s_list)}, h={h}, "
                f"init={config.init}, chains={C}); reduce --langevin-step-size"
            )

        if step >= config.burn_in and (step - config.burn_in) % config.thinning == 0:
            if n_retained < M_per_chain:
                retained[:, n_retained] = z
                n_retained += 1

    if n_retained < M_per_chain:  # pragma: no cover - guarded by total_steps formula
        raise RuntimeError(f"retained {n_retained} < requested {M_per_chain} states")

    # conditioned coordinates must never have moved
    if not torch.equal(x_S_rep, x_S_fixed):
        raise RuntimeError("conditioned coordinates x_{t,S} were modified during sampling")

    # (B*C, M_per_chain, |R|) -> (B, C * M_per_chain, |R|)
    samples = retained.view(B, C * M_per_chain, n_free)

    diagnostics = LangevinDiagnostics(
        mean_score_norm=score_norm_sum / max(1, n_score_evals),
        max_score_norm=score_norm_max,
        mean_update_norm=update_norm_sum / max(1, n_score_evals),
        num_clipped_steps=n_clipped,
        num_retained=int(C * M_per_chain),
    )

    if collect_diagnostics:
        flat = retained.reshape(n_rows * M_per_chain, n_free)
        diagnostics.chain_mean = flat.mean(dim=0).cpu().numpy()
        diagnostics.chain_var = flat.var(dim=0, unbiased=False).cpu().numpy()

        half = max(1, M_per_chain // 2)
        first = retained[:, :half].mean(dim=(0, 1))
        second = retained[:, -half:].mean(dim=(0, 1))
        diagnostics.half_mean_shift = (second - first).cpu().numpy()

        per_chain = retained.view(B, C, M_per_chain, n_free)
        diagnostics.per_chain_mean = per_chain.mean(dim=2).reshape(-1, n_free).cpu().numpy()
        diagnostics.per_chain_var = (
            per_chain.var(dim=2, unbiased=False).reshape(-1, n_free).cpu().numpy())

        if M_per_chain >= 3:
            centred = retained - retained.mean(dim=1, keepdim=True)
            num = (centred[:, :-1] * centred[:, 1:]).mean(dim=(0, 1))
            den = (centred * centred).mean(dim=(0, 1)).clamp_min(1e-30)
            diagnostics.lag1_autocorr = (num / den).cpu().numpy()

    return samples, diagnostics
