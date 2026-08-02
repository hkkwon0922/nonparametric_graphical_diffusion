"""Reverse-DDPM posterior sampling of ``X_0 | X_t = x_t``.

This is a reusable refactor of ``experiments/compute_hessian_network.py::
sample_x0_given_xt_batch``.  The original script keeps working unchanged; this
module is the shared implementation used by the DAG-ordering pipeline (and can
be adopted by the script later without behavioural change).

The two-stage conditional sampler ``sample_x0_S_given_xt_S`` combines:

  Stage A  ULA draws of ``x_{t,R} | x_{t,S}``  (conditional_langevin)
  Stage B  full-dimensional reverse diffusion ``t -> 0`` of the merged state,
           keeping only the ``S`` coordinates of the resulting ``x_0``.

Marginalising over the Stage-A draws approximates ``X_{0,S} | X_{t,S} = x_{t,S}``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from .conditional_langevin import (
    LangevinConfig,
    complement_indices,
    sample_conditional_langevin,
)

__all__ = ["sample_x0_given_xt", "sample_x0_S_given_xt_S", "validate_diffusion_matches_model"]


def validate_diffusion_matches_model(model, diffusion, expected_dim: Optional[int] = None) -> int:
    """Check that ``model`` and ``diffusion`` are mutually consistent.

    Returns the model input dimension ``D``.  Raises ``ValueError`` on a
    mismatch with ``expected_dim`` or on a missing/invalid beta schedule.
    """
    dim = getattr(model, "input_dim", None)
    if dim is None:
        raise ValueError("model does not expose `input_dim`; cannot validate checkpoint config")
    dim = int(dim)
    if expected_dim is not None and dim != int(expected_dim):
        raise ValueError(
            f"model input_dim={dim} does not match expected data dimension {int(expected_dim)}; "
            "the checkpoint was trained on a different feature count"
        )
    timesteps = int(getattr(diffusion, "timesteps", 0))
    if timesteps <= 0 or len(diffusion.alphas_bar) != timesteps:
        raise ValueError(
            f"diffusion has an inconsistent schedule: timesteps={timesteps}, "
            f"len(alphas_bar)={len(diffusion.alphas_bar)}"
        )
    return dim


@torch.inference_mode()
def sample_x0_given_xt(
    model,
    diffusion,
    x_t: torch.Tensor,
    t: int,
    num_draws_per_xt: int = 1,
    device=None,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    apply_clamping: bool = False,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Draw ``L`` posterior samples of ``X_0 | X_t = x_t`` by full reverse diffusion.

    Runs ``t, t-1, ..., 0`` with ``diffusion.p_sample_step``.

    Parameters
    ----------
    x_t: (N, D) tensor of starting states.
    t: int, starting timestep.
    num_draws_per_xt: ``L``, independent reverse trajectories per row of ``x_t``.
    chunk_size: maximum number of rows integrated simultaneously (``None`` = all).
    apply_clamping: kept ``False`` by default — tabular data need not live in [-1, 1].

    Returns
    -------
    (N, L, D) tensor on ``device``.
    """
    if isinstance(x_t, np.ndarray):
        x_t = torch.from_numpy(x_t)
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(0)
    if x_t.ndim != 2:
        raise ValueError(f"x_t must have shape (N, D); got {tuple(x_t.shape)}")

    if device is None:
        device = x_t.device if x_t.is_cuda else next(model.parameters()).device
    device = torch.device(device)
    x_t = x_t.to(device=device, dtype=torch.float32)

    n, dim = x_t.shape
    L = int(num_draws_per_xt)
    if L < 1:
        raise ValueError("num_draws_per_xt must be >= 1")
    t_start = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
    if t_start < 0 or t_start >= int(diffusion.timesteps):
        raise ValueError(f"t={t_start} out of range [0, {int(diffusion.timesteps) - 1}]")

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) if seed is not None else 0)

    x_rep = x_t.repeat_interleave(L, dim=0).contiguous()  # (N*L, D)
    total = x_rep.shape[0]
    step = total if (chunk_size is None or chunk_size >= total) else int(chunk_size)

    out = torch.empty_like(x_rep)
    for start in range(0, total, step):
        end = min(start + step, total)
        chunk = x_rep[start:end].clone()
        t_tensor = torch.full((end - start,), t_start, dtype=torch.int64, device=device)
        for ti in range(t_start, -1, -1):
            t_tensor.fill_(ti)
            chunk = diffusion.p_sample_step(
                denoise_fn=model, x_t=chunk, t=t_tensor,
                clip_denoised=False, return_pred=False, generator=generator,
            )
        out[start:end] = chunk

    x0 = out.view(n, L, dim)
    if apply_clamping:
        x0 = x0.clamp(-1.0, 1.0)
    if not torch.isfinite(x0).all():
        raise FloatingPointError(
            f"reverse diffusion from t={t_start} produced non-finite x_0 samples"
        )
    return x0


@torch.inference_mode()
def sample_x0_S_given_xt_S(
    model,
    diffusion,
    score_adapter,
    x_t_S: torch.Tensor,
    condition_indices: Sequence[int],
    dim: int,
    t: int,
    langevin_config: LangevinConfig,
    reverse_draws_per_xt: int = 1,
    reference_x0: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None,
    chunk_size: Optional[int] = None,
    device=None,
):
    """Two-stage conditional posterior sampler ``X_{0,S} | X_{t,S} = x_{t,S}``.

    Stage A: ULA draws of ``x_{t,R} | x_{t,S}`` using the ``R`` rows of the full
    learned score.
    Stage B: merge with the *fixed* ``x_{t,S}``, run full ``D``-dimensional
    reverse diffusion ``t -> 0``, keep the ``S`` coordinates of ``x_0``.

    The true ``x_{t,R}`` that may have generated an anchor is never used here —
    only the Langevin draws are, so the estimator stays honest.

    Parameters
    ----------
    x_t_S: (B, |S|) conditioning values.
    condition_indices: ``S`` in original ``D``-dimensional indexing.
    reverse_draws_per_xt: ``L``, reverse trajectories per Langevin state.

    Returns
    -------
    x0_S_samples: (B, M_total, |S|) where
        ``M_total = num_chains * num_samples * reverse_draws_per_xt``
        (and ``= reverse_draws_per_xt`` when ``R`` is empty).
    diagnostics: :class:`~models.dag_diffusion.conditional_langevin.LangevinDiagnostics`
    """
    if device is None:
        device = getattr(score_adapter, "device", None) or next(model.parameters()).device
    device = torch.device(device)

    s_list = [int(i) for i in condition_indices]
    r_list = complement_indices(dim, s_list)
    x_S = x_t_S.to(device=device, dtype=torch.float32)
    B = x_S.shape[0]

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) if seed is not None else 0)

    # ---- Stage A ---------------------------------------------------------
    z_R, diagnostics = sample_conditional_langevin(
        score_adapter=score_adapter,
        x_t_S=x_S,
        condition_indices=s_list,
        dim=dim,
        t=t,
        config=langevin_config,
        diffusion=diffusion,
        reference_x0=reference_x0,
        generator=generator,
    )
    M = z_R.shape[1]  # states per anchor

    # ---- merge -----------------------------------------------------------
    s_idx = torch.as_tensor(s_list, dtype=torch.long, device=device)
    r_idx = torch.as_tensor(r_list, dtype=torch.long, device=device)

    x_full = torch.empty((B * M, dim), dtype=torch.float32, device=device)
    x_full[:, s_idx] = x_S.repeat_interleave(M, dim=0)
    if len(r_list) > 0:
        x_full[:, r_idx] = z_R.reshape(B * M, len(r_list))

    # ---- Stage B ---------------------------------------------------------
    x0 = sample_x0_given_xt(
        model=model, diffusion=diffusion, x_t=x_full, t=t,
        num_draws_per_xt=reverse_draws_per_xt, device=device,
        generator=generator, apply_clamping=False, chunk_size=chunk_size,
    )  # (B*M, L, D)

    L = x0.shape[1]
    x0_S = x0[:, :, s_idx].reshape(B, M * L, len(s_list))
    return x0_S, diagnostics
