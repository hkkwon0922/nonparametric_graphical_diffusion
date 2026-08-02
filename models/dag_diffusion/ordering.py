"""Sequential leaf-removal DAG ordering from conditional Tweedie Hessians.

Method
------
At each stage, with ``S`` the set of remaining original variable indices:

1. Draw ``B`` conditioning anchors ``x_{t,S}`` by forward-diffusing ``B`` rows of
   the anchor dataset to time ``t_order`` and restricting to ``S``.
2. For each anchor, approximate ``Cov(X_{0,S} | X_{t,S} = x_{t,S})`` with the
   two-stage sampler (conditional Langevin for the free block, then full
   reverse diffusion).
3. Form the Tweedie Hessian diagonal ``H_ii`` for each ``i in S``.
4. Score each node by the variance of its signed diagonal Hessian **across
   anchors**, ``V_i = Var_b(H_diag[b, i])``, and remove ``argmin_i V_i``.

Repeat until one node remains.  ``leaf_order`` records removals (sinks first);
``topological_order = reverse(leaf_order)`` runs source -> sink.

Scientific caveat
-----------------
The SCORE theorem ("a leaf of a nonlinear additive-noise SCM has constant
diagonal score Hessian") is stated at ``t = 0``, on the *clean* density.  This
implementation evaluates the criterion at a strictly positive diffusion time
``t_order > 0``, where the Gaussian smoothing of ``p_t`` mixes contributions
across variables.  The criterion is therefore **experimental**: it is not
automatically implied by the ``t = 0`` guarantee, and small ``t_order`` is
expected to be closer to the regime where the theory applies (at the cost of a
larger reverse-diffusion variance).

The returned ``fully_connected_order_dag`` is a *complete* DAG consistent with
the estimated order — a re-encoding of the ordering, **not** an estimated sparse
skeleton.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .conditional_langevin import LangevinConfig
from .reverse_posterior import sample_x0_S_given_xt_S, validate_diffusion_matches_model
from .score_adapter import DDPMScoreAdapter
from .tweedie_hessian import estimate_tweedie_hessian, estimate_tweedie_hessian_diagonal

__all__ = [
    "DAGOrderingConfig",
    "DAGOrderingResult",
    "ConditionalDiffusionDAGOrderEstimator",
    "fully_connected_dag_from_order",
]


def fully_connected_dag_from_order(topological_order: Sequence[int]) -> np.ndarray:
    """Complete DAG consistent with ``topological_order`` -> ``(D, D)`` int array.

    ``A[i, j] = 1`` iff ``i`` precedes ``j`` in the order.  This encodes the
    ordering only; it is **not** an estimated sparse structure.
    """
    order = [int(i) for i in topological_order]
    d = len(order)
    adj = np.zeros((d, d), dtype=int)
    for pos_i, i in enumerate(order):
        for j in order[pos_i + 1:]:
            adj[i, j] = 1
    return adj


@dataclass
class DAGOrderingConfig:
    """Configuration of :class:`ConditionalDiffusionDAGOrderEstimator`.

    Attributes
    ----------
    t_order: the single diffusion timestep at which the criterion is evaluated.
        Required; multi-timestep averaging is deliberately not performed since
        no aggregation rule is defined.
    num_anchors: ``B``, conditioning anchors per stage.
    langevin: :class:`LangevinConfig` for the Stage-A sampler.
    reverse_draws_per_xt: reverse trajectories per Langevin state.
    anchor_chunk_size: anchors processed per inner batch.
    sampling_chunk_size: rows integrated simultaneously in reverse diffusion.
    tie_tolerance: relative tolerance for treating two criterion values as tied;
        ties break toward the smallest original index.
    seed: base seed; per-stage seeds derive deterministically from it.
    compute_full_covariance: also store the full (B,|S|,|S|) covariance/Hessian
        per stage.  Diagnostics only; off by default for memory.
    """

    t_order: int
    num_anchors: int = 32
    langevin: LangevinConfig = field(default_factory=LangevinConfig)
    reverse_draws_per_xt: int = 1
    anchor_chunk_size: Optional[int] = None
    sampling_chunk_size: Optional[int] = 4096
    tie_tolerance: float = 1e-12
    seed: int = 120
    compute_full_covariance: bool = False
    verbose: bool = True


@dataclass
class DAGOrderingResult:
    """Output of the ordering estimator.

    Attributes
    ----------
    leaf_order: removal order, sinks first (length ``D``).
    topological_order: ``reverse(leaf_order)``, sources first.
    fully_connected_order_dag: ``(D, D)`` complete DAG encoding the order.
    stage_records: one dict per stage with the full diagnostic payload.
    criterion_by_stage: per-stage ``{original_index: V_i}``.
    """

    leaf_order: List[int]
    topological_order: List[int]
    fully_connected_order_dag: np.ndarray
    stage_records: List[Dict[str, Any]]
    criterion_by_stage: List[Dict[int, float]]
    t_order: int
    feature_names: List[str]
    warnings: List[str] = field(default_factory=list)
    runtime_seconds: float = 0.0
    peak_cuda_memory_bytes: Optional[int] = None

    def to_json_dict(self) -> dict:
        """JSON-serialisable summary (the heavy per-stage arrays are excluded)."""
        return {
            "leaf_order": [int(i) for i in self.leaf_order],
            "topological_order": [int(i) for i in self.topological_order],
            "fully_connected_order_dag": self.fully_connected_order_dag.tolist(),
            "t_order": int(self.t_order),
            "feature_names": list(self.feature_names),
            "criterion_by_stage": [
                {str(k): float(v) for k, v in stage.items()} for stage in self.criterion_by_stage
            ],
            "warnings": list(self.warnings),
            "runtime_seconds": float(self.runtime_seconds),
            "peak_cuda_memory_bytes": (
                int(self.peak_cuda_memory_bytes)
                if self.peak_cuda_memory_bytes is not None else None
            ),
            "note": (
                "fully_connected_order_dag encodes the estimated topological order only; "
                "it is NOT an estimated sparse DAG skeleton."
            ),
        }


class ConditionalDiffusionDAGOrderEstimator:
    """Estimate a topological order via conditional Tweedie Hessian variance.

    Parameters
    ----------
    config: :class:`DAGOrderingConfig`

    Example
    -------
    >>> est = ConditionalDiffusionDAGOrderEstimator(DAGOrderingConfig(t_order=10))
    >>> result = est.fit(X, model, diffusion)          # doctest: +SKIP
    >>> result.topological_order                        # doctest: +SKIP
    """

    def __init__(self, config: DAGOrderingConfig):
        self.config = config
        self.result: Optional[DAGOrderingResult] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        X,
        model,
        diffusion,
        device=None,
        feature_names: Optional[Sequence[str]] = None,
        anchor_data=None,
        reference_x0=None,
    ) -> DAGOrderingResult:
        """Run the full leaf-removal ordering.

        Parameters
        ----------
        X: (N, D) array/tensor of training data (used for anchors and Langevin
            initialisation unless overridden).
        model: trained ``D``-dimensional denoising network.
        diffusion: matching ``GaussianDiffusion``.
        device: torch device; defaults to the model's device.
        feature_names: optional ``D`` names for reporting.
        anchor_data: (N_a, D) rows to forward-diffuse into anchors (default ``X``).
        reference_x0: (N_r, D) rows for ``forward_data`` Langevin init (default ``X``).

        Returns
        -------
        :class:`DAGOrderingResult`
        """
        cfg = self.config
        start_time = time.time()

        if device is None:
            device = next(model.parameters()).device
        device = torch.device(device)

        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32) if not isinstance(X, torch.Tensor) \
            else X.to(dtype=torch.float32)
        X_t = X_t.to(device)
        if X_t.ndim != 2:
            raise ValueError(f"X must have shape (N, D); got {tuple(X_t.shape)}")
        n_samples, dim = X_t.shape

        validate_diffusion_matches_model(model, diffusion, expected_dim=dim)

        t_order = int(cfg.t_order)
        if t_order < 0 or t_order >= int(diffusion.timesteps):
            raise ValueError(
                f"t_order={t_order} out of range [0, {int(diffusion.timesteps) - 1}]")

        anchors_src = X_t if anchor_data is None else torch.as_tensor(
            np.asarray(anchor_data), dtype=torch.float32).to(device)
        ref_src = X_t if reference_x0 is None else torch.as_tensor(
            np.asarray(reference_x0), dtype=torch.float32).to(device)

        names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(dim)]
        if len(names) != dim:
            raise ValueError(f"feature_names must have {dim} entries; got {len(names)}")

        score_adapter = DDPMScoreAdapter(model=model, diffusion=diffusion, device=device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        warnings: List[str] = [
            "Leaf criterion is evaluated at t_order > 0 on the NOISED density; the SCORE "
            "t=0 guarantee does not automatically transfer to positive diffusion times.",
        ]

        # ---- fixed anchor set, shared across all stages -------------------
        # The SAME full-dimensional x_t anchors are reused at every stage and
        # merely restricted to the current S, so candidate nodes within a stage
        # are never compared across different noise realisations.
        B = int(min(cfg.num_anchors, anchors_src.shape[0])) if anchors_src.shape[0] > 0 else 0
        if B < 2:
            raise ValueError(
                f"need at least 2 anchors for a cross-anchor variance criterion; got B={B}"
            )
        if B < cfg.num_anchors:
            warnings.append(
                f"num_anchors reduced from {cfg.num_anchors} to {B} (anchor dataset has "
                f"{anchors_src.shape[0]} rows)."
            )

        anchor_gen = torch.Generator(device=device)
        anchor_gen.manual_seed(int(cfg.seed))
        anchor_row_idx = torch.randperm(
            anchors_src.shape[0], device=device, generator=anchor_gen)[:B]
        x0_anchor = anchors_src[anchor_row_idx]
        t_vec = torch.full((B,), t_order, dtype=torch.int64, device=device)
        anchor_noise = torch.empty_like(x0_anchor).normal_(generator=anchor_gen)
        x_t_anchor_full = diffusion.q_sample(x_0=x0_anchor, t=t_vec, noise=anchor_noise)  # (B, D)

        remaining: List[int] = list(range(dim))
        leaf_order: List[int] = []
        stage_records: List[Dict[str, Any]] = []
        criterion_by_stage: List[Dict[int, float]] = []

        stage = 0
        while len(remaining) > 1:
            stage_start = time.time()
            S = list(remaining)
            stage_seed = int(cfg.seed) + 1000 * (stage + 1)

            if cfg.verbose:
                print(
                    f"[stage {stage:02d}] |S|={len(S)} t={t_order} anchors={B} "
                    f"S={S}", flush=True
                )

            x_t_S = x_t_anchor_full[:, torch.as_tensor(S, dtype=torch.long, device=device)]

            h_diag, stage_diag = self._stage_hessian_diagonal(
                model=model, diffusion=diffusion, score_adapter=score_adapter,
                x_t_S=x_t_S, S=S, dim=dim, t=t_order,
                reference_x0=ref_src, stage_seed=stage_seed, device=device,
            )  # h_diag: (B, |S|) float64

            if not torch.isfinite(h_diag).all():
                raise FloatingPointError(
                    f"non-finite Hessian diagonal at stage {stage} (|S|={len(S)}, t={t_order})"
                )

            # V_i = sample variance ACROSS anchors of the signed diagonal Hessian
            criterion = h_diag.var(dim=0, unbiased=True)          # (|S|,)
            crit_np = criterion.detach().cpu().numpy()

            local_pick = self._argmin_with_tie_break(crit_np, S, cfg.tie_tolerance)
            selected_leaf = int(S[local_pick])

            record = {
                "stage": stage,
                "remaining": [int(i) for i in S],
                "t_order": t_order,
                "anchor_row_indices": anchor_row_idx.detach().cpu().numpy().astype(np.int64),
                "hessian_diag": h_diag.detach().cpu().numpy(),   # (B, |S|)
                "criterion": crit_np,                             # (|S|,)
                "selected_leaf": selected_leaf,
                "selected_local_index": int(local_pick),
                "stage_seed": stage_seed,
                "base_seed": int(cfg.seed),
                "runtime_seconds": time.time() - stage_start,
                "langevin_diagnostics": stage_diag,
            }
            if cfg.compute_full_covariance and "hessian_full" in stage_diag:
                record["hessian_full"] = stage_diag.pop("hessian_full")
                record["cov_full"] = stage_diag.pop("cov_full")

            stage_records.append(record)
            criterion_by_stage.append({int(S[j]): float(crit_np[j]) for j in range(len(S))})

            if cfg.verbose:
                print(
                    f"[stage {stage:02d}] criterion min={crit_np.min():.6g} "
                    f"max={crit_np.max():.6g} -> leaf={selected_leaf} ({names[selected_leaf]}) "
                    f"[{record['runtime_seconds']:.1f}s]", flush=True
                )

            leaf_order.append(selected_leaf)
            remaining.remove(selected_leaf)
            stage += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # |S| == 1: append the survivor without any further sampling.
        leaf_order.append(int(remaining[0]))

        if sorted(leaf_order) != list(range(dim)):
            raise RuntimeError(
                f"leaf_order is not a permutation of range({dim}): {leaf_order}")

        topological_order = list(reversed(leaf_order))
        peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None

        self.result = DAGOrderingResult(
            leaf_order=[int(i) for i in leaf_order],
            topological_order=[int(i) for i in topological_order],
            fully_connected_order_dag=fully_connected_dag_from_order(topological_order),
            stage_records=stage_records,
            criterion_by_stage=criterion_by_stage,
            t_order=t_order,
            feature_names=names,
            warnings=warnings,
            runtime_seconds=time.time() - start_time,
            peak_cuda_memory_bytes=peak_mem,
        )
        return self.result

    # ------------------------------------------------------------------
    def _stage_hessian_diagonal(self, model, diffusion, score_adapter, x_t_S, S, dim, t,
                                reference_x0, stage_seed, device):
        """Hessian diagonal for one stage -> ``((B, |S|) float64, diagnostics dict)``."""
        cfg = self.config
        B = x_t_S.shape[0]
        chunk = cfg.anchor_chunk_size or B
        diag_parts, diag_records = [], []
        hess_parts, cov_parts = [], []

        # First stage: S is the whole index set, so R is empty and the Langevin
        # stage contributes exactly one state per anchor. The posterior-sample
        # budget must then come entirely from independent reverse trajectories,
        # so fold the would-be Langevin multiplicity into the reverse draws.
        # Every later stage keeps the configured reverse_draws_per_xt.
        reverse_draws = int(cfg.reverse_draws_per_xt)
        if len(S) == dim:
            langevin_states = cfg.langevin.num_chains * cfg.langevin.num_samples
            reverse_draws = max(2, reverse_draws * langevin_states)

        for start in range(0, B, chunk):
            end = min(start + chunk, B)
            gen = torch.Generator(device=device)
            gen.manual_seed(stage_seed + start)

            x0_S, lang_diag = sample_x0_S_given_xt_S(
                model=model, diffusion=diffusion, score_adapter=score_adapter,
                x_t_S=x_t_S[start:end], condition_indices=S, dim=dim, t=t,
                langevin_config=cfg.langevin,
                reverse_draws_per_xt=reverse_draws,
                reference_x0=reference_x0, generator=gen,
                chunk_size=cfg.sampling_chunk_size, device=device,
            )  # (b, M_total, |S|)

            if x0_S.shape[1] < 2:
                raise ValueError(
                    f"only {x0_S.shape[1]} posterior sample(s) per anchor; the covariance "
                    "estimator needs >= 2. Increase --langevin-samples, --num-chains, "
                    "or --reverse-draws-per-xt."
                )

            if cfg.compute_full_covariance:
                hess, cov = estimate_tweedie_hessian(x0_S, diffusion, t)
                hess_parts.append(hess.cpu().numpy())
                cov_parts.append(cov.cpu().numpy())
                diag_parts.append(torch.diagonal(hess, dim1=-2, dim2=-1))
            else:
                diag_parts.append(estimate_tweedie_hessian_diagonal(x0_S, diffusion, t))

            diag_records.append(lang_diag.to_dict())
            del x0_S

        diagnostics: Dict[str, Any] = {
            "langevin": diag_records,
            "reverse_draws_per_xt": reverse_draws,
            "free_block_empty": len(S) == dim,
        }
        if cfg.compute_full_covariance:
            diagnostics["hessian_full"] = np.concatenate(hess_parts, axis=0)
            diagnostics["cov_full"] = np.concatenate(cov_parts, axis=0)
        return torch.cat(diag_parts, dim=0), diagnostics

    # ------------------------------------------------------------------
    @staticmethod
    def _argmin_with_tie_break(criterion: np.ndarray, S: Sequence[int], tol: float) -> int:
        """Index into ``S`` of the minimiser; ties break to the smallest original index."""
        crit = np.asarray(criterion, dtype=np.float64)
        best = float(crit.min())
        scale = max(abs(best), 1.0)
        tied = np.flatnonzero(crit <= best + tol * scale)
        # S is ascending in original indexing, so the first tied local index is
        # also the smallest original index; select explicitly to be safe.
        originals = [int(S[j]) for j in tied]
        return int(tied[int(np.argmin(originals))])
