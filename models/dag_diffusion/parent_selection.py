"""Parent (edge) selection from off-diagonal Tweedie Hessian entries.

Once a topological order is fixed, an edge ``i -> l`` may only exist if ``i``
precedes ``l``. Among those candidates, the off-diagonal Hessian entry carries
the parent signal: following DAS (Montagna et al., 2023, Lemma 3), for an ANM
with Gaussian noise and a leaf ``l``,

    E[ |d_{x_j} s_l(X)| ] != 0   <=>   X_j in PA_l.

Two selectors are provided.

``das_hypothesis_test``
    The DAS-style rule. For each ordered candidate pair, test
    ``H0: E[H_{j,l}] = 0`` across anchors with a two-sided one-sample t-test and
    keep the edge when H0 is rejected at level ``alpha``. Optionally applies a
    Benjamini-Hochberg FDR correction, since one test is run per candidate pair.

``multitime_cluster``
    A heuristic that uses the *whole timestep profile* rather than one ``t``.
    Each candidate pair contributes a vector ``(|H_{ij}(t_1)|, ..., |H_{ij}(t_T)|)``;
    the profiles are **standardized per timestep across pairs** (the same
    per-``t`` z-scoring used in the notebook, so no single noisy ``t`` dominates)
    and then split into "edge" / "non-edge" by 2-means. The cluster with the
    higher mean is taken as the edge cluster. This mirrors the clustering step
    the repository's existing `DDPMEstimator` uses for undirected graphs.

Both return a directed adjacency restricted to the supplied order, so the output
is acyclic by construction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "candidate_pairs_from_order",
    "das_hypothesis_test",
    "multitime_cluster",
    "benjamini_hochberg",
]


def candidate_pairs_from_order(topological_order: Sequence[int]) -> List[tuple]:
    """All ``(parent, child)`` pairs admitted by ``topological_order``.

    Returns ``d(d-1)/2`` pairs ``(i, j)`` with ``i`` preceding ``j``.
    """
    order = [int(i) for i in topological_order]
    return [(order[a], order[b])
            for a in range(len(order)) for b in range(a + 1, len(order))]


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return a boolean mask of hypotheses rejected under BH-FDR at ``alpha``."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    reject = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.flatnonzero(passed).max()
        reject[order[: cutoff + 1]] = True
    return reject


def das_hypothesis_test(
    hessian_by_pair: Dict[tuple, np.ndarray],
    topological_order: Sequence[int],
    alpha: float = 0.05,
    fdr_correction: bool = True,
) -> Dict[str, object]:
    """DAS-style edge selection by testing ``E[H_{i,j}] = 0`` across anchors.

    Parameters
    ----------
    hessian_by_pair:
        ``{(i, j): (B,) array}`` of per-anchor **signed** Hessian entries
        ``H_{i,j}`` at the selected timestep. Only pairs admitted by the order
        need be present.
    topological_order: estimated order, sources first.
    alpha: significance level.
    fdr_correction: apply Benjamini-Hochberg across all candidate pairs.

    Returns
    -------
    dict with ``adjacency`` ``(d, d)``, ``pvalues``, ``statistics`` and the
    candidate pair list.
    """
    order = [int(i) for i in topological_order]
    d = len(order)
    pairs = candidate_pairs_from_order(order)

    stats, pvals, used = [], [], []
    for (i, j) in pairs:
        vals = hessian_by_pair.get((i, j))
        if vals is None:
            vals = hessian_by_pair.get((j, i))
        if vals is None:
            continue
        vals = np.asarray(vals, dtype=float).ravel()
        n = vals.size
        if n < 2:
            continue
        sd = vals.std(ddof=1)
        if sd < 1e-30:
            t_stat = np.inf if abs(vals.mean()) > 0 else 0.0
            p = 0.0 if np.isinf(t_stat) else 1.0
        else:
            t_stat = vals.mean() / (sd / np.sqrt(n))
            # two-sided p-value from the t distribution with n-1 dof
            try:
                from scipy import stats as _st
                p = float(2.0 * _st.t.sf(abs(t_stat), df=n - 1))
            except ImportError:  # pragma: no cover - normal approximation
                from math import erfc, sqrt
                p = float(erfc(abs(t_stat) / sqrt(2.0)))
        stats.append(float(t_stat))
        pvals.append(float(p))
        used.append((i, j))

    pvals = np.asarray(pvals, dtype=float)
    if fdr_correction:
        reject = benjamini_hochberg(pvals, alpha=alpha)
    else:
        reject = pvals <= alpha

    adjacency = np.zeros((d, d), dtype=int)
    for keep, (i, j) in zip(reject, used):
        if keep:
            adjacency[i, j] = 1

    return {
        "adjacency": adjacency,
        "pairs": used,
        "pvalues": pvals.tolist(),
        "statistics": stats,
        "alpha": float(alpha),
        "fdr_correction": bool(fdr_correction),
        "method": "das_hypothesis_test",
    }


def multitime_cluster(
    abs_hessian_by_pair: Dict[tuple, np.ndarray],
    topological_order: Sequence[int],
    standardize: bool = True,
    seed: int = 0,
    n_init: int = 10,
    transform: str = "rank",
) -> Dict[str, object]:
    """Cluster per-timestep ``|H_{ij}|`` profiles into edge / non-edge.

    Parameters
    ----------
    abs_hessian_by_pair:
        ``{(i, j): (T,) array}`` of ``E_anchors |H_{i,j}(t)|`` across the
        timestep grid, one profile per candidate pair.
    topological_order: estimated order, sources first.
    standardize:
        Apply the per-timestep normalisation given by ``transform``. Without any
        normalisation the largest-magnitude timesteps dominate the Euclidean
        distance and the clustering degenerates to thresholding a single ``t``.
    transform:
        ``"rank"`` (default) replaces each pair's value at a timestep by its
        rank among pairs, rescaled to ``[0, 1]``; ``"zscore"`` uses the plain
        per-timestep z-score.

        Ranks are the default because ``|H_{ij}|`` is heavy-tailed across pairs:
        with z-scores, k-means splits off a handful of extreme pairs instead of
        finding the edge/non-edge boundary (measured on ER-10 dense: F1 0.30 for
        z-score vs 0.77 for ranks, with the same features and seeds). Ranks are
        invariant to any monotone per-timestep rescaling, so no single timestep
        or outlier can dominate.
    seed, n_init: k-means restarts; the best silhouette score is kept.

    Returns
    -------
    dict with ``adjacency`` ``(d, d)``, cluster labels and the feature matrix.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if transform not in ("rank", "zscore"):
        raise ValueError(f"transform must be 'rank' or 'zscore'; got {transform!r}")

    order = [int(i) for i in topological_order]
    d = len(order)
    pairs = candidate_pairs_from_order(order)

    rows, used = [], []
    for (i, j) in pairs:
        prof = abs_hessian_by_pair.get((i, j))
        if prof is None:
            prof = abs_hessian_by_pair.get((j, i))
        if prof is None:
            continue
        rows.append(np.asarray(prof, dtype=float).ravel())
        used.append((i, j))

    adjacency = np.zeros((d, d), dtype=int)
    if len(used) < 2:
        return {"adjacency": adjacency, "pairs": used, "labels": [],
                "method": "multitime_cluster", "note": "too few candidate pairs to cluster"}

    F = np.vstack(rows)  # (num_pairs, T)
    F_raw = F.copy()
    if standardize:
        if transform == "rank":
            # per-timestep rank across pairs, rescaled to [0, 1]
            denom = max(F.shape[0] - 1, 1)
            F = np.argsort(np.argsort(F, axis=0), axis=0) / denom
        elif transform == "zscore":
            mu = F.mean(axis=0, keepdims=True)
            sd = F.std(axis=0, ddof=1, keepdims=True)
            sd = np.where(sd < 1e-12, 1.0, sd)
            F = (F - mu) / sd
        else:
            raise ValueError(f"transform must be 'rank' or 'zscore'; got {transform!r}")

    best_labels, best_score = None, -np.inf
    for r in range(int(n_init)):
        km = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=int(seed) + r)
        labels = km.fit_predict(F)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(F, labels)
        if score > best_score:
            best_score, best_labels = score, labels
    if best_labels is None:  # pragma: no cover - degenerate features
        return {"adjacency": adjacency, "pairs": used, "labels": [],
                "method": "multitime_cluster", "note": "k-means failed to split"}

    # the cluster with the larger mean magnitude is the "edge" cluster
    means = [F[best_labels == k].mean() for k in range(2)]
    edge_cluster = int(np.argmax(means))
    for lab, (i, j) in zip(best_labels, used):
        if lab == edge_cluster:
            adjacency[i, j] = 1

    return {
        "adjacency": adjacency,
        "pairs": used,
        "labels": best_labels.tolist(),
        "silhouette": float(best_score),
        "edge_cluster": edge_cluster,
        "cluster_means": [float(m) for m in means],
        "standardize": bool(standardize),
        "transform": transform,
        "features_raw": F_raw,
        "method": "multitime_cluster",
    }
