"""Metrics matching Montagna et al. (2023), Appendix D, for comparability.

Definitions used by the paper:

* **TP** — a predicted edge present in the ground-truth *skeleton*
  (direction ignored for TP).
* **FP** — an edge in the predicted skeleton absent from the true skeleton.
* **FN** — a true skeleton edge missing from the prediction, **plus** predicted
  edges whose direction is reversed relative to the ground-truth DAG.
* **F1** = ``TP / (TP + 0.5 (FN + FP))``.
* **FNR-pi** — false negative rate of the *fully connected* DAG encoding the
  estimated order: the fraction of true edges ``i -> j`` for which the order
  places ``j`` before ``i``. Zero iff the order is consistent with every true
  edge.

FNR-pi scores the ordering stage alone; F1/FNR/FPR score the final edge set. We
report both, so the contribution of the parent-selection step is separable from
the contribution of the ordering step.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = ["fnr_pi", "edge_metrics", "summarize_run"]


def _skeleton(adj: np.ndarray) -> np.ndarray:
    a = (np.asarray(adj) != 0).astype(int)
    return ((a + a.T) > 0).astype(int)


def fnr_pi(topological_order: Sequence[int], true_adjacency: np.ndarray) -> float:
    """FNR of the fully connected DAG encoding ``topological_order``.

    Fraction of true edges ``i -> j`` that the order gets backwards. Returns
    ``nan`` when the ground truth has no edges.
    """
    order = [int(i) for i in topological_order]
    A = (np.asarray(true_adjacency) != 0).astype(int)
    pos = {node: k for k, node in enumerate(order)}
    edges = np.argwhere(A == 1)
    if edges.shape[0] == 0:
        return float("nan")
    violated = sum(1 for i, j in edges if pos[int(i)] > pos[int(j)])
    return float(violated) / float(edges.shape[0])


def edge_metrics(pred_adjacency: np.ndarray, true_adjacency: np.ndarray) -> Dict[str, float]:
    """F1 / FNR / FPR of a predicted DAG, using the paper's conventions.

    Returns a dict with ``tp``, ``fp``, ``fn``, ``f1``, ``fnr``, ``fpr``,
    ``num_pred_edges``, ``num_true_edges``.
    """
    P = (np.asarray(pred_adjacency) != 0).astype(int)
    T = (np.asarray(true_adjacency) != 0).astype(int)
    if P.shape != T.shape:
        raise ValueError(f"shape mismatch: pred {P.shape} vs true {T.shape}")
    d = P.shape[0]

    skel_P, skel_T = _skeleton(P), _skeleton(T)
    iu = np.triu_indices(d, k=1)
    sp, st = skel_P[iu], skel_T[iu]

    tp_skel = int(((sp == 1) & (st == 1)).sum())
    fp = int(((sp == 1) & (st == 0)).sum())
    fn_missing = int(((sp == 0) & (st == 1)).sum())

    # a skeleton-correct edge inferred with reversed direction counts as FN
    reversed_edges = 0
    for i, j in np.argwhere(T == 1):
        if P[int(i), int(j)] == 0 and P[int(j), int(i)] == 1:
            reversed_edges += 1

    tp = tp_skel - reversed_edges
    fn = fn_missing + reversed_edges

    denom = tp + 0.5 * (fn + fp)
    f1 = float(tp / denom) if denom > 0 else float("nan")

    num_true = int(st.sum())
    num_neg = int((st == 0).sum())
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "f1": f1,
        "fnr": float(fn / num_true) if num_true > 0 else float("nan"),
        "fpr": float(fp / num_neg) if num_neg > 0 else float("nan"),
        "num_pred_edges": int(P.sum()),
        "num_true_edges": int(T.sum()),
    }


def summarize_run(topological_order: Sequence[int],
                  pred_adjacency: np.ndarray,
                  true_adjacency: np.ndarray) -> Dict[str, object]:
    """Combine ordering and edge metrics for one run."""
    out = {"fnr_pi": fnr_pi(topological_order, true_adjacency)}
    out.update(edge_metrics(pred_adjacency, true_adjacency))
    return out
