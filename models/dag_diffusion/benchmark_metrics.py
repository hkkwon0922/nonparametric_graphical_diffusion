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

__all__ = ["fnr_pi", "edge_metrics", "summarize_run", "ordering_metrics",
           "kendall_tau_vs_valid_orders"]


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


# ---------------------------------------------------------------------------
# ordering-stage metrics
# ---------------------------------------------------------------------------
def _ancestor_matrix(adj: np.ndarray) -> np.ndarray:
    """Transitive closure: ``R[i, j] = 1`` iff ``i`` is an ancestor of ``j``."""
    R = (np.asarray(adj) != 0).astype(bool).copy()
    d = R.shape[0]
    for k in range(d):  # Floyd-Warshall style closure, O(d^3)
        R |= np.outer(R[:, k], R[k, :])
    return R.astype(int)


def kendall_tau_vs_valid_orders(topological_order: Sequence[int],
                                true_adjacency: np.ndarray) -> float:
    """Kendall tau-like agreement restricted to pairs the DAG actually constrains.

    A DAG generally admits many valid topological orders, so comparing against
    one arbitrary reference permutation is misleading. Only **ancestor pairs**
    are constrained: if ``i`` is an ancestor of ``j`` then every valid order puts
    ``i`` first. This returns the fraction of such pairs the estimate gets right,
    rescaled to ``[-1, 1]`` (``1`` = all constrained pairs correct, ``0`` =
    coin-flip, ``-1`` = all reversed). Unconstrained pairs are ignored, so a
    perfect score is attainable by *any* valid order.
    """
    order = [int(i) for i in topological_order]
    R = _ancestor_matrix(true_adjacency)
    pos = {node: k for k, node in enumerate(order)}
    pairs = np.argwhere(R == 1)
    if pairs.shape[0] == 0:
        return float("nan")
    correct = sum(1 for i, j in pairs if pos[int(i)] < pos[int(j)])
    return float(2.0 * correct / pairs.shape[0] - 1.0)


def ordering_metrics(topological_order: Sequence[int],
                     true_adjacency: np.ndarray) -> Dict[str, float]:
    """Performance of the **ordering stage alone**, independent of edge selection.

    Returns
    -------
    ``fnr_pi``
        Fraction of true edges the order reverses (the paper's FNR-pi). 0 is
        perfect; a random order gives ~0.5.
    ``edge_accuracy``
        ``1 - fnr_pi``, i.e. the fraction of true edges the order is consistent
        with. Reported because "higher is better" is easier to read alongside F1.
    ``ancestor_accuracy``
        Same idea over the transitive closure: fraction of *ancestor* pairs
        ordered correctly. Stricter than ``edge_accuracy`` on deep graphs, since
        it also scores indirect constraints.
    ``kendall_tau``
        ``ancestor_accuracy`` rescaled to ``[-1, 1]`` (see
        :func:`kendall_tau_vs_valid_orders`).
    ``num_violated_edges`` / ``num_true_edges``
        Raw counts behind ``fnr_pi``.
    ``is_valid_order``
        ``True`` iff the order is consistent with **every** true edge, i.e. it is
        one of the DAG's valid topological orders.
    """
    order = [int(i) for i in topological_order]
    A = (np.asarray(true_adjacency) != 0).astype(int)
    pos = {node: k for k, node in enumerate(order)}

    edges = np.argwhere(A == 1)
    n_edges = int(edges.shape[0])
    violated = sum(1 for i, j in edges if pos[int(i)] > pos[int(j)])
    fnr = float(violated) / n_edges if n_edges > 0 else float("nan")

    R = _ancestor_matrix(A)
    anc = np.argwhere(R == 1)
    anc_correct = sum(1 for i, j in anc if pos[int(i)] < pos[int(j)])
    anc_acc = float(anc_correct) / anc.shape[0] if anc.shape[0] > 0 else float("nan")

    return {
        "fnr_pi": fnr,
        "edge_accuracy": (1.0 - fnr) if n_edges > 0 else float("nan"),
        "ancestor_accuracy": anc_acc,
        "kendall_tau": (2.0 * anc_acc - 1.0) if anc.shape[0] > 0 else float("nan"),
        "num_violated_edges": int(violated),
        "num_true_edges": n_edges,
        "num_ancestor_pairs": int(anc.shape[0]),
        "is_valid_order": bool(violated == 0),
    }
