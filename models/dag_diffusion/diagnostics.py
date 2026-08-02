"""Evaluation and diagnostics for an estimated topological order.

Only *order*-level metrics are reported.  This phase of the method does not
infer a sparse edge set, so SHD and edge-F1 are deliberately **not** computed —
they would be meaningless against a complete order-encoding DAG.

Metrics
-------
``order_fnr``
    Fraction of true edges ``i -> j`` that the estimated order gets backwards,
    i.e. ``position(i) > position(j)``.  0 means the order is compatible with
    every true edge; there may be many such orders.
``stagewise_leaf_validity``
    At each stage, whether the removed node was a valid leaf (sink) of the
    subgraph induced on the then-remaining set ``S`` — that is, it has no
    outgoing edge to another node of ``S``.  Several nodes can be valid leaves
    simultaneously, so this is the right check rather than comparing against a
    single arbitrary reference order.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "validate_adjacency",
    "is_acyclic",
    "order_fnr",
    "stagewise_leaf_validity",
    "evaluate_ordering",
]


def is_acyclic(adjacency: np.ndarray) -> bool:
    """True if the directed graph (``A[i, j] = 1`` means ``i -> j``) is a DAG.

    Uses Kahn's algorithm on a copy; ``O(D^2)`` in the dense representation.
    """
    adj = (np.asarray(adjacency) != 0).astype(np.int8).copy()
    np.fill_diagonal(adj, adj.diagonal())  # self-loops make it cyclic; keep them
    if adj.diagonal().any():
        return False
    d = adj.shape[0]
    in_deg = adj.sum(axis=0)
    alive = np.ones(d, dtype=bool)
    removed = 0
    while True:
        candidates = np.flatnonzero(alive & (in_deg == 0))
        if candidates.size == 0:
            break
        node = int(candidates[0])
        alive[node] = False
        removed += 1
        in_deg = in_deg - adj[node] * alive.astype(np.int64)
        in_deg[node] = -1
    return removed == d


def validate_adjacency(adjacency, dim: Optional[int] = None) -> np.ndarray:
    """Validate and normalise a ground-truth adjacency to a ``(D, D)`` int array."""
    adj = np.asarray(adjacency)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be square (D, D); got {adj.shape}")
    if dim is not None and adj.shape[0] != int(dim):
        raise ValueError(f"adjacency is {adj.shape[0]}x{adj.shape[0]} but data dimension is {dim}")
    return (adj != 0).astype(int)


def order_fnr(topological_order: Sequence[int], adjacency: np.ndarray) -> Dict[str, Any]:
    """Fraction of true edges violated by the estimated order.

    Parameters
    ----------
    topological_order: estimated order, sources first.
    adjacency: ``(D, D)`` with ``A[i, j] = 1`` meaning ``i -> j``.

    Returns
    -------
    dict with ``order_fnr``, ``num_true_edges``, ``num_violated_edges``,
    ``violated_edges``.
    """
    adj = validate_adjacency(adjacency, dim=len(topological_order))
    position = {int(node): pos for pos, node in enumerate(topological_order)}

    edges = np.argwhere(adj == 1)
    violated = [
        (int(i), int(j)) for i, j in edges if position[int(i)] > position[int(j)]
    ]
    n_edges = int(edges.shape[0])
    return {
        "order_fnr": (len(violated) / n_edges) if n_edges > 0 else float("nan"),
        "num_true_edges": n_edges,
        "num_violated_edges": len(violated),
        "violated_edges": violated,
    }


def stagewise_leaf_validity(leaf_order: Sequence[int], adjacency: np.ndarray) -> Dict[str, Any]:
    """Check, stage by stage, whether the removed node was a valid sink of ``S``.

    Returns
    -------
    dict with ``per_stage`` (list of records) and ``valid_fraction``.
    """
    leaf_order = [int(i) for i in leaf_order]
    adj = validate_adjacency(adjacency, dim=len(leaf_order))
    d = adj.shape[0]

    remaining = set(range(d))
    per_stage: List[Dict[str, Any]] = []
    for stage, node in enumerate(leaf_order):
        S = sorted(remaining)
        # valid leaves of the induced subgraph: no outgoing edge inside S
        valid = [i for i in S if not any(adj[i, j] == 1 for j in S if j != i)]
        per_stage.append({
            "stage": stage,
            "remaining": S,
            "selected": node,
            "valid_leaves": valid,
            "is_valid_leaf": bool(node in valid),
        })
        remaining.discard(node)

    n_checked = max(1, len(per_stage))
    n_valid = sum(1 for r in per_stage if r["is_valid_leaf"])
    return {
        "per_stage": per_stage,
        "valid_fraction": n_valid / n_checked,
        "num_valid_stages": n_valid,
        "num_stages": len(per_stage),
    }


def evaluate_ordering(result, adjacency) -> Dict[str, Any]:
    """Full ground-truth evaluation of a :class:`DAGOrderingResult`.

    Reports order FNR and stagewise leaf validity only — no SHD/edge-F1,
    because no sparse edge set is estimated at this stage.
    """
    adj = validate_adjacency(adjacency, dim=len(result.topological_order))
    acyclic = is_acyclic(adj)
    out = {
        "ground_truth_is_acyclic": bool(acyclic),
        "order_fnr": order_fnr(result.topological_order, adj),
        "stagewise_leaf_validity": stagewise_leaf_validity(result.leaf_order, adj),
        "note": (
            "SHD / edge-F1 are intentionally not reported: this phase estimates a "
            "topological order, not a sparse edge set."
        ),
    }
    if not acyclic:
        out["warning"] = "Ground-truth adjacency is not acyclic; order metrics may be ill-defined."
    return out
