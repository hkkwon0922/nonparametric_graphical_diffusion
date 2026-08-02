"""Synthetic SCM generation following Montagna et al. (2023), arXiv:2310.13387.

Implements the paper's **vanilla** scenario so our diffusion-based estimator can
be compared against its reported numbers:

  * Nonlinear additive noise model  ``X_i := f_i(PA_i) + U_i``  (their Eq. 4).
  * Gaussian noise ``U_i ~ N(0, sigma_i)`` with ``sigma_i ~ U(0.5, 1.0)``
    (Section 3.1).
  * Nonlinear mechanisms ``f_i`` sampled from a **Gaussian process** with a
    unit-bandwidth RBF kernel (Section 3.1 / Appendix B.1):
    ``f_i(X_PA_i) ~ N(0, K(X_PA_i, X_PA_i))``.
  * Erdos-Renyi ground-truth graphs with the density schema of their Table 2:

        nodes      5      10     20     50
        sparse   p=0.1   m=1    m=1    m=2
        dense    p=0.4   m=2    m=4    m=8

    where ``m`` is the average number of edges per node, and 5-node graphs are
    re-sampled until they have at least 2 edges.

The GP is drawn *jointly over the observed sample*: for each node we build the
RBF Gram matrix over its parents' realisations and sample one function value per
row from ``N(0, K)``. This is exactly the "sample the mechanism from a GP"
construction, and it makes each mechanism a fresh nonlinear function of its
parents rather than a fixed parametric form.

Usage
-----
    from data.benchmark_scm import sample_er_dag, simulate_anm_gp

    A = sample_er_dag(num_nodes=20, density="dense", seed=0)
    X = simulate_anm_gp(A, num_samples=1000, seed=0)
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "er_edge_probability",
    "sample_er_dag",
    "simulate_anm_gp",
    "generate_dataset",
    "DENSITY_SCHEMA",
]

# Table 2 of the paper: average edges per node (m), or explicit probability (p).
DENSITY_SCHEMA = {
    5: {"sparse": ("p", 0.1), "dense": ("p", 0.4)},
    10: {"sparse": ("m", 1), "dense": ("m", 2)},
    20: {"sparse": ("m", 1), "dense": ("m", 4)},
    50: {"sparse": ("m", 2), "dense": ("m", 8)},
}


def er_edge_probability(num_nodes: int, density: str) -> float:
    """Edge probability for an ER DAG under the paper's Table 2 schema.

    For rows specified by ``m`` (average edges per node), the total expected
    edge count is ``m * d``, spread over the ``d(d-1)/2`` ordered pairs allowed
    by a topological order, giving ``p = m*d / (d(d-1)/2)``.
    """
    if num_nodes not in DENSITY_SCHEMA:
        raise ValueError(
            f"num_nodes must be one of {sorted(DENSITY_SCHEMA)}; got {num_nodes}")
    if density not in ("sparse", "dense"):
        raise ValueError(f"density must be 'sparse' or 'dense'; got {density!r}")

    kind, value = DENSITY_SCHEMA[num_nodes][density]
    if kind == "p":
        return float(value)
    num_pairs = num_nodes * (num_nodes - 1) / 2.0
    return float(min(1.0, value * num_nodes / num_pairs))


def sample_er_dag(num_nodes: int, density: str = "dense", seed: int = 0,
                  min_edges: int = 2) -> np.ndarray:
    """Sample an Erdos-Renyi DAG -> ``(d, d)`` int adjacency, ``A[i,j]=1`` means ``i -> j``.

    A random permutation fixes a topological order, then each forward pair is
    connected independently with probability :func:`er_edge_probability`.
    Acyclicity is guaranteed by construction. The graph is re-sampled until it
    has at least ``min_edges`` edges (the paper does this for 5-node graphs; we
    apply it uniformly since a graph with no edges is degenerate).
    """
    rng = np.random.default_rng(int(seed))
    prob = er_edge_probability(num_nodes, density)

    for _ in range(1000):
        order = rng.permutation(num_nodes)
        adj = np.zeros((num_nodes, num_nodes), dtype=int)
        for a in range(num_nodes):
            for b in range(a + 1, num_nodes):
                if rng.random() < prob:
                    adj[order[a], order[b]] = 1
        if adj.sum() >= min_edges:
            return adj
    raise RuntimeError(  # pragma: no cover - only with a pathological probability
        f"failed to sample an ER DAG with >= {min_edges} edges (p={prob})")


def topological_order_of(adjacency: np.ndarray) -> list:
    """Return one valid topological order of ``adjacency`` (Kahn's algorithm)."""
    adj = (np.asarray(adjacency) != 0).astype(int)
    d = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int)
    alive = np.ones(d, dtype=bool)
    order = []
    while len(order) < d:
        ready = np.flatnonzero(alive & (in_deg == 0))
        if ready.size == 0:
            raise ValueError("adjacency is cyclic; cannot compute a topological order")
        node = int(ready[0])
        order.append(node)
        alive[node] = False
        in_deg = in_deg - adj[node] * alive
        in_deg[node] = 0
    return order


def _gp_mechanism(parent_values: np.ndarray, rng: np.random.Generator,
                  lengthscale: float = 1.0, jitter: float = 1e-8) -> np.ndarray:
    """Draw ``f(parent_values)`` from a GP with a unit-bandwidth RBF kernel.

    Parameters
    ----------
    parent_values: (n, k) realisations of the ``k`` parents.

    Returns
    -------
    (n,) function values, one per sample.
    """
    n = parent_values.shape[0]
    # Standardise the parents so the unit bandwidth is meaningful regardless of
    # the scale inherited from upstream mechanisms.
    pv = np.asarray(parent_values, dtype=np.float64)
    scale = pv.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    pv = (pv - pv.mean(axis=0)) / scale

    sq = ((pv[:, None, :] - pv[None, :, :]) ** 2).sum(axis=-1)
    K = np.exp(-0.5 * sq / (lengthscale ** 2))
    K[np.diag_indices(n)] += jitter

    # Cholesky with escalating jitter — the Gram matrix is near-singular when
    # samples nearly coincide.
    for extra in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
        try:
            L = np.linalg.cholesky(K + extra * np.eye(n))
            break
        except np.linalg.LinAlgError:
            continue
    else:  # pragma: no cover - eigendecomposition fallback
        vals, vecs = np.linalg.eigh(K)
        L = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))
    return L @ rng.standard_normal(n)


def simulate_anm_gp(adjacency: np.ndarray, num_samples: int = 1000, seed: int = 0,
                    noise_scale_range: Tuple[float, float] = (0.5, 1.0),
                    lengthscale: float = 1.0,
                    standardize_mechanism: bool = True) -> np.ndarray:
    """Simulate the paper's vanilla ANM with GP mechanisms -> ``(n, d)`` data.

    ``X_i = f_i(PA_i) + U_i`` with ``f_i`` a GP draw (unit-bandwidth RBF) and
    ``U_i ~ N(0, sigma_i)``, ``sigma_i ~ U(0.5, 1.0)``.

    ``standardize_mechanism`` rescales each drawn mechanism to unit variance
    before adding noise, which keeps the signal-to-noise ratio comparable across
    nodes and prevents variance from exploding along long causal paths. This
    also suppresses *varsortability* (Reisach et al.), which the paper flags as
    a way benchmarks can be gamed — without it, a node's marginal variance alone
    would leak the causal order.
    """
    adj = (np.asarray(adjacency) != 0).astype(int)
    d = adj.shape[0]
    n = int(num_samples)
    rng = np.random.default_rng(int(seed))

    sigmas = rng.uniform(noise_scale_range[0], noise_scale_range[1], size=d)
    X = np.zeros((n, d), dtype=np.float64)

    for node in topological_order_of(adj):
        parents = np.flatnonzero(adj[:, node] == 1)
        noise = sigmas[node] * rng.standard_normal(n)
        if parents.size == 0:
            X[:, node] = noise
            continue
        f = _gp_mechanism(X[:, parents], rng, lengthscale=lengthscale)
        if standardize_mechanism:
            sd = f.std(ddof=0)
            if sd > 1e-12:
                f = f / sd
        X[:, node] = f + noise

    if not np.isfinite(X).all():
        raise FloatingPointError("simulated data contains non-finite values")
    return X


def generate_dataset(num_nodes: int = 20, density: str = "dense",
                     num_samples: int = 1000, seed: int = 0,
                     lengthscale: float = 1.0,
                     standardize_mechanism: bool = True):
    """Convenience wrapper -> ``(X, adjacency)``.

    The graph and the data share ``seed`` but use separate generator streams, so
    changing ``num_samples`` does not change the sampled graph.
    """
    adjacency = sample_er_dag(num_nodes=num_nodes, density=density, seed=seed)
    X = simulate_anm_gp(adjacency, num_samples=num_samples, seed=seed + 10_000,
                        lengthscale=lengthscale,
                        standardize_mechanism=standardize_mechanism)
    return X, adjacency


def main():
    p = argparse.ArgumentParser(description="Generate benchmark ANM datasets (paper vanilla).")
    p.add_argument("--output-dir", default="data/benchmark", type=str)
    p.add_argument("--num-nodes", default=20, type=int, choices=sorted(DENSITY_SCHEMA))
    p.add_argument("--density", default="dense", choices=["sparse", "dense"])
    p.add_argument("--num-samples", default=1000, type=int)
    p.add_argument("--seeds", default="0", type=str, help="comma-separated seeds")
    args = p.parse_args()

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", out_dir))
    os.makedirs(out_dir, exist_ok=True)

    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        X, adjacency = generate_dataset(
            num_nodes=args.num_nodes, density=args.density,
            num_samples=args.num_samples, seed=seed)
        stem = f"er{args.num_nodes}_{args.density}_n{args.num_samples}_seed{seed}"
        np.save(os.path.join(out_dir, f"{stem}.npy"), X)
        np.save(os.path.join(out_dir, f"{stem}_adjacency.npy"), adjacency)
        print(f"{stem}: X={X.shape} edges={int(adjacency.sum())} "
              f"std={np.round(X.std(axis=0), 3)[:5]}...")


if __name__ == "__main__":
    main()
