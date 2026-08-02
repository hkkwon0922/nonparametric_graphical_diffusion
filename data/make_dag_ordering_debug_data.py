"""Generate a tiny nonlinear additive-noise SCM for DAG-ordering smoke tests.

Default structure is the 3-node chain ``0 -> 1 -> 2``::

    x0 = e0
    x1 = f1(x0) + e1
    x2 = f2(x1) + e2

with nonlinear ``f`` and Gaussian noise, i.e. a nonlinear ANM whose topological
order is identifiable in principle.  Node 2 is the unique sink, node 0 the
unique source.

Usage
-----
    python data/make_dag_ordering_debug_data.py \\
        --output-dir data/dag_ordering_debug --n 2000 --seed 120
"""
import argparse
import os

import numpy as np

__all__ = ["generate_chain_scm"]


def generate_chain_scm(n=2000, seed=120, noise_scale=0.5):
    """Sample ``(n, 3)`` data from ``0 -> 1 -> 2`` plus its ``(3, 3)`` adjacency.

    Returns
    -------
    X: (n, 3) float64
    adjacency: (3, 3) int, ``A[i, j] = 1`` meaning ``i -> j``
    """
    rng = np.random.default_rng(int(seed))
    e = rng.normal(scale=noise_scale, size=(int(n), 3))

    x0 = rng.normal(scale=1.0, size=int(n))
    x1 = np.sin(2.0 * x0) + 0.5 * x0 + e[:, 1]
    x2 = 0.8 * np.tanh(2.0 * x1) + 0.3 * x1 ** 2 + e[:, 2]

    X = np.stack([x0, x1, x2], axis=1)
    adjacency = np.zeros((3, 3), dtype=int)
    adjacency[0, 1] = 1
    adjacency[1, 2] = 1
    return X, adjacency


def main():
    p = argparse.ArgumentParser(description="Generate the D=3 debug SCM dataset.")
    p.add_argument("--output-dir", default="data/dag_ordering_debug", type=str)
    p.add_argument("--n", default=2000, type=int)
    p.add_argument("--seed", default=120, type=int)
    p.add_argument("--noise-scale", default=0.5, type=float)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    X, adjacency = generate_chain_scm(args.n, args.seed, args.noise_scale)

    data_path = os.path.join(args.output_dir, "dag3_chain.npy")
    adj_path = os.path.join(args.output_dir, "dag3_chain_adjacency.npy")
    np.save(data_path, X.astype(np.float64))
    np.save(adj_path, adjacency)

    print(f"data      -> {data_path}  shape={X.shape}")
    print(f"adjacency -> {adj_path}\n{adjacency}")
    print(f"column std: {X.std(axis=0)}")


if __name__ == "__main__":
    main()
