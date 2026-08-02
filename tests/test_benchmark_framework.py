"""Tests for the benchmark framework: data generation, metrics, parent selection."""
import numpy as np
import pytest

from data.benchmark_scm import (
    DENSITY_SCHEMA,
    er_edge_probability,
    generate_dataset,
    sample_er_dag,
    simulate_anm_gp,
    topological_order_of,
)
from models.dag_diffusion.benchmark_metrics import edge_metrics, fnr_pi
from models.dag_diffusion.diagnostics import is_acyclic
from models.dag_diffusion.parent_selection import (
    benjamini_hochberg,
    candidate_pairs_from_order,
    das_hypothesis_test,
    multitime_cluster,
)


# --------------------------------------------------------------------- data
@pytest.mark.parametrize("num_nodes", sorted(DENSITY_SCHEMA))
@pytest.mark.parametrize("density", ["sparse", "dense"])
def test_er_dag_is_acyclic_and_right_size(num_nodes, density):
    A = sample_er_dag(num_nodes, density, seed=3)
    assert A.shape == (num_nodes, num_nodes)
    assert is_acyclic(A), "sampled ER graph must be a DAG"
    assert A.sum() >= 2


def test_density_matches_table2_in_expectation():
    """Average edge count should track the paper's m (edges per node)."""
    counts = [sample_er_dag(20, "dense", seed=s).sum() for s in range(30)]
    # Table 2: ER-20 dense uses m = 4 -> ~80 expected edges
    assert 60 < np.mean(counts) < 100, f"mean edge count {np.mean(counts)} off target ~80"


def test_edge_probability_formula():
    # m-specified rows: p = m*d / (d(d-1)/2)
    assert er_edge_probability(20, "dense") == pytest.approx(4 * 20 / (20 * 19 / 2))
    # p-specified rows pass through
    assert er_edge_probability(5, "dense") == pytest.approx(0.4)


def test_anm_respects_the_graph():
    """A source node must be independent of everything downstream of it."""
    A = np.zeros((3, 3), dtype=int)
    A[0, 1] = 1
    A[1, 2] = 1
    X = simulate_anm_gp(A, num_samples=2000, seed=0)
    assert X.shape == (2000, 3)
    assert np.isfinite(X).all()
    # adjacent variables are dependent; the chain endpoints less so
    c01 = abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    c12 = abs(np.corrcoef(X[:, 1], X[:, 2])[0, 1])
    assert c01 > 0.2 and c12 > 0.2


def test_vstructure_sources_are_marginally_independent():
    A = np.zeros((3, 3), dtype=int)
    A[0, 2] = 1
    A[1, 2] = 1
    X = simulate_anm_gp(A, num_samples=4000, seed=1)
    assert abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1]) < 0.1


def test_topological_order_of_is_valid():
    A = sample_er_dag(10, "dense", seed=7)
    order = topological_order_of(A)
    assert sorted(order) == list(range(10))
    pos = {n: k for k, n in enumerate(order)}
    for i, j in np.argwhere(A == 1):
        assert pos[int(i)] < pos[int(j)]


def test_generate_dataset_is_reproducible():
    X1, A1 = generate_dataset(10, "dense", 200, seed=5)
    X2, A2 = generate_dataset(10, "dense", 200, seed=5)
    np.testing.assert_array_equal(A1, A2)
    np.testing.assert_allclose(X1, X2)


# ------------------------------------------------------------------ metrics
def test_fnr_pi_zero_for_true_order():
    A = sample_er_dag(10, "dense", seed=2)
    assert fnr_pi(topological_order_of(A), A) == 0.0


def test_fnr_pi_one_for_reversed_order():
    A = np.zeros((3, 3), dtype=int)
    A[0, 1] = 1
    A[1, 2] = 1
    assert fnr_pi([2, 1, 0], A) == 1.0


def test_edge_metrics_perfect_recovery():
    A = sample_er_dag(10, "dense", seed=4)
    m = edge_metrics(A, A)
    assert m["f1"] == pytest.approx(1.0)
    assert m["fp"] == 0 and m["fn"] == 0


def test_reversed_edge_counts_as_false_negative():
    """Paper convention: a reversed edge is an FN, not a TP."""
    A = np.zeros((2, 2), dtype=int)
    A[0, 1] = 1
    P = np.zeros((2, 2), dtype=int)
    P[1, 0] = 1
    m = edge_metrics(P, A)
    assert m["tp"] == 0
    assert m["fn"] == 1
    assert m["fp"] == 0  # skeleton is right, so it is not a false positive


def test_empty_prediction_metrics():
    A = np.zeros((4, 4), dtype=int)
    A[0, 1] = A[1, 2] = 1
    m = edge_metrics(np.zeros((4, 4), dtype=int), A)
    assert m["tp"] == 0 and m["fn"] == 2 and m["fp"] == 0
    assert m["f1"] == pytest.approx(0.0)


# --------------------------------------------------------- parent selection
def test_candidate_pairs_respect_order():
    pairs = candidate_pairs_from_order([2, 0, 1])
    assert set(pairs) == {(2, 0), (2, 1), (0, 1)}
    assert len(pairs) == 3


def test_benjamini_hochberg_monotone():
    p = np.array([0.001, 0.02, 0.5, 0.9])
    rej = benjamini_hochberg(p, alpha=0.05)
    assert rej[0] and not rej[-1]
    assert benjamini_hochberg(np.ones(5), alpha=0.05).sum() == 0


def test_das_test_recovers_a_clear_edge():
    """A pair with a large consistent mean is kept; a zero-mean pair is not."""
    rng = np.random.default_rng(0)
    order = [0, 1, 2]
    signed = {
        (0, 1): rng.normal(5.0, 1.0, size=200),   # strong edge
        (0, 2): rng.normal(0.0, 1.0, size=200),   # no edge
        (1, 2): rng.normal(-4.0, 1.0, size=200),  # strong edge (negative sign)
    }
    out = das_hypothesis_test(signed, order, alpha=0.05)
    A = out["adjacency"]
    assert A[0, 1] == 1 and A[1, 2] == 1
    assert A[0, 2] == 0
    assert is_acyclic(A)


def test_das_output_respects_the_order():
    """No edge may point backwards relative to the supplied order."""
    rng = np.random.default_rng(1)
    order = [2, 1, 0]
    signed = {p: rng.normal(9.0, 1.0, size=100) for p in candidate_pairs_from_order(order)}
    A = das_hypothesis_test(signed, order, alpha=0.05)["adjacency"]
    pos = {n: k for k, n in enumerate(order)}
    for i, j in np.argwhere(A == 1):
        assert pos[int(i)] < pos[int(j)]


def test_multitime_cluster_separates_profiles():
    """Edge profiles sit well above non-edge profiles at every timestep.

    Non-edges get *distinct* small values: with rank features, exactly tied rows
    would break ties arbitrarily, which is a property of the fixture rather than
    of the method.
    """
    rng = np.random.default_rng(0)
    order = [0, 1, 2, 3, 4]
    pairs = candidate_pairs_from_order(order)
    true_edges = {(0, 1), (2, 3), (1, 4)}
    decay = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

    profiles = {}
    for p in pairs:
        base = 10.0 if p in true_edges else 0.5
        profiles[p] = base * decay * (1.0 + 0.05 * rng.standard_normal(decay.size))

    out = multitime_cluster(profiles, order, standardize=True, seed=0)
    A = out["adjacency"]
    for (i, j) in true_edges:
        assert A[i, j] == 1, f"missed true edge {(i, j)}"
    assert A.sum() == len(true_edges)
    assert out["transform"] == "rank"


def test_multitime_cluster_zscore_transform_available():
    rng = np.random.default_rng(1)
    order = [0, 1, 2, 3]
    pairs = candidate_pairs_from_order(order)
    true_edges = {(0, 1), (2, 3)}
    profiles = {p: (8.0 if p in true_edges else 0.4) * (1.0 + 0.05 * rng.random(4))
                for p in pairs}
    out = multitime_cluster(profiles, order, standardize=True, seed=0, transform="zscore")
    assert out["transform"] == "zscore"
    for (i, j) in true_edges:
        assert out["adjacency"][i, j] == 1


def test_multitime_cluster_rejects_unknown_transform():
    order = [0, 1]
    with pytest.raises(ValueError, match="transform"):
        multitime_cluster({(0, 1): np.ones(3), (1, 0): np.ones(3)}, order,
                          transform="bogus")


def test_multitime_cluster_standardization_changes_features():
    """Per-timestep z-scoring must actually be applied when requested."""
    order = [0, 1, 2]
    pairs = candidate_pairs_from_order(order)
    # magnitudes differ wildly across t; without standardising, t0 dominates
    profiles = {p: np.array([1000.0, 1.0]) * (5.0 if p == (0, 1) else 1.0) for p in pairs}
    out = multitime_cluster(profiles, order, standardize=True, seed=0)
    assert out["standardize"] is True
    assert out["adjacency"].shape == (3, 3)
    assert is_acyclic(out["adjacency"])


def test_multitime_cluster_output_is_acyclic():
    rng = np.random.default_rng(2)
    order = list(rng.permutation(6))
    profiles = {p: rng.random(4) for p in candidate_pairs_from_order(order)}
    A = multitime_cluster(profiles, order, seed=0)["adjacency"]
    assert is_acyclic(A)


def test_random_baseline_is_not_secretly_an_oracle():
    """The baseline must not reproduce the ground-truth permutation.

    ``sample_er_dag(seed)`` draws its topological permutation from
    ``default_rng(seed)``. If the baseline used the same seed it would recover
    that permutation exactly and score FNR-pi = 0, silently turning the
    "random" reference into an oracle.
    """
    from experiments.run_benchmark_sweep import random_baseline

    fnrs = []
    for seed in range(8):
        A = sample_er_dag(20, "dense", seed=seed)
        fnrs.append(random_baseline(20, A, seed=seed)["fnr_pi"])

    fnrs = np.asarray(fnrs)
    assert (fnrs > 0.0).all(), f"baseline achieved a perfect order: {fnrs}"
    # a random order violates roughly half the edges
    assert 0.2 < fnrs.mean() < 0.8, f"baseline FNR-pi implausible: {fnrs.mean()}"
