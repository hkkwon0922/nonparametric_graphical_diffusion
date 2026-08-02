"""End-to-end smoke test of the DAG-ordering pipeline.

Runs the whole stack on a tiny D=3 nonlinear additive-noise SCM with a briefly
trained model, asserting structural correctness only.  It does **not** assert
that the recovered order is the true one — the sampling budget here is far too
small for a statistically meaningful estimate.
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

from data.make_dag_ordering_debug_data import generate_chain_scm
from models.dag_diffusion.conditional_langevin import LangevinConfig
from models.dag_diffusion.diagnostics import evaluate_ordering, is_acyclic
from models.dag_diffusion.ordering import (
    ConditionalDiffusionDAGOrderEstimator,
    DAGOrderingConfig,
    fully_connected_dag_from_order,
)
from models.dag_diffusion.training import DDPMTrainConfig, train_ddpm

DIM = 3


@pytest.fixture(scope="module")
def tiny_model():
    """A DDPM trained for a handful of epochs on the D=3 chain SCM."""
    X, adjacency = generate_chain_scm(n=300, seed=120)
    cfg = DDPMTrainConfig(
        input_dimension=DIM, mid_features=16, num_temporal_layers=2,
        timesteps=30, epochs=8, batch_size=64, lr=2e-3, seed=120,
    )
    model, diffusion, preproc, history = train_ddpm(
        X, cfg, device="cpu", verbose=False)
    return X.astype(np.float32), adjacency, model, diffusion, history


def make_config(**overrides):
    base = dict(
        t_order=3, num_anchors=4, reverse_draws_per_xt=1,
        langevin=LangevinConfig(num_chains=2, burn_in=10, num_samples=3,
                                thinning=2, step_size=1e-4, init="forward_data"),
        sampling_chunk_size=64, seed=120, verbose=False,
    )
    base.update(overrides)
    return DAGOrderingConfig(**base)


def test_ordering_is_a_valid_permutation(tiny_model):
    X, _, model, diffusion, _ = tiny_model
    est = ConditionalDiffusionDAGOrderEstimator(make_config())
    result = est.fit(X, model, diffusion, device="cpu")

    assert len(result.topological_order) == DIM
    assert sorted(result.topological_order) == list(range(DIM))
    assert sorted(result.leaf_order) == list(range(DIM))
    assert result.topological_order == list(reversed(result.leaf_order))


def test_each_stage_removes_exactly_one_variable(tiny_model):
    X, _, model, diffusion, _ = tiny_model
    est = ConditionalDiffusionDAGOrderEstimator(make_config())
    result = est.fit(X, model, diffusion, device="cpu")

    # D-1 sampling stages; the final survivor is appended without sampling.
    assert len(result.stage_records) == DIM - 1
    for stage, record in enumerate(result.stage_records):
        assert len(record["remaining"]) == DIM - stage
        assert record["selected_leaf"] in record["remaining"]
        assert record["hessian_diag"].shape == (4, DIM - stage)
        assert record["criterion"].shape == (DIM - stage,)

    removed = [r["selected_leaf"] for r in result.stage_records]
    assert len(set(removed)) == len(removed), "a variable was removed twice"

    # each stage's remaining set is the previous one minus the removed node
    for prev, cur in zip(result.stage_records, result.stage_records[1:]):
        expected = [i for i in prev["remaining"] if i != prev["selected_leaf"]]
        assert cur["remaining"] == expected


def test_no_nan_or_inf_anywhere(tiny_model):
    X, _, model, diffusion, _ = tiny_model
    est = ConditionalDiffusionDAGOrderEstimator(make_config())
    result = est.fit(X, model, diffusion, device="cpu")

    for record in result.stage_records:
        assert np.isfinite(record["hessian_diag"]).all()
        assert np.isfinite(record["criterion"]).all()


def test_first_stage_skips_conditional_langevin(tiny_model):
    """With S = {0..D-1} the free block is empty, so ULA must be skipped."""
    X, _, model, diffusion, _ = tiny_model
    est = ConditionalDiffusionDAGOrderEstimator(make_config())
    result = est.fit(X, model, diffusion, device="cpu")

    first = result.stage_records[0]
    assert first["remaining"] == list(range(DIM))
    for lang in first["langevin_diagnostics"]["langevin"]:
        assert lang["extras"].get("skipped") is True

    # later stages DO run Langevin
    second = result.stage_records[1]
    for lang in second["langevin_diagnostics"]["langevin"]:
        assert not lang["extras"].get("skipped", False)
        assert np.isfinite(lang["mean_score_norm"])


def test_original_indexing_is_preserved(tiny_model):
    """Local S indices must map back to original D-dimensional indices."""
    X, _, model, diffusion, _ = tiny_model
    est = ConditionalDiffusionDAGOrderEstimator(make_config())
    result = est.fit(X, model, diffusion, device="cpu")

    for record in result.stage_records:
        local = record["selected_local_index"]
        assert record["remaining"][local] == record["selected_leaf"]
        assert record["remaining"] == sorted(record["remaining"])


def test_fully_connected_order_dag_encodes_the_order(tiny_model):
    X, _, model, diffusion, _ = tiny_model
    est = ConditionalDiffusionDAGOrderEstimator(make_config())
    result = est.fit(X, model, diffusion, device="cpu")

    adj = result.fully_connected_order_dag
    assert adj.shape == (DIM, DIM)
    assert is_acyclic(adj), "an order-encoding DAG must be acyclic"
    assert adj.sum() == DIM * (DIM - 1) // 2, "a complete order DAG has D(D-1)/2 edges"

    order = result.topological_order
    for pos, i in enumerate(order):
        for j in order[pos + 1:]:
            assert adj[i, j] == 1 and adj[j, i] == 0


def test_result_is_json_serialisable(tiny_model):
    X, _, model, diffusion, _ = tiny_model
    est = ConditionalDiffusionDAGOrderEstimator(make_config())
    result = est.fit(X, model, diffusion, device="cpu")

    payload = json.loads(json.dumps(result.to_json_dict()))
    assert payload["topological_order"] == result.topological_order
    assert payload["t_order"] == 3
    assert payload["warnings"], "the positive-t caveat must be recorded"


def test_ground_truth_evaluation_runs(tiny_model):
    X, adjacency, model, diffusion, _ = tiny_model
    est = ConditionalDiffusionDAGOrderEstimator(make_config())
    result = est.fit(X, model, diffusion, device="cpu")

    evaluation = evaluate_ordering(result, adjacency)
    assert evaluation["ground_truth_is_acyclic"] is True
    fnr = evaluation["order_fnr"]
    assert fnr["num_true_edges"] == 2
    assert 0.0 <= fnr["order_fnr"] <= 1.0
    leafv = evaluation["stagewise_leaf_validity"]
    assert leafv["num_stages"] == DIM

    # no sparse-structure metrics are reported at this phase (the explanatory
    # "note" field mentions them by name, so check the metric keys themselves)
    metric_keys = {k.lower() for k in evaluation} | {k.lower() for k in fnr} | {
        k.lower() for k in leafv}
    assert not metric_keys & {"shd", "edge_f1", "f1", "precision", "recall"}


def test_single_anchor_is_rejected(tiny_model):
    """The criterion is a variance ACROSS anchors, so B must be >= 2."""
    X, _, model, diffusion, _ = tiny_model
    est = ConditionalDiffusionDAGOrderEstimator(make_config(num_anchors=1))
    with pytest.raises(ValueError, match="at least 2 anchors"):
        est.fit(X, model, diffusion, device="cpu")


def test_out_of_range_t_order_is_rejected(tiny_model):
    X, _, model, diffusion, _ = tiny_model
    est = ConditionalDiffusionDAGOrderEstimator(make_config(t_order=999))
    with pytest.raises(ValueError, match="t_order"):
        est.fit(X, model, diffusion, device="cpu")


def test_dimension_mismatch_is_rejected(tiny_model):
    """A checkpoint trained on a different D must not be silently accepted."""
    X, _, model, diffusion, _ = tiny_model
    est = ConditionalDiffusionDAGOrderEstimator(make_config())
    wrong = np.concatenate([X, X[:, :1]], axis=1)  # D = 4
    with pytest.raises(ValueError, match="input_dim"):
        est.fit(wrong, model, diffusion, device="cpu")


def test_anchor_set_is_shared_across_stages(tiny_model):
    """The same anchor rows must be reused at every stage."""
    X, _, model, diffusion, _ = tiny_model
    est = ConditionalDiffusionDAGOrderEstimator(make_config())
    result = est.fit(X, model, diffusion, device="cpu")

    first = result.stage_records[0]["anchor_row_indices"]
    for record in result.stage_records[1:]:
        np.testing.assert_array_equal(record["anchor_row_indices"], first)


def test_repeated_fit_with_same_seed_is_deterministic(tiny_model):
    X, _, model, diffusion, _ = tiny_model
    r1 = ConditionalDiffusionDAGOrderEstimator(make_config()).fit(
        X, model, diffusion, device="cpu")
    r2 = ConditionalDiffusionDAGOrderEstimator(make_config()).fit(
        X, model, diffusion, device="cpu")
    assert r1.leaf_order == r2.leaf_order
    np.testing.assert_allclose(
        r1.stage_records[0]["criterion"], r2.stage_records[0]["criterion"], rtol=1e-12)


def test_fully_connected_dag_helper():
    adj = fully_connected_dag_from_order([2, 0, 1])
    assert adj[2, 0] == 1 and adj[2, 1] == 1 and adj[0, 1] == 1
    assert adj[0, 2] == 0 and adj[1, 0] == 0
    assert is_acyclic(adj)


@pytest.mark.slow
def test_runner_writes_all_output_files(tmp_path, repo_root):
    """Full CLI run through the debug config; marked slow (trains a model)."""
    sys.path.insert(0, repo_root)
    from experiments.run_dag_ordering import main as run_main

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    X, adjacency = generate_chain_scm(n=300, seed=120)
    data_path = data_dir / "x.npy"
    adj_path = data_dir / "adj.npy"
    np.save(data_path, X)
    np.save(adj_path, adjacency)

    out_dir = tmp_path / "out"
    run_main([
        "--data-path", str(data_path),
        "--checkpoint-path", str(tmp_path / "ddpm.pt"),
        "--train-if-missing",
        "--output-dir", str(out_dir),
        "--ground-truth-adjacency", str(adj_path),
        "--t-order", "3", "--num-anchors", "4", "--num-chains", "2",
        "--langevin-burn-in", "10", "--langevin-samples", "3",
        "--langevin-thinning", "2", "--langevin-step-size", "1e-4",
        "--reverse-draws-per-xt", "1", "--sampling-chunk-size", "64",
        "--epochs", "8", "--timesteps", "30", "--mid-features", "16",
        "--num-temporal-layers", "2", "--batch-size", "64",
        "--seed", "120", "--device", "cpu", "--quiet",
    ])

    for name in ("config.json", "ordering_result.json", "stage_diagnostics.pt",
                 "runtime.json", "checkpoint_reference.json"):
        assert (out_dir / name).exists(), f"missing output file: {name}"

    payload = json.loads((out_dir / "ordering_result.json").read_text())
    assert sorted(payload["topological_order"]) == list(range(DIM))
    assert "ground_truth_evaluation" in payload
    assert os.path.exists(tmp_path / "ddpm.pt")
