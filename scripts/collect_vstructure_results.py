"""End-to-end ordering + parent estimation on the v-structure ``0 -> 2 <- 1``.

The chain ``0 -> 1 -> 2`` has a single valid topological order. The
v-structure is structurally different in ways that stress the estimator:

* ``x0`` and ``x1`` are **marginally independent** but become dependent given
  the collider ``x2``;
* there are **two** valid topological orders (``[0,1,2]`` and ``[1,0,2]``) — the
  two sources are exchangeable, so scoring must use order-FNR / stagewise leaf
  validity, never equality against one reference permutation;
* the sink ``x2`` has **two** parents, so the off-diagonal parent test must flag
  *both* ``x0`` and ``x1``, not just one.

This script runs, per timestep:

  stage 0 (``S = {0,1,2}``): diagonal-variance criterion -> leaf, plus the
      off-diagonal row ``E|H_{i,i*}|`` against that leaf (parent evidence);
  stage 1 (``S`` pinned to the two remaining sources): the criterion there,
      where **neither** node is a parent of the other — the correct answer is
      that both off-diagonal magnitudes are small.

Usage
-----
    python scripts/collect_vstructure_results.py --device cuda:0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.dag_diffusion.conditional_langevin import LangevinConfig
from models.dag_diffusion.diagnostics import evaluate_ordering
from models.dag_diffusion.ordering import (
    ConditionalDiffusionDAGOrderEstimator,
    DAGOrderingConfig,
)
from models.dag_diffusion.training import load_ddpm_checkpoint
from scripts.collect_offdiag_results import stage_full_hessian


def main():
    p = argparse.ArgumentParser(description="v-structure end-to-end ordering + parents.")
    p.add_argument("--data-path", default="data/dag_ordering_debug/dag3_vstructure.npy")
    p.add_argument("--adjacency-path",
                   default="data/dag_ordering_debug/dag3_vstructure_adjacency.npy")
    p.add_argument("--checkpoint", default="results/dag_ordering/vstruct/ddpm.pt")
    p.add_argument("--output", default="results/dag_ordering_summary/vstructure_summary.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-anchors", default=256, type=int)
    args = p.parse_args()

    for attr in ("data_path", "adjacency_path", "checkpoint", "output"):
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.normpath(os.path.join(REPO_ROOT, val)))

    device = args.device
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; falling back to CPU")
        device = "cpu"

    X = np.load(args.data_path).astype(np.float32)
    A = np.load(args.adjacency_path)
    model, diffusion, _, _ = load_ddpm_checkpoint(args.checkpoint, device, expected_dim=X.shape[1])

    t_values = [1, 2, 3, 5, 8, 12, 20, 30, 50]
    out = {
        "device": str(device),
        "structure": "v-structure 0 -> 2 <- 1",
        "adjacency": A.tolist(),
        "true_parents": {"0": [], "1": [], "2": [0, 1]},
        "valid_topological_orders": [[0, 1, 2], [1, 0, 2]],
        "true_sink": 2,
        "num_anchors": int(args.num_anchors),
        "t_values": t_values,
        "note": (
            "Two topological orders are valid because the sources are exchangeable; "
            "score with order_fnr / stagewise leaf validity, not permutation equality."
        ),
    }

    # ---- full ordering pipeline per timestep --------------------------------
    print("=== full ordering pipeline (v-structure) ===", flush=True)
    out["ordering"] = []
    for t in t_values:
        cfg = DAGOrderingConfig(
            t_order=t, num_anchors=args.num_anchors, anchor_chunk_size=32,
            langevin=LangevinConfig(num_chains=32, burn_in=300, num_samples=16,
                                    thinning=10, step_size=1e-3, init="forward_data"),
            reverse_draws_per_xt=1, sampling_chunk_size=8192, seed=120, verbose=False)
        t0 = time.time()
        r = ConditionalDiffusionDAGOrderEstimator(cfg).fit(X, model, diffusion, device=device)
        wall = time.time() - t0
        ev = evaluate_ordering(r, A)
        rec = {
            "t_order": t,
            "topological_order": r.topological_order,
            "leaf_order": r.leaf_order,
            "order_fnr": ev["order_fnr"]["order_fnr"],
            "leaf_valid": ev["stagewise_leaf_validity"]["num_valid_stages"],
            "leaf_stages": ev["stagewise_leaf_validity"]["num_stages"],
            "first_leaf": r.leaf_order[0],
            "first_leaf_correct": r.leaf_order[0] == 2,
            "criterion_by_stage": [{str(k): float(v) for k, v in s.items()}
                                   for s in r.criterion_by_stage],
            "stage0_criterion": r.stage_records[0]["criterion"].tolist(),
            "wall_seconds": wall,
            "order_is_valid": ev["order_fnr"]["order_fnr"] == 0.0,
        }
        out["ordering"].append(rec)
        print(f"  t={t:3d} order={rec['topological_order']} FNR={rec['order_fnr']:.2f} "
              f"leaf1=x{rec['first_leaf']} valid={rec['leaf_valid']}/{rec['leaf_stages']} "
              f"({wall:.1f}s)", flush=True)

    # ---- stage 0: leaf criterion + off-diagonal parent evidence -------------
    print("=== stage 0 (S={0,1,2}): criterion + off-diagonal ===", flush=True)
    out["stage0"] = []
    for t in t_values:
        rec = stage_full_hessian(X, model, diffusion, device, [0, 1, 2], t, args.num_anchors)
        S, k = rec["S"], rec["selected_local_index"]
        i_star = rec["selected_leaf"]
        am = np.array(rec["abs_mean"])
        rec["offdiag_abs_mean_vs_leaf"] = {
            str(S[j]): float(am[j, k]) for j in range(len(S)) if j != k}
        rec["leaf_correct"] = i_star == 2
        out["stage0"].append(rec)
        desc = "  ".join(f"E|H(x{i},x{i_star})|={v:8.4f}"
                         for i, v in rec["offdiag_abs_mean_vs_leaf"].items())
        print(f"  t={t:3d} leaf=x{i_star}{'' if rec['leaf_correct'] else ' (!)'}  {desc}",
              flush=True)

    # ---- stage 1: the two sources, neither a parent of the other ------------
    print("=== stage 1 (S={0,1}): two exchangeable sources ===", flush=True)
    out["stage1"] = []
    for t in t_values:
        rec = stage_full_hessian(X, model, diffusion, device, [0, 1], t, args.num_anchors)
        S, k = rec["S"], rec["selected_local_index"]
        i_star = rec["selected_leaf"]
        am = np.array(rec["abs_mean"])
        rec["offdiag_abs_mean_vs_leaf"] = {
            str(S[j]): float(am[j, k]) for j in range(len(S)) if j != k}
        # neither source parents the other: either leaf choice is structurally fine
        rec["leaf_correct"] = True
        out["stage1"].append(rec)
        desc = "  ".join(f"E|H(x{i},x{i_star})|={v:8.4f}"
                         for i, v in rec["offdiag_abs_mean_vs_leaf"].items())
        print(f"  t={t:3d} leaf=x{i_star}  {desc}", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    print(f"\nsummary written: {args.output}")


if __name__ == "__main__":
    main()
