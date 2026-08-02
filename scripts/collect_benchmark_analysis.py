"""Re-analysis of saved benchmark Hessians for the notebook.

All of these reuse the ``adjacencies.npz`` files that ``run_causal_benchmark``
already wrote, so nothing here re-runs the diffusion model — it is pure
post-processing of stored per-anchor Hessians.

Produces:
  * ``alpha_sweep``      — DAS F1/FNR/FPR as the significance level is tightened
  * ``transform_compare``— rank vs z-score clustering features
  * ``tsubset_compare``  — which timestep subsets the clustering benefits from
  * ``power_analysis``   — |t| statistics of true edges vs non-edges
  * ``budget_compare``   — the ER-10 anchor-budget comparison (128 vs 512)

Usage
-----
    python scripts/collect_benchmark_analysis.py
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.dag_diffusion.benchmark_metrics import edge_metrics
from models.dag_diffusion.parent_selection import (
    candidate_pairs_from_order,
    das_hypothesis_test,
    multitime_cluster,
)

T_VALUES = [3, 5, 8, 12, 20]


def load_run(path):
    """Load one run's stored adjacencies + Hessians."""
    z = np.load(path)
    A = z["true"]
    order = [int(i) for i in z["order"]]
    H = {t: z[f"hessian_t{t}"] for t in T_VALUES if f"hessian_t{t}" in z}
    return A, order, H


def pair_views(H, order, t_edge=8, t_subset=None):
    pairs = candidate_pairs_from_order(order)
    ts = sorted(H) if t_subset is None else sorted(t_subset)
    signed = {(i, j): H[t_edge][:, i, j] for i, j in pairs}
    profiles = {(i, j): np.array([np.abs(H[t][:, i, j]).mean() for t in ts])
                for i, j in pairs}
    return pairs, signed, profiles


def main():
    p = argparse.ArgumentParser(description="Post-hoc analysis of saved benchmark runs.")
    p.add_argument("--sweep-root", default="results/causal_benchmark/er20_dense")
    p.add_argument("--er10-run", default="results/causal_benchmark/er10_big")
    p.add_argument("--er10-small", default="results/causal_benchmark/er10_dense_seed0")
    p.add_argument("--output", default="results/dag_ordering_summary/benchmark_analysis.json")
    p.add_argument("--seeds", default="0,1,2")
    args = p.parse_args()

    for attr in ("sweep_root", "er10_run", "er10_small", "output"):
        v = getattr(args, attr)
        if not os.path.isabs(v):
            setattr(args, attr, os.path.normpath(os.path.join(REPO_ROOT, v)))

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = {"t_values": T_VALUES, "seeds": seeds}

    # ---- alpha sweep + transform + subsets, averaged over seeds -----------
    alphas = [5e-2, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]
    subsets = [[8], [3], [20], [3, 8], [8, 20], [3, 8, 20], [3, 5, 8], [5, 8, 12], T_VALUES]

    alpha_rows, tr_rows, sub_rows, power_rows = [], [], [], []
    for s in seeds:
        path = os.path.join(args.sweep_root, f"seed{s}", "adjacencies.npz")
        if not os.path.exists(path):
            print(f"[skip] missing {path}")
            continue
        A, order, H = load_run(path)
        pairs, signed, _ = pair_views(H, order)

        for a in alphas:
            P = das_hypothesis_test(signed, order, alpha=a, fdr_correction=True)["adjacency"]
            m = edge_metrics(P, A)
            alpha_rows.append({"seed": s, "alpha": a, "pred": int(P.sum()), **m})

        for tr in ("rank", "zscore"):
            _, _, prof = pair_views(H, order)
            P = multitime_cluster(prof, order, transform=tr, seed=120)["adjacency"]
            m = edge_metrics(P, A)
            tr_rows.append({"seed": s, "transform": tr, "pred": int(P.sum()), **m})

        for sub in subsets:
            _, _, prof = pair_views(H, order, t_subset=sub)
            P = multitime_cluster(prof, order, transform="rank", seed=120)["adjacency"]
            m = edge_metrics(P, A)
            sub_rows.append({"seed": s, "subset": sub, "pred": int(P.sum()), **m})

        # power analysis: |t| of true edges vs non-edges at t_edge = 8
        B = H[8].shape[0]
        te, tn = [], []
        for (i, j) in pairs:
            v = H[8][:, i, j]
            tstat = abs(v.mean()) / (v.std(ddof=1) / np.sqrt(B))
            (te if (A[i, j] or A[j, i]) else tn).append(float(tstat))
        power_rows.append({
            "seed": s, "num_anchors": int(B),
            "edge_t_median": float(np.median(te)), "edge_t_max": float(np.max(te)),
            "nonedge_t_median": float(np.median(tn)), "nonedge_t_max": float(np.max(tn)),
            "edges_above_196": int(np.sum(np.array(te) > 1.96)), "num_edges": len(te),
            "nonedges_above_196": int(np.sum(np.array(tn) > 1.96)), "num_nonedges": len(tn),
        })

    out["alpha_sweep"] = alpha_rows
    out["transform_compare"] = tr_rows
    out["tsubset_compare"] = sub_rows
    out["power_analysis"] = power_rows

    # ---- ER-10 anchor budget comparison ----------------------------------
    budget = []
    for tag, path in (("128 anchors", args.er10_small), ("512 anchors", args.er10_run)):
        f = os.path.join(path, "benchmark_result.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        budget.append({
            "label": tag,
            "num_anchors": r["config"]["num_anchors"],
            "fnr_pi": r["metrics"]["ordering"]["fnr_pi"],
            "das_f1": r["metrics"]["das"]["f1"],
            "cluster_f1": r["metrics"]["cluster"]["f1"],
            "das_pred": r["metrics"]["das"]["num_pred_edges"],
            "cluster_pred": r["metrics"]["cluster"]["num_pred_edges"],
            "num_true_edges": r["metrics"]["das"]["num_true_edges"],
        })
    out["budget_compare"] = budget

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)

    print(f"alpha rows      : {len(alpha_rows)}")
    print(f"transform rows  : {len(tr_rows)}")
    print(f"subset rows     : {len(sub_rows)}")
    print(f"power rows      : {len(power_rows)}")
    print(f"budget rows     : {len(budget)}")
    print(f"written -> {args.output}")


if __name__ == "__main__":
    main()
