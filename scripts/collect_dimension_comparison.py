"""Cross-dimension comparison of the full pipeline at identical settings.

Collects the ``sweep_summary.json`` written by ``run_benchmark_sweep`` for each
graph size and lines them up, so the effect of dimension is isolated: every run
uses the same t_order, anchor budget, Langevin settings, epochs and seeds — only
``D`` changes.

Reports both stages separately (ordering FNR-pi / edge accuracy, and edge
F1/FNR/FPR for each parent selector) plus the oracle-order variants, so the
growth of each error source with ``D`` can be read off directly.

Usage
-----
    python scripts/collect_dimension_comparison.py
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.dag_diffusion.benchmark_metrics import ordering_metrics

METHODS = ["das", "cluster", "das_oracle_order", "cluster_oracle_order"]


def main():
    p = argparse.ArgumentParser(description="Compare the pipeline across graph sizes.")
    p.add_argument("--roots", default=(
        "5:results/causal_benchmark/er5_dense_sweep,"
        "10:results/causal_benchmark/er10_dense_sweep,"
        "20:results/causal_benchmark/er20_dense"))
    p.add_argument("--output",
                   default="results/dag_ordering_summary/dimension_comparison.json")
    args = p.parse_args()

    entries = []
    for chunk in args.roots.split(","):
        d_str, root = chunk.split(":", 1)
        root = root if os.path.isabs(root) else os.path.join(REPO_ROOT, root)
        entries.append((int(d_str), os.path.normpath(root)))

    out = {"per_dimension": [], "note": (
        "identical settings across D (t_order=8, 512 anchors, 64 chains x 32 "
        "retained, 2000 epochs, seeds 0/1/2); only the graph size changes")}

    for d, root in sorted(entries):
        path = os.path.join(root, "sweep_summary.json")
        if not os.path.exists(path):
            print(f"[skip] D={d}: missing {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            S = json.load(f)
        agg = S["aggregate"]

        # recompute the richer ordering metrics from the stored orders
        ord_rows = []
        for seed_rec in S["per_seed"]:
            npz = os.path.join(root, f"seed{seed_rec['data_seed']}", "adjacencies.npz")
            if not os.path.exists(npz):
                continue
            z = np.load(npz)
            ord_rows.append(ordering_metrics([int(i) for i in z["order"]], z["true"]))

        row = {
            "num_nodes": d,
            "num_true_edges_mean": float(np.mean(
                [r["num_true_edges"] for r in ord_rows])) if ord_rows else None,
            "ordering": {
                "fnr_pi_median": agg["ordering_fnr_pi"]["median"],
                "fnr_pi_q25": agg["ordering_fnr_pi"]["q25"],
                "fnr_pi_q75": agg["ordering_fnr_pi"]["q75"],
                "edge_accuracy_mean": float(np.mean(
                    [r["edge_accuracy"] for r in ord_rows])) if ord_rows else None,
                "ancestor_accuracy_mean": float(np.mean(
                    [r["ancestor_accuracy"] for r in ord_rows])) if ord_rows else None,
                "kendall_tau_mean": float(np.mean(
                    [r["kendall_tau"] for r in ord_rows])) if ord_rows else None,
                "violated_mean": float(np.mean(
                    [r["num_violated_edges"] for r in ord_rows])) if ord_rows else None,
                "num_perfect": int(sum(r["is_valid_order"] for r in ord_rows)),
                "num_runs": len(ord_rows),
            },
            "random_baseline": {k: agg["random_baseline"][k]["median"]
                                for k in ("fnr_pi", "f1", "fnr", "fpr")},
            "ordering_seconds_median": float(np.median(
                [r["timings_seconds"]["ordering_seconds"] for r in S["per_seed"]])),
            "total_seconds_median": float(np.median(
                [r["timings_seconds"]["total_seconds"] for r in S["per_seed"]])),
        }
        for m in METHODS:
            row[m] = {k: agg[m][k]["median"] for k in ("f1", "fnr", "fpr")}
        out["per_dimension"].append(row)

    # ---- print ----------------------------------------------------------
    print("=" * 78)
    print("차원별 비교 (동일 설정: t_order=8, anchors=512, seeds 0/1/2)")
    print("=" * 78)
    print(f"\n[순서 단계]")
    print(f"  {'D':>3} {'엣지수':>6} {'FNR-pi':>8} {'edge acc':>9} {'anc acc':>8} "
          f"{'tau':>7} {'뒤집힘':>8} {'perfect':>8}")
    print("  " + "-" * 68)
    for r in out["per_dimension"]:
        o = r["ordering"]
        print(f"  {r['num_nodes']:3d} {r['num_true_edges_mean']:6.0f} "
              f"{o['fnr_pi_median']:8.3f} {o['edge_accuracy_mean']:9.3f} "
              f"{o['ancestor_accuracy_mean']:8.3f} {o['kendall_tau_mean']:7.3f} "
              f"{o['violated_mean']:8.1f} {o['num_perfect']:4d}/{o['num_runs']}")

    print(f"\n[엣지 단계 F1]")
    print(f"  {'D':>3} {'random':>8} {'das':>8} {'cluster':>9} {'das(or)':>9} "
          f"{'clu(or)':>9}")
    print("  " + "-" * 56)
    for r in out["per_dimension"]:
        print(f"  {r['num_nodes']:3d} {r['random_baseline']['f1']:8.3f} "
              f"{r['das']['f1']:8.3f} {r['cluster']['f1']:9.3f} "
              f"{r['das_oracle_order']['f1']:9.3f} "
              f"{r['cluster_oracle_order']['f1']:9.3f}")

    print(f"\n[비용]")
    for r in out["per_dimension"]:
        print(f"  D={r['num_nodes']:3d}  ordering {r['ordering_seconds_median']:7.1f}s"
              f"   전체 {r['total_seconds_median']:7.1f}s")

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.normpath(os.path.join(REPO_ROOT, out_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
