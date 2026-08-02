"""Ordering-stage performance across saved runs, with a random-order reference.

The edge metrics (F1/FNR/FPR) mix two error sources; this isolates the
**ordering stage**. For each saved run it computes FNR-pi, edge accuracy,
ancestor accuracy and the constrained Kendall tau, and compares them against a
Monte-Carlo distribution of *random* permutations on the same graph — so the
reference is the actual chance level for that DAG, not a hand-waved 0.5.

Reads the stored ``adjacencies.npz`` files only; nothing re-runs the model.

Usage
-----
    python scripts/collect_ordering_performance.py
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


def random_order_reference(adjacency, num_draws=2000, seed=0):
    """Monte-Carlo distribution of the ordering metrics under a random permutation."""
    rng = np.random.default_rng(int(seed))
    d = adjacency.shape[0]
    keys = ("fnr_pi", "edge_accuracy", "ancestor_accuracy", "kendall_tau")
    acc = {k: [] for k in keys}
    valid = 0
    for _ in range(int(num_draws)):
        m = ordering_metrics(list(rng.permutation(d)), adjacency)
        for k in keys:
            acc[k].append(m[k])
        valid += int(m["is_valid_order"])
    out = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
               "q05": float(np.percentile(v, 5)), "q95": float(np.percentile(v, 95))}
           for k, v in acc.items()}
    out["prob_valid_order"] = valid / float(num_draws)
    out["num_draws"] = int(num_draws)
    return out


def main():
    p = argparse.ArgumentParser(description="Ordering-stage performance summary.")
    p.add_argument("--runs", default=(
        "results/causal_benchmark/er20_dense/seed0,"
        "results/causal_benchmark/er20_dense/seed1,"
        "results/causal_benchmark/er20_dense/seed2"))
    p.add_argument("--labels", default="seed0,seed1,seed2")
    p.add_argument("--output",
                   default="results/dag_ordering_summary/ordering_performance.json")
    p.add_argument("--num-draws", default=2000, type=int)
    args = p.parse_args()

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.normpath(os.path.join(REPO_ROOT, out_path))

    rows, refs = [], []
    for label, run in zip(labels, runs):
        path = run if os.path.isabs(run) else os.path.join(REPO_ROOT, run)
        npz = os.path.join(path, "adjacencies.npz")
        if not os.path.exists(npz):
            print(f"[skip] missing {npz}")
            continue
        z = np.load(npz)
        A, order = z["true"], [int(i) for i in z["order"]]
        m = ordering_metrics(order, A)
        m["label"] = label
        m["num_nodes"] = int(A.shape[0])
        rows.append(m)
        refs.append(random_order_reference(A, num_draws=args.num_draws,
                                           seed=hash(label) % 10_000))
        print(f"{label}: FNR-pi={m['fnr_pi']:.3f}  edge_acc={m['edge_accuracy']:.3f}  "
              f"anc_acc={m['ancestor_accuracy']:.3f}  tau={m['kendall_tau']:.3f}  "
              f"({m['num_violated_edges']}/{m['num_true_edges']} edges reversed)")

    keys = ("fnr_pi", "edge_accuracy", "ancestor_accuracy", "kendall_tau")
    agg = {k: {"median": float(np.median([r[k] for r in rows])),
               "q25": float(np.percentile([r[k] for r in rows], 25)),
               "q75": float(np.percentile([r[k] for r in rows], 75)),
               "mean": float(np.mean([r[k] for r in rows]))} for k in keys}
    ref_agg = {k: {"mean": float(np.mean([r[k]["mean"] for r in refs])),
                   "std": float(np.mean([r[k]["std"] for r in refs]))} for k in keys}
    ref_agg["prob_valid_order"] = float(np.mean([r["prob_valid_order"] for r in refs]))

    # z-score of our result against the random-order distribution
    zscores = {}
    for k in keys:
        mu = ref_agg[k]["mean"]
        sd = max(ref_agg[k]["std"], 1e-12)
        zscores[k] = float((agg[k]["mean"] - mu) / sd)

    payload = {
        "per_run": rows,
        "aggregate": agg,
        "random_order_reference": ref_agg,
        "zscore_vs_random": zscores,
        "note": ("ancestor_accuracy scores the transitive closure, so indirect "
                 "constraints count too; kendall_tau is that value rescaled to "
                 "[-1, 1]. Any valid topological order attains 1.0."),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)

    print("\naggregate (median):")
    for k in keys:
        print(f"  {k:20s} {agg[k]['median']:.3f}   random {ref_agg[k]['mean']:.3f}"
              f" +/- {ref_agg[k]['std']:.3f}   z = {zscores[k]:+.1f}")
    print(f"  P(random order is valid) = {ref_agg['prob_valid_order']:.2e}")
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
