"""Summarise the scaling study: does more sampling give *exact* ordering recovery?

Reads the two JSON files written by ``experiments/run_scaling_study.py`` and
reports, per graph size, whether perfect recovery (``is_valid_order``, i.e. zero
reversed edges) is attained and how accuracy trends with each axis.

The two axes answer different questions:

* **budget** — more anchors / posterior samples at fixed ``n``. This reduces
  *estimator* variance only; it cannot fix a bias in the criterion itself.
* **data** — larger ``n`` with a retrained model. This improves the learned
  score, so it can in principle reduce both variance and model error.

Note that the budget axis is capped by ``n``: anchors are drawn from data rows,
so ``anchors <= n``. The two axes are therefore not fully independent.

Usage
-----
    python scripts/analyze_scaling.py
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load(path):
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(REPO_ROOT, path))
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_axis(rows, axis_key):
    """Group rows by (num_nodes, axis value) and aggregate over seeds."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["num_nodes"], r[axis_key])].append(r)

    out = []
    for (d, x), rs in sorted(grouped.items()):
        out.append({
            "num_nodes": d, axis_key: x,
            "n_runs": len(rs),
            "edge_accuracy_mean": float(np.mean([r["edge_accuracy"] for r in rs])),
            "edge_accuracy_std": float(np.std([r["edge_accuracy"] for r in rs], ddof=1))
            if len(rs) > 1 else 0.0,
            "kendall_tau_mean": float(np.mean([r["kendall_tau"] for r in rs])),
            "violated_mean": float(np.mean([r["num_violated_edges"] for r in rs])),
            "true_edges_mean": float(np.mean([r["num_true_edges"] for r in rs])),
            "num_perfect": int(sum(r["is_valid_order"] for r in rs)),
            "frac_perfect": float(np.mean([r["is_valid_order"] for r in rs])),
            "ordering_seconds_mean": float(np.mean([r["ordering_seconds"] for r in rs])),
        })
    return out


def report(name, rows, axis_key):
    print(f"\n{'=' * 74}")
    print(f"{name}  (axis = {axis_key})")
    print("=" * 74)
    summ = summarize_axis(rows, axis_key)
    by_d = defaultdict(list)
    for s in summ:
        by_d[s["num_nodes"]].append(s)

    for d in sorted(by_d):
        print(f"\n  D = {d}   (평균 참 엣지 {by_d[d][0]['true_edges_mean']:.0f}개)")
        print(f"  {axis_key:>10} {'edge acc':>16} {'tau':>7} {'뒤집힘':>8} "
              f"{'perfect':>9} {'시간(s)':>8}")
        print("  " + "-" * 66)
        for s in by_d[d]:
            print(f"  {s[axis_key]:>10} {s['edge_accuracy_mean']:8.3f} "
                  f"+/-{s['edge_accuracy_std']:.3f} {s['kendall_tau_mean']:7.3f} "
                  f"{s['violated_mean']:8.1f} {s['num_perfect']:5d}/{s['n_runs']:<3d} "
                  f"{s['ordering_seconds_mean']:8.0f}")
    return summ


def main():
    p = argparse.ArgumentParser(description="Analyse the scaling study.")
    p.add_argument("--budget", default="results/dag_ordering_summary/scaling_budget.json")
    p.add_argument("--data", default="results/dag_ordering_summary/scaling_data.json")
    p.add_argument("--output", default="results/dag_ordering_summary/scaling_summary.json")
    args = p.parse_args()

    payload = {}
    B, D = load(args.budget), load(args.data)

    if B:
        payload["budget"] = report("추정 예산 축 (anchors, n 고정)", B["rows"], "anchors")
        payload["budget_rows"] = B["rows"]
    else:
        print(f"[skip] no budget file at {args.budget}")

    if D:
        payload["data"] = report("데이터 표본 축 (n, 예산 고정)", D["rows"], "n")
        payload["data_rows"] = D["rows"]
    else:
        print(f"[skip] no data file at {args.data}")

    # ---- headline: does perfect recovery ever happen at D=20? -------------
    verdict = {}
    for key, rows in (("budget", B["rows"] if B else []), ("data", D["rows"] if D else [])):
        for d in sorted({r["num_nodes"] for r in rows}):
            sub = [r for r in rows if r["num_nodes"] == d]
            best = max(sub, key=lambda r: r["edge_accuracy"])
            verdict[f"{key}_D{d}"] = {
                "any_perfect": bool(any(r["is_valid_order"] for r in sub)),
                "num_perfect": int(sum(r["is_valid_order"] for r in sub)),
                "num_runs": len(sub),
                "best_edge_accuracy": float(best["edge_accuracy"]),
                "best_setting": {k: best[k] for k in ("n", "anchors", "posterior_samples")},
            }
    payload["verdict"] = verdict

    print(f"\n{'=' * 74}")
    print("PERFECT RECOVERY 여부 요약")
    print("=" * 74)
    for k, v in sorted(verdict.items()):
        mark = "YES" if v["any_perfect"] else "NO "
        print(f"  {k:14s} {mark}  {v['num_perfect']:2d}/{v['num_runs']:2d} 회   "
              f"최고 edge acc {v['best_edge_accuracy']:.3f}")

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.normpath(os.path.join(REPO_ROOT, out_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
