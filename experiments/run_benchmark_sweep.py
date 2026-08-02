"""Multi-seed driver for the causal-discovery benchmark.

Repeats ``run_causal_benchmark`` over seeds and aggregates the metrics the way
Montagna et al. (2023) report them (median / quartiles over 20 seeds), so the
numbers line up with their violin plots.

Also includes a **random baseline** matching their Appendix C.10: sample a
random topological order, take the fully connected DAG it admits, and keep each
edge with probability 0.5.

Examples
--------
    # the paper's headline configuration
    python experiments/run_benchmark_sweep.py \\
        --num-nodes 20 --density dense --num-samples 1000 \\
        --seeds 0,1,2 --output-root results/causal_benchmark/er20_dense \\
        --device cuda:0

    # a cheaper pilot
    python experiments/run_benchmark_sweep.py --num-nodes 10 --seeds 0,1,2
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.run_causal_benchmark import main as run_one
from models.dag_diffusion.benchmark_metrics import edge_metrics, fnr_pi


def random_baseline(num_nodes, true_adjacency, seed=0, keep_prob=0.5):
    """Paper's random baseline (Appendix C.10) -> metrics dict.

    Samples a random topological order, takes the fully connected DAG it
    admits, and keeps each edge with probability ``keep_prob``.

    The generator is deliberately offset away from ``seed``: ``sample_er_dag``
    uses ``default_rng(seed)`` and draws its own permutation first, so reusing
    the same seed here would reproduce the ground-truth graph's permutation
    exactly and hand the "random" baseline a perfect order (FNR-pi = 0).
    """
    rng = np.random.default_rng(int(seed) + 987_654_321)
    order = list(rng.permutation(num_nodes))
    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    for a in range(num_nodes):
        for b in range(a + 1, num_nodes):
            if rng.random() < keep_prob:
                adj[order[a], order[b]] = 1
    out = {"fnr_pi": fnr_pi(order, true_adjacency)}
    out.update(edge_metrics(adj, true_adjacency))
    return out


def summarize(values):
    """Median and quartiles of a metric across seeds (nan-safe)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"median": None, "q25": None, "q75": None, "mean": None, "n": 0}
    return {
        "median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)), "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size),
    }


def build_argparser():
    p = argparse.ArgumentParser(description="Multi-seed causal benchmark sweep.")
    p.add_argument("--num-nodes", type=int, default=20)
    p.add_argument("--density", choices=["sparse", "dense"], default="dense")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--output-root", type=str,
                   default="results/causal_benchmark/sweep")

    p.add_argument("--t-order", type=int, default=8)
    p.add_argument("--num-anchors", type=int, default=128)
    p.add_argument("--anchor-chunk-size", type=int, default=32)
    p.add_argument("--num-chains", type=int, default=16)
    p.add_argument("--langevin-burn-in", type=int, default=200)
    p.add_argument("--langevin-samples", type=int, default=8)
    p.add_argument("--langevin-thinning", type=int, default=5)
    p.add_argument("--langevin-step-size", type=float, default=1e-3)
    p.add_argument("--edge-t-values", type=str, default="3,5,8,12,20")
    p.add_argument("--t-edge", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--cluster-transform", choices=["rank", "zscore"], default="rank")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--mid-features", type=int, default=128)
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--seed", type=int, default=120)
    p.add_argument("--device", type=str, default="cuda:0")
    return p


def main():
    args = build_argparser().parse_args()
    out_root = args.output_root
    if not os.path.isabs(out_root):
        out_root = os.path.normpath(os.path.join(REPO_ROOT, out_root))
    os.makedirs(out_root, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    runs, baselines = [], []
    sweep_start = time.time()

    for k, data_seed in enumerate(seeds, start=1):
        print("=" * 78)
        print(f"[seed {k}/{len(seeds)}] data_seed={data_seed}", flush=True)
        run_dir = os.path.join(out_root, f"seed{data_seed}")
        payload = run_one([
            "--generate",
            "--num-nodes", str(args.num_nodes),
            "--density", args.density,
            "--num-samples", str(args.num_samples),
            "--data-seed", str(data_seed),
            "--output-dir", run_dir,
            "--t-order", str(args.t_order),
            "--num-anchors", str(args.num_anchors),
            "--anchor-chunk-size", str(args.anchor_chunk_size),
            "--num-chains", str(args.num_chains),
            "--langevin-burn-in", str(args.langevin_burn_in),
            "--langevin-samples", str(args.langevin_samples),
            "--langevin-thinning", str(args.langevin_thinning),
            "--langevin-step-size", str(args.langevin_step_size),
            "--edge-t-values", args.edge_t_values,
            "--alpha", str(args.alpha),
            "--cluster-transform", args.cluster_transform,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--mid-features", str(args.mid_features),
            "--timesteps", str(args.timesteps),
            "--seed", str(args.seed),
            "--device", args.device,
            "--quiet",
        ] + ([] if args.t_edge is None else ["--t-edge", str(args.t_edge)]))
        payload["data_seed"] = data_seed
        runs.append(payload)

        A_true = np.load(os.path.join(run_dir, "adjacencies.npz"))["true"]
        baselines.append(random_baseline(args.num_nodes, A_true, seed=data_seed))

    # ---- aggregate ------------------------------------------------------
    methods = ["das", "cluster", "das_oracle_order", "cluster_oracle_order"]
    agg = {
        "ordering_fnr_pi": summarize([r["metrics"]["ordering"]["fnr_pi"] for r in runs]),
        "random_baseline": {
            "fnr_pi": summarize([b["fnr_pi"] for b in baselines]),
            "f1": summarize([b["f1"] for b in baselines]),
            "fnr": summarize([b["fnr"] for b in baselines]),
            "fpr": summarize([b["fpr"] for b in baselines]),
        },
    }
    for m in methods:
        agg[m] = {key: summarize([r["metrics"][m][key] for r in runs])
                  for key in ("f1", "fnr", "fpr")}

    summary = {
        "setting": {
            "scenario": "vanilla ANM, GP mechanisms, Gaussian noise "
                        "(Montagna et al. 2023, Sec. 3.1)",
            "num_nodes": args.num_nodes, "density": args.density,
            "num_samples": args.num_samples, "seeds": seeds,
            "t_order": args.t_order, "num_anchors": args.num_anchors,
        },
        "aggregate": agg,
        "per_seed": [
            {"data_seed": r["data_seed"],
             "fnr_pi": r["metrics"]["ordering"]["fnr_pi"],
             **{m: r["metrics"][m] for m in methods},
             "timings_seconds": r["timings_seconds"]}
            for r in runs
        ],
        "sweep_seconds": time.time() - sweep_start,
    }
    path = os.path.join(out_root, "sweep_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("=" * 78)
    print(f"ER-{args.num_nodes} {args.density}, n={args.num_samples}, "
          f"{len(seeds)} seed(s)")
    o = agg["ordering_fnr_pi"]
    rb = agg["random_baseline"]
    print(f"  ordering FNR-pi : median {o['median']:.3f} "
          f"[{o['q25']:.3f}, {o['q75']:.3f}]   "
          f"(random baseline {rb['fnr_pi']['median']:.3f})")
    for m in methods:
        f1, fnr, fpr = agg[m]["f1"], agg[m]["fnr"], agg[m]["fpr"]
        print(f"  {m:22s} F1 {f1['median']:.3f} [{f1['q25']:.3f}, {f1['q75']:.3f}]  "
              f"FNR {fnr['median']:.3f}  FPR {fpr['median']:.3f}")
    print(f"  random baseline        F1 {rb['f1']['median']:.3f}  "
          f"FNR {rb['fnr']['median']:.3f}  FPR {rb['fpr']['median']:.3f}")
    print(f"summary -> {path}   ({summary['sweep_seconds']:.0f}s)")
    return summary


if __name__ == "__main__":
    main()
