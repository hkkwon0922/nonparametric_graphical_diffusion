"""Regenerate the measurement summary consumed by ``main.ipynb``.

Runs the sweeps reported in the notebook — ``t_order`` sensitivity, anchor
count, seed stability, sampling budget, dimension scaling and CPU/GPU timing —
against a trained checkpoint, and writes a single JSON summary.

Every number in ``main.ipynb`` comes from this script, which drives exactly the
same code path as ``experiments/run_dag_ordering.py``.

Usage
-----
    # first train a model (or point --checkpoint at an existing one)
    python data/make_dag_ordering_debug_data.py --output-dir data/dag_ordering_debug
    python experiments/run_dag_ordering.py \\
        --data-path data/dag_ordering_debug/dag3_chain.npy \\
        --checkpoint-path results/dag_ordering/real_t3/ddpm.pt --train-if-missing \\
        --output-dir results/dag_ordering/real_t3 \\
        --epochs 1500 --timesteps 500 --mid-features 128 \\
        --t-order 3 --num-anchors 64 --device cuda:0

    python scripts/collect_sweep_results.py --device cuda:0

Note
----
CPU and CUDA ``torch.Generator`` streams differ, so results are reproducible
*within* a device but not identical across devices. The notebook discusses this
explicitly; it is not a bug.
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
from models.dag_diffusion.training import DDPMTrainConfig, load_ddpm_checkpoint, train_ddpm


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def run_ordering(X, A, model, diffusion, device, t_order, num_anchors,
                 chains=32, samples=16, burn=300, thin=10, step=1e-3, seed=120):
    """One ordering run -> a JSON-ready record with timings and criteria."""
    cfg = DAGOrderingConfig(
        t_order=t_order, num_anchors=num_anchors, anchor_chunk_size=32,
        langevin=LangevinConfig(num_chains=chains, burn_in=burn, num_samples=samples,
                                thinning=thin, step_size=step, init="forward_data"),
        reverse_draws_per_xt=1, sampling_chunk_size=8192, seed=seed, verbose=False,
    )
    _sync(device)
    t0 = time.time()
    result = ConditionalDiffusionDAGOrderEstimator(cfg).fit(X, model, diffusion, device=device)
    _sync(device)
    wall = time.time() - t0

    ev = evaluate_ordering(result, A)
    return {
        "t_order": t_order, "num_anchors": num_anchors, "seed": seed,
        "chains": chains, "samples": samples, "burn_in": burn, "thinning": thin,
        "topological_order": result.topological_order,
        "leaf_order": result.leaf_order,
        "order_fnr": ev["order_fnr"]["order_fnr"],
        "leaf_valid": ev["stagewise_leaf_validity"]["num_valid_stages"],
        "leaf_stages": ev["stagewise_leaf_validity"]["num_stages"],
        "stage0_criterion": result.stage_records[0]["criterion"].tolist(),
        "criterion_by_stage": [{str(k): float(v) for k, v in s.items()}
                               for s in result.criterion_by_stage],
        "wall_seconds": wall,
        "stage_seconds": [float(s["runtime_seconds"]) for s in result.stage_records],
        "peak_cuda_bytes": result.peak_cuda_memory_bytes,
        "correct": result.topological_order == [0, 1, 2],
    }


def main():
    p = argparse.ArgumentParser(description="Regenerate main.ipynb's measurement summary.")
    p.add_argument("--data-path", default="data/dag_ordering_debug/dag3_chain.npy")
    p.add_argument("--adjacency-path", default="data/dag_ordering_debug/dag3_chain_adjacency.npy")
    p.add_argument("--checkpoint", default="results/dag_ordering/real_t3/ddpm.pt")
    p.add_argument("--output", default="results/dag_ordering_summary/sweep_summary.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-d-scaling", action="store_true", default=False)
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
    model, diffusion, _, _ = load_ddpm_checkpoint(
        args.checkpoint, device, expected_dim=X.shape[1])

    out = {"device": str(device), "true_order": [0, 1, 2], "adjacency": A.tolist()}

    print("=== t_order sweep (B=128) ===", flush=True)
    out["t_sweep"] = []
    for t in [1, 2, 3, 5, 8, 12, 20, 30, 50]:
        rec = run_ordering(X, A, model, diffusion, device, t, 128)
        out["t_sweep"].append(rec)
        print(f"  t={t:3d} crit={np.round(rec['stage0_criterion'], 2)} "
              f"order={rec['topological_order']} {rec['wall_seconds']:.2f}s", flush=True)

    print("=== anchor sweep (t=8) ===", flush=True)
    out["anchor_sweep"] = []
    for b in [4, 8, 16, 32, 64, 128, 256]:
        rec = run_ordering(X, A, model, diffusion, device, 8, b)
        out["anchor_sweep"].append(rec)
        print(f"  B={b:4d} order={rec['topological_order']} "
              f"{rec['wall_seconds']:.2f}s", flush=True)

    print("=== seed stability (t=8, B=128) ===", flush=True)
    out["seed_sweep"] = []
    for s in [120, 1230, 12340, 123450, 1234560]:
        rec = run_ordering(X, A, model, diffusion, device, 8, 128, seed=s)
        out["seed_sweep"].append(rec)
        print(f"  seed={s:8d} order={rec['topological_order']}", flush=True)

    print("=== sampling budget (t=8, B=64) ===", flush=True)
    out["cost_sweep"] = []
    for ch, sm in [(8, 8), (16, 8), (16, 16), (32, 16), (32, 32), (64, 32)]:
        rec = run_ordering(X, A, model, diffusion, device, 8, 64, chains=ch, samples=sm)
        rec["posterior_samples_per_anchor"] = ch * sm
        out["cost_sweep"].append(rec)
        print(f"  chains={ch:3d} retained={sm:3d} M={ch * sm:5d} "
              f"order={rec['topological_order']} {rec['wall_seconds']:.2f}s", flush=True)

    print("=== CPU stability (t=8) ===", flush=True)
    out["cpu_stability"] = []
    cpu_model, cpu_diff, _, _ = load_ddpm_checkpoint(args.checkpoint, "cpu", expected_dim=X.shape[1])
    for b in [64, 128, 256]:
        for s in [120, 1230, 12340]:
            rec = run_ordering(X, A, cpu_model, cpu_diff, "cpu", 8, b, seed=s)
            out["cpu_stability"].append(
                {"B": b, "seed": s, "order": rec["topological_order"],
                 "correct": rec["correct"]})
            print(f"  B={b:4d} seed={s:6d} order={rec['topological_order']} "
                  f"{'OK' if rec['correct'] else 'WRONG'}", flush=True)

    print("=== device comparison (t=8, B=64) ===", flush=True)
    out["device_compare"] = []
    for d in (["cpu", device] if device != "cpu" else ["cpu"]):
        m, dif, _, _ = load_ddpm_checkpoint(args.checkpoint, d, expected_dim=X.shape[1])
        rec = run_ordering(X, A, m, dif, d, 8, 64)
        out["device_compare"].append(
            {"device": d, "ordering_seconds": rec["wall_seconds"],
             "order": rec["topological_order"], "peak_cuda_bytes": rec["peak_cuda_bytes"]})
        print(f"  {d}: {rec['wall_seconds']:.2f}s order={rec['topological_order']}", flush=True)

    out["d_scaling"] = []
    if not args.skip_d_scaling:
        print("=== dimension scaling (freshly trained tiny models) ===", flush=True)
        rng = np.random.default_rng(0)
        for D in [3, 5, 8, 12]:
            Xd = rng.normal(size=(800, D)).astype(np.float32)
            cfg = DDPMTrainConfig(input_dimension=D, mid_features=64, num_temporal_layers=2,
                                  timesteps=100, epochs=60, batch_size=128, lr=2e-3, seed=1)
            t0 = time.time()
            m, dif, _, _ = train_ddpm(Xd, cfg, device, verbose=False)
            t_train = time.time() - t0

            ocfg = DAGOrderingConfig(
                t_order=8, num_anchors=32, anchor_chunk_size=32,
                langevin=LangevinConfig(num_chains=16, burn_in=100, num_samples=8,
                                        thinning=5, step_size=1e-3, init="forward_data"),
                reverse_draws_per_xt=1, sampling_chunk_size=8192, seed=120, verbose=False)
            _sync(device)
            t0 = time.time()
            r = ConditionalDiffusionDAGOrderEstimator(ocfg).fit(Xd, m, dif, device=device)
            _sync(device)
            t_order = time.time() - t0

            out["d_scaling"].append({
                "D": D, "train_seconds": t_train, "ordering_seconds": t_order,
                "num_stages": len(r.stage_records),
                "stage_seconds": [float(s["runtime_seconds"]) for s in r.stage_records],
                "peak_cuda_bytes": r.peak_cuda_memory_bytes})
            print(f"  D={D:3d} train={t_train:.1f}s order={t_order:.2f}s", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    print(f"\nsummary written: {args.output}")


if __name__ == "__main__":
    main()
