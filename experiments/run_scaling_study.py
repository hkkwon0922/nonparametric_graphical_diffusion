"""Does more sampling buy perfect ordering recovery?

Sweeps the two independent axes that "more samples" could mean and records
whether the estimated order becomes **exactly valid** (FNR-pi = 0), not merely
better:

  * ``--mode data``   — data sample size ``n``. A fresh DDPM is trained per
                        ``n``, since the model must see the larger dataset.
  * ``--mode budget`` — estimator budget (anchors, and chains x retained
                        posterior samples) at fixed ``n``. Reuses one trained
                        checkpoint per graph, so this axis is cheap.

Both are run per graph size so the answer can be read against problem
difficulty: a 5-node graph may saturate while a 20-node one does not.

Perfect recovery is the strict criterion — `is_valid_order`, i.e. *zero*
reversed edges. Accuracy that merely improves is reported alongside it, because
"asymptotically better" and "eventually exact" are different claims.

Usage
-----
    python experiments/run_scaling_study.py --mode budget --device cuda:0
    python experiments/run_scaling_study.py --mode data   --device cuda:0
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

from data.benchmark_scm import generate_dataset
from models.dag_diffusion.benchmark_metrics import ordering_metrics
from models.dag_diffusion.conditional_langevin import LangevinConfig
from models.dag_diffusion.ordering import (
    ConditionalDiffusionDAGOrderEstimator,
    DAGOrderingConfig,
)
from models.dag_diffusion.training import (
    DDPMTrainConfig,
    load_ddpm_checkpoint,
    save_ddpm_checkpoint,
    train_ddpm,
)
from models.ddpm.core.ddpm_torch.utils import seed_all


def standardize(X):
    sd = X.std(axis=0)
    return ((X - X.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)).astype(np.float32)


def get_model(X, dim, ckpt_path, device, epochs, mid_features, timesteps,
              batch_size, seed, verbose=False):
    """Load a cached checkpoint or train one."""
    if os.path.exists(ckpt_path):
        model, diffusion, _, _ = load_ddpm_checkpoint(ckpt_path, device, expected_dim=dim)
        return model, diffusion, 0.0
    cfg = DDPMTrainConfig(
        input_dimension=dim, mid_features=mid_features, num_temporal_layers=3,
        timesteps=timesteps, epochs=epochs, batch_size=batch_size, lr=1e-3, seed=seed)
    t0 = time.time()
    model, diffusion, pre, _ = train_ddpm(X, cfg, device, verbose=verbose)
    save_ddpm_checkpoint(ckpt_path, model, cfg, pre, epoch=epochs)
    return model, diffusion, time.time() - t0


def run_ordering(X, model, diffusion, device, t_order, anchors, chains, samples,
                 seed, anchor_chunk=64, burn_in=300, thinning=10, step=1e-3):
    cfg = DAGOrderingConfig(
        t_order=t_order, num_anchors=anchors, anchor_chunk_size=anchor_chunk,
        langevin=LangevinConfig(num_chains=chains, burn_in=burn_in,
                                num_samples=samples, thinning=thinning,
                                step_size=step, init="forward_data",
                                chunk_size=32768),
        reverse_draws_per_xt=1, sampling_chunk_size=32768, seed=seed, verbose=False)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    r = ConditionalDiffusionDAGOrderEstimator(cfg).fit(X, model, diffusion, device=device)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    return r.topological_order, time.time() - t0


def build_argparser():
    p = argparse.ArgumentParser(description="Scaling study: does more sampling give exact recovery?")
    p.add_argument("--mode", choices=["data", "budget"], required=True)
    p.add_argument("--num-nodes", type=str, default="5,10,20")
    p.add_argument("--density", choices=["sparse", "dense"], default="dense")
    p.add_argument("--data-seeds", type=str, default="0,1,2")
    p.add_argument("--t-order", type=int, default=8)

    # data axis
    p.add_argument("--sample-sizes", type=str, default="500,1000,2000,5000,10000")
    p.add_argument("--fixed-anchors", type=int, default=512)
    p.add_argument("--fixed-chains", type=int, default=32)
    p.add_argument("--fixed-samples", type=int, default=16)

    # budget axis
    p.add_argument("--anchor-grid", type=str, default="64,128,256,512,1024,2048")
    p.add_argument("--fixed-n", type=int, default=1000)
    p.add_argument("--scale-chains", action="store_true", default=True,
                   help="also scale chains x retained with the anchor grid")

    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--mid-features", type=int, default=256)
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--seed", type=int, default=120)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ckpt-root", type=str, default="results/scaling/checkpoints")
    p.add_argument("--output", type=str, default=None)
    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; using CPU")
        device = torch.device("cpu")
    seed_all(args.seed)

    ckpt_root = args.ckpt_root
    if not os.path.isabs(ckpt_root):
        ckpt_root = os.path.normpath(os.path.join(REPO_ROOT, ckpt_root))
    os.makedirs(ckpt_root, exist_ok=True)

    out_path = args.output or f"results/dag_ordering_summary/scaling_{args.mode}.json"
    if not os.path.isabs(out_path):
        out_path = os.path.normpath(os.path.join(REPO_ROOT, out_path))

    node_list = [int(x) for x in args.num_nodes.split(",") if x.strip()]
    seeds = [int(x) for x in args.data_seeds.split(",") if x.strip()]
    rows = []
    start = time.time()

    for d in node_list:
        for ds in seeds:
            if args.mode == "data":
                grid = [int(x) for x in args.sample_sizes.split(",") if x.strip()]
            else:
                grid = [int(x) for x in args.anchor_grid.split(",") if x.strip()]

            for g in grid:
                n = g if args.mode == "data" else args.fixed_n
                anchors = args.fixed_anchors if args.mode == "data" else g
                if args.mode == "budget" and args.scale_chains:
                    # keep posterior samples growing with the anchor budget
                    chains = int(min(128, max(16, g // 8)))
                    samples = int(min(64, max(8, g // 32)))
                else:
                    chains, samples = args.fixed_chains, args.fixed_samples

                X_raw, A = generate_dataset(d, args.density, n, seed=ds)
                X = standardize(X_raw)
                anchors_eff = min(anchors, n)

                ckpt = os.path.join(
                    ckpt_root, f"d{d}_{args.density}_n{n}_seed{ds}.pt")
                model, diffusion, t_train = get_model(
                    X, d, ckpt, device, args.epochs, args.mid_features,
                    args.timesteps, args.batch_size, args.seed)

                order, t_ord = run_ordering(
                    X, model, diffusion, device, args.t_order, anchors_eff,
                    chains, samples, args.seed)
                m = ordering_metrics(order, A)
                row = {
                    "mode": args.mode, "num_nodes": d, "density": args.density,
                    "data_seed": ds, "n": n, "anchors": anchors_eff,
                    "chains": chains, "retained": samples,
                    "posterior_samples": chains * samples,
                    "num_true_edges": m["num_true_edges"],
                    "fnr_pi": m["fnr_pi"], "edge_accuracy": m["edge_accuracy"],
                    "ancestor_accuracy": m["ancestor_accuracy"],
                    "kendall_tau": m["kendall_tau"],
                    "num_violated_edges": m["num_violated_edges"],
                    "is_valid_order": m["is_valid_order"],
                    "train_seconds": t_train, "ordering_seconds": t_ord,
                }
                rows.append(row)
                axis = f"n={n}" if args.mode == "data" else f"anchors={anchors_eff}"
                print(f"D={d:2d} seed={ds} {axis:14s} "
                      f"edge_acc={m['edge_accuracy']:.3f} "
                      f"viol={m['num_violated_edges']:2d}/{m['num_true_edges']:2d} "
                      f"{'PERFECT' if m['is_valid_order'] else ''} "
                      f"({t_ord:.0f}s)", flush=True)

                del model, diffusion
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    payload = {"mode": args.mode, "rows": rows,
               "config": {k: v for k, v in vars(args).items()},
               "total_seconds": time.time() - start}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)

    n_perfect = sum(1 for r in rows if r["is_valid_order"])
    print(f"\nperfect recoveries: {n_perfect}/{len(rows)}")
    print(f"written -> {out_path}   ({payload['total_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
