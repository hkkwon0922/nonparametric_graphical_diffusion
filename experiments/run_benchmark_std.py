"""
==============================================================================
DDPM benchmark on standardized inputs (std counterpart of run_benchmark.py)
==============================================================================
Identical to the DDPM path of ``run_benchmark.py`` except that each training
matrix is z-scored column-wise before being handed to ``DDPMEstimator``:

    X <- (X - mean) / std          (per feature, ddof=1)

Everything else -- estimator hyperparameters (batch 100, lr 1e-3, 1000 epochs,
t = 1..30, 5000 samples per t, 128 x0 anchors), the true skeleton, the metrics
and the on-disk layout -- matches the raw benchmark, so the resulting JSONs drop
straight into ``load_all_benchmark_results`` alongside the existing ones.

Results are written to ``results_std/ddpm/{dataset}/N_{n}/results_seed{seed}.json``.

Usage (from the repository root):
    $ python experiments/run_benchmark_std.py --gpu 0 --shard 0 --num_shards 4
==============================================================================
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import time
import json
import glob
import re
import argparse

import numpy as np
import torch

import utils


# Datasets that actually appear in plot_results.ipynb:
#   Figure 1 (3 metrics x 4 datasets) and Figure 2 (1x3, two datasets).
PLOT_DATASETS = [
    "dim20_butterfly",
    "dim20_cop_gau",
    "0.7_dim20_pair_gau",
    "0.3_dim20_pair_gau",
    "dim5_cop_gau",
    "dim6_butterfly",
]
DATA_SIZES = [100, 200, 300, 400, 500, 1000, 2000, 5000, 10000, 20000, 50000]
SEEDS = [120, 1230, 12340, 123450, 1234560]


def parse_args():
    p = argparse.ArgumentParser(description="DDPM benchmark on standardized inputs")
    p.add_argument("--data_dir", type=str, default="../data/raw")
    p.add_argument("--out_dir", type=str, default="../results_std")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index to pin this worker to.")
    p.add_argument("--shard", type=int, default=0, help="This worker's index within the shard set.")
    p.add_argument("--num_shards", type=int, default=1, help="Total number of parallel workers.")
    p.add_argument("--datasets", type=str, default=None,
                   help="Comma-separated dataset override (default: the six plotted ones).")
    return p.parse_args()


def standardize(X):
    """Column-wise z-score; constant columns are left centred (sd -> 1)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd


def build_jobs(datasets):
    """Enumerate (dataset, D, n, seed) in a deterministic order, cheapest first.

    Sorting by n keeps every shard's workload balanced: the expensive n=50000
    runs are spread across shards rather than piling onto whichever worker
    happens to reach them last.
    """
    jobs = []
    for n in DATA_SIZES:
        for ds in datasets:
            match = re.search(r"dim(\d+)", ds)
            D = int(match.group(1)) if match else 20
            for seed in SEEDS:
                jobs.append((ds, D, n, seed))
    return jobs


def main():
    args = parse_args()
    os.chdir(current_dir)  # keep the ../data, ../results_std defaults meaningful

    datasets = args.datasets.split(",") if args.datasets else PLOT_DATASETS
    jobs = build_jobs(datasets)
    mine = [j for k, j in enumerate(jobs) if k % args.num_shards == args.shard]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[shard {args.shard}/{args.num_shards}] device={device} jobs={len(mine)}", flush=True)

    from models.ddpm.ddpm import DDPMEstimator

    done = skipped = failed = 0
    for idx, (ds, D, n, seed) in enumerate(mine):
        out_path = os.path.join(args.out_dir, "ddpm", ds, f"N_{n}", f"results_seed{seed}.json")
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            pattern = os.path.join(args.data_dir, ds, "train", f"*_n{n}_seed{seed}.npy")
            files = glob.glob(pattern)
            if not files:
                raise FileNotFoundError(pattern)

            X = np.load(files[0])[:n, :]
            lam_bar_raw = float(np.trace(np.cov(X, rowvar=False)) / X.shape[1])
            Xs = standardize(X)

            true_skel = utils.extract_true_skeleton(ds, D)

            # seed the estimator with the data seed so each (n, seed) cell is
            # reproducible and independent, exactly as the raw benchmark does.
            est = DDPMEstimator(seed=seed, device=device)

            t0 = time.time()
            est_graph, omega, meta_info = est.fit_predict(Xs)
            elapsed = time.time() - t0

            metrics = utils.calculate_metrics(est_graph, true_skel)

            if meta_info is None:
                meta_info = {}
            meta_info.update({
                "source": "standardized",
                "version": "ddpm_hessian_v1_std",
                "preprocessing": "per-feature z-score (ddof=1)",
                "lambda_bar_raw": lam_bar_raw,
                "lambda_bar_std": 1.0,
                "hyperparams": {
                    "batch_size": est.batch_size, "lr": est.lr, "epochs": est.epochs,
                    "t_list": [est.t_list[0], est.t_list[-1]],
                    "num_samples_per_t": est.num_samples_per_t, "num_x0": 128,
                },
            })

            payload = {
                "model": "DDPM",
                "dataset": ds,
                "N": n,
                "seed": seed,
                "execution_time_sec": round(elapsed, 2),
                "metrics": metrics,
                "meta_info": meta_info,
                "omega": omega.tolist() if hasattr(omega, "tolist") else omega,
            }
            utils.save_json_results(args.out_dir, "ddpm", ds, n, seed, payload)
            done += 1
            print(f"[shard {args.shard}] {idx+1}/{len(mine)} {ds} n={n} seed={seed} "
                  f"HD={metrics['Hamming']} TPR={metrics['TPR']:.2f} FDR={metrics['FDR']:.2f} "
                  f"({elapsed:.0f}s)", flush=True)

        except Exception as exc:  # keep the sweep alive; report at the end
            failed += 1
            print(f"[shard {args.shard}] FAIL {ds} n={n} seed={seed}: {exc}", flush=True)

    print(f"[shard {args.shard}] finished: done={done} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
