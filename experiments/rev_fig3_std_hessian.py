"""
Std-setting Hessians for the Gaussian-copula trajectory figure (plot_results.ipynb, Fig. 3).

Trains a DDPM on standardized ``dim20_cop_gau`` data and stores the averaged
per-timestep Hessian in the same ``[H_dict_avg, extra]`` pickle layout the raw
figure already reads, so the plotting code is unchanged apart from the path.

Output: visualization/data/hessian_dim20_cop_gau_std/n{100,1000}.pickle
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
for p in (project_root, current_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import glob
import time
import pickle
import argparse

import numpy as np
import torch


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dim20_cop_gau")
    ap.add_argument("--seed", type=int, default=120)
    ap.add_argument("--sizes", default="100,1000")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data_dir", default=os.path.join(project_root, "data", "raw"))
    ap.add_argument("--out_dir", default=os.path.join(project_root, "visualization", "data",
                                                     "hessian_dim20_cop_gau_std"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    from models.ddpm.ddpm import DDPMEstimator

    for n in [int(s) for s in args.sizes.split(",")]:
        pattern = os.path.join(args.data_dir, args.dataset, "train", f"*_n{n}_seed{args.seed}.npy")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(pattern)

        X = np.load(files[0])[:n, :]
        lam_raw = float(np.trace(np.cov(X, rowvar=False)) / X.shape[1])
        Xs = standardize(X)

        est = DDPMEstimator(seed=args.seed, device=device)
        est.D = Xs.shape[1]

        t0 = time.time()
        model, diffusion, train_time = est._train_ddpm(Xs)
        H_dict_avg, inf_time = est._compute_hessians_avg(model, diffusion, Xs)
        elapsed = time.time() - t0

        meta = {
            "dataset": args.dataset, "n": n, "seed": args.seed,
            "preprocessing": "per-feature z-score (ddof=1)",
            "lambda_bar_raw": lam_raw, "lambda_bar_std": 1.0,
            "t_list": [est.t_list[0], est.t_list[-1]],
            "num_samples_per_t": est.num_samples_per_t, "num_x0": 128,
            "batch_size": est.batch_size, "lr": est.lr, "epochs": est.epochs,
            "train_seconds": round(train_time, 1), "inference_seconds": round(inf_time, 1),
        }

        out_path = os.path.join(args.out_dir, f"n{n}.pickle")
        with open(out_path, "wb") as fh:
            pickle.dump([H_dict_avg, meta], fh)
        print(f"saved {out_path}  (lam_bar_raw={lam_raw:.4f}, {elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
