"""
Std-setting Hessians for the D=3 Gaussian-chain toy example (toy_example.ipynb).

Draws n samples from N(0, Omega_0^{-1}) with the chain precision used in the
notebook, z-scores them per feature, trains a DDPM and stores the averaged
per-timestep Hessian in the same ``[H_dict_avg, H_dict_std]`` pickle layout the
raw figures already read, so the plotting code is unchanged apart from the path.

This example is the one setting in the paper with lambda_bar > 1
(tr(Sigma_0)/D = 2.897): standardizing pulls it *down* to 1, the opposite
direction from the Gaussian copula (0.15). The zero pattern of Omega_0 is
preserved exactly under the diagonal congruence, so the target graph is unchanged.

Output: visualization/data/hessian_dim3_prec75_gau_std/n{10,20,50,500}.pickle
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
for p in (project_root, current_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import time
import pickle
import argparse

import numpy as np
import torch


# Chain precision from toy_example.ipynb: 1 -- 2 -- 3, with Omega_0[0,2] = 0.
A, B = 0.7, 0.5
OMEGA_0 = np.array([[1.0, A, 0.0],
                    [A, 1.0, B],
                    [0.0, B, 1.0]])


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10,20,50,500")
    ap.add_argument("--seed", type=int, default=120)
    ap.add_argument("--t-max", type=int, default=51, help="exclusive upper bound (t = 1..50)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(project_root, "visualization", "data",
                                                      "hessian_dim3_prec75_gau_std"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    from models.ddpm.ddpm import DDPMEstimator

    Sigma_0 = np.linalg.inv(OMEGA_0)
    lam_raw = float(np.trace(Sigma_0) / OMEGA_0.shape[0])
    print(f"lambda_bar(raw) = {lam_raw:.4f} -> 1.0000 after standardization", flush=True)

    for n in [int(s) for s in args.sizes.split(",")]:
        # Same generative model as the raw figures; the seed is tied to n so each
        # sample size is an independent draw but reproducible.
        rng = np.random.default_rng(args.seed + n)
        X = rng.multivariate_normal(np.zeros(3), Sigma_0, size=n)
        Xs = standardize(X)

        est = DDPMEstimator(seed=args.seed, device=device, t_max=args.t_max)
        est.D = Xs.shape[1]

        t0 = time.time()
        model, diffusion, train_time = est._train_ddpm(Xs)
        H_dict_avg, inf_time = est._compute_hessians_avg(model, diffusion, Xs)
        elapsed = time.time() - t0

        # The raw pickles carry [H_dict_avg, <second dict>]; keep the two-element
        # layout so the notebook's `pickle.load(...)[0]` indexing still works.
        meta = {
            "n": n, "seed": args.seed,
            "preprocessing": "per-feature z-score (ddof=1)",
            "lambda_bar_raw": lam_raw, "lambda_bar_std": 1.0,
            "t_list": [est.t_list[0], est.t_list[-1]],
            "num_samples_per_t": est.num_samples_per_t, "num_x0": min(128, n),
            "batch_size": est.batch_size, "lr": est.lr, "epochs": est.epochs,
            "train_seconds": round(train_time, 1), "inference_seconds": round(inf_time, 1),
        }

        out_path = os.path.join(args.out_dir, f"n{n}.pickle")
        with open(out_path, "wb") as fh:
            pickle.dump([H_dict_avg, meta], fh)
        print(f"saved {out_path}  (t=1..{max(H_dict_avg)}, {elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
