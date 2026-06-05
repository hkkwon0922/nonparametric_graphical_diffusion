"""Compute the per-timestep Hessian of a trained network (sector return) DDPM.

For each diffusion timestep ``t``, estimates the Hessian of ``log p_t(x)`` w.r.t.
the company-return features (entry-wise mean of ``|H|`` over the date samples) and
saves ``{H_dict_avg, H_dict_std}`` to a pickle named like

    hessian_network_<sector>_<ckpt-stem>_x0_<N>_bx0_<B>_S_<S>_t<tmin>-<tmax>.pickle

which is consumed by ``visualization/network_results.ipynb``.

Method per (x0, t):
  1. q_sample x_t from x_0.
  2. Sample S posterior draws x_0 | x_t by full reverse diffusion.
  3. Cov = Cov(x_0 | x_t); H = (alpha_bar / sigma^4) Cov - (1/sigma^2) I.

Uses the in-repo toy DDPM modules (no external dependencies).

Example
-------
    python experiments/compute_hessian_network.py \
        --chkpt-root ./checkpoints/network_2019_connected \
        --sector-csv-dir ./data/network/sector_rt_csv_2019_connected \
        --output-dir ./visualization/data \
        --sectors Industrials --epochs 2000 \
        --num-x0 250 --batch-x0 128 --num-samples-per-t 5000 \
        --t-min 1 --t-max 50 --device cuda:0
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.ddpm.core.ddpm_torch.toy import GaussianDiffusion, get_beta_schedule
from models.ddpm.core.ddpm_torch.toy.toy_model import Decoder5D_0204


# ---------------------------------------------------------------------------
# Core estimation (feature vectors of shape (B, D))
# ---------------------------------------------------------------------------
@torch.inference_mode()
def sample_x0_given_xt_batch(model, diffusion, x_t, t, device, num_samples=10,
                             seed=None, apply_clamping=False):
    """Sample S posterior draws x_0 | x_t for a batch of x_t -> (B, S, D)."""
    if isinstance(x_t, np.ndarray):
        x_t = torch.from_numpy(x_t)
    x_t = x_t.to(device=device, dtype=torch.float32)
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(0)

    B, D = x_t.shape
    S = int(num_samples)
    t_start = int(t.item()) if isinstance(t, torch.Tensor) else int(t)

    x_rep = x_t.repeat_interleave(S, dim=0).contiguous()
    t_tensor = torch.full((B * S,), t_start, dtype=torch.int64, device=device)
    rng = None if seed is None else torch.Generator(device=device).manual_seed(int(seed))

    for ti in range(t_start, -1, -1):
        t_tensor.fill_(ti)
        x_rep = diffusion.p_sample_step(
            denoise_fn=model, x_t=x_rep, t=t_tensor,
            clip_denoised=False, return_pred=False, generator=rng,
        )

    x0 = x_rep.view(B, S, D)
    if apply_clamping:
        x0 = x0.clamp(-1.0, 1.0)
    return x0


@torch.inference_mode()
def estimate_hessian_full_batch(model, diffusion, x_t, t, device,
                                num_samples=1000, seed=None, apply_clamping=False):
    """Hessian of log p_t for a batch of x_t, via posterior covariance of x_0 | x_t.

    Returns H (B, D, D) and Cov (B, D, D).
    """
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(0)
    x_t = x_t.to(device=device, dtype=torch.float32)

    x0_samples = sample_x0_given_xt_batch(
        model=model, diffusion=diffusion, x_t=x_t, t=t, device=device,
        num_samples=num_samples, seed=seed, apply_clamping=apply_clamping,
    )

    mu = x0_samples.mean(dim=1)
    S = x0_samples.shape[1]
    M2 = torch.einsum("bsd,bse->bde", x0_samples, x0_samples) / float(S)
    Cov = M2 - mu.unsqueeze(2) * mu.unsqueeze(1)

    mu2_t = diffusion.alphas_bar[t].to(device).float()
    sigma2_t = (1.0 - mu2_t).clamp_min(1e-12)
    scale = mu2_t / (sigma2_t ** 2)

    H = scale * Cov
    D = H.shape[-1]
    diag_add = (-1.0 / sigma2_t).expand(D)
    H = H + torch.diag(diag_add).unsqueeze(0)
    return H, Cov


@torch.inference_mode()
def compute_hessians_entrywise_avg_over_x0(model, diffusion, test_data, t_values, device,
                                           num_x0=50, batch_x0=128, num_samples_per_t=1000,
                                           seed=123, apply_clamping=False):
    """Entry-wise mean / std of ``|H|`` over x0 anchors, per timestep.

    Returns numpy dicts H_dict_avg[t], H_dict_std[t], each (D, D).
    """
    test_t = torch.as_tensor(test_data, device=device, dtype=torch.float32)
    N, D = test_t.shape
    num_x0 = min(int(num_x0), N)

    sum_abs, sumsq_abs, count = {}, {}, {}

    for start in tqdm(range(0, num_x0, batch_x0), desc="Hessian over x0"):
        end = min(start + batch_x0, num_x0)
        x0_batch = test_t[start:end]
        B = x0_batch.shape[0]

        for t in t_values:
            t_tensor = torch.full((B,), int(t), dtype=torch.int64, device=device)
            x_t_batch = diffusion.q_sample(x_0=x0_batch, t=t_tensor, noise=torch.randn(B, D, device=device))

            H_batch, _ = estimate_hessian_full_batch(
                model=model, diffusion=diffusion, x_t=x_t_batch, t=int(t), device=device,
                num_samples=num_samples_per_t, seed=int(seed + 10 * start + t),
                apply_clamping=apply_clamping,
            )
            abs_H = H_batch.abs()

            if t not in sum_abs:
                sum_abs[t] = abs_H.sum(dim=0)
                sumsq_abs[t] = (abs_H ** 2).sum(dim=0)
                count[t] = B
            else:
                sum_abs[t] += abs_H.sum(dim=0)
                sumsq_abs[t] += (abs_H ** 2).sum(dim=0)
                count[t] += B

    H_dict_avg, H_dict_std = {}, {}
    for t in t_values:
        n = count[t]
        mean = sum_abs[t] / float(n)
        if n > 1:
            var = (sumsq_abs[t] - (sum_abs[t] ** 2) / float(n)) / float(n - 1)
            var = var.clamp_min(0.0)
        else:
            var = torch.zeros_like(mean)
        H_dict_avg[t] = mean.detach().cpu().numpy()
        H_dict_std[t] = var.sqrt().detach().cpu().numpy()
    return H_dict_avg, H_dict_std


# ---------------------------------------------------------------------------
# Data / model helpers
# ---------------------------------------------------------------------------
def load_sector_rt_matrix(csv_path, nan_fill="ffill_bfill"):
    """Load a (Date x Company) return CSV into (T, N) float32 plus column/index labels."""
    import pandas as pd

    raw = pd.read_csv(csv_path)
    if "Date" in raw.columns:
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw = raw.sort_values("Date").set_index("Date")
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if raw.empty:
        raise ValueError(f"Sector CSV is empty after cleaning: {csv_path}")

    fill = {
        "zero": lambda d: d.fillna(0.0),
        "ffill": lambda d: d.ffill().fillna(0.0),
        "bfill": lambda d: d.bfill().fillna(0.0),
        "ffill_bfill": lambda d: d.ffill().bfill().fillna(0.0),
    }
    clean = fill[nan_fill](raw).astype(np.float32)
    return clean.to_numpy(copy=True), list(clean.columns), list(clean.index)


def load_run_config(ckpt_dir):
    cfg_path = os.path.join(ckpt_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model_and_diffusion(args, device, ckpt_path, in_features):
    betas = get_beta_schedule(
        args.beta_schedule, beta_start=args.beta_start,
        beta_end=args.beta_end, timesteps=args.timesteps,
    )
    diffusion = GaussianDiffusion(
        betas=betas, model_mean_type=args.model_mean_type,
        model_var_type=args.model_var_type, loss_type=args.loss_type,
    )
    model = Decoder5D_0204(in_features, args.mid_features, args.num_temporal_layers).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        ckpt_epoch = int(ckpt.get("epoch", -1))
    else:
        model.load_state_dict(ckpt)
        ckpt_epoch = -1
    model.eval()
    return model, diffusion, ckpt_epoch


def build_argparser():
    p = argparse.ArgumentParser(description="Compute the per-timestep Hessian of a trained network DDPM.")

    p.add_argument("--chkpt-root", default="./checkpoints/network_2019_connected", type=str,
                   help="root holding per-sector checkpoint subdirectories")
    p.add_argument("--sector-csv-dir", default="./data/network/sector_rt_csv_2019_connected", type=str)
    p.add_argument("--sector-nan-fill",
                   choices=["zero", "ffill", "bfill", "ffill_bfill"], default="ffill_bfill")
    p.add_argument("--output-dir", default="./visualization/data", type=str)
    p.add_argument("--sectors", default="Industrials", type=str,
                   help="comma-separated sector names to process")
    p.add_argument("--epoch", default=2000, type=int, help="checkpoint epoch to use")

    # diffusion (defaults overridden by each sector's run_config.json when present)
    p.add_argument("--timesteps", default=500, type=int)
    p.add_argument("--beta-schedule",
                   choices=["quad", "linear", "warmup10", "warmup50", "jsd"], default="linear")
    p.add_argument("--beta-start", default=0.001, type=float)
    p.add_argument("--beta-end", default=0.2, type=float)
    p.add_argument("--model-mean-type", choices=["mean", "x_0", "eps"], default="eps")
    p.add_argument("--model-var-type",
                   choices=["learned", "fixed-small", "fixed-large"], default="fixed-large")
    p.add_argument("--loss-type", choices=["kl", "mse"], default="mse")

    # model (Decoder5D_0204); overridden by run_config.json when present
    p.add_argument("--mid-features", default=160, type=int)
    p.add_argument("--num-temporal-layers", default=3, type=int)

    # Hessian estimation
    p.add_argument("--t-min", default=1, type=int)
    p.add_argument("--t-max", default=50, type=int)
    p.add_argument("--num-x0", default=250, type=int)
    p.add_argument("--batch-x0", default=128, type=int)
    p.add_argument("--num-samples-per-t", default=5000, type=int)
    p.add_argument("--apply-clamp", dest="apply_clamping", action="store_true", default=False)
    p.add_argument("--seed", default=1234, type=int)
    p.add_argument("--device", default="cuda:0", type=str)
    return p


def main():
    args = build_argparser().parse_args()

    # Resolve relative path defaults against the repo root, so the script works
    # from any working directory (e.g. when launched from experiments/).
    for attr in ("chkpt_root", "sector_csv_dir", "output_dir"):
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.normpath(os.path.join(REPO_ROOT, val)))

    t_values = list(range(args.t_min, args.t_max + 1))
    sectors = [s.strip() for s in args.sectors.split(",") if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    for sector in sectors:
        sector_ckpt_dir = os.path.join(args.chkpt_root, sector)
        ckpt_path = os.path.join(sector_ckpt_dir, f"ddpm_network_epoch{args.epoch:04d}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Adopt the model/diffusion settings the checkpoint was trained with.
        run_cfg = load_run_config(sector_ckpt_dir)
        for key in ("mid_features", "num_temporal_layers"):
            if key in run_cfg:
                setattr(args, key, int(run_cfg[key]))
        for key in ("model_mean_type", "model_var_type", "loss_type", "beta_schedule"):
            if key in run_cfg:
                setattr(args, key, run_cfg[key])
        for key in ("timesteps",):
            if key in run_cfg:
                setattr(args, key, int(run_cfg[key]))
        for key in ("beta_start", "beta_end"):
            if key in run_cfg:
                setattr(args, key, float(run_cfg[key]))

        csv_path = os.path.join(args.sector_csv_dir, f"sector_{sector}_Rt.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Sector CSV not found: {csv_path}")
        train_data, company_cols, date_index = load_sector_rt_matrix(csv_path, args.sector_nan_fill)
        n_samples, n_features = train_data.shape

        print("=" * 80)
        print(f"sector={sector} | csv={csv_path} | data (Date x Company)={train_data.shape}")
        print(f"checkpoint={ckpt_path}")
        print(f"t in [{args.t_min}, {args.t_max}], num_x0={args.num_x0}, "
              f"batch_x0={args.batch_x0}, S={args.num_samples_per_t}")

        model, diffusion, ckpt_epoch = build_model_and_diffusion(args, device, ckpt_path, n_features)

        num_x0 = min(int(args.num_x0), n_samples)
        H_dict_avg, H_dict_std = compute_hessians_entrywise_avg_over_x0(
            model=model, diffusion=diffusion, test_data=train_data, t_values=t_values,
            device=device, num_x0=num_x0, batch_x0=args.batch_x0,
            num_samples_per_t=args.num_samples_per_t, seed=args.seed,
            apply_clamping=args.apply_clamping,
        )

        ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        out_path = os.path.join(
            args.output_dir,
            f"hessian_network_{sector}_{ckpt_stem}_x0_{num_x0}_bx0_{args.batch_x0}"
            f"_S_{args.num_samples_per_t}_t{args.t_min}-{args.t_max}.pickle",
        )

        payload = {
            "H_dict_avg": H_dict_avg,
            "H_dict_std": H_dict_std,
            "checkpoint": ckpt_path,
            "checkpoint_epoch": ckpt_epoch,
            "sector": sector,
            "sector_csv_path": csv_path,
            "train_data_shape": tuple(train_data.shape),
            "date_sample_count": int(len(date_index)),
            "feature_company_count": int(len(company_cols)),
            "company_columns": company_cols,
            "xt_mode": "forward_diffusion",
            "num_x0": int(num_x0),
            "batch_x0": int(args.batch_x0),
            "num_samples_per_t": int(args.num_samples_per_t),
            "t_values": t_values,
            "timesteps": int(args.timesteps),
            "beta_start": float(args.beta_start),
            "beta_end": float(args.beta_end),
        }
        with open(out_path, "wb") as handle:
            pickle.dump(payload, handle)
        print(f"Hessian saved: {out_path}")


if __name__ == "__main__":
    main()
