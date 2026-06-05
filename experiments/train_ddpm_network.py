"""Train a DDPM on financial network (sector return) data.

Trains a separate diffusion model per sector CSV. Each CSV is a (Date x Company)
return matrix; rows (dates) are the training samples and columns (companies) are
the feature dimension. Produces the checkpoints consumed by
``compute_hessian_network.py`` and the ``network_results.ipynb`` visualization.

Uses the in-repo toy DDPM modules under ``models/ddpm/core/ddpm_torch`` (no
external dependencies).

Example
-------
    python experiments/train_ddpm_network.py \
        --sector-csv-dir ./data/network/sector_rt_csv_2019_connected \
        --output-dir ./checkpoints/network_2019_connected \
        --epochs 2000 --device cuda:0
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
import random

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.ddpm.core.ddpm_torch.utils import seed_all
from models.ddpm.core.ddpm_torch.toy import GaussianDiffusion, get_beta_schedule
from models.ddpm.core.ddpm_torch.toy.toy_model import Decoder5D_0204

def restore_rng_state(path, device):
    state = torch.load(path, map_location="cpu")


    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])

    device_obj = torch.device(device)
    if device_obj.type == "cuda" and state["torch_cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda_rng_state"], device_obj)

    print(f"Restored RNG state from: {path}")

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")
    return total


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip())


def infer_sector_name_from_csv(csv_path):
    stem = Path(csv_path).stem
    if stem.startswith("sector_") and stem.endswith("_Rt"):
        return stem[len("sector_"):-len("_Rt")].replace("_", " ")
    return stem.replace("_", " ")


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
    if nan_fill not in fill:
        raise ValueError(f"Unknown nan_fill: {nan_fill}")
    clean = fill[nan_fill](raw).astype(np.float32)
    return clean.to_numpy(copy=True), list(clean.columns), list(clean.index)


def save_checkpoint(path, epoch, model, optimizer, avg_loss, config):
    torch.save({
        "epoch": epoch,
        "avg_loss": avg_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, path)


def train_one_sector(train_data, output_dir, run_config, args):
    """Train a single sector model and save periodic checkpoints."""
    os.makedirs(output_dir, exist_ok=True)
    n_samples, in_features = train_data.shape

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_data)),
        batch_size=args.batch_size, shuffle=True,
    )
    device = torch.device(args.device)

    betas = get_beta_schedule(
        args.beta_schedule, beta_start=args.beta_start,
        beta_end=args.beta_end, timesteps=args.timesteps,
    )
    diffusion = GaussianDiffusion(
        betas=betas, model_mean_type=args.model_mean_type,
        model_var_type=args.model_var_type, loss_type=args.loss_type,
    )
    model = Decoder5D_0204(in_features, args.mid_features, args.num_temporal_layers).to(device)
    count_parameters(model)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            t = torch.randint(0, diffusion.timesteps, (batch.shape[0],), device=device)
            loss = diffusion.train_losses(model, x_0=batch, t=t).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        history.append({"epoch": epoch, "avg_loss": avg_loss})
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs}: loss={avg_loss:.6f}")

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(output_dir, f"ddpm_network_epoch{epoch:04d}.pt")
            save_checkpoint(ckpt_path, epoch, model, optimizer, avg_loss, run_config)
            print(f"  saved: {ckpt_path}")

        if args.wd_every > 0 and epoch % args.wd_every == 0:
            model.eval()
            with torch.no_grad():
                eval_samples = min(train_data.shape[0], args.wd_num_samples)
                _ = diffusion.p_sample(
                    denoise_fn=model,
                    shape=(eval_samples, model.input_dim),
                    device=device,
                    noise=None,
                    seed=None,
                )

    with open(os.path.join(output_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def build_argparser():
    p = argparse.ArgumentParser(description="Train per-sector DDPMs on financial return data.")

    p.add_argument("--sector-csv-dir", default="./data/network/sector_rt_csv_2019_connected", type=str)
    p.add_argument(
        "--sector-csv-glob",
        nargs="+",
        default=["sector_*_Rt.csv"],
        type=str,
        help="One or more glob patterns, e.g. sector_*_Rt.csv sector_*_LogRt.csv",
    )
    p.add_argument("--sector-nan-fill",
                   choices=["zero", "ffill", "bfill", "ffill_bfill"], default="ffill_bfill")
    p.add_argument("--output-dir", default="./checkpoints/network_2019_connected", type=str)

    # training
    p.add_argument("--epochs", default=4000, type=int)
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--lr", default=5e-4, type=float)
    p.add_argument("--beta1", default=0.9, type=float)
    p.add_argument("--beta2", default=0.999, type=float)
    p.add_argument("--save-every", default=500, type=int, help="save checkpoint every N epochs")
    p.add_argument("--seed", default=1234, type=int)
    p.add_argument("--device", default="cuda:2", type=str)
    p.add_argument("--wd-every", default=200, type=int)
    p.add_argument("--wd-num-samples", default=2000, type=int)

    # diffusion
    p.add_argument("--timesteps", default=500, type=int)
    p.add_argument("--beta-schedule",
                   choices=["quad", "linear", "warmup10", "warmup50", "jsd"], default="linear")
    p.add_argument("--beta-start", default=0.001, type=float)
    p.add_argument("--beta-end", default=0.2, type=float)
    p.add_argument("--model-mean-type", choices=["mean", "x_0", "eps"], default="eps")
    p.add_argument("--model-var-type",
                   choices=["learned", "fixed-small", "fixed-large"], default="fixed-large")
    p.add_argument("--loss-type", choices=["kl", "mse"], default="mse")

    # model (Decoder5D_0204)
    p.add_argument("--mid-features", default=160, type=int)
    p.add_argument("--num-temporal-layers", default=3, type=int)
    return p

def main():
    args = build_argparser().parse_args()
    seed_all(args.seed)

    # Resolve relative path defaults against the repo root, so the script works
    # from any working directory (e.g. when launched from experiments/).
    for attr in ("sector_csv_dir", "output_dir"):
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.normpath(os.path.join(REPO_ROOT, val)))

    sector_dir = Path(args.sector_csv_dir)

    sector_files = []
    for pattern in args.sector_csv_glob:
        sector_files.extend(sector_dir.glob(pattern))

    sector_files = sorted(set(sector_files))

    if not sector_files:
        raise FileNotFoundError(
            f"No sector CSV found in {sector_dir} with pattern(s) {args.sector_csv_glob}"
        )

    print(f"Found {len(sector_files)} sector CSV(s)")
    for pth in sector_files:
        print(f"  - {pth.name}")

    for idx, csv_path in enumerate(sector_files, start=1):
        sector_name = infer_sector_name_from_csv(csv_path)
        train_data, company_cols, date_index = load_sector_rt_matrix(
            csv_path, nan_fill=args.sector_nan_fill)

        print("=" * 80)
        print(f"sector [{idx}/{len(sector_files)}]: {sector_name}  csv={csv_path}")
        print(f"train data (Date x Company): {train_data.shape}")

        run_config = vars(args).copy()
        run_config.update({
            "train_mode": "sector_csv",
            "sector_name": sector_name,
            "resolved_sector_csv_path": str(Path(csv_path).resolve()),
            "date_sample_count": len(date_index),
            "feature_company_count": len(company_cols),
            "company_columns": company_cols,
            "train_data_shape": tuple(train_data.shape),
        })

        train_one_sector(
            train_data=train_data,
            output_dir=os.path.join(args.output_dir, sanitize_name(sector_name)),
            run_config=run_config,
            args=args,
        )


if __name__ == "__main__":
    main()
