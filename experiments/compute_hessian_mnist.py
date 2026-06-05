"""Compute the per-timestep Hessian of a trained MNIST DDPM.

For each diffusion timestep ``t``, this estimates the Hessian of ``log p_t(x)``
w.r.t. the 784 pixels (entry-wise mean of ``|H|`` over a set of x0 anchors), and
saves ``{H_dict_avg, H_dict_std}`` to a pickle consumed by
``visualization/image_results.ipynb`` / ``mnist_graph.py``.

Method per (x0, t):
  1. q_sample x_t from x_0.
  2. Sample S posterior draws x_0 | x_t by full reverse diffusion.
  3. Cov = Cov(x_0 | x_t); H = (alpha_bar / sigma^4) Cov - (1/sigma^2) I.

Results are independent of the number of GPUs used; multi-GPU only shards the x0
anchors. Uses the in-repo DDPM modules (no external dependencies).

Example
-------
    # single GPU
    python experiments/compute_hessian_mnist.py \
        --chkpt-dir ./checkpoints/mnist_small_timesteps \
        --output-dir ./visualization/data \
        --single-gpu --device cuda:0

    # multi-GPU (shards x0 anchors across the listed GPUs)
    python experiments/compute_hessian_mnist.py \
        --chkpt-dir ./checkpoints/mnist_small_timesteps \
        --output-dir ./visualization/data \
        --multi-gpu --gpu-ids 0,1,2,3
"""
import argparse
import glob
import math
import os
import pickle
import re
import sys
import time
import traceback
import multiprocessing as mp

import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.ddpm.core.ddpm_torch.diffusion import GaussianDiffusion, get_beta_schedule
from models.ddpm.core.ddpm_torch.models.unet import UNet


# ---------------------------------------------------------------------------
# Core estimation (operates on image tensors of shape (B, 1, 28, 28))
# ---------------------------------------------------------------------------
@torch.inference_mode()
def estimate_hessian_full_batch_nd(model, diffusion, x_t, t, device,
                                   num_samples=1000, seed=None, apply_clamping=False,
                                   posterior_chunk_size=None, show_chunk_progress=False):
    """Hessian of log p_t for a batch of x_t, via posterior covariance of x_0 | x_t.

    Returns H (B, flat_dim, flat_dim) and Cov (B, flat_dim, flat_dim).
    """
    if isinstance(x_t, np.ndarray):
        x_t = torch.from_numpy(x_t)
    x_t = x_t.to(device=device, dtype=torch.float32)
    if x_t.ndim in (1, 3):
        x_t = x_t.unsqueeze(0)

    B = x_t.shape[0]
    flat_dim = int(x_t[0].numel())
    S = int(num_samples)
    chunk = S if posterior_chunk_size is None else max(1, int(posterior_chunk_size))

    sum_x = torch.zeros((B, flat_dim), device=device, dtype=x_t.dtype)
    sum_xx = torch.zeros((B, flat_dim, flat_dim), device=device, dtype=x_t.dtype)

    base_seed = None if seed is None else int(seed)
    t_start = int(t.item()) if isinstance(t, torch.Tensor) else int(t)

    chunk_starts = list(range(0, S, chunk))
    if show_chunk_progress:
        chunk_starts = tqdm(chunk_starts, desc=f"posterior chunks (t={t_start})", leave=False)

    for s_start in chunk_starts:
        s_end = min(s_start + chunk, S)
        cur_s = s_end - s_start

        # Replicate each x_t cur_s times, then run full reverse diffusion to x_0.
        x_rep = x_t.repeat_interleave(cur_s, dim=0).contiguous()
        t_tensor = torch.full((B * cur_s,), t_start, dtype=torch.int64, device=device)
        rng = None if base_seed is None else torch.Generator(device=device).manual_seed(base_seed + s_start)

        for ti in range(t_start, -1, -1):
            t_tensor.fill_(ti)
            x_rep = diffusion.p_sample_step(
                denoise_fn=model, x_t=x_rep, t=t_tensor,
                clip_denoised=False, return_pred=False, generator=rng,
            )

        x_chunk = x_rep.view(B, cur_s, -1)
        if apply_clamping:
            x_chunk = x_chunk.clamp(-1.0, 1.0)
        sum_x += x_chunk.sum(dim=1)
        sum_xx += torch.einsum("bsd,bse->bde", x_chunk, x_chunk)

    mu = sum_x / float(S)
    Cov = sum_xx / float(S) - mu.unsqueeze(2) * mu.unsqueeze(1)

    mu2_t = diffusion.alphas_bar[t].to(device).float()
    sigma2_t = (1.0 - mu2_t).clamp_min(1e-12)
    scale = mu2_t / (sigma2_t ** 2)

    H = scale * Cov
    diag_add = (-1.0 / sigma2_t).expand(H.shape[-1])
    H = H + torch.diag(diag_add).unsqueeze(0)
    return H, Cov


@torch.inference_mode()
def compute_hessian_stats_over_x0(model, diffusion, x0_data, t_values, device,
                                  batch_x0=8, num_samples_per_t=500, seed=123,
                                  apply_clamping=False, posterior_chunk_size=None,
                                  show_chunk_progress=False, show_batch_progress=True,
                                  worker_tag=None, progress_callback=None):
    """Accumulate sum / sum-of-squares / count of |H| over x0 anchors, per t.

    Returns numpy dicts (sum_abs, sumsq_abs, count) keyed by t so that results
    from multiple shards/GPUs can be summed before computing mean/std.
    """
    test_t = torch.as_tensor(x0_data, device=device, dtype=torch.float32)
    sum_abs, sumsq_abs, count = {}, {}, {}

    total_batches = max(1, int(math.ceil(test_t.shape[0] / float(batch_x0))))
    t_list = list(t_values)
    num_t = len(t_list)
    processed_x0 = 0

    batch_starts = range(0, test_t.shape[0], batch_x0)
    if show_batch_progress:
        batch_starts = tqdm(batch_starts, desc=f"Hessian over x0 [{device}]", leave=False)

    for batch_idx, start in enumerate(batch_starts, start=1):
        end = min(start + batch_x0, test_t.shape[0])
        x0_batch = test_t[start:end]
        B = x0_batch.shape[0]
        processed_before = processed_x0

        for t_idx, t in enumerate(t_list, start=1):
            t_tensor = torch.full((B,), int(t), dtype=torch.int64, device=device)
            x_t_batch = diffusion.q_sample(x_0=x0_batch, t=t_tensor, noise=torch.randn_like(x0_batch))

            H_batch, _ = estimate_hessian_full_batch_nd(
                model=model, diffusion=diffusion, x_t=x_t_batch, t=int(t), device=device,
                num_samples=num_samples_per_t, seed=int(seed + 10 * start + t),
                apply_clamping=apply_clamping, posterior_chunk_size=posterior_chunk_size,
                show_chunk_progress=show_chunk_progress,
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

            if progress_callback is not None:
                progress_callback(
                    batch_idx=batch_idx, total_batches=total_batches,
                    processed_units=int(processed_before * num_t + B * t_idx),
                    total_units=int(test_t.shape[0] * num_t),
                )
        processed_x0 += B

    sum_abs_np = {t: v.detach().cpu().numpy() for t, v in sum_abs.items()}
    sumsq_abs_np = {t: v.detach().cpu().numpy() for t, v in sumsq_abs.items()}
    count_np = {t: int(v) for t, v in count.items()}
    return sum_abs_np, sumsq_abs_np, count_np


def finalize_mean_std(sum_abs, sumsq_abs, count, t_values):
    """Turn accumulated sums into per-t mean and (unbiased) std of |H|."""
    H_dict_avg, H_dict_std = {}, {}
    for t in t_values:
        n = count[t]
        mean = sum_abs[t] / float(n)
        if n > 1:
            var = (sumsq_abs[t] - (sum_abs[t] ** 2) / float(n)) / float(n - 1)
            var = np.clip(var, a_min=0.0, a_max=None)
        else:
            var = np.zeros_like(mean)
        H_dict_avg[t] = mean
        H_dict_std[t] = np.sqrt(var)
    return H_dict_avg, H_dict_std


# ---------------------------------------------------------------------------
# Model / data helpers
# ---------------------------------------------------------------------------
def find_latest_checkpoint(ckpt_dir, pattern="*.pt"):
    candidates = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matching '{pattern}' in: {ckpt_dir}")
    return max(candidates, key=os.path.getmtime)


def build_model_and_diffusion(args, device, ckpt_path):
    betas = get_beta_schedule(
        args.beta_schedule, beta_start=args.beta_start,
        beta_end=args.beta_end, timesteps=args.timesteps,
    )
    diffusion = GaussianDiffusion(
        betas=betas, model_mean_type=args.model_mean_type,
        model_var_type=args.model_var_type, loss_type=args.loss_type,
    )

    ch_multipliers = tuple(int(x) for x in args.ch_mults.split(","))
    apply_attn = tuple(bool(int(x)) for x in args.attn_levels.split(","))
    if len(ch_multipliers) != len(apply_attn):
        raise ValueError("attn-levels length must match ch-mults length")

    in_channels = 1
    out_channels = 2 * in_channels if args.model_var_type == "learned" else in_channels
    model = UNet(
        in_channels=in_channels, hid_channels=args.hid_ch, out_channels=out_channels,
        ch_multipliers=ch_multipliers, num_res_blocks=args.num_res_blocks,
        apply_attn=apply_attn, time_embedding_dim=None,
        drop_rate=args.drop_rate, resample_with_conv=args.resample_with_conv,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        ckpt_epoch = int(ckpt.get("epoch", -1))
    else:
        model.load_state_dict(ckpt)
        ckpt_epoch = -1
    model.eval()
    return model, diffusion, ckpt_epoch


def load_mnist_x0(data_root, n_take):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    ds = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    n_take = min(n_take, len(ds))
    return torch.stack([ds[i][0] for i in range(n_take)], dim=0)  # (n_take, 1, 28, 28)


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------
def _worker(worker_cfg, queue):
    try:
        gpu_id = int(worker_cfg["gpu_id"])
        start, end = int(worker_cfg["start"]), int(worker_cfg["end"])
        args = worker_cfg["args"]
        ckpt_path = worker_cfg["ckpt_path"]
        t_values = worker_cfg["t_values"]

        device = torch.device(f"cuda:{gpu_id}")
        model, diffusion, _ = build_model_and_diffusion(args, device, ckpt_path)
        x0_data = load_mnist_x0(args.data_root, end)[start:end]

        def _cb(**info):
            queue.put({"type": "progress", "gpu_id": gpu_id, **info})

        sum_abs, sumsq_abs, count = compute_hessian_stats_over_x0(
            model=model, diffusion=diffusion, x0_data=x0_data, t_values=t_values,
            device=device, batch_x0=args.batch_x0, num_samples_per_t=args.num_samples_per_t,
            seed=args.seed + 100000 * start, apply_clamping=args.apply_clamping,
            posterior_chunk_size=args.posterior_chunk, show_chunk_progress=False,
            show_batch_progress=False, worker_tag=f"GPU{gpu_id}", progress_callback=_cb,
        )
        queue.put({"type": "result", "ok": True, "gpu_id": gpu_id, "start": start, "end": end,
                   "sum_abs": sum_abs, "sumsq_abs": sumsq_abs, "count": count})
    except Exception:
        queue.put({"type": "error", "ok": False, "gpu_id": worker_cfg.get("gpu_id", "?"),
                   "error": traceback.format_exc()})


def run_multi_gpu(args, ckpt_path, t_values, gpu_ids, n_take):
    n_workers = min(len(gpu_ids), n_take)
    gpu_ids = gpu_ids[:n_workers]
    shard_size = int(math.ceil(n_take / float(n_workers)))
    shards = [(gpu_ids[i], i * shard_size, min((i + 1) * shard_size, n_take))
              for i in range(n_workers) if i * shard_size < n_take]

    print("Shard assignment:")
    for gpu_id, start, end in shards:
        print(f"  GPU{gpu_id}: x0[{start}:{end}] ({end - start} samples)")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for gpu_id, start, end in shards:
        cfg = {"gpu_id": gpu_id, "start": start, "end": end, "args": args,
               "ckpt_path": ckpt_path, "t_values": list(t_values)}
        p = ctx.Process(target=_worker, args=(cfg, queue))
        p.start()
        procs.append(p)

    progress = {gid: {"processed_units": 0, "total_units": (end - start) * len(t_values)}
                for gid, start, end in shards}
    wall0 = time.perf_counter()

    results = []
    while len(results) < len(procs):
        msg = queue.get()
        if msg.get("type") == "progress":
            gid = int(msg["gpu_id"])
            progress[gid]["processed_units"] = int(msg["processed_units"])
            progress[gid]["total_units"] = int(msg["total_units"])
            done = sum(v["processed_units"] for v in progress.values())
            total = max(1, sum(v["total_units"] for v in progress.values()))
            elapsed = max(1e-9, time.perf_counter() - wall0)
            eta = elapsed * (total - done) / max(1, done)
            print(f"[Progress] {100.0 * done / total:5.1f}% | "
                  f"ETA {int(eta // 3600):02d}:{int(eta % 3600 // 60):02d}:{int(eta % 60):02d}")
        elif msg.get("type") == "error":
            for p in procs:
                if p.is_alive():
                    p.terminate()
            raise RuntimeError(f"Worker failed on GPU {msg.get('gpu_id')}:\n{msg.get('error')}")
        else:
            results.append(msg)

    for p in procs:
        p.join()
    for r in results:
        if not r.get("ok", False):
            raise RuntimeError(f"Worker failed on GPU {r.get('gpu_id')}:\n{r.get('error')}")

    # Merge shard accumulators
    t_list = list(results[0]["count"].keys())
    sum_abs = {t: np.zeros_like(results[0]["sum_abs"][t]) for t in t_list}
    sumsq_abs = {t: np.zeros_like(results[0]["sumsq_abs"][t]) for t in t_list}
    count = {t: 0 for t in t_list}
    for r in results:
        for t in t_list:
            sum_abs[t] += r["sum_abs"][t]
            sumsq_abs[t] += r["sumsq_abs"][t]
            count[t] += int(r["count"][t])
    return finalize_mean_std(sum_abs, sumsq_abs, count, t_list)


def run_single_gpu(args, ckpt_path, t_values, n_take):
    device = torch.device(args.device)
    print(f"Computing MNIST Hessian on {device}")
    model, diffusion, _ = build_model_and_diffusion(args, device, ckpt_path)
    x0_data = load_mnist_x0(args.data_root, n_take)
    sum_abs, sumsq_abs, count = compute_hessian_stats_over_x0(
        model=model, diffusion=diffusion, x0_data=x0_data, t_values=t_values,
        device=device, batch_x0=args.batch_x0, num_samples_per_t=args.num_samples_per_t,
        seed=args.seed, apply_clamping=args.apply_clamping,
        posterior_chunk_size=args.posterior_chunk, show_batch_progress=True,
    )
    return finalize_mean_std(sum_abs, sumsq_abs, count, list(t_values))


def build_argparser():
    p = argparse.ArgumentParser(description="Compute the per-timestep Hessian of a trained MNIST DDPM.")

    p.add_argument("--data-root", default="./data/mnist", type=str)
    p.add_argument("--chkpt-dir", default="./checkpoints/mnist_small_timesteps", type=str)
    p.add_argument("--ckpt-pattern", default="*.pt", type=str)
    p.add_argument("--output-dir", default="./visualization/data", type=str)

    # diffusion (must match the trained checkpoint)
    p.add_argument("--timesteps", default=500, type=int)
    p.add_argument("--beta-schedule",
                   choices=["quad", "linear", "warmup10", "warmup50", "jsd"], default="linear")
    p.add_argument("--beta-start", default=0.001, type=float)
    p.add_argument("--beta-end", default=0.2, type=float)
    p.add_argument("--model-mean-type", choices=["mean", "x_0", "eps"], default="eps")
    p.add_argument("--model-var-type",
                   choices=["learned", "fixed-small", "fixed-large"], default="fixed-large")
    p.add_argument("--loss-type", choices=["kl", "mse"], default="mse")

    # UNet (must match the trained checkpoint)
    p.add_argument("--hid-ch", default=128, type=int)
    p.add_argument("--ch-mults", default="1,2,2", type=str)
    p.add_argument("--num-res-blocks", default=2, type=int)
    p.add_argument("--attn-levels", default="0,1,1", type=str)
    p.add_argument("--drop-rate", default=0.0, type=float)
    p.add_argument("--resample-with-conv", action="store_true", default=True)

    # Hessian estimation
    p.add_argument("--t-min", default=1, type=int)
    p.add_argument("--t-max", default=30, type=int)
    p.add_argument("--num-x0", default=128, type=int, help="number of x0 anchors")
    p.add_argument("--batch-x0", default=8, type=int)
    p.add_argument("--num-samples-per-t", default=5000, type=int, help="posterior draws S per (x0, t)")
    p.add_argument("--posterior-chunk", default=128, type=int, help="chunk size over S to cap VRAM")
    p.add_argument("--apply-clamp", dest="apply_clamping", action="store_true", default=False)
    p.add_argument("--seed", default=1234, type=int)

    # device / parallelism
    p.add_argument("--device", default="cuda:0", type=str, help="device for single-GPU mode")
    p.add_argument("--gpu-ids", default="0,1,2,3", type=str, help="comma-separated GPU ids for multi-GPU mode")
    p.add_argument("--multi-gpu", dest="multi_gpu", action="store_true", default=False)
    p.add_argument("--single-gpu", dest="multi_gpu", action="store_false")
    return p


def main():
    args = build_argparser().parse_args()
    t_values = list(range(args.t_min, args.t_max + 1))
    print(f"t in [{args.t_min}, {args.t_max}], num_x0={args.num_x0}, "
          f"batch_x0={args.batch_x0}, S={args.num_samples_per_t}")

    ckpt_path = find_latest_checkpoint(args.chkpt_dir, args.ckpt_pattern)
    print(f"Using checkpoint: {ckpt_path}")

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    use_multi = args.multi_gpu and len(gpu_ids) > 1

    if use_multi:
        print(f"Computing MNIST Hessian in parallel on GPUs: {gpu_ids}")
        H_dict_avg, H_dict_std = run_multi_gpu(args, ckpt_path, t_values, gpu_ids, args.num_x0)
    else:
        H_dict_avg, H_dict_std = run_single_gpu(args, ckpt_path, t_values, args.num_x0)

    _, _, ckpt_epoch = build_model_and_diffusion(
        args, torch.device(f"cuda:{gpu_ids[0]}" if use_multi else args.device), ckpt_path)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    epoch_tag = f"epoch{ckpt_epoch}" if ckpt_epoch >= 0 else "epoch_unknown"
    out_path = os.path.join(
        args.output_dir,
        f"hessian_mnist_{ckpt_stem}_{epoch_tag}_x0_{args.num_x0}_bx0_{args.batch_x0}_S_{args.num_samples_per_t}.pickle",
    )

    payload = {
        "H_dict_avg": H_dict_avg,
        "H_dict_std": H_dict_std,
        "checkpoint": ckpt_path,
        "checkpoint_epoch": ckpt_epoch,
        "multi_gpu": use_multi,
        "gpu_ids": gpu_ids,
        "num_x0": args.num_x0,
        "batch_x0": args.batch_x0,
        "num_samples_per_t": args.num_samples_per_t,
        "t_values": t_values,
        "flattened_dim": int(28 * 28),
    }
    with open(out_path, "wb") as handle:
        pickle.dump(payload, handle)
    print(f"MNIST Hessian results saved to {out_path}")


if __name__ == "__main__":
    main()
