"""End-to-end causal-discovery benchmark: ordering -> parents, one dataset at a time.

A single framework: point it at any ``(n, d)`` data matrix plus its ground-truth
adjacency and it runs the whole pipeline. Nothing but the data path changes
between datasets.

Pipeline
--------
1. Train (or load) one full-dimensional DDPM on the data.
2. **Ordering** — sequential leaf removal on the variance of the diagonal
   Tweedie Hessian (``ConditionalDiffusionDAGOrderEstimator``).
3. **Parent selection** — restricted to the pairs the estimated order admits,
   using off-diagonal Hessian entries. Two selectors are run and reported
   separately:
     * ``das``     — DAS-style test of ``E[H_{i,j}] = 0`` across anchors;
     * ``cluster`` — 2-means on per-timestep ``|H_{i,j}(t)|`` profiles,
                     normalised per timestep across pairs (rank by default).
4. **Evaluation** — FNR-pi for the ordering stage, and F1/FNR/FPR for the edge
   set, following Montagna et al. (2023) Appendix D. Reporting both separates
   the ordering contribution from the parent-selection contribution.

Data conventions
----------------
``--data-path`` is an ``(n, d)`` ``.npy``/``.csv``; ``--adjacency-path`` is a
``(d, d)`` ``.npy`` with ``A[i, j] = 1`` meaning ``i -> j``. Alternatively use
``--generate`` to synthesise the paper's vanilla ANM on an ER graph.

Examples
--------
    # generate + run the paper's vanilla scenario on ER-10 dense
    python experiments/run_causal_benchmark.py --generate \\
        --num-nodes 10 --density dense --num-samples 1000 --data-seed 0 \\
        --output-dir results/causal_benchmark/er10_dense_seed0 --device cuda:0

    # run on your own data
    python experiments/run_causal_benchmark.py \\
        --data-path path/to/X.npy --adjacency-path path/to/A.npy \\
        --output-dir results/causal_benchmark/mydata --device cuda:0
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

from models.dag_diffusion.benchmark_metrics import edge_metrics, fnr_pi
from models.dag_diffusion.conditional_langevin import LangevinConfig
from models.dag_diffusion.ordering import (
    ConditionalDiffusionDAGOrderEstimator,
    DAGOrderingConfig,
)
from models.dag_diffusion.parent_selection import (
    candidate_pairs_from_order,
    das_hypothesis_test,
    multitime_cluster,
)
from models.dag_diffusion.reverse_posterior import sample_x0_S_given_xt_S
from models.dag_diffusion.score_adapter import DDPMScoreAdapter
from models.dag_diffusion.training import (
    DDPMTrainConfig,
    load_ddpm_checkpoint,
    save_ddpm_checkpoint,
    train_ddpm,
)
from models.dag_diffusion.tweedie_hessian import estimate_tweedie_hessian
from models.ddpm.core.ddpm_torch.utils import seed_all


# ---------------------------------------------------------------------------
# full-Hessian evaluation on the FULL variable set (for parent selection)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def full_hessian_over_timesteps(X, model, diffusion, score_adapter, device,
                                t_values, num_anchors, langevin_cfg,
                                reverse_draws=1, anchor_chunk=32, seed=120,
                                sampling_chunk=8192, verbose=True):
    """Per-anchor Tweedie Hessians on ``S = {0..D-1}`` for each ``t`` in ``t_values``.

    The full index set has an empty free block, so no Langevin is needed here:
    the posterior-sample budget comes from independent reverse trajectories.
    This is the same construction the ordering estimator uses at stage 0.

    Returns
    -------
    ``{t: (B, D, D) float64 ndarray}`` of per-anchor Hessians.
    """
    device = torch.device(device)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    n, dim = X_t.shape
    S = list(range(dim))
    draws = max(2, reverse_draws * langevin_cfg.num_chains * langevin_cfg.num_samples)

    out = {}
    for t in t_values:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        idx = torch.randperm(n, device=device, generator=gen)[:num_anchors]
        x0 = X_t[idx]
        t_vec = torch.full((x0.shape[0],), int(t), dtype=torch.int64, device=device)
        noise = torch.empty_like(x0).normal_(generator=gen)
        x_t_full = diffusion.q_sample(x_0=x0, t=t_vec, noise=noise)

        parts = []
        for start in range(0, x_t_full.shape[0], anchor_chunk):
            end = min(start + anchor_chunk, x_t_full.shape[0])
            g = torch.Generator(device=device)
            g.manual_seed(int(seed) + 1000 + start)
            x0_S, _ = sample_x0_S_given_xt_S(
                model=model, diffusion=diffusion, score_adapter=score_adapter,
                x_t_S=x_t_full[start:end], condition_indices=S, dim=dim, t=int(t),
                langevin_config=langevin_cfg, reverse_draws_per_xt=draws,
                reference_x0=X_t, generator=g, chunk_size=sampling_chunk, device=device,
            )
            H, _ = estimate_tweedie_hessian(x0_S, diffusion, int(t))
            parts.append(H.cpu().numpy())
            del x0_S, H
        out[int(t)] = np.concatenate(parts, axis=0)
        if verbose:
            print(f"    [hessian] t={t:3d} shape={out[int(t)].shape}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def build_pair_dicts(hessians, t_edge, order):
    """Split ``{t: (B,D,D)}`` into the per-pair views the two selectors need.

    Returns ``(signed_at_t_edge, abs_profiles)`` where
    ``signed_at_t_edge[(i,j)]`` is ``(B,)`` at ``t_edge`` and
    ``abs_profiles[(i,j)]`` is ``(T,)`` of ``E_anchors |H_ij(t)|``.
    """
    t_sorted = sorted(hessians)
    signed, profiles = {}, {}
    H_edge = hessians[int(t_edge)]
    for (i, j) in candidate_pairs_from_order(order):
        signed[(i, j)] = H_edge[:, i, j]
        profiles[(i, j)] = np.array(
            [np.abs(hessians[t][:, i, j]).mean() for t in t_sorted], dtype=float)
    return signed, profiles


# ---------------------------------------------------------------------------
def load_or_generate(args):
    """Return ``(X, adjacency, provenance_dict)``."""
    if args.generate:
        from data.benchmark_scm import generate_dataset
        X, A = generate_dataset(
            num_nodes=args.num_nodes, density=args.density,
            num_samples=args.num_samples, seed=args.data_seed)
        prov = {"source": "generated", "scenario": "vanilla_anm_gp",
                "num_nodes": args.num_nodes, "density": args.density,
                "num_samples": args.num_samples, "data_seed": args.data_seed}
        return X, A, prov

    if not args.data_path or not args.adjacency_path:
        raise ValueError("provide --data-path and --adjacency-path, or use --generate")
    X = np.load(args.data_path) if args.data_path.endswith(".npy") else \
        np.loadtxt(args.data_path, delimiter=",")
    A = np.load(args.adjacency_path)
    return np.asarray(X, dtype=np.float64), np.asarray(A), {
        "source": "file", "data_path": args.data_path,
        "adjacency_path": args.adjacency_path}


def build_argparser():
    p = argparse.ArgumentParser(description="Ordering + parent selection benchmark.")

    # data
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--adjacency-path", type=str, default=None)
    p.add_argument("--generate", action="store_true", default=False,
                   help="synthesise the paper's vanilla ANM on an ER graph")
    p.add_argument("--num-nodes", type=int, default=20)
    p.add_argument("--density", choices=["sparse", "dense"], default="dense")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--standardize-data", dest="standardize_data",
                   action="store_true", default=True,
                   help="z-score columns before fitting (suppresses varsortability)")
    p.add_argument("--no-standardize-data", dest="standardize_data", action="store_false")

    p.add_argument("--output-dir", type=str, default="./results/causal_benchmark/run")
    p.add_argument("--checkpoint-path", type=str, default=None)

    # ordering
    p.add_argument("--t-order", type=int, default=8)
    p.add_argument("--num-anchors", type=int, default=128)
    p.add_argument("--anchor-chunk-size", type=int, default=32)
    p.add_argument("--num-chains", type=int, default=16)
    p.add_argument("--langevin-burn-in", type=int, default=200)
    p.add_argument("--langevin-samples", type=int, default=8)
    p.add_argument("--langevin-thinning", type=int, default=5)
    p.add_argument("--langevin-step-size", type=float, default=1e-3)
    p.add_argument("--langevin-init", choices=["normal", "forward_data"],
                   default="forward_data")
    p.add_argument("--reverse-draws-per-xt", type=int, default=1)
    p.add_argument("--sampling-chunk-size", type=int, default=8192)

    # parent selection
    p.add_argument("--t-edge", type=int, default=None,
                   help="timestep for the DAS test (defaults to --t-order)")
    p.add_argument("--edge-t-values", type=str, default="3,5,8,12,20",
                   help="comma-separated timesteps for the clustering profiles")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--no-fdr", dest="fdr", action="store_false", default=True)
    p.add_argument("--cluster-transform", choices=["rank", "zscore"], default="rank",
                   help="per-timestep normalisation for the clustering features")

    # training
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mid-features", type=int, default=128)
    p.add_argument("--num-temporal-layers", type=int, default=3)
    p.add_argument("--timesteps", type=int, default=500)

    p.add_argument("--seed", type=int, default=120)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--quiet", action="store_true", default=False)
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    verbose = not args.quiet

    for attr in ("data_path", "adjacency_path", "output_dir", "checkpoint_path"):
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            setattr(args, attr, os.path.normpath(os.path.join(REPO_ROOT, val)))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; falling back to CPU")
        device = torch.device("cpu")
    seed_all(int(args.seed))

    run_start = time.time()
    timings = {}

    X_raw, A_true, prov = load_or_generate(args)
    n, dim = X_raw.shape
    X = X_raw.copy()
    if args.standardize_data:
        X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0))
    X = X.astype(np.float32)

    print("=" * 78)
    print(f"data: {prov.get('source')}  X={X.shape}  true edges={int(A_true.sum())}  "
          f"device={device}  standardized={args.standardize_data}")

    # ---- model ----------------------------------------------------------
    ckpt = args.checkpoint_path or os.path.join(args.output_dir, "ddpm.pt")
    if os.path.exists(ckpt):
        print(f"loading checkpoint: {ckpt}")
        model, diffusion, _, _ = load_ddpm_checkpoint(ckpt, device, expected_dim=dim)
        timings["train_seconds"] = 0.0
    else:
        print(f"training DDPM ({args.epochs} epochs, D={dim})...")
        cfg = DDPMTrainConfig(
            input_dimension=dim, mid_features=args.mid_features,
            num_temporal_layers=args.num_temporal_layers, timesteps=args.timesteps,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            seed=int(args.seed))
        t0 = time.time()
        model, diffusion, preproc, _ = train_ddpm(X, cfg, device, verbose=verbose)
        timings["train_seconds"] = time.time() - t0
        save_ddpm_checkpoint(ckpt, model, cfg, preproc, epoch=args.epochs)
        print(f"checkpoint saved: {ckpt}")

    score_adapter = DDPMScoreAdapter(model=model, diffusion=diffusion, device=device)
    langevin_cfg = LangevinConfig(
        num_chains=args.num_chains, burn_in=args.langevin_burn_in,
        num_samples=args.langevin_samples, thinning=args.langevin_thinning,
        step_size=args.langevin_step_size, init=args.langevin_init,
        chunk_size=args.sampling_chunk_size)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ---- 1) ordering ----------------------------------------------------
    print(f"[1/3] ordering (t={args.t_order}, anchors={args.num_anchors}, D={dim})")
    order_cfg = DAGOrderingConfig(
        t_order=args.t_order, num_anchors=args.num_anchors,
        anchor_chunk_size=args.anchor_chunk_size, langevin=langevin_cfg,
        reverse_draws_per_xt=args.reverse_draws_per_xt,
        sampling_chunk_size=args.sampling_chunk_size, seed=int(args.seed),
        verbose=verbose)
    t0 = time.time()
    result = ConditionalDiffusionDAGOrderEstimator(order_cfg).fit(
        X, model, diffusion, device=device)
    timings["ordering_seconds"] = time.time() - t0
    order = result.topological_order
    order_fnr = fnr_pi(order, A_true)
    print(f"      order FNR-pi = {order_fnr:.4f}   ({timings['ordering_seconds']:.1f}s)")

    # ---- 2) off-diagonal Hessians for parent selection -------------------
    t_edge = int(args.t_edge) if args.t_edge is not None else int(args.t_order)
    t_values = sorted({int(s) for s in args.edge_t_values.split(",") if s.strip()}
                      | {t_edge})
    print(f"[2/3] off-diagonal Hessians at t in {t_values}")
    t0 = time.time()
    hessians = full_hessian_over_timesteps(
        X, model, diffusion, score_adapter, device, t_values,
        num_anchors=args.num_anchors, langevin_cfg=langevin_cfg,
        reverse_draws=args.reverse_draws_per_xt,
        anchor_chunk=args.anchor_chunk_size, seed=int(args.seed),
        sampling_chunk=args.sampling_chunk_size, verbose=verbose)
    timings["hessian_seconds"] = time.time() - t0

    # ---- 3) parent selection (two methods) -------------------------------
    print("[3/3] parent selection")
    signed, profiles = build_pair_dicts(hessians, t_edge, order)

    das = das_hypothesis_test(signed, order, alpha=args.alpha, fdr_correction=args.fdr)
    clust = multitime_cluster(profiles, order, standardize=True, seed=int(args.seed),
                             transform=args.cluster_transform)

    metrics = {
        "ordering": {"fnr_pi": order_fnr,
                     "topological_order": order,
                     "leaf_order": result.leaf_order},
        "das": edge_metrics(das["adjacency"], A_true),
        "cluster": edge_metrics(clust["adjacency"], A_true),
    }
    # an oracle-order upper bound: how good could edge selection be if the
    # ordering were perfect? separates the two error sources.
    from data.benchmark_scm import topological_order_of
    true_order = topological_order_of(A_true)
    signed_o, profiles_o = build_pair_dicts(hessians, t_edge, true_order)
    das_o = das_hypothesis_test(signed_o, true_order, alpha=args.alpha,
                                fdr_correction=args.fdr)
    clust_o = multitime_cluster(profiles_o, true_order, standardize=True,
                                seed=int(args.seed), transform=args.cluster_transform)
    metrics["das_oracle_order"] = edge_metrics(das_o["adjacency"], A_true)
    metrics["cluster_oracle_order"] = edge_metrics(clust_o["adjacency"], A_true)

    for key in ("das", "cluster", "das_oracle_order", "cluster_oracle_order"):
        m = metrics[key]
        print(f"      {key:22s} F1={m['f1']:.3f} FNR={m['fnr']:.3f} FPR={m['fpr']:.3f} "
              f"(pred {m['num_pred_edges']} / true {m['num_true_edges']} edges)")

    # ---- outputs ---------------------------------------------------------
    timings["total_seconds"] = time.time() - run_start
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None

    payload = {
        "provenance": prov,
        "config": {k: v for k, v in vars(args).items()},
        "resolved_device": str(device),
        "data_shape": [int(n), int(dim)],
        "num_true_edges": int(A_true.sum()),
        "t_edge": t_edge,
        "t_values": t_values,
        "metrics": metrics,
        "das_pvalues": das["pvalues"],
        "cluster_silhouette": clust.get("silhouette"),
        "timings_seconds": {k: round(float(v), 3) for k, v in timings.items()},
        "peak_cuda_memory_bytes": peak,
    }
    with open(os.path.join(args.output_dir, "benchmark_result.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    np.savez_compressed(
        os.path.join(args.output_dir, "adjacencies.npz"),
        true=A_true, das=das["adjacency"], cluster=clust["adjacency"],
        das_oracle=das_o["adjacency"], cluster_oracle=clust_o["adjacency"],
        order=np.array(order),
        **{f"hessian_t{t}": hessians[t] for t in t_values})

    print(f"results -> {args.output_dir}   total {timings['total_seconds']:.1f}s"
          + (f"   peak CUDA {peak / 1e9:.3f} GB" if peak else ""))
    return payload


if __name__ == "__main__":
    main()
