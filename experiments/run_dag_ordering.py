"""Recover a DAG topological order from a single full-dimensional DDPM.

Trains (or loads) one ``D``-dimensional DDPM, then runs sequential leaf removal
driven by the variance of the conditional Tweedie Hessian diagonal.  See
``models/dag_diffusion/ordering.py`` for the method and its caveats.

The output is a topological **order**, not a sparse DAG.

Examples
--------
Debug run (CPU, a couple of minutes)::

    python experiments/run_dag_ordering.py --config configs/dag_ordering_debug.json

Full run from a checkpoint::

    python experiments/run_dag_ordering.py \\
        --data-path data/example.npy \\
        --checkpoint-path checkpoints/example/ddpm.pt \\
        --output-dir results/dag_ordering/example \\
        --t-order 10 --num-anchors 32 --num-chains 8 \\
        --langevin-burn-in 500 --langevin-samples 20 --langevin-thinning 20 \\
        --langevin-step-size 1e-4 --langevin-init forward_data \\
        --reverse-draws-per-xt 1 --sampling-chunk-size 256 \\
        --seed 120 --device cuda:0
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
from models.dag_diffusion.training import (
    DDPMTrainConfig,
    load_ddpm_checkpoint,
    save_ddpm_checkpoint,
    train_ddpm,
)
from models.ddpm.core.ddpm_torch.utils import seed_all


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_data(path):
    """Load ``(N, D)`` data from ``.npy`` / ``.npz`` / ``.csv``, plus feature names."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        names = None
    elif ext == ".npz":
        payload = np.load(path)
        key = "X" if "X" in payload else list(payload.keys())[0]
        arr = payload[key]
        names = None
    elif ext in (".csv", ".txt"):
        import pandas as pd
        frame = pd.read_csv(path)
        drop = [c for c in frame.columns if c.lower() in ("date", "index", "unnamed: 0")]
        frame = frame.drop(columns=drop)
        arr = frame.to_numpy()
        names = [str(c) for c in frame.columns]
    else:
        raise ValueError(f"unsupported data extension {ext!r} (use .npy, .npz or .csv)")

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"data must be 2-D (N, D); got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"data at {path} contains non-finite values")
    return arr, names


def estimate_memory_or_raise(args, dim, max_bytes):
    """Reject settings whose peak reverse-diffusion batch would be excessive."""
    states = args.num_anchors * args.num_chains * args.langevin_samples
    rows = states * args.reverse_draws_per_xt
    if args.sampling_chunk_size:
        rows = min(rows, int(args.sampling_chunk_size) * max(1, args.num_anchors))
    approx = rows * dim * 4 * 6  # float32 x a few live buffers per reverse step
    if approx > max_bytes:
        raise ValueError(
            f"requested settings would allocate roughly {approx / 1e9:.1f} GB in the reverse "
            f"diffusion stage (anchors={args.num_anchors}, chains={args.num_chains}, "
            f"langevin_samples={args.langevin_samples}, reverse_draws={args.reverse_draws_per_xt}, "
            f"D={dim}). Lower --sampling-chunk-size / --anchor-chunk-size, or raise "
            f"--max-memory-gb (currently {max_bytes / 1e9:.1f} GB)."
        )
    return approx


def resolve_device(requested):
    """Honour ``--device``, falling back to CPU with a warning when unavailable."""
    dev = torch.device(requested)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            print(f"[warn] CUDA unavailable; falling back to CPU (requested {requested})")
            return torch.device("cpu"), True
        idx = dev.index if dev.index is not None else 0
        if idx >= torch.cuda.device_count():
            print(f"[warn] {requested} not present ({torch.cuda.device_count()} CUDA device(s)); "
                  "falling back to CPU")
            return torch.device("cpu"), True
    return dev, False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="Diffusion-based DAG topological ordering via conditional Tweedie Hessians.")

    p.add_argument("--config", type=str, default=None,
                   help="JSON file of defaults; explicit CLI flags override it")

    # data / io
    p.add_argument("--data-path", type=str, default=None, help="(N, D) .npy/.npz/.csv")
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument("--train-if-missing", action="store_true", default=False,
                   help="train a DDPM when the checkpoint is absent")
    p.add_argument("--output-dir", type=str, default="./results/dag_ordering/run")
    p.add_argument("--ground-truth-adjacency", type=str, default=None,
                   help="optional (D, D) .npy with A[i, j] = 1 meaning i -> j")
    p.add_argument("--feature-names", type=str, default=None,
                   help="comma-separated names, one per column")

    # ordering
    p.add_argument("--t-order", type=int, default=None,
                   help="REQUIRED: the single diffusion timestep for the criterion")
    p.add_argument("--num-anchors", type=int, default=32)
    p.add_argument("--anchor-chunk-size", type=int, default=None)
    p.add_argument("--tie-tolerance", type=float, default=1e-12)
    p.add_argument("--compute-full-covariance", action="store_true", default=False)

    # conditional Langevin
    p.add_argument("--num-chains", type=int, default=8)
    p.add_argument("--langevin-burn-in", type=int, default=500)
    p.add_argument("--langevin-samples", type=int, default=20)
    p.add_argument("--langevin-thinning", type=int, default=20)
    p.add_argument("--langevin-step-size", type=float, default=1e-4)
    p.add_argument("--langevin-init", choices=["normal", "forward_data"], default="forward_data")
    p.add_argument("--langevin-score-clip", type=float, default=None)

    # reverse diffusion
    p.add_argument("--reverse-draws-per-xt", type=int, default=1)
    p.add_argument("--sampling-chunk-size", type=int, default=256)

    # training (used with --train-if-missing)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mid-features", type=int, default=160)
    p.add_argument("--num-temporal-layers", type=int, default=3)
    p.add_argument("--timesteps", type=int, default=500)
    p.add_argument("--beta-schedule",
                   choices=["quad", "linear", "warmup10", "warmup50", "const", "jsd"],
                   default="linear")
    p.add_argument("--beta-start", type=float, default=0.001)
    p.add_argument("--beta-end", type=float, default=0.2)
    p.add_argument("--model-mean-type", choices=["x_0", "eps"], default="eps")
    p.add_argument("--model-var-type",
                   choices=["fixed-small", "fixed-large"], default="fixed-large")
    p.add_argument("--loss-type", choices=["kl", "mse"], default="mse")
    p.add_argument("--standardize", action="store_true", default=False,
                   help="opt-in standardisation; mean/scale are stored in the checkpoint")

    # runtime
    p.add_argument("--seed", type=int, default=120)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-memory-gb", type=float, default=8.0)
    p.add_argument("--quiet", action="store_true", default=False)
    return p


def apply_config_file(parser, args, argv):
    """Merge ``--config`` defaults under any explicitly-passed CLI flag."""
    if not args.config:
        return args
    with open(args.config, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    explicit = set()
    for token in argv:
        if token.startswith("--"):
            explicit.add(token.split("=")[0].lstrip("-").replace("-", "_"))

    for key, value in payload.items():
        if key.startswith("_"):
            continue  # underscore-prefixed keys are comments
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            raise ValueError(f"unknown key {key!r} in config {args.config}")
        if attr not in explicit:
            setattr(args, attr, value)
    return args


def main(argv=None):
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    parser = build_argparser()
    args = parser.parse_args(argv)
    args = apply_config_file(parser, args, argv)

    if args.data_path is None:
        parser.error("--data-path is required (directly or via --config)")
    if args.t_order is None:
        parser.error("--t-order is required: the criterion is evaluated at a single timestep, "
                     "and no multi-timestep aggregation rule is defined")

    for attr in ("data_path", "checkpoint_path", "output_dir", "ground_truth_adjacency"):
        val = getattr(args, attr)
        if val and not os.path.isabs(val):
            setattr(args, attr, os.path.normpath(os.path.join(REPO_ROOT, val)))

    os.makedirs(args.output_dir, exist_ok=True)
    seed_all(int(args.seed))
    device, fell_back = resolve_device(args.device)
    verbose = not args.quiet

    run_start = time.time()
    timings = {}

    # ---- data ----------------------------------------------------------
    X, csv_names = load_data(args.data_path)
    n_samples, dim = X.shape
    names = None
    if args.feature_names:
        names = [s.strip() for s in str(args.feature_names).split(",") if s.strip()]
    elif csv_names:
        names = csv_names
    if names is not None and len(names) != dim:
        raise ValueError(f"feature-names has {len(names)} entries but data has {dim} columns")

    print("=" * 78)
    print(f"data={args.data_path}  shape=({n_samples}, {dim})  device={device}")
    print(f"t_order={args.t_order}  anchors={args.num_anchors}  chains={args.num_chains}")

    estimate_memory_or_raise(args, dim, args.max_memory_gb * 1e9)

    # ---- model: load or train -------------------------------------------
    ckpt_path = args.checkpoint_path
    train_history = None
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"loading checkpoint: {ckpt_path}")
        t0 = time.time()
        model, diffusion, preproc, payload = load_ddpm_checkpoint(
            ckpt_path, device, expected_dim=dim)
        timings["load_seconds"] = time.time() - t0
        trained_now = False
    else:
        if not args.train_if_missing:
            raise FileNotFoundError(
                f"checkpoint not found: {ckpt_path!r}. Pass --train-if-missing to train one."
            )
        print(f"training a new DDPM ({args.epochs} epochs)...")
        train_cfg = DDPMTrainConfig(
            input_dimension=dim, mid_features=args.mid_features,
            num_temporal_layers=args.num_temporal_layers, timesteps=args.timesteps,
            beta_schedule=args.beta_schedule, beta_start=args.beta_start,
            beta_end=args.beta_end, model_mean_type=args.model_mean_type,
            model_var_type=args.model_var_type, loss_type=args.loss_type,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            standardize=bool(args.standardize), seed=int(args.seed),
        )
        t0 = time.time()
        model, diffusion, preproc, train_history = train_ddpm(
            X, train_cfg, device, verbose=verbose)
        timings["train_seconds"] = time.time() - t0

        ckpt_path = args.checkpoint_path or os.path.join(args.output_dir, "ddpm.pt")
        save_ddpm_checkpoint(
            ckpt_path, model, train_cfg, preproc, epoch=args.epochs,
            extra={"data_path": args.data_path, "train_history": train_history},
        )
        print(f"checkpoint saved: {ckpt_path}")
        trained_now = True

    if int(args.t_order) >= int(diffusion.timesteps):
        raise ValueError(
            f"--t-order {args.t_order} is out of range for this model "
            f"(timesteps={int(diffusion.timesteps)})"
        )

    # Ordering runs in the model's own coordinate system: if the checkpoint was
    # trained on standardised data, the criterion is evaluated there too.
    X_model = np.asarray(preproc.transform(X), dtype=np.float32)
    coordinate_space = "standardized" if preproc.enabled else "raw"
    print(f"ordering coordinates: {coordinate_space}")

    # ---- ordering --------------------------------------------------------
    langevin_cfg = LangevinConfig(
        num_chains=int(args.num_chains), burn_in=int(args.langevin_burn_in),
        num_samples=int(args.langevin_samples), thinning=int(args.langevin_thinning),
        step_size=float(args.langevin_step_size), init=args.langevin_init,
        score_norm_clip=args.langevin_score_clip,
        chunk_size=int(args.sampling_chunk_size) if args.sampling_chunk_size else 4096,
    )
    order_cfg = DAGOrderingConfig(
        t_order=int(args.t_order), num_anchors=int(args.num_anchors),
        langevin=langevin_cfg, reverse_draws_per_xt=int(args.reverse_draws_per_xt),
        anchor_chunk_size=args.anchor_chunk_size,
        sampling_chunk_size=args.sampling_chunk_size,
        tie_tolerance=float(args.tie_tolerance), seed=int(args.seed),
        compute_full_covariance=bool(args.compute_full_covariance), verbose=verbose,
    )

    estimator = ConditionalDiffusionDAGOrderEstimator(order_cfg)
    t0 = time.time()
    result = estimator.fit(
        X_model, model, diffusion, device=device, feature_names=names)
    timings["ordering_seconds"] = time.time() - t0

    if fell_back:
        result.warnings.append(f"requested device {args.device} unavailable; ran on CPU")
    if preproc.enabled:
        result.warnings.append(
            "ordering was performed on STANDARDIZED coordinates; the criterion is not "
            "claimed to be scale-invariant."
        )

    print("-" * 78)
    print(f"leaf_order (sinks first): {result.leaf_order}")
    print(f"topological_order        : {result.topological_order}")

    # ---- optional ground truth -------------------------------------------
    evaluation = None
    if args.ground_truth_adjacency:
        adjacency = np.load(args.ground_truth_adjacency)
        evaluation = evaluate_ordering(result, adjacency)
        fnr = evaluation["order_fnr"]
        leafv = evaluation["stagewise_leaf_validity"]
        print(f"order FNR = {fnr['order_fnr']:.4f} "
              f"({fnr['num_violated_edges']}/{fnr['num_true_edges']} edges violated)")
        print(f"stagewise leaf validity = {leafv['num_valid_stages']}/{leafv['num_stages']}")

    # ---- outputs ---------------------------------------------------------
    config_payload = vars(args).copy()
    config_payload.update({
        "resolved_device": str(device),
        "data_shape": [int(n_samples), int(dim)],
        "checkpoint_path": ckpt_path,
        "trained_in_this_run": trained_now,
        "preprocessing": preproc.to_dict(),
        "ordering_coordinate_space": coordinate_space,
        "torch_version": torch.__version__,
    })
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2, default=str)

    result_payload = result.to_json_dict()
    if evaluation is not None:
        result_payload["ground_truth_evaluation"] = evaluation
    with open(os.path.join(args.output_dir, "ordering_result.json"), "w", encoding="utf-8") as h:
        json.dump(result_payload, h, indent=2, default=str)

    torch.save(
        {"stage_records": result.stage_records,
         "leaf_order": result.leaf_order,
         "topological_order": result.topological_order,
         "t_order": result.t_order},
        os.path.join(args.output_dir, "stage_diagnostics.pt"),
    )

    timings["total_seconds"] = time.time() - run_start
    runtime_payload = {
        "timings_seconds": {k: round(float(v), 3) for k, v in timings.items()},
        "peak_cuda_memory_bytes": result.peak_cuda_memory_bytes,
        "peak_cuda_memory_gb": (
            round(result.peak_cuda_memory_bytes / 1e9, 4)
            if result.peak_cuda_memory_bytes is not None else None),
        "device": str(device),
        "stage_runtimes_seconds": [
            round(float(rec["runtime_seconds"]), 3) for rec in result.stage_records],
    }
    with open(os.path.join(args.output_dir, "runtime.json"), "w", encoding="utf-8") as handle:
        json.dump(runtime_payload, handle, indent=2)

    with open(os.path.join(args.output_dir, "checkpoint_reference.json"), "w",
              encoding="utf-8") as handle:
        json.dump({
            "checkpoint_path": ckpt_path,
            "trained_in_this_run": trained_now,
            "input_dimension": int(dim),
            "timesteps": int(diffusion.timesteps),
            "model_mean_type": diffusion.model_mean_type,
            "model_var_type": diffusion.model_var_type,
            "preprocessing": preproc.to_dict(),
        }, handle, indent=2)

    print(f"results written to: {args.output_dir}")
    print(f"total runtime: {timings['total_seconds']:.1f}s")
    if result.peak_cuda_memory_bytes:
        print(f"peak CUDA memory: {result.peak_cuda_memory_bytes / 1e9:.3f} GB")
    return result


if __name__ == "__main__":
    main()
