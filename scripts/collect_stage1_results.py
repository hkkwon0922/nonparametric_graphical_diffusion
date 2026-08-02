"""Measure the stage-1 sub-problem ``S = {0, 1}`` in isolation.

After the true sink (node 2) is removed from the ``0 -> 1 -> 2`` chain, the
remaining sub-problem is the induced sub-DAG ``0 -> 1`` on ``S = {0, 1}``, whose
correct leaf is node **1**.

This stage is scientifically more interesting than stage 0, because it is the
first one where the free block ``R`` is non-empty and the conditional Langevin
sampler actually runs: the marginal ``p_{t,S}`` is obtained by integrating out
``x_{t,2}`` via ULA rather than being available directly.

The full pipeline's own stage 1 conditions on whatever stage 0 happened to
select, so it is not a controlled measurement. Here ``S`` is pinned to
``{0, 1}`` so the sub-problem is always the correct one, isolating the
stage-1 criterion from stage-0 errors.

Also records an "oracle" variant that runs the ordering restricted to the
2-column data matrix ``X[:, [0, 1]]`` with a model trained on those two columns
only — i.e. what a *marginal* (not conditional) estimator would see. Comparing
the two shows what the conditional Langevin step buys.

Usage
-----
    python scripts/collect_stage1_results.py --device cuda:0
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
from models.dag_diffusion.reverse_posterior import sample_x0_S_given_xt_S
from models.dag_diffusion.score_adapter import DDPMScoreAdapter
from models.dag_diffusion.training import DDPMTrainConfig, load_ddpm_checkpoint, train_ddpm
from models.dag_diffusion.tweedie_hessian import estimate_tweedie_hessian_diagonal


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def stage_criterion(X, model, diffusion, device, S, t_order, num_anchors,
                    chains=32, samples=16, burn=300, thin=10, step=1e-3, seed=120,
                    reverse_draws=1):
    """Criterion ``V_i`` for a *pinned* index set ``S`` at one timestep.

    Mirrors exactly what ``ConditionalDiffusionDAGOrderEstimator`` does for one
    stage: build anchors by forward diffusion, restrict to ``S``, run the
    two-stage conditional sampler, form the Tweedie Hessian diagonal, and take
    the variance across anchors.

    Returns a record with ``criterion`` (one entry per element of ``S``),
    the per-anchor Hessian diagonal, and the wall time.
    """
    device = torch.device(device)
    dim = X.shape[1]
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)

    adapter = DDPMScoreAdapter(model=model, diffusion=diffusion, device=device)
    lang = LangevinConfig(num_chains=chains, burn_in=burn, num_samples=samples,
                          thinning=thin, step_size=step, init="forward_data",
                          chunk_size=8192)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    idx = torch.randperm(X_t.shape[0], device=device, generator=gen)[:num_anchors]
    x0_anchor = X_t[idx]
    t_vec = torch.full((num_anchors,), int(t_order), dtype=torch.int64, device=device)
    noise = torch.empty_like(x0_anchor).normal_(generator=gen)
    x_t_full = diffusion.q_sample(x_0=x0_anchor, t=t_vec, noise=noise)

    s_idx = torch.as_tensor(list(S), dtype=torch.long, device=device)
    x_t_S = x_t_full[:, s_idx]

    # match the estimator: when R is empty the sample budget comes from reverse draws
    draws = reverse_draws
    if len(S) == dim:
        draws = max(2, reverse_draws * chains * samples)

    _sync(device)
    t0 = time.time()
    diag_parts, lang_diags = [], []
    for start in range(0, num_anchors, 32):
        end = min(start + 32, num_anchors)
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) + 1000 + start)
        x0_S, ld = sample_x0_S_given_xt_S(
            model=model, diffusion=diffusion, score_adapter=adapter,
            x_t_S=x_t_S[start:end], condition_indices=list(S), dim=dim,
            t=int(t_order), langevin_config=lang, reverse_draws_per_xt=draws,
            reference_x0=X_t, generator=g, chunk_size=8192, device=device,
        )
        diag_parts.append(estimate_tweedie_hessian_diagonal(x0_S, diffusion, int(t_order)))
        lang_diags.append(ld)
    _sync(device)
    wall = time.time() - t0

    h_diag = torch.cat(diag_parts, dim=0)          # (B, |S|)
    crit = h_diag.var(dim=0, unbiased=True)        # (|S|,)
    crit_np = crit.detach().cpu().numpy()

    sel_local = int(np.argmin(crit_np))
    return {
        "S": [int(i) for i in S],
        "t_order": int(t_order), "num_anchors": int(num_anchors), "seed": int(seed),
        "chains": chains, "samples": samples,
        "criterion": crit_np.tolist(),
        "hessian_diag_mean": h_diag.mean(dim=0).detach().cpu().numpy().tolist(),
        "selected_leaf": int(S[sel_local]),
        "wall_seconds": wall,
        "free_block_empty": len(S) == dim,
        "mean_score_norm": float(np.mean([d.mean_score_norm for d in lang_diags]))
        if not len(S) == dim else None,
    }


def main():
    p = argparse.ArgumentParser(description="Measure the pinned S={0,1} stage-1 sub-problem.")
    p.add_argument("--data-path", default="data/dag_ordering_debug/dag3_chain.npy")
    p.add_argument("--checkpoint", default="results/dag_ordering/real_t3/ddpm.pt")
    p.add_argument("--output", default="results/dag_ordering_summary/stage1_summary.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-anchors", default=128, type=int)
    args = p.parse_args()

    for attr in ("data_path", "checkpoint", "output"):
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.normpath(os.path.join(REPO_ROOT, val)))

    device = args.device
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable; falling back to CPU")
        device = "cpu"

    X = np.load(args.data_path).astype(np.float32)
    model, diffusion, _, _ = load_ddpm_checkpoint(args.checkpoint, device, expected_dim=X.shape[1])

    out = {
        "device": str(device),
        "true_chain": "0 -> 1 -> 2",
        "stage0_correct_leaf": 2,
        "stage1_S": [0, 1],
        "stage1_correct_leaf": 1,
        "num_anchors": int(args.num_anchors),
    }

    t_values = [1, 2, 3, 5, 8, 12, 20, 30, 50]

    print("=== stage 1: pinned S={0,1}, conditional (Langevin marginalises x_2) ===", flush=True)
    out["stage1_conditional"] = []
    for t in t_values:
        rec = stage_criterion(X, model, diffusion, device, [0, 1], t, args.num_anchors)
        rec["correct"] = rec["selected_leaf"] == 1
        out["stage1_conditional"].append(rec)
        print(f"  t={t:3d} V=[{rec['criterion'][0]:9.3f}, {rec['criterion'][1]:9.3f}] "
              f"-> leaf x{rec['selected_leaf']} {'OK' if rec['correct'] else 'WRONG'} "
              f"({rec['wall_seconds']:.2f}s)", flush=True)

    print("=== stage 0 for reference: S={0,1,2} ===", flush=True)
    out["stage0_reference"] = []
    for t in t_values:
        rec = stage_criterion(X, model, diffusion, device, [0, 1, 2], t, args.num_anchors)
        rec["correct"] = rec["selected_leaf"] == 2
        out["stage0_reference"].append(rec)
        print(f"  t={t:3d} V={np.round(rec['criterion'], 3)} -> leaf x{rec['selected_leaf']} "
              f"{'OK' if rec['correct'] else 'WRONG'}", flush=True)

    # --- oracle: a model trained on columns [0,1] only -> TRUE 2-D marginal ---
    print("=== oracle: model retrained on X[:, [0,1]] (true 2-D marginal) ===", flush=True)
    X2 = X[:, [0, 1]].copy()
    cfg = DDPMTrainConfig(input_dimension=2, mid_features=128, num_temporal_layers=3,
                          timesteps=500, epochs=1500, batch_size=128, lr=1e-3, seed=120)
    t0 = time.time()
    m2, d2, _, _ = train_ddpm(X2, cfg, device, verbose=False)
    print(f"  (retraining took {time.time() - t0:.1f}s)", flush=True)

    out["stage1_oracle_2d_model"] = []
    for t in t_values:
        rec = stage_criterion(X2, m2, d2, device, [0, 1], t, args.num_anchors)
        rec["correct"] = rec["selected_leaf"] == 1
        out["stage1_oracle_2d_model"].append(rec)
        print(f"  t={t:3d} V=[{rec['criterion'][0]:9.3f}, {rec['criterion'][1]:9.3f}] "
              f"-> leaf x{rec['selected_leaf']} {'OK' if rec['correct'] else 'WRONG'}", flush=True)

    print("=== stage 1 seed stability (t=8) ===", flush=True)
    out["stage1_seed_stability"] = []
    for b in [32, 64, 128, 256]:
        for s in [120, 1230, 12340]:
            rec = stage_criterion(X, model, diffusion, device, [0, 1], 8, b, seed=s)
            ok = rec["selected_leaf"] == 1
            out["stage1_seed_stability"].append(
                {"B": b, "seed": s, "selected_leaf": rec["selected_leaf"], "correct": ok,
                 "criterion": rec["criterion"]})
            print(f"  B={b:4d} seed={s:6d} -> x{rec['selected_leaf']} "
                  f"{'OK' if ok else 'WRONG'}", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    print(f"\nsummary written: {args.output}")


if __name__ == "__main__":
    main()
