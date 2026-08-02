"""Off-diagonal Hessian entries ``H_{i, i*}`` against the selected leaf ``i*``.

Once a stage has picked its leaf ``i*`` (the ``argmin`` of the diagonal-Hessian
variance), the *off-diagonal* row ``H_{i, i*}`` for ``i in S \\ {i*}`` carries the
parent information: in the SCORE line of work, a node ``i`` that is a **parent**
of the leaf shows a non-vanishing cross second derivative, whereas a non-parent
contributes (close to) zero.

So this script reports, per timestep,

    A_i = E_anchors | H_{i, i*}(x_S, t) |     -> LARGE  = likely parent of i*
                                                 near 0 = likely NOT a parent

which complements the diagonal *variance* criterion used for leaf selection.
Both are computed from the same full conditional covariance, so the full
``(B, |S|, |S|)`` Hessian is materialised here (``compute_full_covariance``).

Ground truth on the bundled chain ``0 -> 1 -> 2``:

  * stage 0, ``S = {0,1,2}``, leaf ``i* = 2``  -> parent of 2 is **1**; 0 is not.
  * stage 1, ``S = {0,1}``,   leaf ``i* = 1``  -> parent of 1 is **0**.
    (With only one candidate left this stage cannot discriminate, but the
    magnitude is still informative and is reported for completeness.)

Usage
-----
    python scripts/collect_offdiag_results.py --device cuda:0
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
from models.dag_diffusion.training import load_ddpm_checkpoint
from models.dag_diffusion.tweedie_hessian import estimate_tweedie_hessian


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def stage_full_hessian(X, model, diffusion, device, S, t_order, num_anchors,
                       chains=32, samples=16, burn=300, thin=10, step=1e-3,
                       seed=120, reverse_draws=1):
    """Full Tweedie Hessian for a pinned ``S`` at one timestep.

    Returns a record holding, per anchor, the whole ``(B, |S|, |S|)`` Hessian
    reduced to the summaries the notebook needs:

      ``diag_variance``   (|S|,)      Var_b(H_ii)  -- the leaf criterion
      ``abs_mean``        (|S|,|S|)   E_b |H_ij|   -- parent evidence
      ``signed_mean``     (|S|,|S|)   E_b  H_ij
      ``abs_se``          (|S|,|S|)   standard error of E_b|H_ij| across anchors
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

    draws = reverse_draws
    if len(S) == dim:  # R empty -> budget must come from reverse trajectories
        draws = max(2, reverse_draws * chains * samples)

    _sync(device)
    t0 = time.time()
    hess_parts = []
    for start in range(0, num_anchors, 32):
        end = min(start + 32, num_anchors)
        g = torch.Generator(device=device)
        g.manual_seed(int(seed) + 1000 + start)
        x0_S, _ = sample_x0_S_given_xt_S(
            model=model, diffusion=diffusion, score_adapter=adapter,
            x_t_S=x_t_S[start:end], condition_indices=list(S), dim=dim,
            t=int(t_order), langevin_config=lang, reverse_draws_per_xt=draws,
            reference_x0=X_t, generator=g, chunk_size=8192, device=device,
        )
        hess, _ = estimate_tweedie_hessian(x0_S, diffusion, int(t_order))
        hess_parts.append(hess)
    _sync(device)
    wall = time.time() - t0

    H = torch.cat(hess_parts, dim=0)                      # (B, |S|, |S|) float64
    B = H.shape[0]
    diag = torch.diagonal(H, dim1=-2, dim2=-1)            # (B, |S|)
    diag_var = diag.var(dim=0, unbiased=True)

    absH = H.abs()
    abs_mean = absH.mean(dim=0)                           # (|S|, |S|)
    abs_se = absH.std(dim=0, unbiased=True) / np.sqrt(B)
    signed_mean = H.mean(dim=0)

    sel_local = int(torch.argmin(diag_var).item())
    return {
        "S": [int(i) for i in S],
        "t_order": int(t_order), "num_anchors": int(B), "seed": int(seed),
        "diag_variance": diag_var.cpu().numpy().tolist(),
        "abs_mean": abs_mean.cpu().numpy().tolist(),
        "abs_se": abs_se.cpu().numpy().tolist(),
        "signed_mean": signed_mean.cpu().numpy().tolist(),
        "selected_leaf": int(S[sel_local]),
        "selected_local_index": sel_local,
        "wall_seconds": wall,
    }


def main():
    p = argparse.ArgumentParser(
        description="Off-diagonal Hessian E|H_{i,i*}| against the selected leaf.")
    p.add_argument("--data-path", default="data/dag_ordering_debug/dag3_chain.npy")
    p.add_argument("--checkpoint", default="results/dag_ordering/real_t3/ddpm.pt")
    p.add_argument("--output", default="results/dag_ordering_summary/offdiag_summary.json")
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

    t_values = [1, 2, 3, 5, 8, 12, 20, 30, 50]
    out = {
        "device": str(device),
        "true_chain": "0 -> 1 -> 2",
        "num_anchors": int(args.num_anchors),
        "t_values": t_values,
        # ground truth parents of each node in the chain
        "true_parents": {"0": [], "1": [0], "2": [1]},
        "interpretation": (
            "A_i = E_anchors |H_{i,i*}|. LARGE => i is likely a parent of the "
            "selected leaf i*; near zero => likely NOT a parent."
        ),
    }

    for tag, S, expected_leaf in [("stage0", [0, 1, 2], 2), ("stage1", [0, 1], 1)]:
        print(f"=== {tag}: S={S} (expected leaf x{expected_leaf}) ===", flush=True)
        recs = []
        for t in t_values:
            rec = stage_full_hessian(X, model, diffusion, device, S, t, args.num_anchors)
            i_star = rec["selected_leaf"]
            k = rec["selected_local_index"]
            am = np.array(rec["abs_mean"])
            others = [(S[j], am[j, k]) for j in range(len(S)) if j != k]
            rec["offdiag_abs_mean_vs_leaf"] = {str(i): float(v) for i, v in others}
            rec["leaf_matches_expected"] = (i_star == expected_leaf)
            recs.append(rec)
            desc = "  ".join(f"E|H_(x{i},x{i_star})|={v:9.4f}" for i, v in others)
            print(f"  t={t:3d} leaf=x{i_star}{'' if rec['leaf_matches_expected'] else ' (!)'}"
                  f"  {desc}", flush=True)
        out[tag] = recs

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)
    print(f"\nsummary written: {args.output}")


if __name__ == "__main__":
    main()
