
# Nonparametric undirected graphical model selection using diffusion models [[arXiv]](https://arxiv.org/abs/2606.08468)
Hyeok Kyu Kwon, Myeonggu Kang, Minwoo Chae and Wanjie Wang



## Abstract

Undirected graphical models provide a fundamental framework for representing conditional independence structures among high-dimensional random variables. While undirected graphical model selection has become a central problem in high-dimensional statistics, most existing methods are restricted to parametric settings. In this paper, we develop a nonparametric approach to undirected graphical model selection based on diffusion models. Recent work has shown that diffusion models can adapt to the unknown graph structure of the underlying distribution, yet utilizing these models for explicit graph estimation remains unexplored. To bridge this gap, we introduce a novel diffusion-based method for nonparametric undirected graphical model selection. We establish the model selection consistency of the proposed method and demonstrate its empirical performance through extensive simulations and two real data analyses.

## Overview

To reproduce the numerical experiments in the main paper, readers can run the following notebooks:

```text
├── visualization/
│   ├── plot_results.ipynb          # Simulation results
│   ├── network_results.ipynb       # Network analysis
│   ├── image_results.ipynb         # Image analysis
│   ├── toy_example.ipynb           # Illustrative examples
```

## Reproducing the experiments

## 1. Environment setup

To avoid dependency conflicts between R-based statistical packages and PyTorch-based deep learning models, we use separate CPU and GPU environments.

```bash
# [1] CPU environment: GLASSO, NPN, and SING
# Includes R, TransportMaps, and PyCop
conda env create -f env_cpu.yml
conda activate env_cpu

# [2] GPU environment: L-SING and DDPM
# Includes PyTorch and CUDA
conda env create -f env_gpu.yml
conda activate env_gpu
```

## 2. Simulation pipeline

### Step 1 — Data generation

```bash
cd data
conda activate env_cpu
python generate_data.py
```

### Step 2 — Running the benchmark

```bash
cd experiments

# CPU models (require env_cpu)
python run_benchmark.py --model glasso
python run_benchmark.py --model npn
python run_benchmark.py --model sing --p_order 1
python run_benchmark.py --model sing --p_order 3

# GPU models (require env_gpu)
conda activate env_gpu
python run_benchmark.py --model lsing
python run_benchmark.py --model ddpm
```

## 3. Real data analysis

### Image analysis

```bash
cd experiments

# Train a UNet DDPM on MNIST and compute the pixel-wise Hessian
python train_ddpm_mnist.py # Trains a UNet DDPM on MNIST
python compute_hessian_mnist.py # Computes the per-timestep pixel Hessian
```

### Network analysis

```bash
cd experiments

# Train DDPM, then compute its Hessian
python train_ddpm_network.py # Reads data/network/sector_rt_csv_2019_connected/
python compute_hessian_network.py # Writes the Hessian pickle file
```





---

## 4. Diffusion-based DAG ordering by conditional Tweedie Hessians

An **experimental** extension that recovers a *topological ordering* of a DAG
from a single trained full-dimensional DDPM. It reuses the same
`Decoder5D_0204` network, `GaussianDiffusion` schedule and reverse-sampling
code as the undirected-graph pipeline above; nothing in that pipeline changes.

### What it does and does not produce

- It estimates a **topological order**, i.e. a permutation of `0..D-1`.
- It does **not** infer a sparse DAG skeleton. The
  `fully_connected_order_dag` field in the output is the *complete* DAG
  consistent with the estimated order — an order encoding, nothing more.
  Accordingly, only order-level metrics (order FNR, stagewise leaf validity)
  are reported; SHD and edge-F1 are deliberately omitted.
- The DDPM is trained **once**, on all `D` dimensions. No per-subset model is
  ever retrained.

### Method

With `X_t = sqrt(alpha_bar_t) X_0 + sqrt(1 - alpha_bar_t) eps`, write
`sigma2_t = 1 - alpha_bar_t`. For a remaining set `S`, the second-order Tweedie
identity gives the Hessian of the noised marginal log-density

```
H_S(x_S, t) = grad^2 log p_{t,S}(x_S)
            = alpha_bar_t / sigma2_t^2 * Cov(X_{0,S} | X_{t,S} = x_S)
              - I / sigma2_t
```

(the `sigma_t^4` of the write-up is `sigma2_t ** 2` in code, since
`sigma2_t = 1 - alpha_bar_t`).

At each stage, with `S` the set of remaining variables:

1. **Anchors.** Draw `B` conditioning anchors by forward-diffusing `B` rows of
   the data to `t_order` with `q_sample`, keeping only the `S` coordinates. The
   *same* full-dimensional anchor set is reused at every stage and merely
   restricted to the current `S`, so candidate nodes are never compared across
   different noise realisations.
2. **Conditional Langevin (Stage A).** Sample the free block
   `X_{t,R} | X_{t,S} = x_{t,S}`, `R = {0..D-1} \ S`, with fixed-time
   unadjusted Langevin dynamics. This is possible without retraining because
   `grad_{x_R} log p_t(x_R | x_S) = grad_{x_R} log p_t(x_R, x_S)`, i.e. the `R`
   rows of the learned *full* score with `x_S` pinned.
3. **Reverse diffusion (Stage B).** Merge `x_{t,S}` with the sampled `x_{t,R}`
   into a full `D`-dimensional `x_t`, reverse-diffuse `t -> 0` with
   `p_sample_step`, and retain only the `S` coordinates of the resulting `x_0`.
4. **Covariance and Hessian.** Estimate `Cov(X_{0,S} | X_{t,S})` from those
   samples (unbiased, `M-1` denominator, float64, Welford-style streaming) and
   apply the Tweedie formula.
5. **Leaf removal.** Score each `i in S` by the variance of its **signed**
   diagonal Hessian across the `B` anchors, `V_i = Var_b(H_ii)`, and remove
   `argmin_i V_i`. Ties break toward the smallest original index.

Repeat until one node remains. `leaf_order` records removals (sinks first) and
`topological_order = reverse(leaf_order)` runs source to sink.

At the first stage `S` is all of `{0..D-1}`, so `R` is empty and no Langevin
simulation is performed; the posterior-sample budget is supplied entirely by
independent reverse trajectories instead.

### Scientific caveat

The SCORE theorem — a leaf of a nonlinear additive-noise SCM has constant
diagonal score Hessian — is stated at `t = 0`, on the **clean** density. This
implementation evaluates the criterion at a strictly positive diffusion time
`t_order > 0`, where the Gaussian smoothing of `p_t` mixes contributions across
variables. **The positive-time criterion is experimental and is not
automatically implied by the `t = 0` guarantee.** Smaller `t_order` sits closer
to the regime where the theory applies, but pays for it with a larger
reverse-diffusion variance and a `1 / sigma2_t^2` amplification of covariance
error. `t_order` must be given explicitly; no averaging across timesteps is
performed, because no aggregation rule is defined for it.

### Observed behaviour on the D=3 chain

On the bundled `0 -> 1 -> 2` nonlinear ANM (2000 samples, a DDPM trained for
1500 epochs), the stage-0 criterion behaves as follows — `argmin` is the
selected leaf, and node 2 is the true sink:

| `t_order` | criterion `[x0, x1, x2]` | selected leaf | recovered order |
|---|---|---|---|
| 1 | `[231.3, 206.2, 240.5]` | 1 | `[0, 2, 1]` |
| 2 | `[120.3, 126.3, 126.9]` | 0 | `[1, 2, 0]` |
| 3 | `[93.7, 77.2, 66.7]` | 2 | `[0, 1, 2]` |
| 8 | `[24.2, 7.2, 6.6]` | 2 | `[0, 1, 2]` |
| 20 | `[8.3, 0.9, 0.5]` | 2 | `[0, 1, 2]` |

(128 anchors, 32 chains, 16 retained states.) At `t_order >= 3` the true order
is recovered with order FNR `0.0` and stagewise leaf validity `3/3`. At `t = 1`
and `t = 2` the criterion is flat and the selection is essentially noise: the
`1 / sigma2_t^2` factor amplifies covariance error badly when `sigma2_t` is
small, which is exactly the tension noted in the caveat above.

**Stage 1 (`S = {0, 1}`, the sub-problem after the true sink is removed).** This
is the first stage where the free block `R` is non-empty, so it is the first
that actually exercises the conditional Langevin sampler — stage 0 has
`R = empty` and only reverse-diffuses. Pinning `S = {0, 1}` (correct leaf: node
1) and comparing against an *oracle* DDPM retrained on `X[:, [0,1]]`:

| `t_order` | conditional margin `V(x0)/V(x1)` | oracle margin | conditional picks |
|---|---|---|---|
| 1 | 0.88 | 1.13 | x0 (wrong) |
| 3 | 1.42 | 1.30 | x1 |
| 8 | 3.48 | 3.81 | x1 |
| 20 | 4.58 | 11.18 | x1 |
| 50 | 0.99 | 2.35 | x0 (wrong) |

The conditional margin tracks the oracle up to `t ~ 8`, which is the empirical
evidence that marginalising `x_{t,2}` by Langevin really does approximate the
true marginal — the basis for using one D-dimensional model for every subset.
But it then falls behind: the oracle is correct at all nine timesteps while the
conditional fails at `t = 50`, and the relative error of `V(x1)` against the
oracle grows to 115% at `t = 12` and 208% at `t = 30`. **The ULA marginal
approximation degrades at large `t`** under a fixed step size and burn-in.

So the two stages fail at *opposite ends*: stage 0 at small `t` (the
`1/sigma^4` amplification), stage 1 at large `t` (Langevin degradation). Only
the middle band — roughly `t = 3..20` here — works for both, and where that band
lies for other data is not known in advance. Stage 1 was, however, stable across
all tested anchor counts (B = 32..256) and seeds.

**Parent identification via the off-diagonal.** Leaf *selection* uses the
variance of the diagonal; identifying the selected leaf's **parents** uses the
magnitude of the off-diagonal row against it,

    A_i = E_anchors |H_{i, i*}|,   i in S \ {i*}

where a **large** `A_i` means `i` is likely a parent of `i*` and `A_i ~ 0` means
it is likely not. On the chain, with the leaf correctly selected as node 2, the
true parent is node 1 and node 0 is a non-parent:

| `t_order` | `A_x0` (non-parent) | `A_x1` (true parent) | ratio |
|---|---|---|---|
| 3 | 3.994 ± 0.268 | 4.764 ± 0.307 | 1.19 |
| 8 | 1.392 ± 0.101 | 2.425 ± 0.141 | 1.74 |
| 20 | 0.453 ± 0.036 | 1.355 ± 0.059 | 2.99 |
| 30 | 0.305 ± 0.026 | 0.955 ± 0.042 | 3.13 |
| 50 | 0.205 ± 0.017 | 0.483 ± 0.023 | 2.36 |

The true parent is larger at **all seven** timesteps where the leaf was found,
by well beyond the anchor standard error. Note the ratio *improves* with `t`
here, the opposite of the leaf criterion's preference — the two sub-tasks do not
share an optimal `t_order`. In the linear-Gaussian limit the non-parent entry
vanishes exactly as `alpha_bar -> 1`, so the weak separation at small `t` is
estimation noise (the `1/sigma^4` amplification again), not a property of the
target; `tests/test_offdiag_parent_signal.py` pins this down analytically.

This ranks parent candidates; it does **not** threshold them into a sparse
parent set, so it remains a diagnostic rather than structure estimation.

Two further measured effects:

- **Sampling budget dominates anchor count.** At `t = 8`, `B = 64`, the run
  fails with `M = chains x retained = 64` posterior samples per anchor and
  succeeds from `M >= 128`. `B` scales runtime roughly linearly, so when the
  budget is tight, secure `M` first.
- **Small `B` is seed-dependent.** Across three seeds on CPU, `B = 64` and
  `B = 128` each recover the true order for only some seeds; all three agree
  only at `B = 256`. Results are bit-reproducible *within* a device, but CPU
  and CUDA `torch.Generator` streams differ, so the two devices can disagree
  when the criterion gap is small. **Do not trust a single run** — sweep seeds
  and confirm the order is stable.

This is a single tiny example and should not be read as a general validation of
the method.

A full walkthrough of the method, these measurements and the runtime
comparisons — with figures — is in [`main.ipynb`](main.ipynb). Every number in
it is regenerated by:

```bash
python scripts/collect_sweep_results.py --device cuda:0    # t/anchor/seed/budget sweeps
python scripts/collect_stage1_results.py --device cuda:0   # pinned S={0,1} + oracle comparison
python scripts/collect_offdiag_results.py --device cuda:0  # off-diagonal E|H_{i,i*}| parent evidence
```

### Debug run (CPU, a few minutes)

Settings below exist **only to validate that the code path works**. They are
not scientifically adequate — far too few anchors, posterior samples and
Langevin steps for a trustworthy covariance.

```bash
conda activate env_gpu

# generate the tiny D=3 nonlinear ANM chain 0 -> 1 -> 2
python data/make_dag_ordering_debug_data.py --output-dir data/dag_ordering_debug

python experiments/run_dag_ordering.py --config configs/dag_ordering_debug.json
```

### Checkpoint-based run

```bash
python experiments/run_dag_ordering.py \
    --data-path data/example.npy \
    --checkpoint-path checkpoints/example/ddpm.pt \
    --output-dir results/dag_ordering/example \
    --t-order 10 \
    --num-anchors 32 \
    --num-chains 8 \
    --langevin-burn-in 500 \
    --langevin-samples 20 \
    --langevin-thinning 20 \
    --langevin-step-size 1e-4 \
    --langevin-init forward_data \
    --reverse-draws-per-xt 1 \
    --sampling-chunk-size 256 \
    --seed 120 \
    --device cuda:0
```

Add `--train-if-missing` to train a DDPM when the checkpoint is absent, and
`--ground-truth-adjacency path/to/adjacency.npy` (with `A[i,j] = 1` meaning
`i -> j`) for the optional order FNR / leaf-validity evaluation.

### Cost and tuning

The estimator is nested:

```
(D-1) stages  x  B anchors  x  (chains x retained states)  x  reverse draws  x  t_order reverse steps
```

Network evaluations scale as roughly
`O(D * B * C * M * L * t_order)` plus `O(D * B * C * (burn_in + M * thinning))`
for the Langevin chains, so cost grows quickly in every direction.

| Parameter | Effect |
|---|---|
| `--t-order` | The single timestep of the criterion. Small = closer to the `t=0` theory but noisier (`1/sigma2^2` amplification); large = smoother but more smoothing bias. |
| `--num-anchors` | Sample size of the **variance across anchors** that *is* the criterion. Too few and the ranking is noise. Must be >= 2. |
| `--num-chains`, `--langevin-samples` | Posterior sample count per anchor; drives the covariance accuracy. |
| `--langevin-step-size` | ULA bias vs mixing. Too large diverges (raises `LangevinDivergenceError`). |
| `--langevin-burn-in`, `--langevin-thinning` | Chain equilibration and decorrelation. |
| `--sampling-chunk-size`, `--anchor-chunk-size` | Memory/throughput knobs; lower them on OOM. |
| `--standardize` | Opt-in only. Mean/scale are stored in the checkpoint and reapplied identically at inference; the ordering coordinate space is recorded in the output. No scale-invariance of the criterion is claimed. |

`--max-memory-gb` rejects settings whose reverse-diffusion batch would be
excessive before any allocation happens. `--device` is honoured with a safe
fallback to CPU; no GPU index is hard-coded.

### Outputs

```text
results/dag_ordering/<run>/
├── config.json                 # full resolved configuration
├── ordering_result.json        # leaf_order, topological_order, criterion_by_stage, warnings
├── stage_diagnostics.pt        # per-stage Hessian diagonals, criteria, Langevin diagnostics
├── runtime.json                # timings + peak CUDA memory
└── checkpoint_reference.json   # which model produced the ordering
```

### Tests

```bash
conda activate env_gpu
pytest -q             # score-adapter algebra, analytic Gaussian Langevin,
                      # analytic Tweedie Hessian, end-to-end smoke
pytest -q -m slow     # the expensive CLI round-trip
```
