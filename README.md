
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

**A second structure: the v-structure `0 -> 2 <- 1`.** The chain has one valid
order; the collider is harder — the two sources are marginally independent and
exchangeable (so `[0,1,2]` and `[1,0,2]` are *both* valid, which is why scoring
uses order-FNR and leaf validity rather than permutation equality), and the sink
has **two** parents.

Running the same pipeline end to end (256 anchors, 1500-epoch model):

- **Ordering.** Order FNR `0.0` with leaf validity `3/3` at `t = 3, 8, 12, 20,
  30, 50`; the collider `x2` is correctly removed first. It fails at `t = 1, 2`
  as before, and additionally at `t = 5` — the v-structure is measurably less
  stable than the chain.
- **Parents.** With the leaf correctly at `x2`, *both* `E|H_{x0,x2}|` and
  `E|H_{x1,x2}|` are large at every such `t` — the off-diagonal test flags both
  parents, not just one.
- **Non-edge.** Between the two sources, which have no edge, `E|H_{x0,x1}|`
  decays to **0.075** at `t = 50` (from 2.56 at `t = 8`), whereas the same
  quantity in the chain — where `x0 -> x1` *is* an edge — stays around 0.5. The
  statistic does separate edge from non-edge. Caveat: large `t` also degrades
  the Langevin approximation (see stage 1 above), and both effects push in the
  same direction, so "small, therefore no edge" needs the approximation quality
  checked separately.

Neither structure's parent analysis applies a **threshold** — it ranks
candidates. Turning that into a sparse parent set would need a cutoff or test,
which is out of scope here.

These are two small D=3 examples and should not be read as a general validation
of the method.

A full walkthrough of the method, these measurements and the runtime
comparisons — with figures — is in [`main.ipynb`](main.ipynb). Every number in
it is regenerated by:

```bash
python scripts/collect_sweep_results.py --device cuda:0    # t/anchor/seed/budget sweeps
python scripts/collect_stage1_results.py --device cuda:0   # pinned S={0,1} + oracle comparison
python scripts/collect_offdiag_results.py --device cuda:0  # off-diagonal E|H_{i,i*}| parent evidence

# the v-structure experiment (generate data, train, then sweep)
python data/make_dag_ordering_debug_data.py --structure vstructure \
    --output-dir data/dag_ordering_debug
python experiments/run_dag_ordering.py \
    --data-path data/dag_ordering_debug/dag3_vstructure.npy \
    --ground-truth-adjacency data/dag_ordering_debug/dag3_vstructure_adjacency.npy \
    --checkpoint-path results/dag_ordering/vstruct/ddpm.pt --train-if-missing \
    --output-dir results/dag_ordering/vstruct \
    --epochs 1500 --t-order 8 --num-anchors 256 --device cuda:0
python scripts/collect_vstructure_results.py --device cuda:0
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

---

## 5. Causal-discovery benchmark: ordering + parent selection

A single framework for running the whole pipeline — **topological ordering**
followed by **parent (edge) selection** — on any dataset. Only the data path
changes between datasets.

```bash
# your own data: (n, d) matrix + (d, d) adjacency with A[i,j]=1 meaning i -> j
python experiments/run_causal_benchmark.py \
    --data-path path/to/X.npy --adjacency-path path/to/A.npy \
    --output-dir results/causal_benchmark/mydata --device cuda:0

# or synthesise the benchmark scenario described below
python experiments/run_causal_benchmark.py --generate \
    --num-nodes 20 --density dense --num-samples 1000 --data-seed 0 \
    --output-dir results/causal_benchmark/er20_seed0 --device cuda:0
```

### Data: the "vanilla" scenario of Montagna et al. (2023)

[`data/benchmark_scm.py`](data/benchmark_scm.py) reproduces the correctly
specified setting of [arXiv:2310.13387](https://arxiv.org/abs/2310.13387)
(Section 3.1), so results are comparable with the numbers reported there:

- nonlinear additive noise model `X_i = f_i(PA_i) + U_i`;
- mechanisms `f_i` drawn from a **Gaussian process** with a unit-bandwidth RBF
  kernel (their Appendix B.1);
- Gaussian noise `U_i ~ N(0, sigma_i)`, `sigma_i ~ U(0.5, 1.0)`;
- Erdos-Renyi graphs with their **Table 2** density schema (ER-20 dense uses
  `m = 4`, i.e. ~80 edges; verified empirically at ~80).

Mechanisms are rescaled to unit variance and the data are standardized by
default. Without this, *varsortability* (Reisach et al.) is ~0.63 — marginal
variance alone partly reveals the causal order, which the paper explicitly
warns can game a benchmark. Standardizing brings it to ~0.48 (chance).

### Parent selection: two routes

Both are restricted to pairs the estimated order admits, so the output is
acyclic by construction.

| Method | Rule |
|---|---|
| `das` | DAS-style test of `H0: E[H_{i,j}] = 0` across anchors (a t-test per candidate pair, Benjamini-Hochberg FDR at `--alpha`), following Montagna et al. Lemma 3. |
| `cluster` | 2-means over the per-timestep profile `(|H_{ij}(t_1)|, ..., |H_{ij}(t_T)|)`, normalised **per timestep across pairs**. |

`--cluster-transform` selects the normalisation. The default is **`rank`**, not
`zscore`: `|H_{ij}|` is heavy-tailed across pairs, so z-scored k-means splits
off a few extreme pairs rather than finding the edge boundary (measured on
ER-10 dense: F1 **0.30** for `zscore` vs **0.77** for `rank`, identical features
and seeds). Ranks are invariant to any monotone per-timestep rescaling.

### Metrics

Following the paper's Appendix D, with reversed edges counted as false
negatives:

**Ordering stage** (`ordering_metrics`) — a DAG usually admits many valid
topological orders, so these score only the pairs the graph actually
constrains; *any* valid order attains a perfect value:

| Metric | Meaning | Perfect | Chance |
|---|---|---|---|
| `fnr_pi` | fraction of true edges the order reverses (the paper's FNR-pi) | 0 | ~0.5 |
| `edge_accuracy` | `1 - fnr_pi` | 1 | ~0.5 |
| `ancestor_accuracy` | fraction of **transitive-closure** ancestor pairs ordered correctly — also scores indirect constraints | 1 | ~0.5 |
| `kendall_tau` | `ancestor_accuracy` rescaled to `[-1, 1]` | 1 | ~0 |

**Edge stage** — `f1` / `fnr` / `fpr` of the final edge set, with reversed
edges counted as false negatives.

Every run additionally reports `das_oracle_order` and `cluster_oracle_order`,
which repeat parent selection on the **true** order. The gap between these and
the ordinary results is exactly the damage done by ordering errors, so the two
stages never get conflated.

### Multi-seed sweep

```bash
python experiments/run_benchmark_sweep.py \
    --num-nodes 20 --density dense --num-samples 1000 --seeds 0,1,2 \
    --output-root results/causal_benchmark/er20_dense \
    --epochs 2000 --batch-size 256 --mid-features 256 \
    --t-order 8 --num-anchors 512 --anchor-chunk-size 64 \
    --num-chains 64 --langevin-samples 32 --device cuda:0
```

Aggregates as median / quartiles across seeds (the paper uses 20 seeds and
violin plots) and includes their random baseline (Appendix C.10: random order,
each admitted edge kept with probability 0.5).

**Measured ordering performance** (ER-20 dense, 3 seeds, 512 anchors), against
a Monte-Carlo reference of 2000 random permutations on the same graphs:

| Metric | Estimated (median) | Random permutation | z |
|---|---|---|---|
| FNR-pi | 0.151 | 0.501 ± 0.096 | −3.6 |
| edge accuracy | 0.849 | 0.499 ± 0.096 | +3.6 |
| ancestor accuracy | 0.847 | 0.500 ± 0.097 | +3.8 |
| Kendall tau | 0.693 | −0.001 ± 0.194 | +3.8 |

Roughly 85% of true edges and of transitive-closure ancestor pairs get the
right direction, consistently across seeds (edge accuracy 0.797–0.875). None of
the 2000 random permutations was a valid order. No run produced a *perfect*
order, though — 10–15 edges stay reversed per seed, which is what the
`cluster` 0.644 vs `cluster_oracle_order` 0.706 gap measures.

Regenerate with:

```bash
python scripts/collect_ordering_performance.py
```

**Does more sampling give *exact* recovery? No, beyond small graphs.** Both
axes were swept — estimator budget (anchors and posterior samples, fixed `n`)
and data size (`n` with a retrained model, fixed budget) — over D = 5, 10, 20 x
3 seeds, 108 runs in ~7 hours. Judged strictly by `is_valid_order` (zero
reversed edges):

| Perfect recoveries | D=5 | D=10 | D=20 |
|---|---|---|---|
| budget axis (variance only) | 16/18 | 0/18 | **0/18** |
| data axis (also model quality) | 18/18 | 4/18 | **0/18** |

- **D=5 saturates** on both axes; the budget threshold sits between 128 and 256
  anchors.
- **D=10 is unlocked only by more data** (0/18 on the budget axis vs 4/18 from
  `n >= 5000`), which says the binding constraint there is the *learned score*,
  not estimator variance — spending more budget just estimates a flawed score
  more precisely.
- **D=20 never reaches it.** Edge accuracy does improve (0.647 → 0.885 on the
  budget axis, 0.780 → 0.873 on the data axis) but reversed edges only fall
  from ~17 to ~10 and never to zero, with clear diminishing returns: doubling
  posterior samples from 4096 to 8192 moved accuracy 0.881 → 0.885.

The residual error therefore looks like **criterion bias, not variance** — the
same positive-diffusion-time issue behind the DAS null hypothesis not being
strictly true. More samples estimate the bias more accurately rather than
removing it; getting past it needs a methodological change (a `t -> 0` limit or
an explicit bias correction), not more compute. The axes are also not fully
independent: anchors are drawn from data rows, so `anchors <= n`.

Reproduce with:

```bash
python experiments/run_scaling_study.py --mode budget --device cuda:0
python experiments/run_scaling_study.py --mode data   --device cuda:0
python scripts/analyze_scaling.py
```

**Budget matters.** On ER-10 dense, raising anchors from 128 to 512 moved
ordering FNR-pi from 0.29 to 0.12 and DAS F1 from 0.30 to 0.55. The DAS test is
power-limited: with 128 anchors the true-edge `|t|` statistics had median 1.73
against a 1.96 threshold while *no* non-edge exceeded it — the separation was
real but the sample size was too small to reject. `|t|` grows as `sqrt(anchors)`,
so anchors are the lever.
