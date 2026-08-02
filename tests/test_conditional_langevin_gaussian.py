"""Conditional Langevin sampler validated against an exact Gaussian conditional.

No trained DDPM is involved: the target is a fixed correlated Gaussian whose
score is known in closed form, so the empirical conditional mean/covariance of
the free block can be compared with the analytic Schur-complement answer.
"""
import numpy as np
import pytest
import torch

from models.dag_diffusion.conditional_langevin import (
    LangevinConfig,
    LangevinDivergenceError,
    complement_indices,
    sample_conditional_langevin,
)
from models.dag_diffusion.score_adapter import AnalyticGaussianScoreAdapter


def gaussian_conditional(mean, cov, cond_idx, cond_val, free_idx):
    """Exact ``N(mu_{R|S}, Sigma_{R|S})`` via the Schur complement."""
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    S, R = list(cond_idx), list(free_idx)
    mu_R, mu_S = mean[R], mean[S]
    C_RR = cov[np.ix_(R, R)]
    C_RS = cov[np.ix_(R, S)]
    C_SS = cov[np.ix_(S, S)]
    solve = np.linalg.solve(C_SS, np.asarray(cond_val, dtype=np.float64) - mu_S)
    cond_mean = mu_R + C_RS @ solve
    cond_cov = C_RR - C_RS @ np.linalg.solve(C_SS, C_RS.T)
    return cond_mean, cond_cov


@pytest.fixture(scope="module")
def target():
    """A 3-D correlated Gaussian with a well-conditioned covariance."""
    mean = np.array([0.3, -0.2, 0.5])
    cov = np.array([
        [1.00, 0.50, 0.20],
        [0.50, 1.20, -0.30],
        [0.20, -0.30, 0.80],
    ])
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    return mean, cov


def test_conditional_mean_and_covariance_match_analytic(target):
    mean, cov = target
    dim = 3
    cond_idx = [0]
    free_idx = complement_indices(dim, cond_idx)
    cond_val = np.array([1.1])

    adapter = AnalyticGaussianScoreAdapter(
        torch.tensor(mean), torch.tensor(cov), device="cpu")

    cfg = LangevinConfig(
        num_chains=64, burn_in=3000, num_samples=400, thinning=5,
        step_size=5e-3, init="normal",
    )
    x_S = torch.tensor(cond_val, dtype=torch.float32).reshape(1, 1)
    samples, diag = sample_conditional_langevin(
        score_adapter=adapter, x_t_S=x_S, condition_indices=cond_idx,
        dim=dim, t=0, config=cfg, seed=17,
    )

    assert samples.shape == (1, cfg.num_chains * cfg.num_samples, len(free_idx))
    flat = samples[0].double().numpy()

    exp_mean, exp_cov = gaussian_conditional(mean, cov, cond_idx, cond_val, free_idx)
    emp_mean = flat.mean(axis=0)
    emp_cov = np.cov(flat, rowvar=False, ddof=1)

    # ULA has an O(h) discretisation bias on top of Monte-Carlo error; the
    # tolerances below are loose enough for that but tight enough to catch a
    # wrong conditional (e.g. sampling the marginal instead).
    np.testing.assert_allclose(emp_mean, exp_mean, atol=0.06)
    np.testing.assert_allclose(emp_cov, exp_cov, atol=0.10)


def test_conditioning_on_two_coordinates(target):
    mean, cov = target
    dim = 3
    cond_idx = [0, 2]
    free_idx = complement_indices(dim, cond_idx)
    cond_val = np.array([0.8, -0.4])

    adapter = AnalyticGaussianScoreAdapter(
        torch.tensor(mean), torch.tensor(cov), device="cpu")
    cfg = LangevinConfig(
        num_chains=64, burn_in=3000, num_samples=400, thinning=5,
        step_size=5e-3, init="normal",
    )
    x_S = torch.tensor(cond_val, dtype=torch.float32).reshape(1, 2)
    samples, _ = sample_conditional_langevin(
        score_adapter=adapter, x_t_S=x_S, condition_indices=cond_idx,
        dim=dim, t=0, config=cfg, seed=23,
    )

    flat = samples[0].double().numpy()
    exp_mean, exp_cov = gaussian_conditional(mean, cov, cond_idx, cond_val, free_idx)
    np.testing.assert_allclose(flat.mean(axis=0), exp_mean, atol=0.06)
    np.testing.assert_allclose(flat.var(axis=0, ddof=1), np.diag(exp_cov), atol=0.10)


def test_conditioned_coordinates_stay_exactly_fixed(target):
    """The sampler must never write into the conditioned block."""
    mean, cov = target
    dim = 3
    cond_idx = [1]
    cond_val = torch.tensor([[0.7]], dtype=torch.float32)

    seen = []

    class RecordingAdapter(AnalyticGaussianScoreAdapter):
        def score(self, x_t, t=0):
            seen.append(x_t[:, cond_idx].clone())
            return super().score(x_t, t)

    adapter = RecordingAdapter(torch.tensor(mean), torch.tensor(cov), device="cpu")
    cfg = LangevinConfig(num_chains=4, burn_in=10, num_samples=5, thinning=2,
                         step_size=1e-2, init="normal")
    x_S_in = cond_val.clone()
    samples, _ = sample_conditional_langevin(
        score_adapter=adapter, x_t_S=x_S_in, condition_indices=cond_idx,
        dim=dim, t=0, config=cfg, seed=5,
    )

    assert torch.equal(x_S_in, cond_val)  # input tensor untouched
    assert seen, "score was never evaluated"
    for observed in seen:
        assert torch.all(observed == 0.7), "conditioned coordinate drifted during ULA"
    assert samples.shape == (1, 20, 2)


def test_empty_free_block_skips_sampling(target):
    """S = {0..D-1} -> R empty: no Langevin simulation, empty sample tensor."""
    mean, cov = target
    dim = 3

    class ExplodingAdapter(AnalyticGaussianScoreAdapter):
        def score(self, x_t, t=0):  # pragma: no cover - must never be called
            raise AssertionError("score must not be evaluated when R is empty")

    adapter = ExplodingAdapter(torch.tensor(mean), torch.tensor(cov), device="cpu")
    cfg = LangevinConfig(num_chains=4, burn_in=100, num_samples=5, thinning=2)
    x_S = torch.randn(6, dim)

    samples, diag = sample_conditional_langevin(
        score_adapter=adapter, x_t_S=x_S, condition_indices=list(range(dim)),
        dim=dim, t=0, config=cfg, seed=1,
    )
    assert samples.shape == (6, 1, 0)
    assert diag.extras.get("skipped") is True


def test_multiple_anchors_are_independent(target):
    """Different anchors must yield different conditional means."""
    mean, cov = target
    dim = 3
    cond_idx = [0]
    adapter = AnalyticGaussianScoreAdapter(
        torch.tensor(mean), torch.tensor(cov), device="cpu")
    # Many chains + heavier thinning: the residual here is dominated by
    # Monte-Carlo noise across correlated chains, not by ULA's O(h) bias.
    cfg = LangevinConfig(num_chains=256, burn_in=3000, num_samples=400, thinning=10,
                         step_size=5e-3, init="normal")
    cond_vals = np.array([[-1.5], [1.5]])
    x_S = torch.tensor(cond_vals, dtype=torch.float32)

    samples, _ = sample_conditional_langevin(
        score_adapter=adapter, x_t_S=x_S, condition_indices=cond_idx,
        dim=dim, t=0, config=cfg, seed=11,
    )
    assert samples.shape[0] == 2
    for b in range(2):
        exp_mean, _ = gaussian_conditional(mean, cov, cond_idx, cond_vals[b], [1, 2])
        emp = samples[b].double().numpy().mean(axis=0)
        np.testing.assert_allclose(emp, exp_mean, atol=0.08)


def test_exploding_chain_raises(target):
    """A wildly oversized step size must raise, not silently return garbage."""
    mean, cov = target

    class BlowUpAdapter:
        device = torch.device("cpu")

        def score(self, x_t, t=0):
            return 1e12 * x_t

    cfg = LangevinConfig(num_chains=2, burn_in=50, num_samples=2, thinning=1,
                         step_size=1.0, init="normal", max_abs_value=1e6)
    with pytest.raises(LangevinDivergenceError):
        sample_conditional_langevin(
            score_adapter=BlowUpAdapter(), x_t_S=torch.zeros(1, 1),
            condition_indices=[0], dim=3, t=0, config=cfg, seed=3,
        )


def test_diagnostics_are_populated(target):
    mean, cov = target
    adapter = AnalyticGaussianScoreAdapter(
        torch.tensor(mean), torch.tensor(cov), device="cpu")
    cfg = LangevinConfig(num_chains=4, burn_in=50, num_samples=8, thinning=2,
                         step_size=5e-3, init="normal")
    _, diag = sample_conditional_langevin(
        score_adapter=adapter, x_t_S=torch.zeros(2, 1), condition_indices=[0],
        dim=3, t=0, config=cfg, seed=7,
    )
    assert np.isfinite(diag.mean_score_norm)
    assert np.isfinite(diag.max_score_norm)
    assert diag.chain_mean.shape == (2,)
    assert diag.per_chain_mean.shape == (2 * 4, 2)
    assert diag.lag1_autocorr.shape == (2,)
    assert diag.num_retained == 32
    assert isinstance(diag.to_dict(), dict)
