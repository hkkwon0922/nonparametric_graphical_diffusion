"""Analytic validation of the Tweedie Hessian formula.

For ``X_0 ~ N(0, Sigma_0)`` and ``X_t = sqrt(a) X_0 + sqrt(1-a) eps`` the noised
marginal is exactly ``X_t ~ N(0, a Sigma_0 + (1-a) I)``, so

    grad^2 log p_t(x) = -[a Sigma_0 + (1-a) I]^{-1}

everywhere.  Feeding the exact posterior covariance ``Cov(X_0 | X_t)`` into the
Tweedie formula must reproduce that matrix.  This pins down the
``alpha_bar / sigma2**2`` scaling: writing ``sigma2**4`` or dropping
``alpha_bar`` fails immediately.
"""
import numpy as np
import pytest
import torch

from models.dag_diffusion.moments import BatchedWelford
from models.dag_diffusion.tweedie_hessian import (
    alpha_bar_and_sigma2,
    estimate_conditional_covariance,
    estimate_tweedie_hessian,
    estimate_tweedie_hessian_diagonal,
    tweedie_hessian_from_covariance,
)
from models.ddpm.core.ddpm_torch.toy import GaussianDiffusion, get_beta_schedule

TIMESTEPS = 60


def make_diffusion():
    betas = get_beta_schedule("linear", beta_start=0.001, beta_end=0.2, timesteps=TIMESTEPS)
    return GaussianDiffusion(betas=betas, model_mean_type="eps",
                             model_var_type="fixed-large", loss_type="mse")


def exact_posterior_cov(sigma0, alpha_bar):
    """``Cov(X_0 | X_t)`` for a centred Gaussian prior — independent of ``x_t``.

    ``Cov = Sigma_0 - a Sigma_0 (a Sigma_0 + (1-a) I)^{-1} Sigma_0``.
    """
    d = sigma0.shape[0]
    marginal = alpha_bar * sigma0 + (1.0 - alpha_bar) * np.eye(d)
    return sigma0 - alpha_bar * sigma0 @ np.linalg.solve(marginal, sigma0)


def exact_marginal_hessian(sigma0, alpha_bar):
    """``grad^2 log p_t = -[a Sigma_0 + (1-a) I]^{-1}``."""
    d = sigma0.shape[0]
    return -np.linalg.inv(alpha_bar * sigma0 + (1.0 - alpha_bar) * np.eye(d))


@pytest.fixture(scope="module")
def sigma0():
    s = np.array([
        [1.30, 0.40, 0.10],
        [0.40, 0.90, -0.25],
        [0.10, -0.25, 1.10],
    ])
    assert np.all(np.linalg.eigvalsh(s) > 0)
    return s


@pytest.mark.parametrize("t", [1, 5, 17, 40, 59])
def test_tweedie_matches_exact_marginal_hessian(sigma0, t):
    diffusion = make_diffusion()
    alpha_bar, sigma2 = alpha_bar_and_sigma2(diffusion, t)
    a = float(alpha_bar)

    cov = torch.tensor(exact_posterior_cov(sigma0, a), dtype=torch.float64).unsqueeze(0)
    got = tweedie_hessian_from_covariance(cov, alpha_bar, sigma2)[0].numpy()
    expected = exact_marginal_hessian(sigma0, a)

    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("t", [3, 25])
def test_subset_S_hessian_matches_marginal_of_S(sigma0, t):
    """Restricting to ``S`` must give the Hessian of the *marginal* on ``S``."""
    diffusion = make_diffusion()
    alpha_bar, sigma2 = alpha_bar_and_sigma2(diffusion, t)
    a = float(alpha_bar)

    S = [0, 2]
    sigma0_S = sigma0[np.ix_(S, S)]  # marginal prior on S

    cov_S = torch.tensor(exact_posterior_cov(sigma0_S, a), dtype=torch.float64).unsqueeze(0)
    got = tweedie_hessian_from_covariance(cov_S, alpha_bar, sigma2)[0].numpy()
    expected = exact_marginal_hessian(sigma0_S, a)
    np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-9)


def test_wrong_scaling_would_fail(sigma0):
    """Guard: sigma2**4 or a missing alpha_bar must NOT reproduce the answer."""
    diffusion = make_diffusion()
    alpha_bar, sigma2 = alpha_bar_and_sigma2(diffusion, 10)
    a, s2 = float(alpha_bar), float(sigma2)
    cov = exact_posterior_cov(sigma0, a)
    expected = exact_marginal_hessian(sigma0, a)
    eye = np.eye(sigma0.shape[0])

    wrong_pow4 = a / (s2 ** 4) * cov - eye / s2
    wrong_no_alpha = 1.0 / (s2 ** 2) * cov - eye / s2
    assert not np.allclose(wrong_pow4, expected, rtol=1e-3, atol=1e-3)
    assert not np.allclose(wrong_no_alpha, expected, rtol=1e-3, atol=1e-3)


def test_monte_carlo_covariance_recovers_hessian(sigma0):
    """Sampled posterior draws converge to the analytic Hessian."""
    diffusion = make_diffusion()
    t = 12
    alpha_bar, _ = alpha_bar_and_sigma2(diffusion, t)
    a = float(alpha_bar)

    post_cov = exact_posterior_cov(sigma0, a)
    rng = np.random.default_rng(0)
    M = 400_000
    draws = rng.multivariate_normal(np.zeros(3), post_cov, size=M)
    samples = torch.tensor(draws, dtype=torch.float64).unsqueeze(0)  # (1, M, 3)

    hess, cov = estimate_tweedie_hessian(samples, diffusion, t)
    expected = exact_marginal_hessian(sigma0, a)

    np.testing.assert_allclose(cov[0].numpy(), post_cov, atol=0.02)
    # sigma2^-2 amplifies covariance error, so compare with a matching tolerance
    scale = a / float(1.0 - a) ** 2
    np.testing.assert_allclose(hess[0].numpy(), expected, atol=0.02 * scale)


def test_diagonal_helper_matches_full_diagonal(sigma0):
    diffusion = make_diffusion()
    t = 8
    rng = np.random.default_rng(1)
    a = float(diffusion.alphas_bar[t])
    draws = rng.multivariate_normal(np.zeros(3), exact_posterior_cov(sigma0, a), size=5000)
    samples = torch.tensor(draws, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1)

    hess, _ = estimate_tweedie_hessian(samples, diffusion, t)
    diag = estimate_tweedie_hessian_diagonal(samples, diffusion, t)
    np.testing.assert_allclose(
        diag.numpy(), torch.diagonal(hess, dim1=-2, dim2=-1).numpy(), rtol=1e-10, atol=1e-12)


def test_covariance_requires_two_samples():
    diffusion = make_diffusion()
    single = torch.zeros(1, 1, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="M >= 2|at least 2"):
        estimate_conditional_covariance(single)
    with pytest.raises(ValueError, match="M >= 2|at least 2"):
        estimate_tweedie_hessian_diagonal(single, diffusion, 5)


def test_unbiased_denominator_is_m_minus_one():
    """Covariance must use the M-1 denominator by default."""
    x = torch.tensor([[[1.0], [3.0], [5.0]]], dtype=torch.float64)  # (1, 3, 1)
    cov = estimate_conditional_covariance(x, unbiased=True)
    biased = estimate_conditional_covariance(x, unbiased=False)
    assert float(cov[0, 0, 0]) == pytest.approx(4.0)      # var with ddof=1
    assert float(biased[0, 0, 0]) == pytest.approx(8.0 / 3.0)


def test_streaming_welford_matches_batch_covariance():
    """Chunked streaming updates equal a single-shot covariance."""
    rng = np.random.default_rng(7)
    data = rng.normal(size=(3, 5000, 4))
    full = estimate_conditional_covariance(torch.tensor(data), unbiased=True)

    acc = BatchedWelford(3, 4, dtype=torch.float64)
    for start in range(0, 5000, 137):
        acc.update(torch.tensor(data[:, start:start + 137]))
    assert acc.count == 5000
    np.testing.assert_allclose(acc.covariance().numpy(), full.numpy(), rtol=1e-9, atol=1e-11)


def test_hessian_is_not_absolute_valued(sigma0):
    """The diagonal must stay negative for a Gaussian — no abs() anywhere."""
    diffusion = make_diffusion()
    t = 20
    a = float(diffusion.alphas_bar[t])
    cov = torch.tensor(exact_posterior_cov(sigma0, a), dtype=torch.float64).unsqueeze(0)
    alpha_bar, sigma2 = alpha_bar_and_sigma2(diffusion, t)
    hess = tweedie_hessian_from_covariance(cov, alpha_bar, sigma2)
    assert (torch.diagonal(hess, dim1=-2, dim2=-1) < 0).all()


def test_out_of_range_timestep_raises():
    diffusion = make_diffusion()
    with pytest.raises(ValueError, match="out of range"):
        alpha_bar_and_sigma2(diffusion, TIMESTEPS)
