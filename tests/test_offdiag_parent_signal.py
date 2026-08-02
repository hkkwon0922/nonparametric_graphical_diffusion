"""The off-diagonal Hessian entry ``H_{i,j}`` carries the parent signal.

Leaf selection uses the *variance of the diagonal*. Identifying the leaf's
**parents** instead uses the *magnitude of the off-diagonal* row against the
selected leaf: for a linear-Gaussian SCM the exact noised Hessian is

    H(t) = -[alpha_bar Sigma_0 + (1 - alpha_bar) I]^{-1}

and on the chain ``0 -> 1 -> 2`` the entry ``H_{0,2}`` (non-parent of the leaf)
vanishes as ``alpha_bar -> 1`` while ``H_{1,2}`` (true parent) stays away from
zero. These tests pin that behaviour down analytically, so the notebook's
"large |H_{i,i*}| => parent" reading is anchored to something exact rather than
to one Monte-Carlo run.
"""
import numpy as np
import pytest
import torch

from models.dag_diffusion.tweedie_hessian import tweedie_hessian_from_covariance


def chain_sigma0(a=0.9, b=0.8, noise_scale=1.0):
    """Covariance of the linear-Gaussian chain ``x1 = a x0 + e``, ``x2 = b x1 + e``."""
    B = np.array([[0.0, 0.0, 0.0],
                  [a, 0.0, 0.0],
                  [0.0, b, 0.0]])
    M = np.linalg.inv(np.eye(3) - B)
    return M @ (noise_scale ** 2 * np.eye(3)) @ M.T


def exact_noised_hessian(sigma0, alpha_bar):
    d = sigma0.shape[0]
    return -np.linalg.inv(alpha_bar * sigma0 + (1.0 - alpha_bar) * np.eye(d))


@pytest.mark.parametrize("alpha_bar", [0.99, 0.9, 0.7, 0.5])
def test_true_parent_dominates_non_parent(alpha_bar):
    """|H_{1,2}| (true parent of leaf 2) must exceed |H_{0,2}| (non-parent)."""
    sigma0 = chain_sigma0()
    H = exact_noised_hessian(sigma0, alpha_bar)
    assert abs(H[1, 2]) > abs(H[0, 2]), (
        f"parent signal inverted at alpha_bar={alpha_bar}: "
        f"|H[1,2]|={abs(H[1, 2]):.4f} <= |H[0,2]|={abs(H[0, 2]):.4f}"
    )


def test_non_parent_entry_vanishes_at_zero_noise():
    """As alpha_bar -> 1 (t -> 0) the non-parent entry goes to zero.

    This is the exact statement behind "near zero => not a parent"; it holds in
    the clean-density limit, not at arbitrary positive t.
    """
    sigma0 = chain_sigma0()
    prev = None
    for alpha_bar in [0.5, 0.7, 0.9, 0.99, 0.999]:
        H = exact_noised_hessian(sigma0, alpha_bar)
        cur = abs(H[0, 2])
        if prev is not None and alpha_bar >= 0.9:
            assert cur < prev, "non-parent entry should shrink as alpha_bar -> 1"
        prev = cur
    assert prev < 1e-2, f"non-parent entry did not vanish: {prev}"


def test_parent_discrimination_degrades_with_noise():
    """The parent/non-parent ratio shrinks as the diffusion time grows.

    Motivates keeping ``t_order`` moderate: heavy smoothing mixes the
    coordinates and washes out the off-diagonal contrast.
    """
    sigma0 = chain_sigma0()
    ratios = []
    for alpha_bar in [0.99, 0.9, 0.7, 0.5, 0.3]:
        H = exact_noised_hessian(sigma0, alpha_bar)
        ratios.append(abs(H[1, 2]) / max(abs(H[0, 2]), 1e-12))
    assert ratios == sorted(ratios, reverse=True), f"ratio not monotone: {ratios}"
    assert ratios[0] > 50, "expected strong discrimination near t=0"
    assert ratios[-1] < 5, "expected weak discrimination at heavy noise"


def test_tweedie_reproduces_offdiagonal_from_posterior_covariance():
    """The off-diagonal read from Tweedie matches the exact noised Hessian.

    Guards the whole path used by ``collect_offdiag_results.py``: posterior
    covariance -> Tweedie -> off-diagonal entry.
    """
    sigma0 = chain_sigma0()
    alpha_bar = 0.8
    sigma2 = 1.0 - alpha_bar

    marginal = alpha_bar * sigma0 + sigma2 * np.eye(3)
    post_cov = sigma0 - alpha_bar * sigma0 @ np.linalg.solve(marginal, sigma0)

    cov = torch.tensor(post_cov, dtype=torch.float64).unsqueeze(0)
    H = tweedie_hessian_from_covariance(cov, alpha_bar, sigma2)[0].numpy()
    expected = exact_noised_hessian(sigma0, alpha_bar)

    np.testing.assert_allclose(H, expected, rtol=1e-9, atol=1e-11)
    # and the parent ordering survives the round trip
    assert abs(H[1, 2]) > abs(H[0, 2])


def test_hessian_is_symmetric():
    """H_{i,j} == H_{j,i}: the off-diagonal row against the leaf is well defined."""
    sigma0 = chain_sigma0()
    alpha_bar = 0.7
    sigma2 = 1.0 - alpha_bar
    marginal = alpha_bar * sigma0 + sigma2 * np.eye(3)
    post_cov = sigma0 - alpha_bar * sigma0 @ np.linalg.solve(marginal, sigma0)
    cov = torch.tensor(post_cov, dtype=torch.float64).unsqueeze(0)
    H = tweedie_hessian_from_covariance(cov, alpha_bar, sigma2)[0].numpy()
    np.testing.assert_allclose(H, H.T, rtol=1e-12, atol=1e-13)
