"""
==============================================================================
Copula Gaussian Data Generation Module
==============================================================================
This module generates synthetic data from a Gaussian copula with mixture Beta
marginals. It is designed to evaluate undirected graph estimation methods.

Data Generation Process:
1. Generates an AR(1) correlation matrix for the underlying Gaussian copula.
2. Samples from the Gaussian copula using the specified correlation matrix.
3. Transforms the uniform marginals of the copula into mixture Beta
   distributions using a fast numerical inversion technique (interpolation
   over a pre-computed grid) to avoid slow sample-by-sample bisection.
4. Scales the final samples to the range [-1, 1].

Dependencies: numpy, scipy, pycop
==============================================================================
"""

import numpy as np
from scipy.stats import beta
from pycop import simulation


def ar_matrix(dim, rho):
    """
    Generates a d x d AR(1) correlation matrix.

    Args:
        dim (int): Dimension of the matrix.
        rho (float): Correlation parameter.

    Returns:
        np.ndarray: AR(1) correlation matrix of shape (dim, dim).
    """
    idx = np.arange(dim)
    return rho ** np.abs(np.subtract.outer(idx, idx))


def mixture_beta_ppf_grid(u, alphas, betas, weights, grid_size=4096):
    """
    Computes the Percent Point Function (Inverse CDF) for a mixture of Beta
    distributions using fast grid interpolation.

    Args:
        u (np.ndarray): Uniform samples in [0, 1], shape (n,).
        alphas, betas, weights (np.ndarray): Parameters for the mixture components.
        grid_size (int): Number of points for the interpolation grid.
                         Higher means more accurate but slower.

    Returns:
        np.ndarray: Transformed samples.
    """
    u = np.asarray(u, dtype=np.float64)
    u = np.clip(u, 0.0, 1.0)

    # Create a grid on [0, 1]
    xg = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)

    # Compute the mixture CDF on the grid (vectorized)
    Fg = np.zeros_like(xg)
    for w, a, b in zip(weights, alphas, betas):
        Fg += w * beta.cdf(xg, a, b)

    # Enforce monotonicity to avoid numerical issues
    Fg = np.maximum.accumulate(Fg)
    Fg[0] = 0.0
    Fg[-1] = 1.0

    # Ensure unique values for stable interpolation
    Fg_u, idx = np.unique(Fg, return_index=True)
    xg_u = xg[idx]

    # Inverse transform via interpolation
    return np.interp(u, Fg_u, xg_u)


def generate_cop_gau(dim=20, n=1000, seed=1234, margin_seed=1234, rho=0.8, grid_size=4096):
    """
    Generates dataset using a Gaussian copula with mixture Beta marginals.

    Args:
        dim (int): Number of variables (D).
        n (int): Number of samples (N).
        seed (int): Random seed for copula sampling.
        margin_seed (int): Random seed for marginal parameter generation.
        rho (float): AR(1) correlation coefficient.
        grid_size (int): Grid size for the fast PPF computation.

    Returns:
        np.ndarray: Generated dataset of shape (n, dim) scaled to [-1, 1].
    """
    # 1. Generate Copula Samples
    Sigma = ar_matrix(dim, rho)
    np.random.seed(seed)
    cop_samples = simulation.simu_gaussian(dim, n, Sigma)
    cop_samples = np.vstack(cop_samples).T.astype(np.float64)

    # 2. Setup Marginals (Different mixture beta for each dimension)
    df = 2
    np.random.seed(margin_seed)
    alphas = np.random.chisquare(df, (dim, 2)) + 1.5
    betas = np.random.chisquare(df, (dim, 2)) + 1.5
    weights = np.ones((dim, 2), dtype=np.float64) * 0.5

    # 3. Apply fast inversion to transform marginals
    samples = np.empty_like(cop_samples, dtype=np.float64)
    for j in range(dim):
        samples[:, j] = mixture_beta_ppf_grid(
            cop_samples[:, j],
            alphas[j], betas[j], weights[j],
            grid_size=grid_size
        )

    # 4. Scale the final samples to the range [-1, 1]
    samples = (2.0 * samples - 1.0).astype(np.float32)
    return samples