"""
==============================================================================
Pair Gaussian Data Generation Module
==============================================================================
This module generates synthetic data from a Pairwise Gaussian structure.
The covariance matrix is block-diagonal with user-defined correlation (rho).

Dependencies: numpy
==============================================================================
"""

import numpy as np


def create_population_gaussian_pair(dim: int, correlation: float = 0.8):
    """
    Generates the population covariance and precision matrices.
    """
    cov = np.eye(dim)
    prec = np.eye(dim)
    adj = np.zeros((dim, dim))

    det = 1.0 - correlation ** 2
    p_diag = 1.0 / det
    p_off = -correlation / det

    for i in range(0, dim, 2):
        if i + 1 < dim:
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1

            cov[i, i + 1] = correlation
            cov[i + 1, i] = correlation

            prec[i, i] = p_diag
            prec[i + 1, i + 1] = p_diag
            prec[i, i + 1] = p_off
            prec[i + 1, i] = p_off

    return adj, prec, cov


def generate_pair_gau(dim: int, n: int, seed: int, correlation: float = 0.8) -> np.ndarray:
    """
    Generates n samples from the Pairwise Gaussian structure.

    Args:
        dim (int): Dimension of the data.
        n (int): Number of samples to generate.
        seed (int): Random seed.
        correlation (float): Signal size (rho) between adjacent pairs.

    Returns:
        np.ndarray: Generated dataset of shape (n, dim).
    """
    _, _, cov = create_population_gaussian_pair(dim, correlation)

    if seed is not None:
        np.random.seed(seed)

    mean = np.zeros(dim)
    samples = np.random.multivariate_normal(mean, cov, size=n)

    return samples