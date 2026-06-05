"""
==============================================================================
Butterfly Data Generation Module
==============================================================================
This module generates synthetic data for the Butterfly structure.
It creates a block-diagonal dependency where only adjacent pairs are dependent.

Dependencies: numpy
==============================================================================
"""

import numpy as np


def generate_butterfly(dim: int, n: int, seed: int) -> np.ndarray:
    """
    Generates n samples from the Butterfly dependency structure.

    Args:
        dim (int): Dimension of the data. Must be an even number.
        n (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Generated dataset of shape (n, dim).
    """
    assert dim % 2 == 0, "Dimension must be an even number for butterfly structure."

    rng = np.random.default_rng(seed)
    r = dim // 2

    P = rng.standard_normal(size=(n, r))
    W = rng.standard_normal(size=(n, r))
    Q = W * P

    # Interleave to create pairs [P1, Q1, P2, Q2, ...]
    samples = np.stack([P, Q], axis=-1).reshape(n, dim)

    return samples