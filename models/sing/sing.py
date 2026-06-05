"""
==============================================================================
Sparsity Identification for Non-Gaussian (SING) Estimator
==============================================================================
This module serves as a standardized wrapper for the SING algorithm.
It leverages measure transport (Transport Maps) to estimate conditional
independence in non-Gaussian distributions by computing the Generalized
Precision Matrix via the Hessian of the log-density.

Methodology Highlights:
1. Approximates the non-Gaussian target distribution using a monotone
   triangular transport map driven by unconstrained monotonic neural networks
   or polynomial expansions.
2. Extracts conditional independencies using the variance-normalized
   generalized precision matrix.

Integration by: Hyeok Kyu Kwon & Myeonggu Kang (2026)
Dependencies: numpy, TransportMaps, core SING modules
==============================================================================
"""

import sys
import numpy as np
import warnings

# Suppress TransportMaps internal warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=UserWarning)

# Import core algorithms
from .core.SparsityIdentificationNonGaussian import SING
from .core.NodeOrdering import ReverseCholesky


class SINGEstimator:
    """
    A wrapper class for the SING algorithm ensuring consistent I/O formats
    with standard baseline estimators (e.g., GlassoEstimator, NPNEstimator).
    """

    def __init__(self, p_order=3, delta=1.0, reg_type='L2', reg_alpha=1e-1):
        """
        Args:
            p_order (int): Order of the polynomial/expansion for the Transport Map.
                           (p=1 corresponds to a well-specified Gaussian model,
                            p>1 captures non-Gaussian structures like Butterfly).
            delta (float): Thresholding parameter for the generalized precision matrix.
            reg_type (str): Regularization type for density estimation (e.g., 'L2').
            reg_alpha (float): Regularization penalty term to prevent overfitting
                               in small sample size (N < D) settings.
        """
        self.p_order = p_order
        self.delta = delta

        if reg_type is not None:
            self.REG = {'type': reg_type, 'alpha': reg_alpha}
        else:
            self.REG = None

        # Default ordering strategy based on Baptista et al.
        self.ordering = ReverseCholesky()

    def _standardize_features(self, X):
        """
        Standardizes the data to zero mean and unit variance.
        Crucial for the numerical stability of measure transport optimization.
        """
        mean_vec = np.mean(X, axis=0)
        X_centered = X - mean_vec

        var_vec = np.var(X_centered, axis=0)
        var_vec[var_vec == 0] = 1e-8  # Prevent division by zero

        inv_std = np.diag(1.0 / np.sqrt(var_vec))
        X_scaled = np.dot(X_centered, inv_std)

        return X_scaled

    def fit_predict(self, X):
        """
        Fits the SING model to the data and extracts the structural skeleton.

        Args:
            X (np.ndarray): Data matrix of shape (n_samples, n_features).

        Returns:
            est_graph (np.ndarray): Binary adjacency matrix of shape (D, D).
            rec_omega (np.ndarray): Estimated generalized precision matrix of shape (D, D).
            meta_info (dict): Additional hyperparameters used during execution.
        """
        D = X.shape[1]

        # 1. Data Preprocessing
        X_processed = self._standardize_features(X)

        # 2. Execute Core SING Pipeline
        # Note: 'results_df' contains iteration logs which can be discarded in benchmark
        rec_omega, _ = SING(
            data=X_processed,
            p_order=self.p_order,
            ordering=self.ordering,
            delta=self.delta,
            offset=0,
            REG=self.REG,
            plotting=False
        )

        # 3. Adjacency Extraction (Structural Skeleton)
        # SING inherently thresholds based on its internal variance estimates.
        # Elements exactly equal to 0 imply conditional independence.
        est_graph = np.zeros((D, D), dtype=int)
        est_graph[np.nonzero(rec_omega)] = 1
        np.fill_diagonal(est_graph, 0)

        # Symmetrize the undirected graph structure
        est_graph = np.maximum(est_graph, est_graph.T)

        meta_info = {
            "p_order": self.p_order,
            "delta": self.delta
        }

        return est_graph, rec_omega, meta_info