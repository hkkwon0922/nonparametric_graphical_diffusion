"""
==============================================================================
L-SING (Scalable Estimation of Nonparametric Markov Networks)
==============================================================================
This module implements L-SING, which scales the generalized precision matrix
estimation by learning node-wise conditional distributions using Unconstrained
Monotonic Neural Networks (UMNNs).

Original Code Base & Inspiration:
- Author: Sarah Liaw (California Institute of Technology)
- Repository: https://github.com/SarahLiaw/L-SING
- Adapted for benchmark integration by Hyeok Kyu Kwon & Myeonggu Kang (2026)

Dependencies: torch, numpy, core L-SING modules
==============================================================================
"""

import os
import copy
import time
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Import isolated core logic
from .core.UMNN import MonotonicNN
from .core.computesk import test_map, test_losses


class LSINGEstimator:
    """
    Standardized wrapper for the L-SING algorithm.
    Includes built-in time profiling to separate training and inference costs.
    """

    def __init__(self, hidden_layers=[64, 64, 64], nb_steps=50, lr=0.01,
                 num_epochs=100, lambdas=[1, 0.1, 0.01, 0.001, 0],
                 patience=10, tau=0.2, device=None):
        self.hidden_layers = hidden_layers
        self.nb_steps = nb_steps
        self.lr = lr
        self.num_epochs = num_epochs
        self.lambdas = lambdas
        self.patience = patience
        self.tau = tau  # Edge-selection threshold (e.g., 0.2 from validation)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def _to_tensor(self, X):
        return torch.tensor(X, dtype=torch.float32, device=self.device)

    def _train_node(self, kth, X_train, X_val, num_features):
        """Optimizes UMNN for a single node conditional distribution."""
        non_kth = [idx for idx in range(num_features) if idx != kth]
        n_samples = X_train.shape[0]

        best_val_overall = float("inf")
        best_model = None

        for reg_lambda in self.lambdas:
            Sk = MonotonicNN(num_features, self.hidden_layers, self.nb_steps, self.device).to(self.device)
            optimizer = optim.Adam(Sk.parameters(), lr=self.lr)

            best_val_this = float("inf")
            early_stop_counter = 0
            best_model_this_lambda = None

            for epoch in range(self.num_epochs):
                zk = X_train.detach().requires_grad_(True)
                h = zk[:, non_kth]
                x = zk[:, [kth]]

                sk_zi = Sk(x, h)
                jacobian = torch.autograd.grad(sk_zi, x, torch.ones_like(sk_zi), create_graph=True)[0]

                nll_loss = (0.5 * sk_zi ** 2 - torch.log(jacobian)).sum(axis=0) / n_samples
                regulariser = torch.sqrt((jacobian ** 2).sum(axis=0) / n_samples)
                total_loss = nll_loss + reg_lambda * regulariser

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                Sk_zi_val, jacobian_val = test_map(X_val, non_kth, kth, Sk)
                val_nll = float(test_losses(Sk_zi_val, jacobian_val)[1])

                if val_nll < best_val_this:
                    best_val_this = val_nll
                    early_stop_counter = 0
                    best_model_this_lambda = copy.deepcopy(Sk)
                else:
                    early_stop_counter += 1

                if early_stop_counter >= self.patience:
                    break

            if best_val_this < best_val_overall:
                best_val_overall = best_val_this
                best_model = best_model_this_lambda

        return best_model

    def _compute_omega(self, models_list, data_tensor, num_features):
        """Computes the generalized precision matrix via cross-derivatives."""
        precision_matrix = np.zeros((num_features, num_features))

        for j in range(num_features):
            Sj = models_list[j]
            Sj.eval()
            kth = j
            non_kth = [idx for idx in range(num_features) if idx != kth]

            zk = data_tensor.detach().requires_grad_(True)
            h = zk[:, non_kth]
            x = zk[:, [kth]]
            sk_zi = Sj(x, h)

            for i in range(num_features):
                if i == j:
                    precision_matrix[j, i] = 1.0
                    continue

                # 2nd-order cross derivative computation
                first_derivative = torch.autograd.grad(sk_zi, zk, torch.ones_like(sk_zi), create_graph=True)[0]
                first_derivative_log = torch.log(torch.abs(first_derivative))
                second_derivative = \
                torch.autograd.grad(first_derivative_log[:, [kth]], zk, torch.ones_like(first_derivative_log[:, [kth]]),
                                    create_graph=True)[0]
                third_derivative = \
                torch.autograd.grad(second_derivative[:, [kth]], zk, torch.ones_like(second_derivative[:, [kth]]),
                                    create_graph=True)[0]
                second = torch.abs(third_derivative[:, [i]]).mean().item()

                first_half = -0.5 * (sk_zi ** 2)
                first_half_derivative = \
                torch.autograd.grad(first_half, zk, torch.ones_like(first_half), create_graph=True)[0]
                second_half_derivative = torch.autograd.grad(first_half_derivative[:, [kth]], zk,
                                                             torch.ones_like(first_half_derivative[:, [kth]]),
                                                             create_graph=True)[0]
                first = torch.abs(second_half_derivative[:, [i]]).mean().item()

                precision_matrix[j, i] = first + second

        # Symmetrize and Normalize
        matrix = np.array(precision_matrix)
        symmetric_matrix = (matrix.T + matrix) / 2.0
        np.fill_diagonal(symmetric_matrix, 0)

        max_value = np.max(symmetric_matrix) if np.max(symmetric_matrix) > 0 else 1.0
        normalized_matrix = symmetric_matrix / max_value
        np.fill_diagonal(normalized_matrix, 1)

        return normalized_matrix

    def fit_predict(self, X):
        """
        Executes L-SING with temporal profiling.
        Splits data into Train (60%), Val (20%), Test (20%) internally.
        """
        N, D = X.shape
        n_train = int(0.6 * N)
        n_val = int(0.2 * N)

        X_train = self._to_tensor(X[:n_train, :])
        X_val = self._to_tensor(X[n_train:n_train + n_val, :])
        X_test = self._to_tensor(X[n_train + n_val:, :])

        best_models = []

        # ==========================================
        # [1] Phase 1: Training UMNNs
        # ==========================================
        train_start = time.time()
        for kth in range(D):
            Sk_best = self._train_node(kth, X_train, X_val, D)
            best_models.append(Sk_best)
        train_duration = time.time() - train_start

        # ==========================================
        # [2] Phase 2: Inference (Omega Computation)
        # ==========================================
        inf_start = time.time()
        normalized_omega = self._compute_omega(best_models, X_test, D)
        inf_duration = time.time() - inf_start

        # ==========================================
        # [3] Graph Thresholding
        # ==========================================
        est_graph = (np.abs(normalized_omega) > self.tau).astype(int)
        np.fill_diagonal(est_graph, 0)

        meta_info = {
            "tau_threshold": self.tau,
            "time_breakdown": {
                "training_seconds": round(train_duration, 2),
                "inference_seconds": round(inf_duration, 2),
                "total_seconds": round(train_duration + inf_duration, 2)
            }
        }

        return est_graph, normalized_omega, meta_info