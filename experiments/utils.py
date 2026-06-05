"""
==============================================================================
Benchmark Utilities for Graph Structure Learning
==============================================================================
Provides common functions for data loading, ground-truth generation,
adjacency matrix extraction (thresholding), and performance evaluation.
==============================================================================
"""

import os
import json
import numpy as np


# ==============================================================================
# [1] Ground Truth Generators
# ==============================================================================
def make_chain_adj_matrix(D: int) -> np.ndarray:
    adj = np.zeros((D, D), dtype=int)
    for i in range(D - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


def make_butterfly_adj_matrix(D: int) -> np.ndarray:
    A = np.zeros((D, D), dtype=int)
    for k in range(D // 2):
        i, j = 2 * k, 2 * k + 1
        A[i, j] = 1
        A[j, i] = 1
    np.fill_diagonal(A, 0)
    return A


def get_true_adj_matrix(dataset_name: str, D: int) -> np.ndarray:
    if dataset_name.endswith("cop_gau"):
        return make_chain_adj_matrix(D)
    if dataset_name.endswith("butterfly") or dataset_name.endswith("pair_gau"):
        return make_butterfly_adj_matrix(D)
    raise ValueError(f"Unknown dataset rule: {dataset_name}")


def extract_true_skeleton(dataset_name: str, D: int) -> np.ndarray:
    """Returns the binary undirected skeleton with zero diagonals."""
    true_adj = get_true_adj_matrix(dataset_name, D)
    true_skel = ((true_adj + true_adj.T) > 0).astype(int)
    np.fill_diagonal(true_skel, 0)
    return true_skel


# ==============================================================================
# [2] Adjacency Matrix Extraction (Thresholding)
# ==============================================================================
def extract_adjacency(omega: np.ndarray, method: str, tau: float = None) -> np.ndarray:
    """
    Converts estimated precision/Hessian matrices into binary adjacency matrices.

    Args:
        omega: Estimated matrix \hat{\Omega}.
        method: The algorithm used (e.g., "lsing", "glasso", "sing", "ddpm").
        tau: Custom threshold. If None, defaults to method-specific values.

    Returns:
        Binary adjacency matrix.
    """
    omega_sym = (omega + omega.T) / 2.0

    # 1. Scaled Method (L-SING)
    if method.lower() == "lsing":
        threshold = tau if tau is not None else 0.2
        # Scale by the maximum off-diagonal absolute value
        max_val = np.nanmax(np.abs(np.triu(omega_sym, k=1)))
        if max_val != 0 and not np.isnan(max_val):
            omega_scaled = omega_sym / max_val
        else:
            omega_scaled = omega_sym.copy()

        omega_scaled = np.nan_to_num(omega_scaled, nan=0.0)
        pred_adj = (np.abs(omega_scaled) >= threshold).astype(int)

    # 2. Unscaled Methods (GLASSO, NPN, SING)
    else:
        # Apply a strict machine-zero threshold (1e-12)
        threshold = tau if tau is not None else 1e-12
        omega_sym = np.nan_to_num(omega_sym, nan=0.0)
        pred_adj = (np.abs(omega_sym) >= threshold).astype(int)

    # Ensure undirected graph constraints
    np.fill_diagonal(pred_adj, 0)
    pred_adj = np.maximum(pred_adj, pred_adj.T)
    return pred_adj


# ==============================================================================
# [3] Evaluation Metrics
# ==============================================================================
def calculate_metrics(pred_adj: np.ndarray, true_adj: np.ndarray) -> dict:
    """
    Computes structure recovery metrics focusing on the upper triangle
    to avoid double-counting undirected edges.
    """
    D = pred_adj.shape[0]
    idx_upper = np.triu_indices(D, k=1)

    pred_edges = pred_adj[idx_upper]
    true_edges = true_adj[idx_upper]

    TP = np.sum((pred_edges == 1) & (true_edges == 1))
    FP = np.sum((pred_edges == 1) & (true_edges == 0))
    FN = np.sum((pred_edges == 0) & (true_edges == 1))
    TN = np.sum((pred_edges == 0) & (true_edges == 0))

    TPR = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    FDR = float(FP / (FP + TP)) if (FP + TP) > 0 else 0.0
    FPR = float(FP / (FP + TN)) if (FP + TN) > 0 else 0.0

    Precision = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    F1 = float(2 * (Precision * TPR) / (Precision + TPR)) if (Precision + TPR) > 0 else 0.0
    Hamming = int(FP + FN)

    return {
        "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
        "TPR": TPR, "FDR": FDR, "FPR": FPR,
        "F1": F1, "Hamming": Hamming
    }


# ==============================================================================
# [4] I/O Operations
# ==============================================================================
def save_json_results(base_dir: str, method: str, dataset: str, n: int, seed: int, data_dict: dict):
    """
    Saves the experimental results into a structured directory tree:
    results/{method}/{dataset}/N_{n}/results_seed{seed}.json
    """
    target_dir = os.path.join(base_dir, method.lower(), dataset, f"N_{n}")
    os.makedirs(target_dir, exist_ok=True)

    file_path = os.path.join(target_dir, f"results_seed{seed}.json")
    with open(file_path, "w") as f:
        json.dump(data_dict, f, indent=2)
    return file_path