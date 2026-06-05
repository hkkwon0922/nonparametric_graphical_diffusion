"""
==============================================================================
GLASSO (Graphical Lasso) Estimator
==============================================================================
This module implements the Graphical Lasso algorithm for estimating sparse
undirected graphs (Gaussian Graphical Models). It computes a regularization
path and selects the optimal precision matrix using the Extended Bayesian
Information Criterion (EBIC).

Original Code Base & Inspiration:
- Author: Zhao Lyu (PhD student, University of Chicago)
- Repository: https://github.com/zhao-lyu/GGM
- Modified and Integrated by: Hyeok Kyu Kwon & Myeonggu Kang (2026)

Dependencies: numpy, rpy2, R (with 'glasso' package installed)
==============================================================================
"""

import sys
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

# ==============================================================================
# [0] R Environment Initialization
# ==============================================================================
glasso_r_code = """
library(glasso)
get_refitted_path <- function(S, n_alphas=100) {
  S_off <- abs(S)
  diag(S_off) <- 0
  lam_max <- max(S_off)
  lam_min <- lam_max * 0.01 
  alphas <- exp(seq(log(lam_max), log(lam_min), length.out = n_alphas))
  alphas <- sort(alphas, decreasing = TRUE) 

  path_res <- glassopath(S, rholist = alphas, trace = 0, penalize.diagonal = FALSE)
  refitted_precs <- list()

  for (i in 1:length(alphas)) {
    prec_lasso <- path_res$wi[,,i]
    adj_temp <- ifelse(abs(prec_lasso) > 1e-12, 1, 0)
    diag(adj_temp) <- 1
    zeroes_idx <- which(adj_temp == 0, arr.ind = TRUE)

    if (nrow(zeroes_idx) > 0) {
      tryCatch({
        refit_obj <- glasso(S, rho = 0, zero = zeroes_idx, trace = 0, penalize.diagonal = FALSE)
        prec_refit <- refit_obj$wi
      }, error = function(e) { prec_refit <- prec_lasso }) 
    } else {
      prec_refit <- prec_lasso
    }
    refitted_precs[[i]] <- prec_refit
  }
  return(list(precs = refitted_precs, alphas = path_res$rholist))
}
"""

try:
    robjects.r(glasso_r_code)
    get_refitted_path_r = robjects.globalenv['get_refitted_path']
except Exception as e:
    print(f"CRITICAL: R compilation failed for GLASSO. Ensure R and the 'glasso' package are installed.\nError: {e}")
    sys.exit(1)


# ==============================================================================
# [1] GLASSO Estimator Class
# ==============================================================================
class GlassoEstimator:
    """
    A wrapper class for the GLASSO algorithm ensuring consistent I/O formats
    with other baseline models in the repository.
    """

    def __init__(self, n_alphas=100, gamma=0.5, threshold=1e-12):
        """
        Args:
            n_alphas (int): Number of lambda penalties to evaluate along the path.
            gamma (float): Tuning parameter for the EBIC penalty term.
            threshold (float): Threshold to determine exact zeros in the precision matrix.
        """
        self.n_alphas = n_alphas
        self.gamma = gamma
        self.threshold = threshold

    def _generate_path(self, X):
        """Generates the regularization path using R's glasso via rpy2."""
        S = np.corrcoef(X, rowvar=False)
        with localconverter(robjects.default_converter + numpy2ri.converter):
            r_result = get_refitted_path_r(S, self.n_alphas)
            precs_list = list(r_result[0])
            alphas = np.array(r_result[1])
        return precs_list, alphas, S

    def _select_best_ebic(self, precs_list, alphas, S, n):
        """Selects the best precision matrix from the path using EBIC."""
        best_ebic = np.inf
        best_idx = -1
        p = S.shape[0]

        for i, prec in enumerate(precs_list):
            prec_np = np.array(prec)
            sign, log_det = np.linalg.slogdet(prec_np)

            if sign <= 0:
                continue

            tr_val = np.sum(S * prec_np)
            log_lik = (n / 2) * (log_det - tr_val)

            # Count non-zero edges in the upper triangle
            upper_edges = np.sum((np.abs(prec_np) > self.threshold) & (np.triu(np.ones((p, p)), k=1) == 1))

            bic_term = upper_edges * np.log(n)
            ebic_penalty = 4 * self.gamma * upper_edges * np.log(p)

            ebic = -2 * log_lik + bic_term + ebic_penalty

            if ebic < best_ebic:
                best_ebic = ebic
                best_idx = i

        if best_idx == -1:
            raise ValueError(
                "EBIC selection failed. All precision matrices were invalid (e.g., non-positive definite).")

        best_alpha = alphas[best_idx]
        best_prec = np.array(precs_list[best_idx])

        return best_alpha, best_prec

    def fit_predict(self, X):
        """
        Fits the GLASSO model to the data and returns the estimated graph.

        Args:
            X (np.ndarray): Data matrix of shape (n_samples, n_features).

        Returns:
            best_adj (np.ndarray): Binary adjacency matrix of shape (D, D).
            best_prec (np.ndarray): The chosen precision matrix of shape (D, D).
            best_alpha (float): The optimal penalty parameter chosen by EBIC.
        """
        n_samples = X.shape[0]

        # 1. Compute Path
        precs_list, alphas, S = self._generate_path(X)

        # 2. Model Selection via EBIC
        best_alpha, best_prec = self._select_best_ebic(precs_list, alphas, S, n_samples)

        # 3. Adjacency Matrix Thresholding & Symmetrization
        best_adj = (np.abs(best_prec) > self.threshold).astype(int)
        np.fill_diagonal(best_adj, 0)
        best_adj = np.maximum(best_adj, best_adj.T)

        return best_adj, best_prec, best_alpha