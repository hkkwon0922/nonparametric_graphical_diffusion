"""
==============================================================================
Nonparanormal (NPN) + Refitted GLASSO Estimator
==============================================================================
This module implements a semiparametric approach for estimating high-dimensional
undirected graphs. It relaxes the Gaussian assumption by transforming the data
using the Nonparanormal (NPN) truncation method, followed by a Graphical Lasso
path exploration. Finally, it refits the precision matrix using Maximum
Likelihood Estimation (MLE) constrained by the estimated graph structure to
reduce bias.

Dependencies & Acknowledgments:
- This implementation utilizes the R 'huge' package for NPN transformation.
  (https://cran.r-project.org/web/packages/huge/index.html)
- R 'glasso' package is used for path generation and constrained refitting.
- Pipeline design and integration by Hyeok Kyu Kwon & Myeonggu Kang.

Dependencies: numpy, rpy2, R (with 'huge' and 'glasso' packages installed)
==============================================================================
"""

import sys
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

# Monkey-patching for legacy numpy compatibility with rpy2/pandas
np.int = int
np.float = float
np.bool = np.bool_

# ==============================================================================
# [0] R Environment Initialization
# ==============================================================================
r_pipeline_code = """
library(huge)
library(glasso)
library(Matrix)

# 1. NPN Transformation
get_npn_data <- function(X) {
  X_npn <- huge.npn(X, npn.func = "truncation", verbose = FALSE)
  return(X_npn)
}

# 2. Custom GLASSO Path Exploration and Refitting
get_refitted_path <- function(S, n_alphas=100, min_ratio=0.3) {
  S_off <- abs(S)
  diag(S_off) <- 0
  lam_max <- max(S_off)
  lam_min <- lam_max * min_ratio 
  alphas <- exp(seq(log(lam_max), log(lam_min), length.out = n_alphas))
  alphas <- sort(alphas, decreasing = TRUE) 

  path_res <- glassopath(S, rholist = alphas, trace = 0, penalize.diagonal = FALSE)
  refitted_precs <- list()

  for (i in 1:length(alphas)) {
    prec_lasso <- path_res$wi[,,i]
    adj_temp <- ifelse(abs(prec_lasso) > 1e-12, 1, 0)
    diag(adj_temp) <- 1
    zeroes_idx <- which(adj_temp == 0, arr.ind = TRUE)

    # Refit with MLE under zero-constraints
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
  return(list(precs = refitted_precs, alphas = alphas))
}
"""

try:
    robjects.r(r_pipeline_code)
    get_npn_data_r = robjects.globalenv['get_npn_data']
    get_refitted_path_r = robjects.globalenv['get_refitted_path']
except Exception as e:
    print(f"CRITICAL: R compilation failed for NPN pipeline. Ensure R, 'huge', and 'glasso' are installed.\nError: {e}")
    sys.exit(1)


# ==============================================================================
# [1] NPNEstimator Class
# ==============================================================================
class NPNEstimator:
    """
    Wrapper class for the NPN + Refitted GLASSO algorithm.
    Ensures identical I/O format as the standard GlassoEstimator.
    """

    def __init__(self, n_alphas=100, min_ratio=0.3, gamma=0.5, threshold=1e-12):
        """
        Args:
            n_alphas (int): Number of lambda penalties along the path.
            min_ratio (float): Ratio of lam_min to lam_max for path sequence.
            gamma (float): Tuning parameter for the EBIC penalty term.
            threshold (float): Threshold to determine exact zeros.
        """
        self.n_alphas = n_alphas
        self.min_ratio = min_ratio
        self.gamma = gamma
        self.threshold = threshold

    def _execute_npn_and_path(self, X):
        """Applies NPN truncation, calculates correlation, and generates refitted path."""
        with localconverter(robjects.default_converter + numpy2ri.converter):
            # 1. Truncation NPN Transformation
            X_npn = np.array(get_npn_data_r(X))

            # 2. Scale-invariant correlation matrix
            S = np.corrcoef(X_npn, rowvar=False)

            # 3. Path Generation & Refitting
            r_result = get_refitted_path_r(S, self.n_alphas, self.min_ratio)
            precs_list = list(r_result[0])
            alphas = np.array(r_result[1])

        return precs_list, alphas, S

    def _select_best_ebic(self, precs_list, alphas, S, n):
        """Selects the best refitted precision matrix using EBIC."""
        best_ebic = np.inf
        best_idx = 0  # Default to 0 in case of failure
        p = S.shape[0]

        for i, prec in enumerate(precs_list):
            prec_np = np.array(prec)
            sign, log_det = np.linalg.slogdet(prec_np)

            # Skip non-positive definite matrices
            if sign <= 0:
                continue

            tr_val = np.sum(S * prec_np)
            log_lik = (n / 2) * (log_det - tr_val)

            # Count upper triangular edges
            upper_edges = np.sum((np.abs(prec_np) > self.threshold) & (np.triu(np.ones((p, p)), k=1) == 1))

            bic_term = upper_edges * np.log(n)
            ebic_penalty = 4 * self.gamma * upper_edges * np.log(p)

            ebic = -2 * log_lik + bic_term + ebic_penalty

            if ebic < best_ebic:
                best_ebic = ebic
                best_idx = i

        best_alpha = alphas[best_idx]
        best_prec = np.array(precs_list[best_idx])

        return best_alpha, best_prec

    def fit_predict(self, X):
        """
        Executes the full pipeline: NPN -> Path -> Refit -> EBIC Selection.

        Args:
            X (np.ndarray): Data matrix of shape (n_samples, n_features).

        Returns:
            best_adj (np.ndarray): Binary adjacency matrix of shape (D, D).
            best_prec (np.ndarray): Refitted precision matrix of shape (D, D).
            best_alpha (float): Optimal penalty parameter chosen.
        """
        n_samples = X.shape[0]

        # 1. NPN Transform & Refitted Path
        precs_list, alphas, S = self._execute_npn_and_path(X)

        # 2. Model Selection
        best_alpha, best_prec = self._select_best_ebic(precs_list, alphas, S, n_samples)

        # 3. Adjacency Extraction & Symmetrization
        best_adj = (np.abs(best_prec) > self.threshold).astype(int)
        np.fill_diagonal(best_adj, 0)
        best_adj = np.maximum(best_adj, best_adj.T)

        return best_adj, best_prec, best_alpha