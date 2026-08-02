"""Diffusion-based DAG topological-order recovery via conditional Tweedie Hessians.

A single full-dimensional DDPM is trained once.  At each stage, with ``S`` the
set of remaining variables, the pipeline

  1. conditions on ``x_{t,S}`` and samples the free block ``X_{t,R} | X_{t,S}``
     with fixed-time Langevin dynamics driven by the ``R`` rows of the learned
     full score,
  2. reverse-diffuses the merged ``D``-dimensional state to ``x_0``,
  3. keeps the ``S`` coordinates to estimate ``Cov(X_{0,S} | X_{t,S})``,
  4. converts that covariance to the Hessian of ``log p_{t,S}`` via Tweedie,
  5. removes the node whose signed diagonal Hessian varies least across anchors.

The output is a **topological order**, not a sparse DAG.  See
:mod:`models.dag_diffusion.ordering` for the scientific caveat on evaluating the
criterion at a positive diffusion time.
"""
from .conditional_langevin import (
    LangevinConfig,
    LangevinDiagnostics,
    LangevinDivergenceError,
    complement_indices,
    sample_conditional_langevin,
)
from .diagnostics import evaluate_ordering, order_fnr, stagewise_leaf_validity
from .moments import BatchedWelford
from .ordering import (
    ConditionalDiffusionDAGOrderEstimator,
    DAGOrderingConfig,
    DAGOrderingResult,
    fully_connected_dag_from_order,
)
from .reverse_posterior import sample_x0_S_given_xt_S, sample_x0_given_xt
from .score_adapter import AnalyticGaussianScoreAdapter, DDPMScoreAdapter
from .tweedie_hessian import (
    estimate_conditional_covariance,
    estimate_tweedie_hessian,
    estimate_tweedie_hessian_diagonal,
    tweedie_hessian_from_covariance,
)

__all__ = [
    "AnalyticGaussianScoreAdapter",
    "BatchedWelford",
    "ConditionalDiffusionDAGOrderEstimator",
    "DAGOrderingConfig",
    "DAGOrderingResult",
    "DDPMScoreAdapter",
    "LangevinConfig",
    "LangevinDiagnostics",
    "LangevinDivergenceError",
    "complement_indices",
    "estimate_conditional_covariance",
    "estimate_tweedie_hessian",
    "estimate_tweedie_hessian_diagonal",
    "evaluate_ordering",
    "fully_connected_dag_from_order",
    "order_fnr",
    "sample_conditional_langevin",
    "sample_x0_S_given_xt_S",
    "sample_x0_given_xt",
    "stagewise_leaf_validity",
    "tweedie_hessian_from_covariance",
]
