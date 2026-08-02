"""Shared, public DDPM training / checkpoint utilities.

``DDPMEstimator._train_ddpm`` is private and tied to the undirected-graph
benchmark, so the DAG-ordering runner does not call it.  This module provides an
equivalent public path that reuses exactly the same building blocks
(``get_beta_schedule``, ``GaussianDiffusion``, ``Decoder5D_0204``,
``diffusion.train_losses``) and writes a self-describing checkpoint.

Standardisation is **opt-in**.  When enabled, the training mean/scale are stored
in the checkpoint and applied identically at inference; the checkpoint records
whether downstream analysis runs on raw or standardised coordinates.  No claim
is made that the ordering criterion is scale-invariant.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ..ddpm.core.ddpm_torch.toy import GaussianDiffusion, get_beta_schedule
from ..ddpm.core.ddpm_torch.toy.toy_model import Decoder5D_0204
from ..ddpm.core.ddpm_torch.utils import seed_all

__all__ = ["DDPMTrainConfig", "Preprocessor", "build_model_and_diffusion",
           "train_ddpm", "save_ddpm_checkpoint", "load_ddpm_checkpoint"]


@dataclass
class DDPMTrainConfig:
    """Architecture, diffusion schedule and optimisation settings."""

    input_dimension: int = 0
    mid_features: int = 160
    num_temporal_layers: int = 3
    timesteps: int = 500
    beta_schedule: str = "linear"
    beta_start: float = 0.001
    beta_end: float = 0.2
    model_mean_type: str = "eps"
    model_var_type: str = "fixed-large"
    loss_type: str = "mse"
    epochs: int = 1000
    batch_size: int = 100
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    standardize: bool = False
    seed: int = 120

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Preprocessor:
    """Opt-in standardisation with explicitly stored mean/scale.

    ``enabled=False`` (default) is the identity map; nothing is applied
    silently.
    """

    def __init__(self, mean: Optional[np.ndarray] = None,
                 scale: Optional[np.ndarray] = None, enabled: bool = False):
        self.enabled = bool(enabled)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float64)

    @classmethod
    def fit(cls, X: np.ndarray, enabled: bool = False) -> "Preprocessor":
        """Fit on ``(N, D)`` data; a zero/degenerate scale is floored at 1.0."""
        X = np.asarray(X, dtype=np.float64)
        if not enabled:
            return cls(enabled=False)
        mean = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(mean=mean, scale=scale, enabled=True)

    def transform(self, X):
        """Apply ``(x - mean) / scale`` when enabled, else return ``X`` unchanged."""
        if not self.enabled:
            return X
        if isinstance(X, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=X.dtype, device=X.device)
            scale = torch.as_tensor(self.scale, dtype=X.dtype, device=X.device)
            return (X - mean) / scale
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.scale

    def inverse_transform(self, X):
        """Undo :meth:`transform`."""
        if not self.enabled:
            return X
        if isinstance(X, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=X.dtype, device=X.device)
            scale = torch.as_tensor(self.scale, dtype=X.dtype, device=X.device)
            return X * scale + mean
        return np.asarray(X, dtype=np.float64) * self.scale + self.mean

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mean": None if self.mean is None else self.mean.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
            "kind": "standardize" if self.enabled else "identity",
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "Preprocessor":
        if not payload:
            return cls(enabled=False)
        return cls(
            mean=payload.get("mean"), scale=payload.get("scale"),
            enabled=bool(payload.get("enabled", False)),
        )


def build_model_and_diffusion(config: DDPMTrainConfig, device) -> Tuple[Any, Any]:
    """Instantiate ``(Decoder5D_0204, GaussianDiffusion)`` from ``config``."""
    if int(config.input_dimension) <= 0:
        raise ValueError("config.input_dimension must be set to the data dimension D")
    betas = get_beta_schedule(
        config.beta_schedule, beta_start=config.beta_start,
        beta_end=config.beta_end, timesteps=config.timesteps,
    )
    diffusion = GaussianDiffusion(
        betas=betas, model_mean_type=config.model_mean_type,
        model_var_type=config.model_var_type, loss_type=config.loss_type,
    )
    model = Decoder5D_0204(
        int(config.input_dimension), int(config.mid_features), int(config.num_temporal_layers)
    ).to(torch.device(device))
    return model, diffusion


def train_ddpm(X: np.ndarray, config: DDPMTrainConfig, device, verbose: bool = True):
    """Train a DDPM on ``(N, D)`` data.

    Returns
    -------
    (model, diffusion, preprocessor, history) where ``history`` is a list of
    ``{"epoch", "avg_loss"}`` records.
    """
    device = torch.device(device)
    seed_all(int(config.seed))

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must have shape (N, D); got {X.shape}")
    config.input_dimension = int(X.shape[1])

    pre = Preprocessor.fit(X, enabled=bool(config.standardize))
    X_use = np.asarray(pre.transform(X), dtype=np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_use)),
        batch_size=int(config.batch_size), shuffle=True,
    )
    model, diffusion = build_model_and_diffusion(config, device)
    optimizer = Adam(model.parameters(), lr=float(config.lr),
                     betas=(float(config.beta1), float(config.beta2)))

    history = []
    start = time.time()
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        total, nb = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            t = torch.randint(0, diffusion.timesteps, (batch.shape[0],), device=device)
            loss = diffusion.train_losses(model, x_0=batch, t=t).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            nb += 1
        avg = total / max(1, nb)
        history.append({"epoch": epoch, "avg_loss": avg})
        if verbose and (epoch % max(1, int(config.epochs) // 10) == 0 or epoch == 1):
            print(f"  epoch {epoch:5d}/{config.epochs}  loss={avg:.6f}", flush=True)

    model.eval()
    if verbose:
        print(f"  training finished in {time.time() - start:.1f}s", flush=True)
    return model, diffusion, pre, history


def save_ddpm_checkpoint(path: str, model, config: DDPMTrainConfig,
                         preprocessor: Preprocessor, epoch: int,
                         extra: Optional[Dict[str, Any]] = None) -> str:
    """Write a self-describing checkpoint and return its path.

    The payload records the architecture, the full diffusion schedule, the
    preprocessing metadata and the training seed, so that inference can rebuild
    an exactly matching model/diffusion pair.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "input_dimension": int(config.input_dimension),
        "mid_features": int(config.mid_features),
        "num_temporal_layers": int(config.num_temporal_layers),
        "timesteps": int(config.timesteps),
        "beta_schedule": config.beta_schedule,
        "beta_start": float(config.beta_start),
        "beta_end": float(config.beta_end),
        "model_mean_type": config.model_mean_type,
        "model_var_type": config.model_var_type,
        "loss_type": config.loss_type,
        "preprocessing": preprocessor.to_dict(),
        "training_seed": int(config.seed),
        "train_config": config.to_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_ddpm_checkpoint(path: str, device, expected_dim: Optional[int] = None):
    """Rebuild ``(model, diffusion, preprocessor, payload)`` from a checkpoint.

    Raises ``ValueError`` when the stored ``input_dimension`` disagrees with
    ``expected_dim``, or when the payload predates this format and lacks the
    fields needed to rebuild the diffusion schedule.
    """
    device = torch.device(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            f"{path} is not a recognised DDPM checkpoint (missing 'model_state_dict')")

    required = ("input_dimension", "mid_features", "num_temporal_layers", "timesteps",
                "beta_schedule", "beta_start", "beta_end", "model_mean_type",
                "model_var_type", "loss_type")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(
            f"checkpoint {path} is missing required fields {missing}; it was not written by "
            "models.dag_diffusion.training.save_ddpm_checkpoint"
        )

    config = DDPMTrainConfig(
        input_dimension=int(payload["input_dimension"]),
        mid_features=int(payload["mid_features"]),
        num_temporal_layers=int(payload["num_temporal_layers"]),
        timesteps=int(payload["timesteps"]),
        beta_schedule=payload["beta_schedule"],
        beta_start=float(payload["beta_start"]),
        beta_end=float(payload["beta_end"]),
        model_mean_type=payload["model_mean_type"],
        model_var_type=payload["model_var_type"],
        loss_type=payload["loss_type"],
        seed=int(payload.get("training_seed", 0)),
        standardize=bool(payload.get("preprocessing", {}).get("enabled", False)),
    )
    if expected_dim is not None and config.input_dimension != int(expected_dim):
        raise ValueError(
            f"checkpoint input_dimension={config.input_dimension} does not match data "
            f"dimension {int(expected_dim)}"
        )

    model, diffusion = build_model_and_diffusion(config, device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    pre = Preprocessor.from_dict(payload.get("preprocessing"))
    return model, diffusion, pre, payload
