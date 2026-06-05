"""
==============================================================================
Denoising Diffusion Probabilistic Model (DDPM) Estimator
==============================================================================
This module implements a graph structure estimation pipeline using DDPMs.
It learns the data distribution via a diffusion process and extracts the
conditional independence structure by analyzing the second-order derivatives
(Hessian) of the estimated score function across multiple noise scales.
Finally, K-Means clustering is applied to the Hessian series to recover the
binary adjacency matrix.

Original Code Base:
- Author: Hyeok Kyu Kwon (Senior Researcher)
- Integrated by: Myeonggu Kang (2026)

Dependencies: torch, numpy, sklearn, core DDPM modules
==============================================================================
"""

import time
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Import isolated core logic (formerly ddpm_github)
try:
    from .core.ddpm_torch.toy import get_beta_schedule, GaussianDiffusion
    from .core.ddpm_torch.toy.toy_model import Decoder5D_0204
    from .core.ddpm_torch.utils import seed_all
except ImportError as e:
    raise ImportError(
        f"CRITICAL: Failed to load DDPM core modules. Ensure 'core' directory is properly set up. Details: {e}")


class DDPMEstimator:
    """
    Standardized wrapper for the DDPM-based graph estimation algorithm.
    Encapsulates training, Hessian-based inference, and clustering.
    """

    def __init__(self, mid_features=160, num_temporal_layers=3,
                 timesteps=500, beta_start=0.001, beta_end=0.2,
                 beta_schedule="linear", batch_size=100, lr=0.001,
                 epochs=1000, t_min=1, t_max=31, num_samples_per_t=5000,
                 seed=120, device=None):

        self.mid_features = mid_features
        self.num_temporal_layers = num_temporal_layers
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        # Inference (Hessian) Params
        self.t_list = list(range(t_min, t_max))
        self.num_samples_per_t = num_samples_per_t
        self.seed = seed

        # Fixed hyperparameters from original implementation
        self.beta1, self.beta2 = 0.9, 0.999
        self.model_mean_type = "eps"
        self.model_var_type = "fixed-large"
        self.loss_type = "mse"
        self.apply_clamping = False

        if device is None:
            self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def _train_ddpm(self, X_train):
        """Trains the DDPM model on the provided data matrix."""
        seed_all(self.seed)
        train_tensor = torch.FloatTensor(X_train)
        trainloader = DataLoader(TensorDataset(train_tensor), batch_size=self.batch_size, shuffle=True)

        betas = get_beta_schedule(self.beta_schedule, beta_start=self.beta_start,
                                  beta_end=self.beta_end, timesteps=self.timesteps)
        diffusion = GaussianDiffusion(
            betas=betas, model_mean_type=self.model_mean_type,
            model_var_type=self.model_var_type, loss_type=self.loss_type
        )

        model = Decoder5D_0204(self.D, self.mid_features, self.num_temporal_layers).to(self.device)
        optimizer = Adam(model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        start_time = time.time()
        for epoch in range(self.epochs):
            model.train()
            for (batch,) in trainloader:
                batch = batch.to(self.device)
                t = torch.randint(0, diffusion.timesteps, (batch.shape[0],), device=self.device)
                loss = diffusion.train_losses(model, x_0=batch, t=t).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        train_time = time.time() - start_time
        model.eval()
        return model, diffusion, train_time

    @torch.inference_mode()
    def _estimate_hessian_full_batch(self, model, diffusion, x_t, t, num_samples, seed_val):
        """Generates samples and estimates the Hessian for a given noise scale t."""
        x_t = torch.as_tensor(x_t, device=self.device, dtype=torch.float32)
        if x_t.ndim == 1: x_t = x_t.unsqueeze(0)
        B, D = x_t.shape
        S = int(num_samples)

        x_rep = x_t.repeat_interleave(S, dim=0).contiguous()
        t_start = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        t_tensor = torch.full((B * S,), t_start, dtype=torch.int64, device=self.device)
        rng = torch.Generator(device=self.device).manual_seed(int(seed_val)) if seed_val is not None else None

        for ti in range(t_start, -1, -1):
            t_tensor.fill_(ti)
            x_rep = diffusion.p_sample_step(
                denoise_fn=model, x_t=x_rep, t=t_tensor,
                clip_denoised=False, return_pred=False, generator=rng
            )

        x0_samples = x_rep.view(B, S, D)
        if self.apply_clamping:
            x0_samples = x0_samples.clamp(-1.0, 1.0)

        mu = x0_samples.mean(dim=1)
        M2 = torch.einsum('bsd,bse->bde', x0_samples, x0_samples) / float(S)
        Cov = M2 - mu.unsqueeze(2) * mu.unsqueeze(1)

        mu2_t = diffusion.alphas_bar[t].to(self.device).float()
        sigma2_t = (1.0 - mu2_t).clamp_min(1e-12)
        scale = mu2_t / (sigma2_t ** 2)

        H = scale * Cov
        diag_add = (-1.0 / sigma2_t).expand(D)
        H = H + torch.diag(diag_add).unsqueeze(0)
        return H

    @torch.inference_mode()
    def _compute_hessians_avg(self, model, diffusion, X_train, num_x0=128, batch_x0=128):
        """Computes the average Hessian across multiple time steps."""
        start_time = time.time()
        test_t = torch.as_tensor(X_train, device=self.device, dtype=torch.float32)
        N, D = test_t.shape
        num_x0 = min(int(num_x0), N)
        sum_abs, count = {}, {}

        for start in range(0, num_x0, batch_x0):
            end = min(start + batch_x0, num_x0)
            x0_batch = test_t[start:end]
            B = x0_batch.shape[0]

            for t in self.t_list:
                t_tensor = torch.full((B,), int(t), dtype=torch.int64, device=self.device)
                noise = torch.randn(B, D, device=self.device)
                x_t_batch = diffusion.q_sample(x_0=x0_batch, t=t_tensor, noise=noise)

                H_batch = self._estimate_hessian_full_batch(
                    model, diffusion, x_t_batch, int(t),
                    self.num_samples_per_t, int(self.seed + 10 * start + t)
                )

                abs_H = H_batch.abs()
                if t not in sum_abs:
                    sum_abs[t] = abs_H.sum(dim=0)
                    count[t] = B
                else:
                    sum_abs[t] += abs_H.sum(dim=0)
                    count[t] += B

        H_dict_avg = {}
        for t in self.t_list:
            H_dict_avg[t] = (sum_abs[t] / float(count[t])).detach().cpu().numpy()

        inference_time = time.time() - start_time
        return H_dict_avg, inference_time

    def _extract_adj_via_kmeans(self, H_dict_avg, fix_k=2, n_init=10):
        """Applies K-Means clustering to the Hessian series to extract the adjacency matrix."""
        series_list, pair_list = [], []

        for i in range(self.D):
            for j in range(i + 1, self.D):
                H_vals = [H_dict_avg[t][i, j] for t in self.t_list]
                series_list.append(np.array(H_vals))
                pair_list.append((i, j))

        X = np.vstack(series_list)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        best_score = -1.0
        best_labels = None

        for r in range(n_init):
            kmeans = KMeans(n_clusters=fix_k, init='k-means++', n_init=1, random_state=self.seed + r)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_labels = labels

        cluster_means = [X[(best_labels == k)].mean() for k in range(fix_k)]
        best_cluster = np.argmax(cluster_means)

        adj = np.zeros((self.D, self.D), dtype=int)
        for row_idx, (i, j) in enumerate(pair_list):
            if best_labels[row_idx] == best_cluster:
                adj[i, j] = 1
                adj[j, i] = 1

        return adj

    def fit_predict(self, X):
        """
        Executes the full DDPM pipeline: Train -> Hessian Inference -> Clustering.

        Args:
            X (np.ndarray): Data matrix of shape (n_samples, n_features).

        Returns:
            est_graph (np.ndarray): Binary adjacency matrix of shape (D, D).
            omega (None): DDPM does not output a continuous precision matrix natively.
            meta_info (dict): Time profiling and execution metadata.
        """
        self.D = X.shape[1]
        # 1. Train
        model, diffusion, train_time = self._train_ddpm(X)

        # 2. Inference (Hessian Computation)
        # Assuming inference uses a subset (num_x0) of the training data as per original script
        H_dict_avg, inf_time = self._compute_hessians_avg(model, diffusion, X)

        # 3. Graph Extraction (K-Means Clustering)
        est_graph = self._extract_adj_via_kmeans(H_dict_avg)

        meta_info = {
            "time_breakdown": {
                "training_seconds": round(train_time, 2),
                "inference_seconds": round(inf_time, 2),
                "total_seconds": round(train_time + inf_time, 2)
            }
        }

        # Return est_graph as omega as well to maintain output signature compatibility
        return est_graph, est_graph.astype(float), meta_info