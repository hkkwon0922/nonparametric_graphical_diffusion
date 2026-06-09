
# Nonparametric undirected graphical model selection using diffusion models
Hyeok Kyu Kwon, Myeonggu Kang, Minwoo Chae and Wanjie Wang [arXiv](https://arxiv.org/abs/2606.08468)



## Abstract

Undirected graphical models provide a fundamental framework for representing conditional independence structures among high-dimensional random variables. While undirected graphical model selection has become a central problem in high-dimensional statistics, most existing methods are restricted to parametric settings. In this paper, we develop a nonparametric approach to undirected graphical model selection based on diffusion models. Recent work has shown that diffusion models can adapt to the unknown graph structure of the underlying distribution, yet utilizing these models for explicit graph estimation remains unexplored. To bridge this gap, we introduce a novel diffusion-based method for nonparametric undirected graphical model selection. We establish the model selection consistency of the proposed method and demonstrate its empirical performance through extensive simulations and two real data analyses.

## Overview

To reproduce the numerical experiments in the main paper, readers can run the following notebooks:

```text
├── visualization/
│   ├── plot_results.ipynb          # Simulation results
│   ├── network_results.ipynb       # Network analysis
│   ├── image_results.ipynb         # Image analysis
│   ├── toy_example.ipynb           # Illustrative examples
```

## Reproducing the experiments

## 1. Environment setup

To avoid dependency conflicts between R-based statistical packages and PyTorch-based deep learning models, we use separate CPU and GPU environments.

```bash
# [1] CPU environment: GLASSO, NPN, and SING
# Includes R, TransportMaps, and PyCop
conda env create -f env_cpu.yml
conda activate env_cpu

# [2] GPU environment: L-SING and DDPM
# Includes PyTorch and CUDA
conda env create -f env_gpu.yml
conda activate env_gpu
```

## 2. Simulation pipeline

### Step 1 — Data generation

```bash
cd data
conda activate env_cpu
python generate_data.py
```

### Step 2 — Running the benchmark

```bash
cd experiments

# CPU models (require env_cpu)
python run_benchmark.py --model glasso
python run_benchmark.py --model npn
python run_benchmark.py --model sing --p_order 1
python run_benchmark.py --model sing --p_order 3

# GPU models (require env_gpu)
conda activate env_gpu
python run_benchmark.py --model lsing
python run_benchmark.py --model ddpm
```

## 3. Real data analysis

### Image analysis

```bash
cd experiments

# Train a UNet DDPM on MNIST and compute the pixel-wise Hessian
python train_ddpm_mnist.py # Trains a UNet DDPM on MNIST
python compute_hessian_mnist.py # Computes the per-timestep pixel Hessian
```

### Network analysis

```bash
cd experiments

# Train DDPM, then compute its Hessian
python train_ddpm_network.py # Reads data/network/sector_rt_csv_2019_connected/
python compute_hessian_network.py # Writes the Hessian pickle file
```




