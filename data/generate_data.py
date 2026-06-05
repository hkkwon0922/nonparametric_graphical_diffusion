"""
==============================================================================
Batch Data Generator for Graph Structure Learning
==============================================================================
Generates all datasets required for the benchmark pipeline.
Parses complex dataset names (e.g., '0.1_dim20_pair_gau') to automatically
extract signal size (correlation) and dimensionality.

Usage:
    $ cd data
    $ python generate_data.py
==============================================================================
"""

import os
import re
import numpy as np
from tqdm import tqdm

# Import generator modules
from cop_gau import generate_cop_gau
from butterfly import generate_butterfly
from pair_gau import generate_pair_gau

def main():
    # ==========================================================================
    # [1] Data Generation Settings
    # ==========================================================================
    # [Default] Quick Test Setting for Sanity Check
    # datasets = [
    #     "dim5_cop_gau", "dim6_butterfly", "dim6_pair_gau"
    # ]
    # data_sizes = [100, 200, 300, 400, 500, 1000]
    # seeds = [120, 1230, 12340]

    # --------------------------------------------------------------------------
    # [Paper Reproduction] Full settings used in the benchmark
    # Uncomment the following lines to generate the complete dataset.
    # --------------------------------------------------------------------------
    datasets = [
        "dim5_cop_gau", "dim6_butterfly",
        "dim20_butterfly", "dim20_cop_gau", "0.3_dim20_pair_gau", "0.7_dim20_pair_gau",
    ]
    data_sizes = [100, 200, 300, 400, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    seeds = [120, 1230, 12340, 123450, 1234560]
    # ==========================================================================

    output_base_dir = "../data/raw"
    os.makedirs(output_base_dir, exist_ok=True)

    total_iters = len(datasets) * len(data_sizes) * len(seeds)
    print(f"Starting batch data generation. Total files: {total_iters}")

    with tqdm(total=total_iters, desc="Generating Data") as pbar:
        for ds_name in datasets:

            # Parse dataset name using Regex
            # Pattern matching: (rho_)?dim(D)_(type)
            # Example 1: "0.1_dim20_pair_gau" -> rho=0.1, D=20, type="pair_gau"
            # Example 2: "dim6_butterfly"     -> rho=None, D=6, type="butterfly"
            match = re.match(r"(?:([0-9\.]+)_)?dim(\d+)_([a-z_]+)", ds_name)

            if not match:
                raise ValueError(f"Invalid dataset naming convention: {ds_name}")

            rho_str = match.group(1)
            D = int(match.group(2))
            dist_type = match.group(3)

            # Set default correlation for pair_gau if not explicitly provided
            rho = float(rho_str) if rho_str else 0.8

            target_dir = os.path.join(output_base_dir, ds_name, "train")
            os.makedirs(target_dir, exist_ok=True)

            for n in data_sizes:
                for seed in seeds:
                    file_name = f"{dist_type}_n{n}_seed{seed}.npy"
                    save_path = os.path.join(target_dir, file_name)

                    # Skip generation if file already exists
                    if os.path.exists(save_path):
                        pbar.update(1)
                        continue

                    # Call appropriate generator based on parsed distribution type
                    if dist_type == "cop_gau":
                        X = generate_cop_gau(dim=D, n=n, seed=seed, rho=rho)
                    elif dist_type == "butterfly":
                        X = generate_butterfly(dim=D, n=n, seed=seed)
                    elif dist_type == "pair_gau":
                        X = generate_pair_gau(dim=D, n=n, seed=seed, correlation=rho)
                    else:
                        raise ValueError(f"Unknown distribution type: {dist_type}")

                    # Save array to disk
                    np.save(save_path, X)
                    pbar.update(1)

    print("All datasets generated successfully.")

if __name__ == "__main__":
    main()