"""
==============================================================================
Unified Benchmark Orchestrator for Graph Structure Learning
==============================================================================
Executes specified graphical model estimation algorithms across multiple
datasets, sample sizes (N), and random seeds.

Usage Example (Run from the 'experiments' directory):
    $ conda activate env_cpu
    $ python run_benchmark.py --model glasso
==============================================================================
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import argparse
import traceback
import re
import glob
import numpy as np
from tqdm import tqdm

# Import common utility module
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Graph Structure Learning Benchmark")
    parser.add_argument("--model", type=str, required=True,
                        choices=["glasso", "npn", "sing", "lsing", "ddpm"],
                        help="Target algorithm to evaluate.")

    parser.add_argument("--p_order", type=int, default=3,
                        help="Polynomial order for Transport Map (Only used when --model sing).")

    # Adjusted paths assuming execution from the 'experiments' directory
    parser.add_argument("--data_dir", type=str, default="../data/raw",
                        help="Base directory where generated .npy data files are stored.")
    parser.add_argument("--out_dir", type=str, default="../results",
                        help="Base directory to save evaluation results in JSON format.")
    return parser.parse_args()


def get_estimator(args):
    """
    Dynamically imports and initializes the specified model's wrapper class.
    """
    if args.model == "glasso":
        from models.glasso.glasso import GlassoEstimator
        return GlassoEstimator()
    elif args.model == "npn":
        from models.npn.npn import NPNEstimator
        return NPNEstimator()
    elif args.model == "sing":
        from models.sing.sing import SINGEstimator
        # Pass the command-line p_order value directly to the model
        return SINGEstimator(p_order=args.p_order)
    elif args.model == "lsing":
        from models.lsing.lsing import LSINGEstimator
        return LSINGEstimator()
    elif args.model == "ddpm":
        from models.ddpm.ddpm import DDPMEstimator
        return DDPMEstimator()
    else:
        raise ValueError(f"Unknown model: {args.model}")


def main():
    args = parse_args()

    # ==========================================================================
    # [1] Benchmark Settings
    # ==========================================================================
    # [Default] Quick Sanity Check
    datasets = [
        "dim5_cop_gau", "dim6_butterfly"
    ]
    data_sizes = [100, 200, 300, 400, 500, 1000]
    seeds = [120, 1230, 12340]

    # # --------------------------------------------------------------------------
    # # [Paper Reproduction] Full settings used in the benchmark
    # # Uncomment the following lines to reproduce the exact results from the paper.
    # # --------------------------------------------------------------------------
    # datasets = [
    #     "dim5_cop_gau", "dim6_butterfly",
    #     "dim20_butterfly", "dim20_cop_gau", "0.3_dim20_pair_gau", "0.7_dim20_pair_gau",
    # ]
    # data_sizes = [100, 200, 300, 400, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    # seeds = [120, 1230, 12340, 123450, 1234560]
    # # ==========================================================================

    total_iters = len(datasets) * len(data_sizes) * len(seeds)
    print(f"Starting Benchmark for Model: [{args.model.upper()}]")
    if args.model == "sing":
        print(f"SING specific configuration: p_order = {args.p_order}")
    print(f"Total target iterations: {total_iters}")

    # Directory mapping logic for SING (e.g., 'sing/p_1')
    save_method_name = f"sing/p_{args.p_order}" if args.model == "sing" else args.model

    # 2. Initialize Estimator
    estimator = get_estimator(args)

    with tqdm(total=total_iters, desc="Benchmark Progress") as pbar:
        for ds in datasets:
            match = re.search(r"dim(\d+)", ds)
            D = int(match.group(1)) if match else 20

            for n in data_sizes:
                for seed in seeds:
                    try:
                        # Check path adjusted to use save_method_name
                        check_path = os.path.join(args.out_dir, save_method_name, ds, f"N_{n}",
                                                  f"results_seed{seed}.json")
                        if os.path.exists(check_path):
                            pbar.update(1)
                            continue

                        search_pattern = os.path.join(args.data_dir, ds, "train", f"*_n{n}_seed{seed}.npy")
                        file_list = glob.glob(search_pattern)
                        if not file_list:
                            raise FileNotFoundError(f"Missing data pattern: {search_pattern}")

                        X_full = np.load(file_list[0])
                        X_batch = X_full[:n, :]
                        true_skel = utils.extract_true_skeleton(ds, D)

                        start_time = time.time()
                        est_graph, omega, meta_info = estimator.fit_predict(X_batch)
                        exec_time = time.time() - start_time

                        metrics = utils.calculate_metrics(est_graph, true_skel)

                        # Inject p_order metadata into the JSON output
                        if meta_info is None:
                            meta_info = {}
                        if args.model == "sing":
                            meta_info["p_order"] = args.p_order

                        results_dict = {
                            "model": args.model.upper(),
                            "dataset": ds,
                            "N": n,
                            "seed": seed,
                            "execution_time_sec": round(exec_time, 2),
                            "metrics": metrics,
                            "meta_info": meta_info,
                            "omega": omega.tolist() if hasattr(omega, "tolist") else omega
                        }

                        # Save path mapping (e.g., out_dir/sing/p_1/...)
                        utils.save_json_results(args.out_dir, save_method_name, ds, n, seed, results_dict)

                    except FileNotFoundError as e:
                        tqdm.write(f"Data not found: {e}")
                    except Exception as e:
                        tqdm.write(f"Error [{ds} N={n} Seed={seed}]: {e}")
                        # tqdm.write(traceback.format_exc())
                    finally:
                        pbar.update(1)

    print(f"Benchmark for {args.model.upper()} completed successfully.")


if __name__ == "__main__":
    main()
