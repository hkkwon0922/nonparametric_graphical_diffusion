"""
Helpers for the MNIST graph-selection visualization.

The pipeline recovers a graph over the 28x28 MNIST pixel grid from a trained
DDPM's Hessian:

  1. Load the per-timestep Hessian (``H_dict_avg``: t -> 784x784 matrix).
  2. For one anchor pixel, standardize ``|H[i, j]|`` per timestep and show that
     the trajectories separate by L1 grid distance (locality).
  3. Stack ``|H|`` over all unordered pixel pairs, standardize per timestep, and
     run KMeans(K=2). The high-magnitude cluster is the selected graph.
  4. Draw the selected graph on the 28x28 grid.

Data is kept self-contained under ``./data/``:
  - ``hessian_mnist_*.pickle``   : the DDPM Hessian payload
  - ``mnist_reference_image.npy``: one non-empty MNIST digit, for backgrounds
"""

import glob
import os
import pickle
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------
# Grid / pipeline constants
# --------------------------------------------------------------------------
GRID = 28
N_PIX = GRID * GRID                      # 784
ANCHOR_ROW = 15
ANCHOR_COL = 16
ANCHOR_FLAT = ANCHOR_ROW * GRID + ANCHOR_COL
T_LIST = list(range(1, 31))              # timesteps used (t = 1..30)
N_T = len(T_LIST)
KM_RANDOM_STATE = 100
KM_N_INIT = 30
HIGH = 1                                 # cluster label for the high-magnitude group

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_hessian(data_dir=DATA_DIR):
    """Load the MNIST Hessian payload and return ``H_dict_avg`` (t -> matrix)."""
    candidates = sorted(glob.glob(os.path.join(data_dir, "hessian_mnist_*.pickle")))
    if not candidates:
        raise FileNotFoundError(f"No MNIST hessian pickle found in: {data_dir}")
    path = max(candidates, key=os.path.getmtime)
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, dict) and "H_dict_avg" in obj:
        h_dict_avg = obj["H_dict_avg"]
    elif isinstance(obj, (list, tuple)) and len(obj) >= 1:
        h_dict_avg = obj[0]
    else:
        raise ValueError("Unsupported pickle format for MNIST Hessian payload")
    return h_dict_avg, path


def load_reference_image(data_dir=DATA_DIR):
    """Load the pre-extracted reference MNIST digit (28x28 float array)."""
    return np.load(os.path.join(data_dir, "mnist_reference_image.npy"))


# --------------------------------------------------------------------------
# (2) Local view: |H| trajectories from one anchor pixel
# --------------------------------------------------------------------------
def anchor_trajectories(h_dict_avg, anchor_flat=ANCHOR_FLAT,
                        anchor_row=ANCHOR_ROW, anchor_col=ANCHOR_COL, t_list=T_LIST):
    """Per-pixel ``|H[anchor, j]|`` trajectories with L1 distance and z-scores.

    Returns a list of dicts with keys: row, col, j_idx, l1, vals, vals_std.
    """
    records = []
    for r in range(GRID):
        for c in range(GRID):
            if (r, c) == (anchor_row, anchor_col):
                continue
            j = r * GRID + c
            d_l1 = abs(r - anchor_row) + abs(c - anchor_col)
            vals = [abs(h_dict_avg[t][anchor_flat, j]) for t in t_list]
            records.append({"row": r, "col": c, "j_idx": j, "l1": d_l1, "vals": vals})

    vals_matrix = np.array([rec["vals"] for rec in records], dtype=float)
    mu_t = vals_matrix.mean(axis=0, keepdims=True)
    sd_t = vals_matrix.std(axis=0, keepdims=True)
    vals_std = (vals_matrix - mu_t) / np.where(sd_t > 0, sd_t, 1.0)
    for rec, zrow in zip(records, vals_std):
        rec["vals_std"] = zrow
    return records


# --------------------------------------------------------------------------
# (3) Global clustering over all pixel pairs
# --------------------------------------------------------------------------
def global_clustering(h_dict_avg, t_list=T_LIST,
                      random_state=KM_RANDOM_STATE, n_init=KM_N_INIT):
    """KMeans(K=2) on per-t standardized ``|H|`` over all unordered pixel pairs.

    Returns a dict with:
      iu, ju            : upper-triangular pair indices over the 784 pixels
      vals_global_std   : (n_pairs, n_t) standardized trajectories
      labels_g          : cluster label per pair (0 = low-mag, 1 = high-mag)
      centers_g         : (2, n_t) cluster centroids in the same ordering
    """
    iu, ju = np.triu_indices(N_PIX, k=1)
    n_pairs = iu.size

    vals_global = np.empty((n_pairs, len(t_list)), dtype=np.float32)
    for k, t in enumerate(t_list):
        h_t = np.abs(np.asarray(h_dict_avg[t]))
        vals_global[:, k] = h_t[iu, ju]

    mu_g = vals_global.mean(axis=0, keepdims=True)
    sd_g = vals_global.std(axis=0, keepdims=True)
    vals_global_std = (vals_global - mu_g) / np.where(sd_g > 0, sd_g, 1.0)

    km = KMeans(n_clusters=2, init="k-means++", n_init=n_init, random_state=random_state)
    labels_g = km.fit_predict(vals_global_std)

    # Re-label so cluster 0 = low-magnitude, cluster 1 = high-magnitude
    order = np.argsort([vals_global_std[labels_g == k].mean() for k in range(2)])
    remap = {old: new for new, old in enumerate(order)}
    labels_g = np.array([remap[l] for l in labels_g])
    centers_g = km.cluster_centers_[order]

    return {
        "iu": iu, "ju": ju,
        "vals_global_std": vals_global_std,
        "labels_g": labels_g,
        "centers_g": centers_g,
    }


# --------------------------------------------------------------------------
# (4) Selected graph: high-magnitude cluster on the grid
# --------------------------------------------------------------------------
def selected_graph(cluster, high=HIGH):
    """Build edge geometry for the high-magnitude cluster.

    Returns a dict with:
      sel       : indices (into iu/ju) of selected edges
      segments  : (n_sel, 2, 2) line segments in (col, row) coordinates
      edge_l1   : L1 grid distance per selected edge
      deg_grid  : (28, 28) per-pixel degree over all selected edges
    """
    iu, ju, labels_g = cluster["iu"], cluster["ju"], cluster["labels_g"]
    sel = np.where(labels_g == high)[0]

    src_r, src_c = iu[sel] // GRID, iu[sel] % GRID
    dst_r, dst_c = ju[sel] // GRID, ju[sel] % GRID
    edge_l1 = np.abs(src_r - dst_r) + np.abs(src_c - dst_c)

    segments = np.stack([np.column_stack([src_c, src_r]),
                         np.column_stack([dst_c, dst_r])], axis=1)

    deg = np.zeros(N_PIX, dtype=int)
    np.add.at(deg, iu[sel], 1)
    np.add.at(deg, ju[sel], 1)
    deg_grid = deg.reshape(GRID, GRID)

    return {
        "sel": sel,
        "segments": segments,
        "edge_l1": edge_l1,
        "deg_grid": deg_grid,
        "src_rc": (src_r, src_c),
        "dst_rc": (dst_r, dst_c),
    }


def degree_grid_for_mask(cluster, graph, mask):
    """Per-pixel degree (28x28) restricted to the selected edges where ``mask`` is True."""
    iu, ju = cluster["iu"], cluster["ju"]
    sel = graph["sel"]
    deg = np.zeros(N_PIX, dtype=int)
    np.add.at(deg, iu[sel][mask], 1)
    np.add.at(deg, ju[sel][mask], 1)
    return deg.reshape(GRID, GRID)


def l1_distribution(edge_l1, top=10):
    """Return a sorted list of (L1, count) for the selected edges."""
    return sorted(Counter(edge_l1.tolist()).items())[:top]
