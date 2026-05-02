# experiments/run_sim3_spd.py (Fully Corrected Version)

import sys
import os
import numpy as np
import scipy.linalg
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from data.simulations import generate_spd_data_with_labels
from models.amqr_engine import AMQR_Engine
from utils.visualization import plot_spd_3x5_comparison

from models.baselines import (
    get_nw_tube,
    get_riemannian_l2_tube,
    get_isotropic_geodesic_tube,
    get_kde_mode_tube
)


def compute_lem_distance_matrix(Y_matrices):
    """Calculate the exact Log-Euclidean Riemannian distance matrix for SPD matrices"""
    if Y_matrices.ndim == 2:
        N, flat_dim = Y_matrices.shape
        dim_mat = int(np.sqrt(flat_dim))
        Y_matrices_3d = Y_matrices.reshape(N, dim_mat, dim_mat)
    else:
        N, dim_mat, _ = Y_matrices.shape
        Y_matrices_3d = Y_matrices

    print(f"Mapping {N} {dim_mat}x{dim_mat} SPD matrices to the Riemannian tangent space (Log-space)...")
    log_Y_flat = np.array([scipy.linalg.logm(M + np.eye(dim_mat) * 1e-6).real.flatten() for M in Y_matrices_3d])

    lem_dist_matrix = squareform(pdist(log_Y_flat, metric='euclidean'))
    return lem_dist_matrix


if __name__ == "__main__":
    print("========================================================")
    print(" Sim 3: Dynamic SPD Regression (5-Method Ultimate Showdown)")
    print("========================================================")

    DIM = 3
    # Generate data, note that R_true is now SPD_true
    Y_spd, true_labels, SPD_true = generate_spd_data_with_labels(N=400, dim=DIM, random_state=42)

    # --- Pre-compute the exact Riemannian metric ---
    lem_dist_matrix = compute_lem_distance_matrix(Y_spd)

    # --- AMQR Model Parameter Configuration ---
    amqr_params = {
        'ref_dist': 'uniform',
        'use_log_squash': False,
        'use_knn': False,  # Because we directly provide the distance matrix
        'd_int': int(DIM * (DIM + 1) / 2),
        'epsilon': 0.0,
        'max_samples': 2000
    }

    # --- Unified Execution of All Models ---
    # In this static scenario, we only evaluate the cross-section at a single time point t=5.0
    T = np.linspace(0, 10, len(Y_spd))
    target_t = 5.0
    window_size = 10.0  # Use a large window to cover all data

    # 1. AMQR (Proposed)
    print("\n[1/5] Running AMQR...")
    amqr = AMQR_Engine(**amqr_params)
    a_med, a_ranks = amqr.fit_predict(Y_spd, y_dist_m=lem_dist_matrix)

    # 2. NW Mean (Baseline)
    print("\n[2/5] Running NW Mean...")
    nw_traj, _ = get_nw_tube(T, Y_spd, t_eval=[target_t], window_size=window_size)
    nw_med = nw_traj[0][1]
    nw_ranks = rankdata(np.linalg.norm(Y_spd - nw_med.flatten(), axis=1)) / len(Y_spd)

    # 3. Riemannian L2 Mean (Baseline)
    print("\n[3/5] Running Riemannian L2 Mean...")
    f_l2_traj, f_l2_ranks_full = get_riemannian_l2_tube(T, Y_spd, t_eval=[target_t], window_size=window_size)
    f_l2_med = f_l2_traj[0][1]

    # 4. Geodesic L1 Medoid (Baseline)
    print("\n[4/5] Running Geodesic L1 Medoid...")
    f_l1_traj, f_l1_ranks_full = get_isotropic_geodesic_tube(T, Y_spd, t_eval=[target_t], window_size=window_size,
                                                             y_dist_m=lem_dist_matrix)
    f_l1_med = f_l1_traj[0][1]

    # 5. KDE Mode (Baseline)
    print("\n[5/5] Running KDE Mode...")
    kde_traj, kde_ranks_full = get_kde_mode_tube(T, Y_spd, t_eval=[target_t], window_size=window_size)
    kde_med = kde_traj[0][1]

    # --- Organize Results for Plotting ---
    results_static = {
        'nw': {'med': nw_med, 'ranks': nw_ranks},
        'f_l2': {'med': f_l2_med, 'ranks': f_l2_ranks_full},
        'f_l1': {'med': f_l1_med, 'ranks': f_l1_ranks_full},
        'kde': {'med': kde_med, 'ranks': kde_ranks_full},
        'amqr': {'med': a_med, 'ranks': a_ranks}
    }

    # --- Render the Final Comparison Plot ---
    print("\nRendering the 3x5 Ultimate Comparison Plot...")
    save_img_path = os.path.join(PROJECT_ROOT, "results", "figures", f"Fig5_SPD_{DIM}D_Comparison_3x5.pdf")

    plot_spd_3x5_comparison(Y_spd, results_static, dim=DIM, filename=save_img_path)

    print(f"\n{DIM}x{DIM} dimensional SPD matrix experiment completed successfully!")
