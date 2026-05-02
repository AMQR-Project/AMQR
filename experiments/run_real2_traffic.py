import sys
import os
import numpy as np
import scipy.linalg
from scipy.interpolate import interp1d
from scipy.stats import rankdata

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

from models.amqr_engine import AMQR_Engine
from data.real_data import load_pems_traffic_and_locations
from utils.visualization import plot_traffic_tube_validation, plot_spatial_grid, plot_local_matrix_grid


def compute_lem_anomaly_scores(Y_raw_flat, Y_reg_flat, dim=20):
    print("Calculating Log-Euclidean Riemannian manifold anomaly absolute distance...")
    N = Y_raw_flat.shape[0]
    scores = np.zeros(N)
    for i in range(N):
        M_raw = Y_raw_flat[i].reshape(dim, dim) + np.eye(dim) * 1e-5
        M_reg = Y_reg_flat[i].reshape(dim, dim) + np.eye(dim) * 1e-5
        log_raw = scipy.linalg.logm(M_raw).real
        log_reg = scipy.linalg.logm(M_reg).real
        scores[i] = np.linalg.norm(log_raw - log_reg)
    return scores


if __name__ == "__main__":
    print("========================================================")
    print("Pipeline: Urban Traffic Topology Regression & Mapping")
    print("========================================================")

    # Configure file paths
    h5_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'pems-bay.h5')
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'graph_sensor_locations_bay.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'graph_sensor_locations.csv')
    save_dir = os.path.join(PROJECT_ROOT, "results", "figures")
    DIM = 20

    # Load data
    Y_flat, T, timestamps, coords, sensor_ids = load_pems_traffic_and_locations(h5_path, csv_path, num_nodes=DIM)

    # ========================================================
    # Pre-compute Log-Euclidean Riemannian metric matrix
    # ========================================================
    print("Pre-computing global Log-Euclidean Riemannian distance matrix (Metric-Agnostic injection)...")
    from scipy.spatial.distance import pdist, squareform

    # 1. Extract all real SPD matrices and map them to the tangent space (Log-space)
    Y_matrices = Y_flat.reshape(-1, DIM, DIM)
    Y_log_space = np.array([scipy.linalg.logm(M + np.eye(DIM) * 1e-5).real for M in Y_matrices])

    # 2. Calculate Euclidean distance in the tangent space, which is mathematically exactly equivalent to the Log-Euclidean geodesic distance on the SPD manifold
    # Completely abandon ambient Euclidean distance!
    lem_dist_matrix = squareform(pdist(Y_log_space.reshape(len(Y_flat), -1), metric='euclidean'))
    # ========================================================

    # ========================================================
    # Global MLE Intrinsic Dimension Estimation (Global Intrinsic Dimension)
    # ========================================================
    print("Globally estimating the effective intrinsic dimension of the traffic manifold (Global MLE)...")
    k_mle = min(10, len(Y_flat) - 1)
    # Utilize the pre-calculated global exact Riemannian distance matrix
    dists = np.sort(lem_dist_matrix, axis=1)[:, 1:k_mle + 2]
    dists = np.maximum(dists, 1e-9)
    r_k = dists[:, -1:]
    mle_val = (k_mle - 1) / np.sum(np.log(r_k / dists[:, :-1]), axis=1)
    global_d_int = int(np.round(np.mean(mle_val)))
    global_d_int = max(1, global_d_int)  # Force capping

    print(f"Global MLE automatically detected effective intrinsic dimension: d_int = {global_d_int} (much smaller than ambient dimension 210)")

    # Run AMQR engine, injecting the globally stable d_int
    print(f"Running AMQR engine...")
    amqr = AMQR_Engine(ref_dist='uniform', epsilon=0.0, d_int=global_d_int,
                       use_log_squash=False, use_knn=False)

    # Inject the Riemannian metric matrix y_dist_m into the engine
    t_eval_hours = np.arange(0, 24, 1)
    trajectory, _ = amqr.fit_predict(Y_flat, y_dist_m=lem_dist_matrix, T=T, t_eval=t_eval_hours, window_size=2.0)

    t_sparse = np.array([t for t, _ in trajectory])
    Y_reg_sparse = np.array([med for _, med in trajectory])

    # Complete interpolation and calculate ranks
    print("Completing sample expectations via nearest neighbor mapping...")
    interpolator = interp1d(t_sparse, Y_reg_sparse, axis=0, kind='nearest', fill_value="extrapolate")
    Y_reg_flat_full = interpolator(T)

    # The anomaly_scores here are absolute distances
    anomaly_scores = compute_lem_anomaly_scores(Y_flat, Y_reg_flat_full, dim=DIM)
    a_ranks = rankdata(anomaly_scores) / len(anomaly_scores)

    print("\nStarting full pipeline visualization rendering...")
    # 1. 1x3 global pipeline plot (must use global a_ranks)
    plot_traffic_tube_validation(T, Y_flat, t_sparse, Y_reg_sparse, a_ranks, save_dir, dim=DIM)

    # 2. 3x4 geospatial topology plot (pass absolute distance scores, calculate locally inside)
    plot_spatial_grid(Y_flat, anomaly_scores, T, coords, save_dir, dim=DIM)

    # 3. 3x4 microscopic matrix heatmap (pass absolute distance scores, calculate locally inside)
    plot_local_matrix_grid(Y_flat, anomaly_scores, T, save_dir, dim=DIM)

    print("\nPipeline execution complete! All top-tier journal charts have been generated.")
