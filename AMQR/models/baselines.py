import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.neighbors import KernelDensity, kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from tqdm import tqdm


def _sliding_window_runner(T, Y, t_eval, window_size, core_estimator_func, y_dist_m=None, **kwargs):
    """Unified sliding window scheduler, converting static point estimates to conditional manifold regression"""
    N = len(Y)
    trajectory_med = []
    final_ranks = np.ones(N)

    step_size = t_eval[1] - t_eval[0] if len(t_eval) > 1 else window_size / 5.0

    for t_c in tqdm(t_eval, desc=f"Running {core_estimator_func.__name__}"):
        idx = np.where(np.abs(T - t_c) <= window_size / 2.0)[0]
        if len(idx) < 15:
            continue

        Y_c = Y[idx]

        y_dist_m_c = y_dist_m[np.ix_(idx, idx)] if y_dist_m is not None else None

        # Call the specific underlying core estimator
        med_c, depths_c = core_estimator_func(Y_c, y_dist_m_c=y_dist_m_c, **kwargs)

        trajectory_med.append((t_c, med_c))

        # Inner-core slice assignment logic (maintaining an absolutely fair assignment mechanism with AMQR)
        inner_condition = (T[idx] >= t_c - step_size / 2.0) & (T[idx] < t_c + step_size / 2.0)
        inner_in_idx = np.where(inner_condition)[0]
        if len(inner_in_idx) > 0:
            # Convert depth to rank (0~1)
            ranks_c = rankdata(depths_c) / len(Y_c)
            final_ranks[idx[inner_in_idx]] = ranks_c[inner_in_idx]

    return trajectory_med, final_ranks


# ---------------------------------------------------------------------
# 1. Classic Euclidean Mean (Nadaraya-Watson L2 Mean)
# ---------------------------------------------------------------------
def get_nw_tube(T, Y, t_eval, window_size=1.0):

    N = len(Y)
    Y_flat = Y.reshape(N, -1)
    trajectory_med = []
    final_ranks = np.ones(N)

    step_size = t_eval[1] - t_eval[0] if len(t_eval) > 1 else window_size / 5.0

    for t_c in tqdm(t_eval, desc="Running NW L2 Mean"):
        idx = np.where(np.abs(T - t_c) <= window_size / 2.0)[0]
        if len(idx) < 15: continue

        Y_c_flat = Y_flat[idx]
        weights = np.exp(-((T[idx] - t_c) ** 2) / (2 * (window_size / 2) ** 2))
        weights /= (np.sum(weights) + 1e-9)

        med_c_flat = np.sum(weights[:, None] * Y_c_flat, axis=0)
        trajectory_med.append((t_c, med_c_flat.reshape(Y.shape[1:])))

        residuals = np.linalg.norm(Y_c_flat - med_c_flat, axis=1)
        ranks_c = rankdata(residuals) / len(Y_c_flat)

        inner_condition = (T[idx] >= t_c - step_size / 2.0) & (T[idx] < t_c + step_size / 2.0)
        inner_in_idx = np.where(inner_condition)[0]
        if len(inner_in_idx) > 0:
            final_ranks[idx[inner_in_idx]] = ranks_c[inner_in_idx]

    return trajectory_med, final_ranks


# ---------------------------------------------------------------------
# 2. Intrinsic Fréchet Regression (Intrinsic Fréchet L2 Mean)
# ---------------------------------------------------------------------
def _frechet_l2_mean_core(Y_c, y_dist_m_c=None, k_neighbors=15):

    if y_dist_m_c is not None:
        Cy_geo = y_dist_m_c
    else:
        Y_c_flat = Y_c.reshape(len(Y_c), -1)
        k = min(k_neighbors, len(Y_c) - 1)
        A = kneighbors_graph(Y_c_flat, n_neighbors=k, mode='distance', include_self=False)
        Cy_geo = shortest_path(A, method='D', directed=False)
        if np.isinf(Cy_geo).any(): Cy_geo[np.isinf(Cy_geo)] = np.nanmax(Cy_geo[Cy_geo != np.inf]) * 2.0

    # L2 Mean: Minimize the sum of squared geodesic distances
    l2_sums = np.sum(Cy_geo ** 2, axis=1)
    idx = np.argmin(l2_sums)
    return Y_c[idx], Cy_geo[:, idx]


def get_frechet_regression_tube(T, Y, t_eval, window_size=1.0, y_dist_m=None, k_neighbors=15):
    return _sliding_window_runner(T, Y, t_eval, window_size, _frechet_l2_mean_core, y_dist_m=y_dist_m,
                                  k_neighbors=k_neighbors)


# ---------------------------------------------------------------------
# 3. Intrinsic Isotropic Median (Intrinsic L1 Median + Isotropic Caps)
# ---------------------------------------------------------------------
def _geodesic_l1_median_core(Y_c, y_dist_m_c=None, k_neighbors=15):

    if y_dist_m_c is not None:
        Cy_geo = y_dist_m_c
    else:
        Y_c_flat = Y_c.reshape(len(Y_c), -1)
        k = min(k_neighbors, len(Y_c) - 1)
        A = kneighbors_graph(Y_c_flat, n_neighbors=k, mode='distance', include_self=False)
        Cy_geo = shortest_path(A, method='D', directed=False)
        if np.isinf(Cy_geo).any(): Cy_geo[np.isinf(Cy_geo)] = np.nanmax(Cy_geo[Cy_geo != np.inf]) * 2.0

    # L1 Median: Minimize the sum of absolute geodesic distances
    l1_sums = np.sum(Cy_geo, axis=1)
    idx = np.argmin(l1_sums)
    return Y_c[idx], Cy_geo[:, idx]


def get_isotropic_geodesic_tube(T, Y, t_eval, window_size=1.0, y_dist_m=None, k_neighbors=15):
    return _sliding_window_runner(T, Y, t_eval, window_size, _geodesic_l1_median_core, y_dist_m=y_dist_m,
                                  k_neighbors=k_neighbors)


# ---------------------------------------------------------------------
# 4. Density Mode Regression (Sliding-Window KDE Mode)
# ---------------------------------------------------------------------
def _kde_mode_core(Y_c, y_dist_m_c=None):
    Y_c_flat = Y_c.reshape(len(Y_c), -1)


    if Y_c_flat.shape[1] > 3:
        pca = PCA(n_components=min(3, len(Y_c))).fit(Y_c_flat)
        Y_eval = pca.transform(Y_c_flat)
    else:
        Y_eval = Y_c_flat

    kde = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(Y_eval)
    log_dens = kde.score_samples(Y_eval)
    idx_mode = np.argmax(log_dens)

    return Y_c[idx_mode], -log_dens


# models/baselines.py (add the following new content at the end of the file)

import scipy.linalg


# ---------------------------------------------------------------------
# 5. Riemannian Statistical Baselines on SPD Manifolds (Riemannian Baselines on SPD)
# ---------------------------------------------------------------------

def _riemannian_l2_mean_core(Y_c, y_dist_m_c=None, **kwargs):
    """
    Accurately calculate the Fréchet L2 Mean (Riemannian mean) of SPD matrices under the Log-Euclidean metric.
    This is a 'synthesis' process, and the result is usually not among the original sample points.
    Mathematical formula: Mean = exp( 1/N * sum( log(M_i) ) )
    """
    if Y_c.ndim == 2:
        N, flat_dim = Y_c.shape
        dim = int(np.sqrt(flat_dim))
        Y_matrices = Y_c.reshape(N, dim, dim)
    else:
        N, dim, _ = Y_c.shape
        Y_matrices = Y_c

    # 1. Map to the tangent space via matrix logarithm
    log_sum = np.zeros((dim, dim))
    for M in Y_matrices:
        # Add a perturbation term to ensure numerical stability
        log_sum += scipy.linalg.logm(M + np.eye(dim) * 1e-6).real

    # 2. Calculate the standard mean in the tangent space (Euclidean space)
    log_mean = log_sum / N

    # 3. Map back to the SPD manifold via matrix exponential
    riemannian_mean = scipy.linalg.expm(log_mean).real

    # Calculate the distance from all points to this newly synthesized mean as depth
    # Note: This requires recalculating the distance because the mean is a new point
    log_riemannian_mean = scipy.linalg.logm(riemannian_mean + np.eye(dim) * 1e-6).real
    log_Y_matrices = np.array([scipy.linalg.logm(M + np.eye(dim) * 1e-6).real for M in Y_matrices])

    depths = np.linalg.norm(log_Y_matrices.reshape(N, -1) - log_riemannian_mean.flatten(), axis=1)

    return riemannian_mean, depths


def get_riemannian_l2_tube(T, Y, t_eval, window_size=1.0, y_dist_m=None):
    """
    Generate a sliding window regression trajectory of the Riemannian L2 mean for SPD matrices.
    """
    # Note: y_dist_m is no longer needed here because the mean is synthesized and the distance needs to be recalculated
    return _sliding_window_runner(T, Y, t_eval, window_size, _riemannian_l2_mean_core)

def get_kde_mode_tube(T, Y, t_eval, window_size=1.0):
    return _sliding_window_runner(T, Y, t_eval, window_size, _kde_mode_core)
