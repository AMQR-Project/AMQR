import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.neighbors import KernelDensity, kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from tqdm import tqdm


# =====================================================================
# 🛡️ 流形统计学顶刊基线全家桶 (Conditional Regression 统一接口版)
# =====================================================================

def _sliding_window_runner(T, Y, t_eval, window_size, core_estimator_func, y_dist_m=None, **kwargs):
    """统一的滑动窗口调度器，将静态点估计转化为条件流形回归"""
    N = len(Y)
    trajectory_med = []
    final_ranks = np.ones(N)

    step_size = t_eval[1] - t_eval[0] if len(t_eval) > 1 else window_size / 5.0

    for t_c in tqdm(t_eval, desc=f"Running {core_estimator_func.__name__}"):
        idx = np.where(np.abs(T - t_c) <= window_size / 2.0)[0]
        if len(idx) < 15:
            continue

        Y_c = Y[idx]

        # 🌟 修复不公平对比：允许基线模型使用全局切片的真实测地线距离
        y_dist_m_c = y_dist_m[np.ix_(idx, idx)] if y_dist_m is not None else None

        # 调用具体的底层核心估计器
        med_c, depths_c = core_estimator_func(Y_c, y_dist_m_c=y_dist_m_c, **kwargs)

        trajectory_med.append((t_c, med_c))

        # Inner-core 切片赋值逻辑 (与 AMQR 保持绝对公平的赋值机制)
        inner_condition = (T[idx] >= t_c - step_size / 2.0) & (T[idx] < t_c + step_size / 2.0)
        inner_in_idx = np.where(inner_condition)[0]
        if len(inner_in_idx) > 0:
            # 深度转为秩 (0~1)
            ranks_c = rankdata(depths_c) / len(Y_c)
            final_ranks[idx[inner_in_idx]] = ranks_c[inner_in_idx]

    return trajectory_med, final_ranks


# ---------------------------------------------------------------------
# 1. 经典欧氏均值 (Nadaraya-Watson L2 Mean)
# ---------------------------------------------------------------------
def get_nw_tube(T, Y, t_eval, window_size=1.0):
    # ...[保持原样，NW 不需要流形距离矩阵] ...
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
# 2. 内蕴 Fréchet 回归 (Intrinsic Fréchet L2 Mean) - SOTA 对照组
# ---------------------------------------------------------------------
def _frechet_l2_mean_core(Y_c, y_dist_m_c=None, k_neighbors=15):
    # 🌟 修复：优先使用传入的全局切片距离矩阵，保证与 AMQR 拓扑信息对等
    if y_dist_m_c is not None:
        Cy_geo = y_dist_m_c
    else:
        Y_c_flat = Y_c.reshape(len(Y_c), -1)
        k = min(k_neighbors, len(Y_c) - 1)
        A = kneighbors_graph(Y_c_flat, n_neighbors=k, mode='distance', include_self=False)
        Cy_geo = shortest_path(A, method='D', directed=False)
        if np.isinf(Cy_geo).any(): Cy_geo[np.isinf(Cy_geo)] = np.nanmax(Cy_geo[Cy_geo != np.inf]) * 2.0

    # L2 Mean: 最小化测地线距离的平方和
    l2_sums = np.sum(Cy_geo ** 2, axis=1)
    idx = np.argmin(l2_sums)
    return Y_c[idx], Cy_geo[:, idx]


def get_frechet_regression_tube(T, Y, t_eval, window_size=1.0, y_dist_m=None, k_neighbors=15):
    return _sliding_window_runner(T, Y, t_eval, window_size, _frechet_l2_mean_core, y_dist_m=y_dist_m,
                                  k_neighbors=k_neighbors)


# ---------------------------------------------------------------------
# 3. 内蕴各向同性中位数 (Intrinsic L1 Median + Isotropic Caps) - 强力对照组
# ---------------------------------------------------------------------
def _geodesic_l1_median_core(Y_c, y_dist_m_c=None, k_neighbors=15):
    # 🌟 修复：优先使用传入的全局切片距离矩阵
    if y_dist_m_c is not None:
        Cy_geo = y_dist_m_c
    else:
        Y_c_flat = Y_c.reshape(len(Y_c), -1)
        k = min(k_neighbors, len(Y_c) - 1)
        A = kneighbors_graph(Y_c_flat, n_neighbors=k, mode='distance', include_self=False)
        Cy_geo = shortest_path(A, method='D', directed=False)
        if np.isinf(Cy_geo).any(): Cy_geo[np.isinf(Cy_geo)] = np.nanmax(Cy_geo[Cy_geo != np.inf]) * 2.0

    # L1 Median: 最小化测地线绝对距离之和
    l1_sums = np.sum(Cy_geo, axis=1)
    idx = np.argmin(l1_sums)
    return Y_c[idx], Cy_geo[:, idx]


def get_isotropic_geodesic_tube(T, Y, t_eval, window_size=1.0, y_dist_m=None, k_neighbors=15):
    return _sliding_window_runner(T, Y, t_eval, window_size, _geodesic_l1_median_core, y_dist_m=y_dist_m,
                                  k_neighbors=k_neighbors)


# ---------------------------------------------------------------------
# 4. 密度寻模回归 (Sliding-Window KDE Mode)
# ---------------------------------------------------------------------
def _kde_mode_core(Y_c, y_dist_m_c=None):
    Y_c_flat = Y_c.reshape(len(Y_c), -1)

    # 🛡️ 学术防御注释：KDE 在高维空间会遭遇严重的维数灾难。
    # 为了让基线模型能够正常运行，我们强制将其投影到前 3 个主成分上。
    if Y_c_flat.shape[1] > 3:
        pca = PCA(n_components=min(3, len(Y_c))).fit(Y_c_flat)
        Y_eval = pca.transform(Y_c_flat)
    else:
        Y_eval = Y_c_flat

    kde = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(Y_eval)
    log_dens = kde.score_samples(Y_eval)
    idx_mode = np.argmax(log_dens)

    return Y_c[idx_mode], -log_dens


# models/baselines.py (在文件末尾添加以下新内容)

import scipy.linalg


# ---------------------------------------------------------------------
# 5. [新增] SPD 流形上的黎曼统计基线 (Riemannian Baselines on SPD)
# ---------------------------------------------------------------------

def _riemannian_l2_mean_core(Y_c, y_dist_m_c=None, **kwargs):
    """
    精确计算 SPD 矩阵在 Log-Euclidean 度量下的 Fréchet L2 Mean (黎曼均值)。
    这是一个“合成”过程，结果通常不在原始样本点中。

    数学公式: Mean = exp( 1/N * sum( log(M_i) ) )
    """
    if Y_c.ndim == 2:
        N, flat_dim = Y_c.shape
        dim = int(np.sqrt(flat_dim))
        Y_matrices = Y_c.reshape(N, dim, dim)
    else:
        N, dim, _ = Y_c.shape
        Y_matrices = Y_c

    # 1. 通过矩阵对数映射到切空间
    log_sum = np.zeros((dim, dim))
    for M in Y_matrices:
        # 添加扰动项以保证数值稳定性
        log_sum += scipy.linalg.logm(M + np.eye(dim) * 1e-6).real

    # 2. 在切空间（欧氏空间）中计算标准均值
    log_mean = log_sum / N

    # 3. 通过矩阵指数映射回SPD流形
    riemannian_mean = scipy.linalg.expm(log_mean).real

    # 计算所有点到这个新合成的均值的距离作为深度
    # 注意：这需要重新计算距离，因为均值是新点
    log_riemannian_mean = scipy.linalg.logm(riemannian_mean + np.eye(dim) * 1e-6).real
    log_Y_matrices = np.array([scipy.linalg.logm(M + np.eye(dim) * 1e-6).real for M in Y_matrices])

    depths = np.linalg.norm(log_Y_matrices.reshape(N, -1) - log_riemannian_mean.flatten(), axis=1)

    return riemannian_mean, depths


def get_riemannian_l2_tube(T, Y, t_eval, window_size=1.0, y_dist_m=None):
    """
    为 SPD 矩阵生成黎曼 L2 均值的滑动窗口回归轨迹。
    """
    # 注意：这里不再需要 y_dist_m，因为均值是合成的，距离需要重新计算
    return _sliding_window_runner(T, Y, t_eval, window_size, _riemannian_l2_mean_core)

def get_kde_mode_tube(T, Y, t_eval, window_size=1.0):
    return _sliding_window_runner(T, Y, t_eval, window_size, _kde_mode_core)
