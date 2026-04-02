import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA

# =====================================================================
# 🛡️ 基线方法全家桶 (经过轻量级改造以适应滑动窗口)
# =====================================================================

def get_frechet_tube(Y):
    Y_flat = Y.reshape(len(Y), -1)
    Cy = cdist(Y_flat, Y_flat, metric='euclidean')
    l1_sums = np.sum(Cy, axis=1)
    idx = np.argmin(l1_sums)
    return Y[idx], rankdata(Cy[:, idx]) / len(Y)


def get_kde_tube(Y):
    Y_flat = Y.reshape(len(Y), -1)
    # 针对高维情况进行安全降维
    if Y_flat.shape[1] > 3:
        pca = PCA(n_components=min(3, len(Y))).fit(Y_flat)
        Y_eval = pca.transform(Y_flat)
    else:
        Y_eval = Y_flat

    kde = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(Y_eval)
    log_dens = kde.score_samples(Y_eval)
    idx_mode = np.argmax(log_dens)
    return Y[idx_mode], rankdata(-log_dens) / len(Y)


def get_nw_tube(T, Y, bandwidth=1.0):
    """NW 是原生支持 T 的条件期望模型"""
    N = len(Y)
    Y_flat = Y.reshape(N, -1)
    nw_means_flat = np.zeros_like(Y_flat)

    for i, t_i in enumerate(T):
        weights = np.exp(-((T - t_i) ** 2) / (2 * bandwidth ** 2))
        weights /= (np.sum(weights) + 1e-9)
        nw_means_flat[i] = np.sum(weights[:, None] * Y_flat, axis=0)

    residuals = np.linalg.norm(Y_flat - nw_means_flat, axis=1)
    nw_ranks = rankdata(residuals) / N
    nw_means = nw_means_flat.reshape(Y.shape)
    return nw_means, nw_ranks
