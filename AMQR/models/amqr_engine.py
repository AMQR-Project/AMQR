import ot
import warnings
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import qmc, norm, laplace, rankdata
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph, NearestNeighbors, KNeighborsRegressor

warnings.filterwarnings("ignore")


# =====================================================================
# 👑 核心类：AMQR 万能流形分位数回归引擎 (工业封装版)
# =====================================================================
class AMQR_Engine:
    """
    Auto-conditioned Manifold Quantile Regression (AMQR)
    特性:
    1. 动态/强制本征维数探测 (MLE)
    2. 破缺噪声免疫的精确最优传输 (Exact GW with Symmetry Breaking)
    3. 自动地标采样与样本外补全 (Landmark Out-of-Sample Extension)
    4. 自动条件滑动窗口回归 (Conditional Sliding Window)
    """

    def __init__(self, ref_dist='uniform', epsilon=0.0, d_int=None,
                 use_knn=True, k_neighbors=15, max_samples=2500, use_log_squash=False):
        self.ref_dist = ref_dist.lower()
        self.epsilon = epsilon
        self.d_int = d_int
        self.use_knn = use_knn
        self.k_neighbors = k_neighbors
        self.max_samples = max_samples  # 🌟 控制 O(N^3) 复杂度的核心阀门
        self.use_log_squash = use_log_squash

    def _generate_latent_qmc(self, d, N):
        sampler = qmc.Halton(d=d, scramble=False)
        u_grid = np.clip(sampler.random(n=N), 1e-4, 1 - 1e-4)
        Z = np.zeros((N, d))

        if self.ref_dist == 'uniform':
            if d == 1:
                Z = (u_grid * 2) - 1.0
            elif d == 2:
                r = np.sqrt(u_grid[:, 0])
                theta = 2 * np.pi * u_grid[:, 1]
                Z = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            else:
                Z = (u_grid * 2) - 1.0
        elif self.ref_dist == 'gaussian':
            Z = norm.ppf(u_grid, loc=0, scale=1.0)
        elif self.ref_dist == 'laplace':
            Z = laplace.ppf(u_grid, loc=0, scale=1.0)

        Z[0] = np.zeros(d)
        return Z

    def _fit_predict_core(self, Y, y_dist_m=None):
        """底层纯粹的数学对齐引擎 (支持直接传入预计算距离矩阵 y_dist_m)"""
        N = len(Y)
        Y_flat = Y.reshape(N, -1)

        # =========================================================
        # 1. 计算或直接使用流形测地线距离 Cy
        # =========================================================
        if y_dist_m is not None:
            Cy = y_dist_m.copy()  # 🌟 直接使用外部传入的真实度量 (如 Wasserstein 距离)
        else:
            if self.use_knn:
                k = min(self.k_neighbors, N - 1)
                A = kneighbors_graph(Y_flat, n_neighbors=k, mode='distance', include_self=False)
                Cy = shortest_path(A, method='D', directed=False)
                if np.isinf(Cy).any():
                    Cy[np.isinf(Cy)] = np.nanmax(Cy[Cy != np.inf]) * 2.0
            else:
                Cy = cdist(Y_flat, Y_flat, metric='euclidean')

        # =========================================================
        # 2. MLE 本征维数探测 (支持从距离矩阵直接推断)
        # =========================================================
        if self.d_int is not None:
            final_d = self.d_int
        else:
            k_mle = min(10, N - 1)
            if y_dist_m is not None:
                # 如果传入了距离矩阵，直接对距离矩阵排序取前 k 个最近邻
                dists = np.sort(Cy, axis=1)[:, 1:k_mle + 2]
            else:
                nn = NearestNeighbors(n_neighbors=k_mle + 1).fit(Y_flat)
                dists, _ = nn.kneighbors(Y_flat)
                dists = dists[:, 1:]

            dists = np.maximum(dists, 1e-9)
            r_k = dists[:, -1:]
            mle_val = (k_mle - 1) / np.sum(np.log(r_k / dists[:, :-1]), axis=1)
            final_d = int(np.round(np.mean(mle_val)))
            final_d = max(1, min(final_d, 3))  # 强制封顶

        # =========================================================
        # 3. 构建目标空间与打乱对称性
        # =========================================================
        Z_ref = self._generate_latent_qmc(final_d, N)
        Cz = cdist(Z_ref, Z_ref, metric='euclidean')

        Cz_norm = Cz / (Cz.max() + 1e-9)
        if self.use_log_squash:
            Cy_proc = np.log1p(Cy)
        else:
            Cy_proc = Cy
        Cy_norm = Cy_proc / (np.nanmax(Cy_proc) + 1e-9)

        noise_z = np.random.uniform(0, 1e-8, size=Cz_norm.shape)
        Cz_norm += (noise_z + noise_z.T) / 2.0  # 强制对称化
        np.fill_diagonal(Cz_norm, 0)

        noise_y = np.random.uniform(0, 1e-8, size=Cy_norm.shape)
        Cy_norm += (noise_y + noise_y.T) / 2.0
        np.fill_diagonal(Cy_norm, 0)

        # =========================================================
        # 4. 极速求解 GW
        # =========================================================
        py, pz = ot.unif(N), ot.unif(N)
        if self.epsilon > 0:
            gw_plan = ot.gromov.entropic_gromov_wasserstein(
                Cz_norm, Cy_norm, pz, py, 'square_loss', epsilon=self.epsilon, max_iter=100)
        else:
            gw_plan = ot.gromov.gromov_wasserstein(
                Cz_norm, Cy_norm, pz, py, 'square_loss', max_iter=50, tol=1e-4)

        # =========================================================
        # 5. 提取深度与排名
        # =========================================================
        Y_mapped_to_Z = np.dot(gw_plan.T, Z_ref) * N
        amqr_depths = np.linalg.norm(Y_mapped_to_Z, axis=1)
        med_idx = np.argmin(amqr_depths)
        ranks = rankdata(amqr_depths) / N

        return Y[med_idx], ranks

    def fit_predict(self, Y, y_dist_m=None, T=None, t_eval=None, window_size=1.0):
        """
        新增 y_dist_m 参数。支持按索引优雅切片传入子循环！
        """
        N = len(Y)

        if T is None:
            if N <= self.max_samples:
                med, ranks = self._fit_predict_core(Y, y_dist_m=y_dist_m)
                return med, ranks
            else:
                sub_idx = np.random.choice(N, size=self.max_samples, replace=False)
                Y_sub = Y[sub_idx]
                # 同步切片距离矩阵
                y_dist_m_sub = y_dist_m[np.ix_(sub_idx, sub_idx)] if y_dist_m is not None else None

                med, ranks_sub = self._fit_predict_core(Y_sub, y_dist_m=y_dist_m_sub)

                knn_ext = KNeighborsRegressor(n_neighbors=3).fit(Y_sub.reshape(self.max_samples, -1), ranks_sub)
                ranks_full = knn_ext.predict(Y.reshape(N, -1))
                return med, ranks_full

        else:
            if t_eval is None:
                t_eval = np.linspace(T.min(), T.max(), 50)
            step_size = t_eval[1] - t_eval[0] if len(t_eval) > 1 else window_size / 5.0

            trajectory_med = []
            final_ranks = np.ones(N)

            for t_c in tqdm(t_eval):
                idx = np.where(np.abs(T - t_c) <= window_size / 2.0)[0]
                if len(idx) < 15:
                    continue

                Y_c = Y[idx]
                # 🌟 动态时序滑动窗口同样完美支持预计算距离矩阵的局部切片！
                y_dist_m_c = y_dist_m[np.ix_(idx, idx)] if y_dist_m is not None else None

                inner_condition = (T[idx] >= t_c - step_size / 2.0) & (T[idx] < t_c + step_size / 2.0)
                inner_in_idx = np.where(inner_condition)[0]

                med_c, ranks_c = self.fit_predict(Y=Y_c, y_dist_m=y_dist_m_c, T=None)
                trajectory_med.append((t_c, med_c))

                if len(inner_in_idx) > 0:
                    final_ranks[idx[inner_in_idx]] = ranks_c[inner_in_idx]

            return trajectory_med, final_ranks
