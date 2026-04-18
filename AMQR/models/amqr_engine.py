import ot
import warnings
import numpy as np
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes
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

    def _run_with_oos_protection(self, Y, y_dist_m=None):
        """
        🌟 统一的算力防火墙：触发地标隔离与核正则化内蕴投影 (Kernel-regularized Intrinsic Projection)
        """
        N = len(Y)
        if N <= self.max_samples:
            return self._fit_predict_core(Y, y_dist_m=y_dist_m)

        # 🚨 触发 OOS 保护机制
        core_idx = np.random.choice(N, size=self.max_samples, replace=False)
        oos_idx = np.setdiff1d(np.arange(N), core_idx)

        Y_core = Y[core_idx]
        y_dist_m_core = y_dist_m[np.ix_(core_idx, core_idx)] if y_dist_m is not None else None

        # 1. 仅对核心地标运行精确 GW
        med_core, ranks_core, z_core, S_x = self._fit_predict_core(Y_core, y_dist_m=y_dist_m_core, return_scale=True)

        # 2. 提取 OOS 点到 Core 点的纯正内蕴距离
        if y_dist_m is not None:
            dist_oos_to_core = y_dist_m[np.ix_(oos_idx, core_idx)]
        else:
            dist_oos_to_core = cdist(Y[oos_idx].reshape(len(oos_idx), -1), Y_core.reshape(len(core_idx), -1))

        # 3. 核正则化平滑内蕴投影 (Strictly aligned with Paper Step 3)
        k_nn = min(int(np.log(len(core_idx)) * 2), len(core_idx)) # k \asymp \log N_x
        nearest_core_indices = np.argsort(dist_oos_to_core, axis=1)[:, :k_nn]
        nearest_dists = np.take_along_axis(dist_oos_to_core, nearest_core_indices, axis=1)

        # 使用高斯核替代原始 IDW，保证等度连续性 (Equicontinuity)
        sigma = np.median(nearest_dists) + 1e-8
        weights = np.exp(-(nearest_dists ** 2) / (2 * sigma ** 2))
        weights /= np.sum(weights, axis=1, keepdims=True)

        z_oos = np.zeros((len(oos_idx), z_core.shape[1]))
        for i in range(len(oos_idx)):
            z_oos[i] = np.average(z_core[nearest_core_indices[i]], axis=0, weights=weights[i])

        # 4. 拼装全局 Z 坐标
        z_full = np.zeros((N, z_core.shape[1]))
        z_full[core_idx] = z_core
        z_full[oos_idx] = z_oos

        # 🌟 严格按照 Eq. (4) 乘回局部伸缩因子 S(x)
        raw_depths = S_x * np.linalg.norm(z_full, axis=1)

        # 🌟 动态计算参考空间的期望范数，完美适配任何维度的几何体
        expected_norm = np.sqrt(np.mean(np.linalg.norm(z_core, axis=1)**2))
        depths = raw_depths / (expected_norm + 1e-9)

        ranks_full = rankdata(depths) / N

        return med_core, ranks_full, z_full

    def _generate_latent_qmc(self, d, N):
        # 🌟 修复维度耦合：生成 d+1 维序列，分离方向与半径的随机源
        sampler = qmc.Halton(d=d+1, scramble=False)
        u_grid = np.clip(sampler.random(n=N), 1e-4, 1 - 1e-4)
        Z = np.zeros((N, d))

        if self.ref_dist == 'uniform':
            if d == 1:
                Z = (u_grid[:, 0:1] * 2) - 1.0
            elif d == 2:
                r = np.sqrt(u_grid[:, 0])
                theta = 2 * np.pi * u_grid[:, 1]
                Z = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            else:
                # 前 d 维用于方向，第 d+1 维用于半径，彻底消除相关性
                Z_dir = norm.ppf(u_grid[:, :d])
                Z_dir /= (np.linalg.norm(Z_dir, axis=1, keepdims=True) + 1e-9)
                radius = u_grid[:, d] ** (1.0 / d)
                Z = Z_dir * radius[:, None]
        elif self.ref_dist == 'gaussian':
            Z = norm.ppf(u_grid[:, :d], loc=0, scale=1.0)
        elif self.ref_dist == 'laplace':
            Z = laplace.ppf(u_grid[:, :d], loc=0, scale=1.0)

        Z[0] = np.zeros(d)
        return Z

    def _fit_predict_core(self, Y, y_dist_m=None, return_scale=False):
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
            final_d = max(1, final_d)  # 强制封顶

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

        # 🌟 提取局部伸缩因子 S(x)
        S_x = np.nanmax(Cy_proc)
        Cy_norm = Cy_proc / (S_x + 1e-9)

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

        # 🌟 严格按照 Eq. (4) 乘回局部伸缩因子 S(x)
        raw_depths = S_x * np.linalg.norm(Y_mapped_to_Z, axis=1)

        # 🌟 动态计算参考空间的期望范数进行标准化
        expected_norm = np.sqrt(np.mean(np.linalg.norm(Z_ref, axis=1) ** 2))
        amqr_depths = raw_depths / (expected_norm + 1e-9)

        # 寻找最小深度及容差范围内的所有候选点
        min_depth = np.min(amqr_depths)
        tolerance = 1e-9
        candidates = np.where(np.abs(amqr_depths - min_depth) < tolerance)[0]

        if len(candidates) == 1:
            med_idx = candidates[0]
        else:
            # 触发 Secondary Fréchet Refinement
            # 利用前文已计算好的流形距离矩阵 Cy (N x N)
            print(f"⚠️ 触发几何平局打破机制！候选点数量: {len(candidates)}")
            # 向量化计算每个候选点到当前邻域内所有点的距离之和
            frechet_sums = np.sum(Cy[candidates, :], axis=1)
            best_local_idx = np.argmin(frechet_sums)
            med_idx = candidates[best_local_idx]

        ranks = rankdata(amqr_depths) / N

        if return_scale:
            return Y[med_idx], ranks, Y_mapped_to_Z, S_x

        return Y[med_idx], ranks, Y_mapped_to_Z

    def fit_predict(self, Y, y_dist_m=None, T=None, t_eval=None, window_size=1.0):
        N = len(Y)

        # ==========================================
        # 分支 A：静态全局流形 (无时间序列)
        # ==========================================
        if T is None:
            # 直接调用带有 OOS 保护的引擎
            med, ranks, _ = self._run_with_oos_protection(Y, y_dist_m)
            return med, ranks

        # ==========================================
        # 分支 B：条件滑动窗口 (动态时空组装)
        # ==========================================
        if t_eval is None:
            t_eval = np.linspace(T.min(), T.max(), 50)

        step_size = t_eval[1] - t_eval[0] if len(t_eval) > 1 else window_size / 5.0

        trajectory_med = []
        final_ranks = np.ones(N)

        # 用于 Procrustes 同步的状态变量
        prev_z = None
        prev_idx = None

        for t_c in tqdm(t_eval):
            idx = np.where(np.abs(T - t_c) <= window_size / 2.0)[0]
            if len(idx) < 15:
                continue

            Y_c = Y[idx]
            y_dist_m_c = y_dist_m[np.ix_(idx, idx)] if y_dist_m is not None else None

            # 🌟 关键修改：用 _run_with_oos_protection 替代 _fit_predict_core
            # 现在，即使局部窗口涌入一万个点，也不会 OOM，系统会自动截取 max_samples 计算
            med_c, ranks_c, z_c = self._run_with_oos_protection(Y=Y_c, y_dist_m=y_dist_m_c)

            # --- 下方的正交普氏同步逻辑 (Procrustes) 完全保持您的原样 ---
            if prev_z is not None:
                common_global_idx, curr_local_idx, prev_local_idx = np.intersect1d(
                    idx, prev_idx, return_indices=True
                )
                if len(common_global_idx) > z_c.shape[1]:
                    R, _ = orthogonal_procrustes(z_c[curr_local_idx], prev_z[prev_local_idx])
                    z_c = z_c @ R

            prev_z = z_c
            prev_idx = idx

            trajectory_med.append((t_c, med_c))

            # Inner-core 切片赋值逻辑
            inner_condition = (T[idx] >= t_c - step_size / 2.0) & (T[idx] < t_c + step_size / 2.0)
            inner_in_idx = np.where(inner_condition)[0]
            if len(inner_in_idx) > 0:
                final_ranks[idx[inner_in_idx]] = ranks_c[inner_in_idx]

        return trajectory_med, final_ranks
