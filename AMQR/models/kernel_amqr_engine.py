import ot
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.metrics.pairwise import pairwise_kernels

class Kernel_AMQR_Engine:
    """
    Kernel Auto-conditioned Manifold Quantile Regression (Kernel-AMQR)
    利用核技巧 (Kernel Trick) 将数据隐式映射到无限维 RKHS 空间中，
    在完美的平坦空间内提取流形距离，再通过 GW 映射到目标圆盘。
    """

    def __init__(self, ref_dist='uniform', epsilon=0.0, d_int=2,
                 kernel='rbf', gamma=None, degree=3, coef0=1,
                 use_log_squash=False):
        """
        :param kernel: 'rbf', 'poly', 'linear', 或者 'precomputed' (直接传入核矩阵)
        :param gamma: RBF 或 Poly 核的带宽参数 (None 则自动按 1/n_features 推断)
        """
        self.ref_dist = ref_dist.lower()
        self.epsilon = epsilon
        self.d_int = d_int  # 核空间通常无限维，目标空间 d_int 建议强制指定 (如画靶心图用 2)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.use_log_squash = use_log_squash

    def _generate_latent_qmc(self, d, N):
        """生成目标空间的 QMC 均匀圆盘点 (沿用你原版引擎的优秀设计)"""
        from scipy.stats import qmc
        sampler = qmc.Halton(d=d, scramble=True)
        sample = sampler.random(n=N)
        if self.ref_dist == 'uniform':
            if d == 2:
                r = np.sqrt(sample[:, 0])
                theta = 2 * np.pi * sample[:, 1]
                return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            return sample * 2 - 1
        else:
            from scipy.stats import norm
            return norm.ppf(sample)

    def fit_predict(self, Y):
        """
        直接输入原始特征 Y，或者在 kernel='precomputed' 时输入核矩阵 K
        """
        N = len(Y)
        Y_flat = Y.reshape(N, -1) if self.kernel != 'precomputed' else Y

        # =========================================================
        # 1. 计算核矩阵 K (Kernel Matrix)
        # =========================================================
        if self.kernel == 'precomputed':
            K = Y_flat.copy()
        else:
            # 动态组装当前核函数支持的参数字典
            kernel_kwargs = {}
            if self.gamma is not None:
                kernel_kwargs['gamma'] = self.gamma
            if self.kernel == 'poly':
                kernel_kwargs['degree'] = self.degree
                kernel_kwargs['coef0'] = self.coef0
            elif self.kernel == 'sigmoid':
                kernel_kwargs['coef0'] = self.coef0

            K = pairwise_kernels(Y_flat, Y_flat, metric=self.kernel, **kernel_kwargs)

        # =========================================================
        # 2. 从核矩阵提取 RKHS 空间中的纯正欧氏距离 Cy
        # D^2(x, y) = K(x,x) + K(y,y) - 2K(x,y)
        # =========================================================
        K_diag = np.diag(K)
        # 巧妙利用广播机制计算距离平方矩阵
        Cy_sq = K_diag[:, None] + K_diag[None, :] - 2 * K
        Cy_sq = np.maximum(Cy_sq, 0)  # 防止浮点数误差导致微小负数
        Cy = np.sqrt(Cy_sq)

        # =========================================================
        # 3. 目标圆盘构建与 GW 映射 (沿用 AMQR 逻辑)
        # =========================================================
        Z_ref = self._generate_latent_qmc(self.d_int, N)
        Cz = cdist(Z_ref, Z_ref, metric='euclidean')

        Cz_norm = Cz / (Cz.max() + 1e-9)
        Cy_proc = np.log1p(Cy) if self.use_log_squash else Cy
        Cy_norm = Cy_proc / (np.nanmax(Cy_proc) + 1e-9)

        # 注入对称破缺噪声
        noise_z = np.random.uniform(0, 1e-8, size=Cz_norm.shape)
        Cz_norm += (noise_z + noise_z.T) / 2.0
        np.fill_diagonal(Cz_norm, 0)

        noise_y = np.random.uniform(0, 1e-8, size=Cy_norm.shape)
        Cy_norm += (noise_y + noise_y.T) / 2.0
        np.fill_diagonal(Cy_norm, 0)

        # 求解 GW
        py, pz = ot.unif(N), ot.unif(N)
        if self.epsilon > 0:
            gw_plan = ot.gromov.entropic_gromov_wasserstein(
                Cz_norm, Cy_norm, pz, py, 'square_loss', epsilon=self.epsilon, max_iter=100)
        else:
            gw_plan = ot.gromov.gromov_wasserstein(
                Cz_norm, Cy_norm, pz, py, 'square_loss', max_iter=50, tol=1e-4)

        # =========================================================
        # 4. 提取深度与排名
        # =========================================================
        Y_mapped_to_Z = np.dot(gw_plan.T, Z_ref) * N
        amqr_depths = np.linalg.norm(Y_mapped_to_Z, axis=1)
        med_idx = np.argmin(amqr_depths)
        ranks = rankdata(amqr_depths) / N

        # 注意：如果输入的是 precomputed 核矩阵，Y[med_idx] 返回的是核矩阵的某一行
        return Y[med_idx], ranks
