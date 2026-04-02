import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.metrics.pairwise import pairwise_kernels
import ot


# =====================================================================
# 👑 Kernel AMQR Engine (MLE 自适应高维球体版)
# =====================================================================
class Kernel_AMQR_Engine_MLE:
    def __init__(self, kernel='rbf', gamma=None):
        self.kernel = kernel
        self.gamma = gamma

    def _generate_latent_hyperball(self, d, N):
        """生成目标空间的 d 维均匀超球体 (Uniform Hyperball)"""
        # 1. 在 d 维标准正态分布中采样，得到方向向量
        points = np.random.normal(0, 1, (N, d))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # 2. 映射到球壳上
        directions = points / norms
        # 3. 均匀半径采样 (d 维球的体积与 r^d 成正比)
        radii = np.random.uniform(0, 1, (N, 1)) ** (1.0 / d)
        return directions * radii

    def fit_predict(self, Y):
        N = len(Y)
        Y_flat = Y.reshape(N, -1)

        # 1. 计算核矩阵 K
        K = pairwise_kernels(Y_flat, Y_flat, metric=self.kernel, gamma=self.gamma)

        # 2. 从无限维 RKHS 空间中提取纯正流形距离 Cy
        K_diag = np.diag(K)
        Cy_sq = K_diag[:, None] + K_diag[None, :] - 2 * K
        Cy_sq = np.maximum(Cy_sq, 0)
        Cy = np.sqrt(Cy_sq)

        # ==========================================
        # 🌟 3. MLE 本征维数估计 (直接利用 Cy 距离)
        # ==========================================
        k_mle = min(10, N - 1)
        # 对每行距离排序，取前 k_mle 个最近邻
        dists = np.sort(Cy, axis=1)[:, 1:k_mle + 2]
        dists = np.maximum(dists, 1e-9)
        r_k = dists[:, -1:]

        # MLE 公式
        mle_val = (k_mle - 1) / np.sum(np.log(r_k / dists[:, :-1]), axis=1)
        final_d = int(np.round(np.mean(mle_val)))
        final_d = max(1, min(final_d, 20))  # 设定合理的最高维度上限，防止计算爆炸
        print(f"🌟 [MLE] 引擎自动推断出本征维数为: {final_d} 维！")

        # 4. 目标 d 维高维球构建
        Z_ref = self._generate_latent_hyperball(final_d, N)
        Cz = cdist(Z_ref, Z_ref, metric='euclidean')

        Cz_norm = Cz / (Cz.max() + 1e-9)
        Cy_norm = Cy / (np.nanmax(Cy) + 1e-9)

        # 注入对称破缺噪声
        noise_z = np.random.uniform(0, 1e-8, size=Cz_norm.shape)
        Cz_norm += (noise_z + noise_z.T) / 2.0
        np.fill_diagonal(Cz_norm, 0)

        noise_y = np.random.uniform(0, 1e-8, size=Cy_norm.shape)
        Cy_norm += (noise_y + noise_y.T) / 2.0
        np.fill_diagonal(Cy_norm, 0)

        # 5. GW 最优传输映射
        py, pz = ot.unif(N), ot.unif(N)
        gw_plan = ot.gromov.gromov_wasserstein(
            Cz_norm, Cy_norm, pz, py, 'square_loss', max_iter=50, tol=1e-4)

        # 6. 计算分位数 (高维球体内的欧氏范数)
        Y_mapped_to_Z = np.dot(gw_plan.T, Z_ref) * N
        amqr_depths = np.linalg.norm(Y_mapped_to_Z, axis=1)

        ranks = rankdata(amqr_depths) / N

        return ranks, final_d


# =====================================================================
# 🎨 极简切片图 (1x3 画布)
# =====================================================================
def plot_3x5_grid(images, ranks, filename="Quantile_Images_3x5.png"):
    print("🎨 正在渲染 3x5 分位数矩阵切片图...")
    fig, axes = plt.subplots(3, 5, figsize=(15, 9), facecolor='white')

    target_quantiles = [0.1, 0.5, 0.9]
    row_titles = ['u ≈ 0.1\n(Core / Healthy)', 'u ≈ 0.5\n(Normal Edge)', 'u ≈ 0.9\n(Anomaly)']

    for i, q in enumerate(target_quantiles):
        # 找到距离目标分位数最近的 5 个样本
        idx_sorted = np.argsort(np.abs(ranks - q))
        selected_idx = idx_sorted[:5]

        # 设置行标题
        axes[i, 0].set_ylabel(row_titles[i], fontsize=16, fontweight='bold',
                              rotation=0, labelpad=80, ha='center', va='center')

        for j, idx in enumerate(selected_idx):
            ax = axes[i, j]
            img_inverted = 1.0 - images[idx]  # 反色处理，白底黑字更清晰

            ax.imshow(img_inverted, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

            # 画一个浅色的边框区分不同图片
            for spine in ax.spines.values():
                spine.set_color('gray')
                spine.set_linewidth(1)

            # 在底部标注该样本准确的 AMQR 深度得分
            ax.text(0.5, -0.15, f"u = {ranks[idx]:.3f}",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, color='darkred', fontweight='bold')

    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    plt.suptitle("Kernel-AMQR: 5 Nearest Samples per Target Quantile",
                 fontsize=22, fontweight='bold', y=0.98)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 渲染完毕！请查看图片: {filename}")


# =====================================================================
# 🚀 主执行流
# =====================================================================
if __name__ == "__main__":
    print("📥 正在加载 MNIST 数据...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    mask = (mnist.target == '8')
    X_digit = mnist.data[mask][:400]

    # 归一化，这对 RBF 核极其重要！
    X_digit_scaled = X_digit / 255.0
    images = X_digit_scaled.reshape(-1, 28, 28)

    print("⏳ 正在运行带 MLE 的 Kernel AMQR 引擎...")
    gamma_val = 1.0 / X_digit_scaled.shape[1]
    engine = Kernel_AMQR_Engine_MLE(kernel='rbf', gamma=gamma_val)

    ranks, estimated_d = engine.fit_predict(X_digit_scaled)

    plot_3x5_grid(images, ranks, "Quantile_Images_3x5.jpg")