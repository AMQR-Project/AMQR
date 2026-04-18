import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# 接入项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# 🌟 直接调用您统一封装的 Kernel 引擎
from models.kernel_amqr_engine import Kernel_AMQR_Engine


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

            for spine in ax.spines.values():
                spine.set_color('gray')
                spine.set_linewidth(1)

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
    from scipy.spatial.distance import pdist  # 用于计算中位数启发式

    print("========================================================")
    print(" 🌟 Appendix: Implicit RKHS Geometry via Kernel-AMQR")
    print("========================================================")

    # 🚨 锁死随机种子
    np.random.seed(42)

    print("📥 正在加载 MNIST 数据...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    mask = (mnist.target == '8')
    X_digit = mnist.data[mask][:400]

    # 归一化
    X_digit_scaled = X_digit / 255.0
    images = X_digit_scaled.reshape(-1, 28, 28)

    # =====================================================================
    # 🌟 修复 1：使用中位数启发式 (Median Heuristic) 精确锚定 RKHS 流形尺度
    # =====================================================================
    print("⏳ 正在计算 RBF 核的 Median Heuristic...")
    pairwise_sq_dists = pdist(X_digit_scaled, metric='sqeuclidean')
    median_sq_dist = np.median(pairwise_sq_dists)
    # 确保不会除以 0
    gamma_val = 1.0 / (median_sq_dist + 1e-8)
    print(f"   -> 动态计算得到最佳 Gamma: {gamma_val:.6f}")

    print("⏳ 正在运行统一版 Kernel AMQR 引擎 (Pathway C)...")

    # =====================================================================
    # 🌟 修复 2：提升本征维度 d_int，避免 RKHS 中的高维特征被压扁
    # =====================================================================
    engine = Kernel_AMQR_Engine(
        kernel='rbf',
        gamma=gamma_val,
        d_int=2,
        epsilon=0.0,  # Exact GW 零熵模糊
        use_log_squash=False
    )

    med, ranks = engine.fit_predict(X_digit_scaled)

    # 输出图片到指定统一结果目录
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "figures"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig_Appendix_Kernel_MNIST.pdf")
    plot_3x5_grid(images, ranks, save_path)
