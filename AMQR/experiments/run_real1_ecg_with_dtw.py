import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.signal import find_peaks
from scipy.stats import rankdata

try:
    from scipy.datasets import electrocardiogram
except ImportError:
    from scipy.misc import electrocardiogram

try:
    from tslearn.metrics import cdist_dtw
except ImportError:
    print("⚠️ 致命错误: 未检测到 tslearn 库。请运行 'pip install tslearn'。")
    sys.exit(1)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from models.amqr_engine import AMQR_Engine


def plot_single_panel(ax, X_grid, Y_curves, ranks, center_curve, nw_mean, title, center_label, rank_label):
    """绘制单个面板的辅助函数"""
    cmap = cm.get_cmap('viridis_r')
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # 渐变染色
    for i, curve in enumerate(Y_curves):
        r = ranks[i]
        color = cmap(norm(r))
        alpha_val = max(0.15, 0.8 - 0.7 * r)
        ax.plot(X_grid, curve, color=color, alpha=alpha_val, linewidth=1.0, zorder=1)

    # NW 均值
    ax.plot(X_grid, nw_mean, color='blue', linewidth=3, linestyle='--',
            label='NW Mean (Amplitude Dampening)', zorder=3)

    # 中心曲线
    ax.plot(X_grid, center_curve, color='crimson', linewidth=4, label=center_label, zorder=5)

    # 完善图表元素
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (mV)")
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    return cmap, norm


if __name__ == "__main__":
    np.random.seed(42)

    print("========================================================")
    print(" 🌟 Real Application 2: AMQR vs. DTW (Dual Panel)")
    print("========================================================")

    # 1. 提取并切割 ECG 数据
    ecg_signal = electrocardiogram()
    fs = 360
    signal_segment = ecg_signal[:60 * fs]
    peaks, _ = find_peaks(signal_segment, height=1.0, distance=fs // 2)

    window = int(0.25 * fs)
    Y_curves = []
    for p in peaks:
        if p - window > 0 and p + window < len(signal_segment):
            shift = np.random.randint(-15, 15)
            heartbeat = signal_segment[p - window + shift: p + window + shift]
            Y_curves.append(heartbeat)

    Y_curves = np.array(Y_curves)
    X_grid = np.linspace(-0.25, 0.25, Y_curves.shape[1])
    N = len(Y_curves)
    nw_mean = np.mean(Y_curves, axis=0)

    # 2. 计算 DTW 基线
    print("⏳ 计算 DTW 距离矩阵与 Medoid...")
    dtw_matrix = cdist_dtw(Y_curves)
    dtw_medoid_idx = np.argmin(dtw_matrix.sum(axis=1))
    dtw_medoid = Y_curves[dtw_medoid_idx]
    dtw_ranks = rankdata(dtw_matrix[dtw_medoid_idx]) / N

    # 3. 计算 AMQR 方法
    print("⏳ 执行 AMQR 高维流形拓扑对齐...")
    amqr = AMQR_Engine(ref_dist='uniform', use_knn=True, k_neighbors=5, d_int=None, epsilon=0.01)
    amqr_median, amqr_ranks = amqr.fit_predict(Y_curves, T=None)

    # 4. 创建上下双面板图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    # 绘制 Panel A (DTW)
    cmap1, norm1 = plot_single_panel(
        axes[0], X_grid, Y_curves, dtw_ranks, dtw_medoid, nw_mean,
        title="(A) Baseline: ECG Center-Outward Ranking via DTW Fréchet Depth",
        center_label="DTW Medoid (Empirical Selection)",
        rank_label="DTW Distance Rank ($u_{DTW}$)"
    )
    sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
    sm1.set_array([])
    cbar1 = fig.colorbar(sm1, ax=axes[0], pad=0.01)
    cbar1.set_label('DTW Distance Rank ($u_{DTW}$)', rotation=270, labelpad=15, fontweight='bold')

    # 绘制 Panel B (AMQR)
    cmap2, norm2 = plot_single_panel(
        axes[1], X_grid, Y_curves, amqr_ranks, amqr_median, nw_mean,
        title="(B) Proposed: ECG Template Extraction via AMQR Topological Depth",
        center_label="AMQR Topological Median (Barycentric Projection)",
        rank_label="AMQR Topological Quantile Rank ($u$)"
    )
    sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, ax=axes[1], pad=0.01)
    cbar2.set_label('AMQR Topological Quantile Rank ($u$)', rotation=270, labelpad=15, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "figures"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig8_Dual_Comparison.jpg")
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"🎉 双面板图表已生成: {save_path}")