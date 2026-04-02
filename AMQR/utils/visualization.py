# utils/visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import rankdata
from scipy.linalg import logm
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from utils.metrics import compute_spd_all_props


def plot_combined_motivation_1x4(data_dict, save_path=None):
    """
    绘制 1x4 终极动机对比图 (Straight vs Bent / Frechet vs AMQR)
    """
    fig, axes = plt.subplots(1, 4, figsize=(32, 8), facecolor='white')
    cmap_choice = 'plasma'
    scatter_s, scatter_a, scatter_lw = 45, 0.95, 0.2

    plots_data = [
        (axes[0], data_dict['straight']['Y'], data_dict['straight']['f_med'], data_dict['straight']['f_ranks'],
         "Linear Anisotropy\nFréchet $L^1$ Median", '#c0392b', 'X', "Fréchet Median", [-12, 12]),
        (axes[1], data_dict['straight']['Y'], data_dict['straight']['a_med'], data_dict['straight']['a_ranks'],
         "Linear Anisotropy\nProposed AMQR Model", '#2980b9', '*', "AMQR Median", [-12, 12]),
        (axes[2], data_dict['bent']['Y'], data_dict['bent']['f_med'], data_dict['bent']['f_ranks'],
         "Non-linear Manifold\nFréchet $L^1$ Median", '#c0392b', 'X', "Fréchet Median", [-14, 14]),
        (axes[3], data_dict['bent']['Y'], data_dict['bent']['a_med'], data_dict['bent']['a_ranks'],
         "Non-linear Manifold\nProposed AMQR Model", '#2980b9', '*', "AMQR Median", [-14, 14])
    ]

    last_scatter = None
    legend_handles, legend_labels = [], []

    for ax, Y, med, ranks, title, color, marker, label, lims in plots_data:
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=ranks, cmap=cmap_choice,
                        s=scatter_s, alpha=scatter_a, edgecolor='white', lw=scatter_lw, zorder=1)
        last_scatter = sc

        marker_size = 500 if marker == 'X' else 650
        med_scatter = ax.scatter(med[0], med[1], color='#00FF00', s=marker_size, marker=marker,
                                 edgecolor='black', lw=2, zorder=5, label=label)

        if label not in legend_labels:
            legend_labels.append(label)
            legend_handles.append(med_scatter)

        ax.set_title(title, fontsize=22, fontweight='bold', color=color, pad=20)
        ax.set_aspect('equal')
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[0], lims[1])
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.axvline(0, color='gray', lw=0.5, ls='--')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)

    plt.subplots_adjust(left=0.02, right=0.86, wspace=0.1)

    fig.legend(legend_handles, legend_labels, loc='lower left', bbox_to_anchor=(0.87, 0.65),
               fontsize=16, framealpha=1.0, facecolor='white', edgecolor='black', title="Center Markers",
               title_fontsize=18)

    cbar_ax = fig.add_axes([0.87, 0.18, 0.015, 0.45])
    cbar = fig.colorbar(last_scatter, cax=cbar_ax, orientation='vertical')
    cbar.set_label("Quantile Percentile\n(Depth: 0% to 100%)", fontsize=16, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=14)

    if save_path:
        # 🌟 自动创建缺失的文件夹，防止报错
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 图片已成功保存至: {save_path}")

    plt.show()


def plot_2x4_spiral_experiment(T_sorted, P_sorted, GT_3D, t_traj, centers, masks, target_t=5.0, save_path=None):
    """
    渲染 3D 穿孔流形追踪的 2x4 终极对比图 (上层: 3D宏观, 下层: 2D截面微观)
    """
    # 3D 轨迹平滑
    traj_3d = {k: gaussian_filter1d(v, sigma=3, axis=0) for k, v in centers.items()}

    fig = plt.figure(figsize=(32, 16))
    try: plt.style.use('seaborn-v0_8-darkgrid')
    except: pass

    methods = ['nw', 'f', 'kde', 'a']
    colors_core = ['#e74c3c', '#e67e22', '#9b59b6', '#27ae60']
    colors_traj = ['#c0392b', '#d35400', '#8e44ad', '#1e8449']

    titles_3d = [
        "NW Euclidean Mean (3D View)\nPulled into the hollow void",
        "Fréchet L1 Median (3D View)\nIsotropic distance misguides the core",
        "SW-KDE Density Mode (3D View)\nDensity fragments around the gap",
        "AMQR Complete Model (3D View)\nFlawless manifold tracking across the gap"
    ]

    titles_2d = [
        f"Cross-Section @ X={target_t} (NW)\nMean falls into vacuum",
        f"Cross-Section @ X={target_t} (Fréchet)\n20% tube cuts through empty space",
        f"Cross-Section @ X={target_t} (KDE)\nGap shatters the continuous tube",
        f"Cross-Section @ X={target_t} (AMQR)\nOT maps correctly over the 135° gap"
    ]

    # --- 上半部分：宏观 3D 图 ---
    for i, method in enumerate(methods):
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        ax.scatter(P_sorted[~masks[method], 0], P_sorted[~masks[method], 1], P_sorted[~masks[method], 2],
                   color='#d5d5d5', s=2, alpha=0.1)
        ax.scatter(P_sorted[masks[method], 0], P_sorted[masks[method], 1], P_sorted[masks[method], 2],
                   color=colors_core[i], s=10, alpha=0.6, label="Top 20% Mask")
        ax.plot(traj_3d[method][:, 0], traj_3d[method][:, 1], traj_3d[method][:, 2], color=colors_traj[i], lw=5,
                label="Estimated Trajectory")
        ax.plot(GT_3D[:, 0], GT_3D[:, 1], GT_3D[:, 2], color='black', ls='--', lw=3, label="Ground Truth Core")

        ax.set_title(titles_3d[i], fontsize=18, fontweight='bold', pad=15)
        ax.set_xlim(0, 10); ax.set_ylim(-4, 4); ax.set_zlim(-4, 4)
        ax.view_init(elev=20, azim=-45)
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        if i == 0: ax.legend(loc='upper right', fontsize=14)

    # --- 下半部分：微观 2D 截面染色图 ---
    window = 0.3
    idx_slice = np.where(np.abs(T_sorted - target_t) <= window)[0]
    idx_t = np.argmin(np.abs(t_traj - target_t))
    gt_idx = np.argmin(np.abs(GT_3D[:, 0] - target_t))

    for i, method in enumerate(methods):
        ax = fig.add_subplot(2, 4, i + 5)
        ax.scatter(P_sorted[idx_slice, 1], P_sorted[idx_slice, 2], color='#d5d5d5', s=25, alpha=0.6,
                   label="Raw Slice (with 135° Gap)")

        mask_slice = masks[method][idx_slice]
        ax.scatter(P_sorted[idx_slice][mask_slice, 1], P_sorted[idx_slice][mask_slice, 2], color=colors_core[i], s=70,
                   alpha=0.9, edgecolor='white', label="Top 20% Mask")

        exact_center = centers[method][idx_t]
        ax.scatter(exact_center[1], exact_center[2], marker='X', color=colors_traj[i], s=500, edgecolor='black', lw=2,
                   label="Estimated Center")
        ax.scatter(GT_3D[gt_idx, 1], GT_3D[gt_idx, 2], marker='*', color='gold', s=700, edgecolor='black', lw=1.5,
                   label="Ground Truth Mode")

        circle = plt.Circle((0, 0), 3.0, color='blue', fill=False, linestyle=':', linewidth=2, alpha=0.4)
        ax.add_patch(circle)

        ax.set_title(titles_2d[i], fontsize=18, fontweight='bold')
        ax.set_aspect('equal'); ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
        if i == 0: ax.legend(loc='lower right', fontsize=13)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 图片已成功保存至: {save_path}")
    plt.show()


def plot_bimodal_crescent_1x4(Y, meds, ranks, save_path=None):
    """
    绘制双峰月牙流形的 1x4 终极对比图
    meds 和 ranks 是包含 4 种方法结果的列表
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), facecolor='white')
    titles = [
        "NW Mean",
        "Fréchet L1 Median",
        "KDE Density Mode",
        "AMQR (Proposed)"
    ]

    for i in range(4):
        ax = axes[i]
        # 画全量数据，颜色表示 Depth (黄色为外围，深色为核心)
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=ranks[i], cmap='viridis_r', s=30, alpha=0.5)

        # 标出前 20% 的核心管
        mask_20 = ranks[i] <= 0.20
        ax.scatter(Y[mask_20, 0], Y[mask_20, 1], color='red', s=40, alpha=0.9, label='Top 20% Quantile')

        # 标出计算出的中心点
        ax.scatter(meds[i][0], meds[i][1], marker='*', color='#00FF00', s=800, edgecolor='black', lw=2,
                   label='Estimated Center')

        ax.set_title(titles[i], fontsize=16, fontweight='bold', pad=15)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-2, 6)
        ax.set_aspect('equal')
        ax.set_xticks([]);
        ax.set_yticks([])

        if i == 0:
            ax.legend(loc='lower center', fontsize=12)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 双峰月牙对比图已保存至: {save_path}")
    plt.show()


def plot_dynamic_functional_2x4(T, X_grid, Y, t_eval, GT_surface, centers_dict, target_t=7.5, save_path=None):
    fig = plt.figure(figsize=(32, 14), facecolor='white')
    methods = ['nw', 'f', 'kde', 'a']
    titles_global = ["NW Mean Surface", "Fréchet L1 Surface", "SW-KDE Surface", "AMQR Surface (Proposed)"]
    titles_cross = ["Cross-section (NW)", "Cross-section (Fréchet)", "Cross-section (KDE)", "Cross-section (AMQR)"]
    colors = ['#c0392b', '#d35400', '#8e44ad', '#27ae60']

    # ==========================================
    # 上半场：全局时空热力图 (Spatiotemporal Heatmap)
    # ==========================================
    for i, method in enumerate(methods):
        ax = fig.add_subplot(2, 4, i + 1)
        est_surface = np.array(centers_dict[method])

        # 绘制热力图
        c = ax.pcolormesh(t_eval, X_grid, est_surface.T, cmap='viridis', shading='auto', vmin=0, vmax=2.5)

        # 叠加真实的相位轨迹线 (Ground Truth Phase)
        true_phase = 2.0 * np.sin(t_eval * np.pi / 5.0)
        ax.plot(t_eval, true_phase, color='red', linestyle='--', lw=3, label="True Phase")

        # 标出截面位置
        ax.axvline(target_t, color='white', linestyle=':', lw=2, label=f"Slice @ T={target_t}")

        ax.set_title(titles_global[i], fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel("Time (T)", fontsize=14)
        ax.set_ylabel("Functional Domain (X)", fontsize=14)
        if i == 0: ax.legend(loc='upper right', fontsize=12)
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04).set_label("Amplitude", fontsize=12)

    # ==========================================
    # 下半场：截面曲线对比图 (Cross-sectional Slice)
    # ==========================================
    # 提取截面周边的数据点
    window = 1.0
    idx_slice = np.where(np.abs(T - target_t) <= window)[0]
    Y_slice = Y[idx_slice]

    # 找到最接近 target_t 的时间戳索引
    t_idx = np.argmin(np.abs(t_eval - target_t))

    for i, method in enumerate(methods):
        ax = fig.add_subplot(2, 4, i + 5)

        # 底层：原始截面数据
        ax.plot(X_grid, Y_slice.T, color='#bdc3c7', alpha=0.15, lw=1, zorder=1)

        # 顶层：估算出的中心曲线
        est_curve = centers_dict[method][t_idx]
        ax.plot(X_grid, est_curve, color=colors[i], lw=4, zorder=5, label='Estimated Center')

        # 真实曲线
        gt_curve = GT_surface[t_idx]
        ax.plot(X_grid, gt_curve, color='black', linestyle='--', lw=3, zorder=6, label='Ground Truth')

        ax.set_title(titles_cross[i], fontsize=20, fontweight='bold', color=colors[i], pad=15)
        ax.set_xlim(-5, 5);
        ax.set_ylim(-0.5, 3.5)
        ax.set_xticks([]);
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_linewidth(1.5); spine.set_color('#7f8c8d')
        if i == 0: ax.legend(loc='upper right', fontsize=14)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 动态泛函削峰图已保存至: {save_path}")
    plt.show()


def plot_spd_3x4_comparison(Y_flat, results_dict, dim=2, R_true=None, filename=None):
    fig, axes = plt.subplots(3, 4, figsize=(28, 18), facecolor='white')
    raw_dets, raw_eigs, raw_log_Y = compute_spd_all_props(Y_flat, dim=dim)
    pca = PCA(n_components=2)
    Y_pca = pca.fit_transform(raw_log_Y)

    # ==========================================
    # 🌟 新增：在循环外，一次性计算 Ground Truth 的所有属性和坐标
    # ==========================================
    if R_true is not None:
        true_det, true_eig, true_log = compute_spd_all_props(np.array([R_true]), dim=dim)
        true_pca = pca.transform(true_log)[0]  # 利用已经 fit 好的 PCA 直接投影！
    # ==========================================

    col_titles = ["NW (L2 Mean)", "Fréchet (L1 Median)", "SW-KDE (Mode)", "Proposed AMQR (OT)"]
    row_titles = [f"1. Determinant (Volume)", "2. Eigen Spectrum (Max vs Min)", "3. Manifold Embedding"]
    cmap_choice = 'plasma'
    u_target = 0.20

    for j, m_key in enumerate(['nw', 'frechet', 'kde', 'amqr']):
        med = results_dict[m_key]['med']
        ranks = results_dict[m_key]['ranks']
        m_det, m_eig, m_log = compute_spd_all_props(np.array([med]), dim=dim)
        m_pca = pca.transform(m_log)[0]
        mask = ranks <= u_target

        m_color = '#c0392b' if j < 2 else ('#8e44ad' if j == 2 else '#27ae60')
        symbol = '*' if m_key == 'amqr' else 'X'
        size = 800 if m_key == 'amqr' else 500

        # --- Row 1: Determinant Histogram ---
        ax1 = axes[0, j]

        # 🌟 1. 设置“微距镜头”的 X 轴最大值 (比如 2.5，足以看清 0.5 附近的核心分布)
        zoom_max = 2.5
        # 重新划分这 2.5 范围内的精细柱子
        custom_bins = np.linspace(0, zoom_max, 50)

        # 🌟 2. 用 np.clip 把大体积的离群点强行“拍”在 2.5 的边界上，不让它们撑大 X 轴
        clipped_raw = np.clip(raw_dets, 0, zoom_max)
        clipped_mask = np.clip(raw_dets[mask], 0, zoom_max)

        # 3. 画灰色背景
        ax1.hist(clipped_raw, bins=custom_bins, color='#bdc3c7', alpha=0.4, edgecolor='white', zorder=1)
        # 4. 画彩色核心
        ax1.hist(clipped_mask, bins=custom_bins, color=m_color, alpha=0.8, edgecolor='white', zorder=2,
                 label="Top 20% Core")

        # 5. 真实的体积线 (半透明金柱)
        if R_true is not None:
            ax1.axvline(true_det[0], color='gold', lw=6, ls='-', alpha=0.7, zorder=3, label="True Det")

        # 6. 估计的体积线 (黑色虚线)
        ax1.axvline(m_det[0], color='black', lw=2.5, ls='--', zorder=4, label=f"Det: {m_det[0]:.2f}")

        # 🌟 7. 锁死 X 轴的显示范围！
        ax1.set_xlim(0, zoom_max)

        ax1.set_title(col_titles[j], fontsize=20, fontweight='bold', pad=15, color=m_color)
        if j == 0: ax1.set_ylabel(row_titles[0], fontsize=16, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=12)

        # --- Row 2: Eigenvalue Spectrum (Max Eig vs Min Eig) ---
        ax2 = axes[1, j]
        sc = ax2.scatter(raw_eigs[:, -1], raw_eigs[:, 0], c=ranks, cmap=cmap_choice, s=50, alpha=0.7, edgecolor='w',
                         lw=0.2)

        # 🌟 新增：在特征值图上画出真实的特征值位置 (巨大金星垫底)
        if R_true is not None:
            ax2.scatter(true_eig[0, -1], true_eig[0, 0], marker='*', color='gold', s=1000, edgecolor='black', lw=1.5,
                        zorder=4)

        ax2.scatter(m_eig[0, -1], m_eig[0, 0], marker=symbol, color='#00FF00', s=size, edgecolor='black', lw=2,
                    zorder=5)
        ax2.set_xlim(np.min(raw_eigs[:, -1]) - 1, np.max(raw_eigs[:, -1]) + 1)
        ax2.set_ylim(np.min(raw_eigs[:, 0]) - 0.2, np.max(raw_eigs[:, 0]) + 0.5)
        if j == 0: ax2.set_ylabel(row_titles[1], fontsize=16, fontweight='bold')

        # --- Row 3: Log-Euclidean PCA ---
        ax3 = axes[2, j]
        ax3.scatter(Y_pca[:, 0], Y_pca[:, 1], c=ranks, cmap=cmap_choice, s=50, alpha=0.7, edgecolor='w', lw=0.2)

        # 🌟 新增：在 PCA 降维图上画出真实坐标 (巨大金星垫底)
        if R_true is not None:
            ax3.scatter(true_pca[0], true_pca[1], marker='*', color='gold', s=1000, edgecolor='black', lw=1.5, zorder=4)

        ax3.scatter(m_pca[0], m_pca[1], marker=symbol, color='#00FF00', s=size, edgecolor='black', lw=2, zorder=5)
        if j == 0: ax3.set_ylabel(row_titles[2], fontsize=16, fontweight='bold')

    for ax in axes.flat:
        ax.set_xticks([]);
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_color('#bdc3c7')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax).set_label("Quantile Percentile Depth", fontsize=16, fontweight='bold')

    plt.subplots_adjust(left=0.05, right=0.90, wspace=0.1, hspace=0.15)
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ 高维 SPD 评测图已保存至: {filename}")
    plt.show()


def compute_traces(Y_flat, dim=20):
    """计算矩阵的迹，代表总体波动体积"""
    return np.array([np.trace(y.reshape(dim, dim)) for y in Y_flat])


def plot_traffic_tube_validation(T, Y_flat, t_sparse, Y_reg_sparse, a_ranks, save_dir, dim=20):
    """绘制 1x3 的交通流形管道演化与 PCA 拓扑图"""
    print("🎨 正在渲染 1x3 交通流形管道演化图...")
    fig, axes = plt.subplots(1, 3, figsize=(28, 8), facecolor='white')
    cmap_choice = 'plasma'

    raw_traces = compute_traces(Y_flat, dim)
    reg_traces_sparse = compute_traces(Y_reg_sparse, dim)

    # --- 视图 A: 交通流形渐变管道 ---
    ax1 = axes[0]
    jitter = np.random.uniform(-0.35, 0.35, len(T))
    T_jitter = T + jitter

    # 排序以防止高分异常点被掩盖
    sort_idx = np.argsort(a_ranks)
    T_jitter_sorted, raw_traces_sorted, a_ranks_sorted = T_jitter[sort_idx], raw_traces[sort_idx], a_ranks[sort_idx]

    sc_tube = ax1.scatter(T_jitter_sorted, raw_traces_sorted, c=a_ranks_sorted,
                          cmap=cmap_choice, s=30, alpha=0.85, zorder=3, vmin=0.0, vmax=1.0, edgecolor='none')

    ax1.plot(t_sparse, reg_traces_sparse, color='#00FF00', lw=4, zorder=5, label='AMQR Healthy Quantile Tube Center')
    ax1.scatter(t_sparse, reg_traces_sparse, color='#00FF00', s=80, edgecolor='black', lw=1.5, zorder=6)

    ax1.set_title("1. Time-of-Day Traffic Manifold Tube", fontsize=18, fontweight='bold', pad=15)
    ax1.set_xlabel("Hour of Day (0:00 - 23:59)", fontsize=14)
    ax1.set_ylabel("Network Volatility (Trace, Log Scale)", fontsize=14)
    ax1.set_yscale('log')
    ax1.set_xticks(np.arange(0, 25, 4))
    ax1.legend(loc='upper left', fontsize=12)

    # --- 视图 B: 切空间 PCA ---
    ax2 = axes[1]
    Y_log_flat = np.array([logm(y.reshape(dim, dim) + np.eye(dim) * 1e-5).real.flatten() for y in Y_flat])
    Y_pca = PCA(n_components=2).fit_transform(Y_log_flat)
    Y_pca_sorted = Y_pca[sort_idx]

    sc_pca = ax2.scatter(Y_pca_sorted[:, 0], Y_pca_sorted[:, 1], c=a_ranks_sorted,
                         cmap=cmap_choice, s=40, alpha=0.8, edgecolor='none', vmin=0.0, vmax=1.0)
    ax2.set_title("2. Tangent Space PCA Topology", fontsize=18, fontweight='bold', pad=15)

    # --- 视图 C: 箱线图 ---
    ax3 = axes[2]
    labels = ['Core\n(0-20%)', 'Low\n(20-40%)', 'Mid\n(40-60%)', 'High\n(60-80%)', 'Anomaly\n(80-100%)']
    rank_categories = pd.cut(a_ranks, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=labels, include_lowest=True)
    sns.boxplot(x=rank_categories, y=np.log10(raw_traces), ax=ax3, palette='plasma_r', width=0.6, fliersize=3)
    ax3.set_title("3. Unsupervised Traffic Anomaly Isolation", fontsize=18, fontweight='bold', pad=15)
    ax3.set_ylabel("Raw Network Volatility ($\log_{10}$ Trace)", fontsize=14)

    cbar_ax = fig.add_axes([0.62, 0.15, 0.012, 0.7])
    fig.colorbar(sc_pca, cax=cbar_ax).set_label("AMQR Anomaly Quantile", fontsize=16, fontweight='bold')

    plt.subplots_adjust(left=0.06, right=0.95, wspace=0.3)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "Fig_1x3_Traffic_Tube.jpg")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 1x3 管道图谱已保存至: {save_path}")


def plot_spatial_grid(Y_flat, scores, T, coords, save_dir, dim=20):
    """
    绘制 3x4 地理拓扑空间演化网格图 (局部条件分位数版)
    注意：这里传入的是绝对异常距离 scores，而不是全局 a_ranks！
    """
    print("🎨 正在渲染 3x4 地理空间拓扑演化大图 (Local Quantiles)...")
    target_hours = [6, 12, 18, 0]
    col_labels = ["6:00\n(Morning Wake-up)", "12:00\n(Noon Steady)", "18:00\n(Evening Rush)", "0:00\n(Midnight Sleep)"]
    row_labels = ["Local u ≈ 0.1\n(Healthy Core)", "Local u ≈ 0.5\n(Congestion)", "Local u ≈ 0.9\n(Severe Gridlock)"]

    fig, axes = plt.subplots(3, 4, figsize=(24, 18), facecolor='#f8f9fa')

    # 定义画线阈值
    THRESHOLD_RED = 0.8
    THRESHOLD_BLUE = -0.5
    THRESHOLD_GRAY = 0.2

    for col_idx, hour in enumerate(target_hours):
        hour_mask = (T == hour)
        indices = np.where(hour_mask)[0]
        if len(indices) == 0: continue

        hour_Y = Y_flat[indices]

        # 🌟 核心升级：独立计算“当前小时内”的局部排秩！
        local_scores = scores[indices]
        hour_ranks = rankdata(local_scores) / len(local_scores)

        # 在局部的三大分位数区间内寻找最真实的代表日 (Medoid)
        mask_u0 = (hour_ranks <= 0.2)
        mask_u05 = (hour_ranks >= 0.4) & (hour_ranks <= 0.6)
        mask_u1 = (hour_ranks >= 0.8)

        for row_idx, mask in enumerate([mask_u0, mask_u05, mask_u1]):
            ax = axes[row_idx, col_idx]
            if np.sum(mask) == 0:
                ax.axis('off')
                continue

            sub_Y = hour_Y[mask]
            sub_ranks = hour_ranks[mask]

            # 找到最贴近目标的真实样本
            target_u = [0.1, 0.5, 0.9][row_idx]
            medoid_idx = np.argmin(np.abs(sub_ranks - target_u))
            real_cov = sub_Y[medoid_idx].reshape(dim, dim)

            # 转化为相关系数矩阵
            d = np.sqrt(np.diag(real_cov))
            d[d == 0] = 1e-5
            corr_mat = real_cov / np.outer(d, d)

            # --- 开始在当前子图上绘制地理信息 ---
            ax.scatter(coords[:, 0], coords[:, 1], s=40, c='#2c3e50', zorder=5, edgecolor='w', lw=1)

            for i in range(dim):
                for j in range(i + 1, dim):
                    c_val = corr_mat[i, j]
                    # 断流撕裂 (强负相关) -> 蓝色
                    if c_val < THRESHOLD_BLUE:
                        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                                color='dodgerblue', linewidth=abs(c_val) * 3.5, alpha=0.9, zorder=2)
                    # 死锁拥堵 (强正相关) -> 红色
                    elif c_val > THRESHOLD_RED:
                        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                                color='crimson', linewidth=c_val * 3, alpha=0.75, zorder=1)
                    # 微弱关联 -> 灰色
                    elif abs(c_val) > THRESHOLD_GRAY:
                        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                                color='gray', linewidth=0.3, alpha=0.2, zorder=0)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#bdc3c7')

            if row_idx == 0: ax.set_title(col_labels[col_idx], fontsize=18, fontweight='bold', pad=15)
            if col_idx == 0: ax.set_ylabel(row_labels[row_idx], fontsize=18, fontweight='bold', labelpad=15)

    # 底部图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='crimson', lw=4, label='Gridlock (Severe Positive Corr)'),
        Line2D([0], [0], color='dodgerblue', lw=4, label='Flow Severed (Severe Negative Corr)'),
        Line2D([0], [0], color='gray', lw=1, label='Normal Synergy')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=16, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle("Spatial Topological Evolution of Traffic Networks\n(Time-of-Day vs. Local AMQR Manifold Quantile)",
                 fontsize=26, fontweight='bold', y=0.98)
    plt.subplots_adjust(wspace=0.05, hspace=0.1, bottom=0.08)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "Fig_3x4_Spatial_Evolution.jpg")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 3x4 地理拓扑演化大图(局部条件版)已保存至: {save_path}")


def plot_local_matrix_grid(Y_flat, scores, T, save_dir, dim=20):
    """
    绘制 3x4 流形拓扑演化热力图 (基于局部条件分位数 Local Quantiles)
    """
    print("🎨 正在渲染 3x4 局部条件分位数拓扑热力图...")
    target_hours = [6, 12, 18, 0]
    col_labels = ["6:00\n(Morning Wake-up)", "12:00\n(Noon Steady)", "18:00\n(Evening Rush)", "0:00\n(Midnight Sleep)"]
    row_labels = ["Local u ≈ 0.1\n(Healthy Core)", "Local u ≈ 0.5\n(Congestion)", "Local u ≈ 0.9\n(Severe Anomaly)"]

    fig, axes = plt.subplots(3, 4, figsize=(22, 16), facecolor='white')

    for col_idx, hour in enumerate(target_hours):
        hour_mask = (T == hour)
        indices = np.where(hour_mask)[0]
        if len(indices) == 0: continue

        hour_Y = Y_flat[indices]

        # 🌟 核心：独立计算“当前小时内”的局部排秩
        local_scores = scores[indices]
        hour_ranks = rankdata(local_scores) / len(local_scores)

        # 定义局部分位数区间
        mask_u0 = (hour_ranks <= 0.2)
        mask_u05 = (hour_ranks >= 0.4) & (hour_ranks <= 0.6)
        mask_u1 = (hour_ranks >= 0.8)

        for row_idx, mask in enumerate([mask_u0, mask_u05, mask_u1]):
            ax = axes[row_idx, col_idx]
            if np.sum(mask) == 0:
                ax.axis('off')
                continue

            sub_Y = hour_Y[mask]
            sub_ranks = hour_ranks[mask]

            # 找到最贴近目标的真实历史样本 (Medoid)
            target_u = [0.1, 0.5, 0.9][row_idx]
            medoid_idx = np.argmin(np.abs(sub_ranks - target_u))
            real_cov = sub_Y[medoid_idx].reshape(dim, dim)

            # 转化为相关系数矩阵，聚焦纯粹的拓扑形态
            d = np.sqrt(np.diag(real_cov))
            d[d == 0] = 1e-5
            corr_mat = real_cov / np.outer(d, d)

            # 🌟 修改 1：彻底屏蔽 seaborn 自带的 cbar，保证 12 个子图尺寸绝对统一！
            sns.heatmap(corr_mat, ax=ax, cmap='coolwarm', vmin=-1, vmax=1,
                        square=True, cbar=False,
                        xticklabels=False, yticklabels=False)

            # 边框和标题美化
            if row_idx == 0: ax.set_title(col_labels[col_idx], fontsize=18, fontweight='bold', pad=20)
            if col_idx == 0: ax.set_ylabel(row_labels[row_idx], fontsize=18, fontweight='bold', labelpad=20)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#333333')
                spine.set_linewidth(1)

    plt.suptitle("Microscopic Topological Evolution\n(Time-of-Day vs. Local AMQR Manifold Quantile)",
                 fontsize=24, fontweight='bold', y=1.02)

    # 将主体子图区域向左挤压一点，给右侧全局 Colorbar 留出充足空间
    plt.subplots_adjust(right=0.9, wspace=0.1, hspace=0.15)

    # 🌟 修改 2：构建全局统一的 ScalarMappable 颜色映射条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    sm = cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])  # 必须加上这句，初始化空数据

    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Pearson Correlation", fontsize=18, fontweight='bold', labelpad=20)
    cbar.ax.tick_params(labelsize=14)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "Fig_3x4_Local_Matrix_Heatmap.jpg")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 3x4 局部条件热力大图已保存至: {save_path}")


def plot_eeg_tube_validation(T, Y_flat, t_sparse, Y_reg_sparse, a_ranks, labels, anomaly_scores, save_dir, dim=23):
    """
    绘制 1x3 的脑电波 (EEG) 流形演化与无监督发病检测图
    """
    print("🎨 正在渲染 1x3 脑电波流形诊断图谱...")
    fig, axes = plt.subplots(1, 3, figsize=(28, 8), facecolor='white')
    cmap_choice = 'plasma'

    raw_traces = np.array([np.trace(y.reshape(dim, dim)) for y in Y_flat])
    reg_traces_sparse = np.array([np.trace(y.reshape(dim, dim)) for y in Y_reg_sparse])

    # 提取医生标注的真实发病时间段 (用于画红色警报背景)
    seizure_indices = np.where(labels == 1)[0]
    if len(seizure_indices) > 0:
        seizure_start_t = T[seizure_indices[0]]
        seizure_end_t = T[seizure_indices[-1]]
    else:
        seizure_start_t, seizure_end_t = None, None

    # --- 视图 A: 脑网络流形渐变管道 (伴随发病期高亮) ---
    ax1 = axes[0]
    # 排序以防高分点被遮挡
    sort_idx = np.argsort(a_ranks)
    T_sorted = T[sort_idx]
    raw_traces_sorted = raw_traces[sort_idx]
    a_ranks_sorted = a_ranks[sort_idx]

    sc_tube = ax1.scatter(T_sorted, raw_traces_sorted, c=a_ranks_sorted,
                          cmap=cmap_choice, s=25, alpha=0.9, zorder=3, vmin=0.0, vmax=1.0, edgecolor='none')

    # 🌟 核心医学佐证：画出真实的癫痫发作区间 (红色背景)
    if seizure_start_t is not None:
        ax1.axvspan(seizure_start_t, seizure_end_t, color='lightcoral', alpha=0.3, zorder=1,
                    label='Ground Truth: Seizure')

    ax1.plot(t_sparse, reg_traces_sparse, color='#00FF00', lw=3, zorder=5, label='AMQR Healthy Baseline')
    ax1.scatter(t_sparse, reg_traces_sparse, color='#00FF00', s=60, edgecolor='black', lw=1.5, zorder=6)

    ax1.set_title("1. EEG Functional Connectivity Manifold Tube", fontsize=18, fontweight='bold', pad=15)
    ax1.set_xlabel("Time (Seconds)", fontsize=14)
    ax1.set_ylabel("Brain Network Volatility (Log Trace)", fontsize=14)
    ax1.set_yscale('log')
    ax1.legend(loc='upper right', fontsize=12)

    # --- 视图 B: 切空间 PCA (健康 vs 癫痫 拓扑撕裂) ---
    ax2 = axes[1]
    import scipy.linalg
    Y_log_flat = np.array([scipy.linalg.logm(y.reshape(dim, dim) + np.eye(dim) * 1e-5).real.flatten() for y in Y_flat])
    from sklearn.decomposition import PCA
    Y_pca = PCA(n_components=2).fit_transform(Y_log_flat)
    Y_pca_sorted = Y_pca[sort_idx]

    sc_pca = ax2.scatter(Y_pca_sorted[:, 0], Y_pca_sorted[:, 1], c=a_ranks_sorted,
                         cmap=cmap_choice, s=35, alpha=0.8, edgecolor='none', vmin=0.0, vmax=1.0)

    # 如果有真实的癫痫点，在 PCA 里用红色空心圆圈圈出来，证明它们确实在拓扑边缘！
    if seizure_start_t is not None:
        seizure_pca = Y_pca[labels == 1]
        ax2.scatter(seizure_pca[:, 0], seizure_pca[:, 1], facecolors='none', edgecolors='red',
                    s=80, lw=1.5, zorder=4, label='True Seizure States')
        ax2.legend(loc='upper right')

    ax2.set_title("2. Tangent Space PCA (Pathological Shift)", fontsize=18, fontweight='bold', pad=15)
    ax2.set_xlabel("Principal Component 1", fontsize=14)

    # --- 视图 C: 无监督异常得分追踪 (医学时序诊断图) ---
    ax3 = axes[2]
    # 画出模型给出的每一秒的绝对异常得分
    ax3.plot(T, anomaly_scores, color='#2c3e50', lw=2, zorder=3, label='AMQR Anomaly Score')
    ax3.fill_between(T, anomaly_scores, color='#34495e', alpha=0.2, zorder=2)

    # 再次画出红色的真实发病底色，形成强烈对比！
    if seizure_start_t is not None:
        ax3.axvspan(seizure_start_t, seizure_end_t, color='lightcoral', alpha=0.3, zorder=1)
        # 在发病区域标上文字
        ax3.text(seizure_start_t + 10, np.max(anomaly_scores) * 0.9, 'Seizure\nOnset',
                 color='red', fontweight='bold', fontsize=12)

    ax3.set_title("3. Unsupervised Zero-Shot Seizure Detection", fontsize=18, fontweight='bold', pad=15)
    ax3.set_xlabel("Time (Seconds)", fontsize=14)
    ax3.set_ylabel("AMQR Absolute Topological Distance", fontsize=14)
    ax3.legend(loc='upper left', fontsize=12)

    # 统一 Colorbar
    cbar_ax = fig.add_axes([0.62, 0.15, 0.012, 0.7])
    fig.colorbar(sc_pca, cax=cbar_ax).set_label("AMQR Anomaly Quantile", fontsize=16, fontweight='bold')

    plt.subplots_adjust(left=0.06, right=0.95, wspace=0.3)

    import os
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "Fig_1x3_EEG_Detection.jpg")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ EEG 脑电无监督诊断图谱已保存至: {save_path}")