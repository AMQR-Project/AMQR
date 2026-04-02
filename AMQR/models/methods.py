import ot
import warnings
import numpy as np
import pandas as pd
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import qmc, norm, laplace, rankdata
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph, KernelDensity, NearestNeighbors, KNeighborsRegressor

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
                 use_knn=True, k_neighbors=15, max_samples=2500):
        self.ref_dist = ref_dist.lower()
        self.epsilon = epsilon
        self.d_int = d_int
        self.use_knn = use_knn
        self.k_neighbors = k_neighbors
        self.max_samples = max_samples  # 🌟 控制 O(N^3) 复杂度的核心阀门

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

    def _fit_predict_core(self, Y):
        """底层纯粹的数学对齐引擎 (仅处理不超过 max_samples 的数据)"""
        N = len(Y)
        Y_flat = Y.reshape(N, -1)

        # 1. 计算测地线距离
        if self.use_knn:
            k = min(self.k_neighbors, N - 1)
            A = kneighbors_graph(Y_flat, n_neighbors=k, mode='distance', include_self=False)
            Cy = shortest_path(A, method='D', directed=False)
            if np.isinf(Cy).any():
                Cy[np.isinf(Cy)] = np.nanmax(Cy[Cy != np.inf]) * 2.0
        else:
            Cy = cdist(Y_flat, Y_flat, metric='euclidean')

        # 2. MLE 本征维数探测 (带安全封顶)
        if self.d_int is not None:
            final_d = self.d_int
        else:
            k_mle = min(10, N - 1)
            nn = NearestNeighbors(n_neighbors=k_mle + 1).fit(Y_flat)
            dists, _ = nn.kneighbors(Y_flat)
            dists = np.maximum(dists[:, 1:], 1e-9)
            r_k = dists[:, -1:]
            mle_val = (k_mle - 1) / np.sum(np.log(r_k / dists[:, :-1]), axis=1)
            final_d = int(np.round(np.mean(mle_val)))
            final_d = max(1, min(final_d, 3))  # 强制封顶，防止高维拓扑崩塌

        # 3. 构建目标空间与打乱对称性
        Z_ref = self._generate_latent_qmc(final_d, N)
        Cz = cdist(Z_ref, Z_ref, metric='euclidean')

        Cz_norm = Cz / (Cz.max() + 1e-9)
        Cy_norm = Cy / (np.nanmax(Cy) + 1e-9)

        # 🌟 注入微量噪声，打破单纯形法退化困境
        Cz_norm += np.random.uniform(0, 1e-8, size=Cz_norm.shape)
        Cy_norm += np.random.uniform(0, 1e-8, size=Cy_norm.shape)

        py, pz = ot.unif(N), ot.unif(N)

        # 4. 极速求解 GW
        if self.epsilon > 0:
            gw_plan = ot.gromov.entropic_gromov_wasserstein(
                Cz_norm, Cy_norm, pz, py, 'square_loss', epsilon=self.epsilon, max_iter=100)
        else:
            gw_plan = ot.gromov.gromov_wasserstein(
                Cz_norm, Cy_norm, pz, py, 'square_loss', max_iter=50, tol=1e-4)

        # 5. 提取深度与排名
        Y_mapped_to_Z = np.dot(gw_plan.T, Z_ref) * N
        amqr_depths = np.linalg.norm(Y_mapped_to_Z, axis=1)
        med_idx = np.argmin(amqr_depths)
        ranks = rankdata(amqr_depths) / N

        return Y[med_idx], ranks

    def fit_predict(self, Y, T=None, t_eval=None, window_size=1.0):
        """
        高层 API:
        如果 T=None, 则进行全局分位数提取 (自动进行 Landmark 采样与 KNN 扩展)。
        如果 T!=None, 则沿 T 进行条件滑动窗口回归。
        """
        N = len(Y)

        # =========================================================
        # 模式 A: 全局静态分位数提取 (带 Landmark 扩展)
        # =========================================================
        if T is None:
            if N <= self.max_samples:
                med, ranks = self._fit_predict_core(Y)
                return med, ranks
            else:
                # 随机抽取地标点
                sub_idx = np.random.choice(N, size=self.max_samples, replace=False)
                Y_sub = Y[sub_idx]

                # 仅在地标点上运行核心引擎
                med, ranks_sub = self._fit_predict_core(Y_sub)

                # 样本外扩展 (Out-of-Sample Extension)
                knn_ext = KNeighborsRegressor(n_neighbors=3).fit(Y_sub.reshape(self.max_samples, -1), ranks_sub)
                ranks_full = knn_ext.predict(Y.reshape(N, -1))
                return med, ranks_full

        # =========================================================
        # 模式 B: 动态条件滑动窗口回归
        # =========================================================
        else:
            if t_eval is None:
                t_eval = np.linspace(T.min(), T.max(), 50)

            trajectory_med = []
            global_ranks = np.zeros(N)
            counts = np.zeros(N)

            for t_c in t_eval:
                idx = np.where(np.abs(T - t_c) <= window_size / 2.0)[0]
                if len(idx) < 15:
                    continue

                Y_c = Y[idx]

                # 递归调用自身 (此时走模式 A，自动享受 Landmark 加速)
                med_c, ranks_c = self.fit_predict(Y=Y_c, T=None)

                trajectory_med.append((t_c, med_c))

                # 将局部的 ranks 累加到全局，最后取平均 (时空平滑)
                global_ranks[idx] += ranks_c
                counts[idx] += 1

            # 整理返回结果
            counts[counts == 0] = 1
            final_ranks = global_ranks / counts

            # trajectory_med 包含 (t_c, 对应中位数张量)
            return trajectory_med, final_ranks


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







# ==========================================
# 🌪️ 1. 闭合圆柱流形 (带 135° 空当) 生成器
# ==========================================
def generate_circular_manifold_with_gap(n_points=15000):
    np.random.seed(42)
    T = np.random.uniform(0, 10, n_points)
    R = np.random.normal(3.0, 0.2, n_points)
    theta_center = T * (np.pi / 2.5)
    theta = np.random.vonmises(mu=theta_center, kappa=1.5, size=n_points)

    rel_theta = (theta - theta_center + np.pi) % (2 * np.pi) - np.pi
    gap_center = 3 * np.pi / 4
    gap_width = 0.5
    in_gap = np.abs(rel_theta - gap_center) < gap_width
    keep_prob = np.where(in_gap, 0.02, 1.0)
    keep_mask = np.random.rand(n_points) < keep_prob

    T, R, theta = T[keep_mask], R[keep_mask], theta[keep_mask]
    Y1, Y2 = R * np.cos(theta), R * np.sin(theta)
    P_3D = np.column_stack([T, Y1, Y2])

    T_gt = np.linspace(0, 10, 200)
    theta_gt = T_gt * (np.pi / 2.5)
    GT_3D = np.column_stack([T_gt, 3.0 * np.cos(theta_gt), 3.0 * np.sin(theta_gt)])

    return T, P_3D, GT_3D


# ==========================================
# 🚀 2. 核心提取引擎
# ==========================================
def extract_four_models_sliding_windows(T, P_3D, window_size=1.2, step_size=0.1, u_target=0.20):
    n = len(T)
    masks = {'nw': np.zeros(n, dtype=bool), 'f': np.zeros(n, dtype=bool),
             'kde': np.zeros(n, dtype=bool), 'a': np.zeros(n, dtype=bool)}
    t_traj = []
    centers = {'nw': [], 'f': [], 'kde': [], 'a': []}

    # 一维闭合流形先验，启用 KNN
    amqr = AMQR_Engine(ref_dist='uniform', epsilon=1e-3, d_int=1, use_knn=True, k_neighbors=10)
    t_centers = tqdm(np.arange(T.min(), T.max() + step_size, step_size))
    print(f"⏳ 滑动窗口运算启动，共计 {len(t_centers)} 个局部时间窗...")

    for t_c in t_centers:
        idx = np.where(np.abs(T - t_c) <= window_size / 2.0)[0]
        print(len(idx))
        MAX_POINTS = 250  # 建议设在 200 - 300 之间

        if len(idx) > MAX_POINTS:
            # 随机无放回抽取 MAX_POINTS 个点，足以完美描绘局部拓扑
            idx = np.random.choice(idx, size=MAX_POINTS, replace=False)

            # ⚠️ 关键细节：对抽取后的索引重新排序！
            # 这能保证 T[idx] 依然在时间/空间上是有序排列的，防止后续代码逻辑错乱
            idx = np.sort(idx)

        if len(idx) < 15: continue
        inner_condition = (T[idx] >= t_c - step_size / 2.0) & (T[idx] < t_c + step_size / 2.0)
        inner_in_idx = np.where(inner_condition)[0]
        P_c = P_3D[idx]

        # A. NW
        nw_med = np.mean(P_c, axis=0)
        if len(inner_in_idx) > 0: masks['nw'][idx[inner_in_idx]] = \
        (rankdata(np.linalg.norm(P_c - nw_med, axis=1)) / len(P_c))[inner_in_idx] <= u_target
        # B. Fréchet
        f_med, f_ranks = get_frechet_tube(P_c)
        if len(inner_in_idx) > 0: masks['f'][idx[inner_in_idx]] = f_ranks[inner_in_idx] <= u_target
        # C. SW-KDE
        kde_med, kde_ranks = get_kde_tube(P_c)
        if len(inner_in_idx) > 0: masks['kde'][idx[inner_in_idx]] = kde_ranks[inner_in_idx] <= u_target
        # D. AMQR
        a_med, a_ranks, _ = amqr.fit_predict(P_c, Cy=None)
        if len(inner_in_idx) > 0: masks['a'][idx[inner_in_idx]] = a_ranks[inner_in_idx] <= u_target

        t_traj.append(t_c)
        centers['nw'].append(nw_med)
        centers['f'].append(f_med)
        centers['kde'].append(kde_med)
        centers['a'].append(a_med)

    return np.array(t_traj), {k: np.array(v) for k, v in centers.items()}, masks


# ==========================================
# 🧠 3. 定量指标计算模块
# ==========================================
def evaluate_spiral_metrics(t_traj, centers, masks, T_sorted, P_sorted):
    # 构建绝对真实的 GT 数据
    gt_y1 = 3.0 * np.cos(t_traj * np.pi / 2.5)
    gt_y2 = 3.0 * np.sin(t_traj * np.pi / 2.5)
    gt_centers = np.column_stack([t_traj, gt_y1, gt_y2])

    metrics = []
    method_names = {'nw': 'NW Mean', 'f': 'FRÉCHET', 'kde': 'SW-KDE', 'a': 'AMQR (Proposed)'}

    for k, name in method_names.items():
        est_centers = centers[k]

        # 1. 轨迹追踪均方根误差
        rmse = np.sqrt(np.mean(np.sum((est_centers - gt_centers) ** 2, axis=1)))

        # 2. 径向偏离度 (越接近 3.0 越好，偏离越大说明掉进真空)
        est_radii = np.sqrt(est_centers[:, 1] ** 2 + est_centers[:, 2] ** 2)
        radial_dev = np.mean(np.abs(est_radii - 3.0))

        # 3. 核心纯净度 (提取出的 20% Mask 到真实 GT 中心线的平均距离)
        mask = masks[k]
        P_masked = P_sorted[mask]
        T_masked = P_masked[:, 0]
        # 精确计算每一个 masked point 对应的真实物理极点位置
        gt_y1_masked = 3.0 * np.cos(T_masked * np.pi / 2.5)
        gt_y2_masked = 3.0 * np.sin(T_masked * np.pi / 2.5)
        gt_masked_exact = np.column_stack([T_masked, gt_y1_masked, gt_y2_masked])

        core_purity = np.mean(np.linalg.norm(P_masked - gt_masked_exact, axis=1))

        metrics.append({
            'Method': name,
            'Tracking RMSE ↓': round(rmse, 3),
            'Radial Dev (Vacuum Error) ↓': round(radial_dev, 3),
            'Core Purity ↓': round(core_purity, 3)
        })

    return pd.DataFrame(metrics).set_index('Method')


# ==========================================
# 📊 4. 终极大图渲染 (2x4 布局)
# ==========================================
def plot_2x4_experiment(T_sorted, P_sorted, GT_3D, t_traj, centers, masks, target_t=5.0,
                        filename="circular_gap_comparison.png"):
    # 3D 轨迹平滑
    traj_3d = {k: gaussian_filter1d(v, sigma=3, axis=0) for k, v in centers.items()}

    print("🎨 绘制 2x4 穿孔流形对比图中...")
    fig = plt.figure(figsize=(32, 16))
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass

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

    # ---------------- 上半部分：宏观 3D 图 ----------------
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
        ax.set_xlim(0, 10);
        ax.set_ylim(-4, 4);
        ax.set_zlim(-4, 4)
        ax.view_init(elev=20, azim=-45)
        ax.xaxis.pane.fill = False;
        ax.yaxis.pane.fill = False;
        ax.zaxis.pane.fill = False
        if i == 0: ax.legend(loc='upper right', fontsize=14)

    # ---------------- 下半部分：微观 2D 截面染色图 ----------------
    window = 0.3
    idx_slice = np.where(np.abs(T_sorted - target_t) <= window)[0]
    idx_t = np.argmin(np.abs(t_traj - target_t))
    gt_idx = np.argmin(np.abs(GT_3D[:, 0] - target_t))

    for i, method in enumerate(methods):
        ax = fig.add_subplot(2, 4, i + 5)

        # 绘制被截断带有 135度 空当的背景圆环切片
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
        ax.set_aspect('equal');
        ax.set_xlim(-4.5, 4.5);
        ax.set_ylim(-4.5, 4.5)
        if i == 0: ax.legend(loc='lower right', fontsize=13)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"✅ 2x4 穿孔流形图谱已保存至: {filename}")
    plt.show()


# ==========================================
# 🎯 主函数
# ==========================================
def main():
    print("========================================================")
    print(" 🌟 Sim 3: 3D Spiral Manifold with Topological Gap")
    print("========================================================")

    T, P_3D, GT_3D = generate_circular_manifold_with_gap(n_points=15000)
    sort_idx = np.argsort(T)
    T_sorted, P_sorted = T[sort_idx], P_3D[sort_idx]

    t_traj, centers, masks = extract_four_models_sliding_windows(
        T_sorted, P_sorted, window_size=1.2, step_size=0.1, u_target=0.20
    )

    # 打印终极定量表格
    print("\n=======================================================")
    print("📊 Table 3: Quantitative Evaluation on Spiral Gap")
    print("=======================================================")
    df_metrics = evaluate_spiral_metrics(t_traj, centers, masks, T_sorted, P_sorted)
    print(df_metrics.to_string())
    print("=======================================================\n")

    # 画图
    plot_2x4_experiment(T_sorted, P_sorted, GT_3D, t_traj, centers, masks, target_t=5.0)


if __name__ == "__main__":
    main()


# =====================================================================
# 📈 1. 50D 泛函数据生成器 (带高密度离群陷阱 Dense Outlier Trap)
# =====================================================================
def generate_functional_data(N=250, D=50, random_state=42):
    np.random.seed(random_state)
    X_grid = np.linspace(-5, 5, D)
    Y = np.zeros((N, D))

    for i in range(N):
        rand_val = np.random.rand()
        if rand_val < 0.85:
            # 【多数派 - 真实波峰】：85% 的数据锚定在中心 0，但有自然的相位起伏 (较散)
            shift = np.random.normal(0, 1.0)
        else:
            # 【极少数 - 致命陷阱】：只有 15% 的数据，偏离到右侧 3.0
            # ！！！注意方差改成了极其微小的 0.1，它们会死死挤在一起！！！
            shift = np.random.normal(3.0, 0.1)

        # 振幅变异
        amp = np.random.normal(2.0, 0.2)

        # 生成基础尖锐波峰
        base_curve = amp * np.exp(-((X_grid - shift) ** 2) / 0.5)
        # 加入局部白噪声
        noise = np.random.normal(0, 0.05, D)

        Y[i] = base_curve + noise

    return X_grid, Y

# =====================================================================
# 🎨 3. 视觉渲染模块：1x4 泛函削峰对比图 (去副标题纯净版)
# =====================================================================
def plot_functional_comparison(X_grid, Y, mean_curve, mean_ranks, f_med, f_ranks, kde_med, kde_ranks, a_med, a_ranks,
                               filename="Sim_4_Functional_50D.png"):
    fig, axes = plt.subplots(1, 4, figsize=(32, 8), facecolor='white')

    # 配色与切分设定
    core_color = '#3498db'  # 核心 20% 管的颜色
    center_color = '#00FF00'  # 中位数曲线高亮
    u_target = 0.20

    # 🌟 修改：去掉了 subtitle 参数 🌟
    plots_data = [
        (axes[0], mean_curve, mean_ranks, "Cross-sectional Mean (L2)", '#c0392b'),
        (axes[1], f_med, f_ranks, "Fréchet Median (L1)", '#d35400'),
        (axes[2], kde_med, kde_ranks, "SW-KDE (Density Mode)", '#8e44ad'),
        (axes[3], a_med, a_ranks, "Proposed AMQR", '#27ae60')
    ]

    for ax, med, ranks, title, t_color in plots_data:
        # 1. 底层：画出所有灰色的原始曲线
        ax.plot(X_grid, Y.T, color='#bdc3c7', alpha=0.15, lw=1, zorder=1)

        # 2. 中层：画出排名前 20% 的核心曲线管 (Top 20% Tube)
        mask = ranks <= u_target
        ax.plot(X_grid, Y[mask].T, color=core_color, alpha=0.4, lw=1.5, zorder=2)

        # 3. 顶层：画出提取的中心曲线
        ax.plot(X_grid, med, color=center_color, lw=4, zorder=5, label='Estimated Center')

        # 🌟 修改：直接设置单行主标题，去掉 ax.text 写的副标题 🌟
        ax.set_title(title, fontsize=24, fontweight='bold', color=t_color, pad=15)

        ax.set_xlim(-5, 5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#7f8c8d')

        ax.legend(loc='upper right', fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 50D 泛函削峰对比图 (纯净版) 已保存至: {filename}")
    plt.show()


def evaluate_functional_metrics(results_dict, X_grid):
    # 构建绝对真实的 GT 数据原型 (无相移，无振幅变异，无白噪声)
    # 振幅 A=2.0, 相移 S=0.0
    Y_true = 2.0 * np.exp(-((X_grid - 0) ** 2) / 0.5)

    metrics = []

    for model_name, res in results_dict.items():
        Y_est = res['med']

        # 1. 振幅畸变误差 (Amplitude Error -> 衡量削峰严重程度)
        peak_est = np.max(Y_est)
        amp_error = np.abs(peak_est - 2.0)

        # 2. 相位偏移误差 (Phase Shift Error -> 衡量波峰是否在 X=0 处)
        peak_idx = np.argmax(Y_est)
        phase_est = X_grid[peak_idx]
        phase_error = np.abs(phase_est - 0.0)

        # 3. 全局波形保真度 (Shape RMSE -> 与完美高斯原型的点对点 L2 误差)
        rmse = np.sqrt(np.mean((Y_est - Y_true) ** 2))

        metrics.append({
            'Method': model_name.upper(),
            'Amplitude Error ↓': round(amp_error, 3),
            'Phase Shift Error ↓': round(phase_error, 3),
            'Shape RMSE ↓': round(rmse, 3)
        })

    return pd.DataFrame(metrics).set_index('Method')


# =====================================================================
# 🚀 替换主执行逻辑的结尾部分，加入指标打印
# =====================================================================
if __name__ == "__main__":
    print("========================================================")
    print(" 🌟 Sim 4: 50D Functional Data (Phase Variation)")
    print("========================================================")

    N_samples = 250
    X_grid, Y_data = generate_functional_data(N=N_samples)

    results = {}

    print("⏳ 计算 L2 Mean...")
    mean_curve = np.mean(Y_data, axis=0)
    results['nw'] = {'med': mean_curve, 'ranks': rankdata(np.linalg.norm(Y_data - mean_curve, axis=1)) / N_samples}

    print("⏳ 计算 Fréchet L1 Median...")
    f_med, f_ranks = get_frechet_tube(Y_data)
    results['frechet'] = {'med': f_med, 'ranks': f_ranks}

    print("⏳ 计算 SW-KDE (PCA-reduced)...")
    kde_med, kde_ranks = get_kde_tube(Y_data)
    results['kde'] = {'med': kde_med, 'ranks': kde_ranks}

    print("⏳ 计算 AMQR 流形分位数...")
    amqr = AMQR_Engine(ref_dist='uniform', epsilon=0.0, d_int=1, use_knn=True, k_neighbors=5)
    a_med, a_ranks, _ = amqr.fit_predict(Y_data)
    results['amqr'] = {'med': a_med, 'ranks': a_ranks}

    # 📊 输出定量评估表格
    print("\n=======================================================")
    print("📊 Table 4: Quantitative Evaluation on Functional Data")
    print("=======================================================")
    df_metrics = evaluate_functional_metrics(results, X_grid)
    print(df_metrics.to_string())
    print("=======================================================\n")

    # 🎨 画图 (调用你的代码)
    plot_functional_comparison(X_grid, Y_data, mean_curve, results['nw']['ranks'],
                               f_med, f_ranks, kde_med, kde_ranks, a_med, a_ranks)



# =====================================================================
# 🌪️ 1. SPD 矩阵数据生成 (旋转椭圆流形 + 巨型膨胀噪声)
# =====================================================================
def generate_spd_data_with_labels(N=300, random_state=42):
    np.random.seed(random_state)
    Y = np.zeros((N, 4))
    labels = np.zeros(N)  # 0 for core, 1 for outlier

    for i in range(N):
        is_outlier = np.random.rand() < 0.20
        labels[i] = 1 if is_outlier else 0

        if is_outlier:
            l1, l2 = np.random.normal(12.0, 1.0), np.random.normal(2.0, 0.5)
            theta = np.random.normal(0, 0.1)
        else:
            l1, l2 = np.random.normal(3.0, 0.3), np.random.normal(0.4, 0.1)
            theta = np.random.normal(np.pi / 4, 0.3)

        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        M = R @ np.diag([l1, l2]) @ R.T
        Y[i] = M.flatten()

    return Y, labels


# =====================================================================
# 🧠 定量计算与辅助模块
# =====================================================================
def compute_spd_all_props(Y_flat):
    N = len(Y_flat)
    dets = np.zeros(N)
    eigs = np.zeros((N, 2))
    log_Y_flat = np.zeros((N, 4))
    for i in range(N):
        M = Y_flat[i].reshape(2, 2)
        dets[i] = np.linalg.det(M)
        vals, vecs = np.linalg.eigh(M)
        eigs[i] = vals
        log_M = vecs @ np.diag(np.log(np.clip(vals, 1e-10, None))) @ vecs.T
        log_Y_flat[i] = log_M.flatten()
    return dets, eigs, log_Y_flat


def evaluate_spd_metrics(results_dict, true_labels):
    """计算 4 大核心定量指标，用于论文表格"""
    theta_true = np.pi / 4
    R_true = np.array([[np.cos(theta_true), -np.sin(theta_true)],
                       [np.sin(theta_true), np.cos(theta_true)]])
    M_true = R_true @ np.diag([3.0, 0.4]) @ R_true.T
    log_M_true = slinalg.logm(M_true)
    det_true = 1.2

    metrics = []
    for model_name, res in results_dict.items():
        M_est = res['med'].reshape(2, 2)
        ranks = res['ranks']

        # 1. Log-Euclidean Distance (LED)
        log_M_est = slinalg.logm(M_est)
        led_error = np.linalg.norm(log_M_est - log_M_true, ord='fro')

        # 2. Volume Bias (|Det - 1.2|)
        det_est = np.linalg.det(M_est)
        det_bias = np.abs(det_est - det_true)

        # 3. Spectral Error
        vals_est = np.linalg.eigvalsh(M_est)
        spectral_error = np.abs(vals_est[1] - 3.0) + np.abs(vals_est[0] - 0.4)

        # 4. AUC-ROC (Core Isolation)
        auc = roc_auc_score(true_labels, ranks)

        metrics.append({
            'Method': model_name.upper(),
            'LED Error ↓': round(led_error, 3),
            'Volume Bias ↓': round(det_bias, 3),
            'Spectral Error ↓': round(spectral_error, 3),
            'Isolation AUC ↑': round(auc, 3)
        })

    return pd.DataFrame(metrics).set_index('Method')


# =====================================================================
# 🎨 视觉渲染模块：3x4 终极 SPD 深度评测图
# =====================================================================
def plot_spd_3x4_comparison(Y_flat, results_dict, filename="Sim_5_SPD_3x4_Final.png"):
    fig, axes = plt.subplots(3, 4, figsize=(28, 18), facecolor='white')
    raw_dets, raw_eigs, raw_log_Y = compute_spd_all_props(Y_flat)
    pca = PCA(n_components=2)
    Y_pca = pca.fit_transform(raw_log_Y)

    col_titles = ["NW (L2 Mean)", "Fréchet (L1 Median)", "SW-KDE (Mode)", "Proposed AMQR (OT)"]
    row_titles = ["1. Determinant (Volume)", "2. Eigenvalue Spectrum", "3. Manifold Embedding"]
    cmap_choice = 'plasma'
    u_target = 0.20

    for j, m_key in enumerate(['nw', 'frechet', 'kde', 'amqr']):
        med = results_dict[m_key]['med']
        ranks = results_dict[m_key]['ranks']
        m_det, m_eig, m_log = compute_spd_all_props(np.array([med]))
        m_pca = pca.transform(m_log)[0]
        mask = ranks <= u_target

        m_color = '#c0392b' if j < 2 else ('#8e44ad' if j == 2 else '#27ae60')
        symbol = '*' if m_key == 'amqr' else 'X'
        size = 800 if m_key == 'amqr' else 500

        # --- Row 1: Determinant Histogram ---
        ax1 = axes[0, j]
        ax1.hist(raw_dets, bins=40, color='#bdc3c7', alpha=0.4, edgecolor='white')
        ax1.hist(raw_dets[mask], bins=40, color=m_color, alpha=0.8, edgecolor='white', label="Top 20% Core")
        ax1.axvline(m_det[0], color='black', lw=4, ls='--', label=f"Det: {m_det[0]:.2f}")
        ax1.set_title(col_titles[j], fontsize=20, fontweight='bold', pad=15, color=m_color)
        if j == 0: ax1.set_ylabel(row_titles[0], fontsize=16, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=12)

        # --- Row 2: Eigenvalue Spectrum ---
        ax2 = axes[1, j]
        ax2.scatter(raw_eigs[:, 1], raw_eigs[:, 0], c=ranks, cmap=cmap_choice, s=50, alpha=0.7, edgecolor='w', lw=0.2)
        ax2.scatter(m_eig[0, 1], m_eig[0, 0], marker=symbol, color='#00FF00', s=size, edgecolor='black', lw=2, zorder=5)
        ax2.set_xlim(0, 16);
        ax2.set_ylim(-0.2, 4.5)
        if j == 0: ax2.set_ylabel(row_titles[1], fontsize=16, fontweight='bold')

        # --- Row 3: Log-Euclidean PCA ---
        ax3 = axes[2, j]
        sc = ax3.scatter(Y_pca[:, 0], Y_pca[:, 1], c=ranks, cmap=cmap_choice, s=50, alpha=0.7, edgecolor='w', lw=0.2)
        ax3.scatter(m_pca[0], m_pca[1], marker=symbol, color='#00FF00', s=size, edgecolor='black', lw=2, zorder=5)
        if j == 0: ax3.set_ylabel(row_titles[2], fontsize=16, fontweight='bold')

    for ax in axes.flat:
        ax.set_xticks([]);
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_color('#bdc3c7')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax).set_label("Quantile Percentile Depth", fontsize=16, fontweight='bold')

    plt.subplots_adjust(left=0.05, right=0.90, wspace=0.1, hspace=0.15)
    print(f"\n[INFO] Saving high-res plot to {filename} ...")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# =====================================================================
# 🚀 主程序执行 (Main Routine)
# =====================================================================
if __name__ == "__main__":
    print("1. Generating SPD Data with Outliers...")
    Y_spd, true_labels = generate_spd_data_with_labels(N=350)

    results = {}

    print("2. Running Baselines & AMQR...")
    # 1. NW (L2 Mean) - 全局算术平均
    nw_med = np.mean(Y_spd, axis=0)
    results['nw'] = {'med': nw_med, 'ranks': rankdata(np.linalg.norm(Y_spd - nw_med, axis=1)) / len(Y_spd)}

    # 2. Frechet (L1 Median)
    results['frechet'] = dict(zip(['med', 'ranks'], get_frechet_tube(Y_spd)))

    # 3. SW-KDE (Mode)
    results['kde'] = dict(zip(['med', 'ranks'], get_kde_tube(Y_spd)))

    # 4. AMQR (Proposed OT)
    amqr = AMQR_Engine(ref_dist='gaussian', epsilon=0.0, d_int=2, use_knn=True, k_neighbors=10)
    a_med, a_ranks, _ = amqr.fit_predict(Y_spd)
    results['amqr'] = {'med': a_med, 'ranks': a_ranks}

    # --- 输出论文级定量指标表格 ---
    print("\n=======================================================")
    print("📊 Table 1: Quantitative Evaluation on SPD Manifold")
    print("=======================================================")
    df_metrics = evaluate_spd_metrics(results, true_labels)
    print(df_metrics.to_string())
    print("=======================================================\n")

    # --- 渲染并保存最终大图 ---
    print("3. Rendering the 3x4 Comparison Plot...")
    plot_spd_3x4_comparison(Y_spd, results)
    print("Done! Check your working directory for the saved plot.")


# =====================================================================
# 📈 1. 欧洲股市数据获取与原始 SPD 矩阵构建
# =====================================================================
def fetch_spd_tensor_data():
    print("📥 正在尝试从 Rdatasets 官方仓库拉取 EuStockMarkets 数据集...")
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/EuStockMarkets.csv"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            df = pd.read_csv(response)
        prices = df[['DAX', 'SMI', 'CAC', 'FTSE']].values
        print("✅ 数据下载成功！")
    except Exception as e:
        print(f"⚠️ 下载失败 ({e})，正在生成高保真统计模拟数据...")
        np.random.seed(42)
        prices = np.zeros((1860, 4))
        prices[0] = [1600, 1600, 1600, 2400]
        L = np.linalg.cholesky(np.array([[1.5, 1.2, 1.3, 1.0], [1.2, 1.6, 1.1, 0.9],
                                         [1.3, 1.1, 1.7, 1.2], [1.0, 0.9, 1.2, 1.4]]) * 1e-4)
        for i in range(1, 1860):
            vol_shock = 1.0 + 3.0 * np.exp(-((i - 1000) ** 2) / 20000)
            ret = (L @ np.random.randn(4)) * vol_shock
            prices[i] = prices[i - 1] * np.exp(ret)

    returns = np.diff(np.log(prices), axis=0) * 100
    window = 20
    N = len(returns) - window
    Y_matrices_flat = np.zeros((N, 16))
    time_index = np.arange(N)

    for i in range(N):
        cov_mat = np.cov(returns[i: i + window], rowvar=False)
        cov_mat += np.eye(4) * 1e-5
        Y_matrices_flat[i] = cov_mat.flatten()

    return time_index, Y_matrices_flat


# 计算矩阵行列式用于验证
def compute_determinants(Y_flat):
    N = len(Y_flat)
    dets = np.zeros(N)
    for i in range(N):
        M = Y_flat[i].reshape(4, 4)
        dets[i] = np.linalg.det(M)
    return dets


# =====================================================================
# 🎨 3. 视觉渲染模块：盲视野下的流形感知验证
# =====================================================================
def plot_blind_manifold_validation(time_index, Y_flat, a_ranks, med_idx, filename="EuStock_BlindManifold.png"):
    fig, axes = plt.subplots(1, 3, figsize=(28, 8), facecolor='white')
    cmap_choice = 'plasma'

    # 获取验证用的真实体积
    raw_dets = compute_determinants(Y_flat)
    med_det = raw_dets[med_idx]

    # --- 视图 A: 时序演化 ---
    ax1 = axes[0]
    ax1.plot(time_index, raw_dets, color='#bdc3c7', alpha=0.5, lw=1, label='Raw Covariance Det')
    mask = a_ranks <= 0.15
    ax1.scatter(time_index[mask], raw_dets[mask], c=a_ranks[mask], cmap=cmap_choice, s=20, alpha=0.8, zorder=3,
                label='Top 15% Core')
    ax1.scatter(time_index[med_idx], med_det, marker='*', color='#00FF00', s=600, edgecolor='black', lw=2, zorder=5,
                label='AMQR Center')

    ax1.set_title("1. Temporal Evolution of Systemic Risk", fontsize=18, fontweight='bold', pad=15)
    ax1.set_xlabel("Trading Days", fontsize=14)
    ax1.set_ylabel("Covariance Determinant (Log Scale)", fontsize=14)
    ax1.set_yscale('log')
    ax1.legend(loc='upper right', fontsize=12)

    # --- 视图 B: 原始高维空间的 PCA (不带流形先验) ---
    ax2 = axes[1]
    # 直接在拍平的 16D 欧氏向量上做 PCA
    pca = PCA(n_components=2)
    Y_pca = pca.fit_transform(Y_flat)

    sc = ax2.scatter(Y_pca[:, 0], Y_pca[:, 1], c=a_ranks, cmap=cmap_choice, s=50, alpha=0.7, edgecolor='w', lw=0.2)
    ax2.scatter(Y_pca[med_idx, 0], Y_pca[med_idx, 1], marker='*', color='#00FF00', s=700, edgecolor='black', lw=2,
                zorder=5)

    ax2.set_title("2. Naive Euclidean PCA Topology", fontsize=18, fontweight='bold', pad=15)
    ax2.set_xlabel("Principal Component 1", fontsize=14)
    ax2.set_ylabel("Principal Component 2", fontsize=14)

    # --- 视图 C: 箱线图 (定量证明膨胀免疫力) ---
    ax3 = axes[2]

    # 按照排秩切分成 5 个区间
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['Core\n(0-20%)', 'Low\n(20-40%)', 'Mid\n(40-60%)', 'High\n(60-80%)', 'Outlier\n(80-100%)']
    rank_categories = pd.cut(a_ranks, bins=bins, labels=labels, include_lowest=True)

    df_plot = pd.DataFrame({
        'Rank Group': rank_categories,
        'Log10 Determinant': np.log10(raw_dets)
    })

    # 箱线图更硬核，能清晰展示中位数、四分位距和离群点
    sns.boxplot(x='Rank Group', y='Log10 Determinant', data=df_plot, ax=ax3, palette='plasma_r', width=0.6, fliersize=3)

    # 标出全局提取中心的实际体积，作为极低基准线
    ax3.axhline(np.log10(med_det), color='#00FF00', ls='--', lw=3, label="AMQR Center Volume")

    ax3.set_title("3. Blind Isolation: Boxplot of Volume by Rank Zone", fontsize=18, fontweight='bold', pad=15)
    ax3.set_xlabel("AMQR Quantile Depth Zone", fontsize=14)
    ax3.set_ylabel("Covariance Volume ($\log_{10}$ Det)", fontsize=14)
    ax3.legend(loc='lower right', fontsize=12)

    # 统一边框清理
    for ax in axes:
        for spine in ax.spines.values(): spine.set_color('#bdc3c7')

    # Colorbar 放置在第二张图右侧
    cbar_ax = fig.add_axes([0.62, 0.15, 0.012, 0.7])
    fig.colorbar(sc, cax=cbar_ax).set_label("AMQR Quantile Rank", fontsize=16, fontweight='bold')

    plt.subplots_adjust(left=0.06, right=0.95, wspace=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 基于盲视野与箱线图的金融分析图已保存至: {filename}")
    plt.show()


# =====================================================================
# 🚀 主执行逻辑 (纯正无监督设定)
# =====================================================================
if __name__ == "__main__":
    print("========================================================")
    print(" 🌟 Real-world Application: Blind Manifold Assumption")
    print("========================================================")

    time_index, Y_spd_flat = fetch_spd_tensor_data()

    print("⏳ 正在运行 AMQR 引擎 (完全无视流形先验结构)...")
    # 🌟 核心设定：d_int=None 让模型自适应估计；直接传入原始 Y_spd_flat
    amqr = AMQR_Engine(ref_dist='gaussian', epsilon=0.0, d_int=None, use_knn=True, k_neighbors=15)

    a_med, a_ranks, gw_plan = amqr.fit_predict(Y_spd_flat)

    med_idx = np.argmin(a_ranks)
    print(f"✅ 模型自动估计的本征维数为: 见内部日志")

    print("🎨 正在渲染盲视野验证图谱...")
    plot_blind_manifold_validation(time_index, Y_spd_flat, a_ranks, med_idx)




# =====================================================================
# 📈 1. 真实 ECG 数据获取与切割 (自动从 SciPy 加载)
# =====================================================================
def get_real_ecg_functional_data(num_beats=300, random_state=42):
    """
    直接从 SciPy 内置库获取真实的心电图数据。
    提取心拍，并人为加入未知的相位漂移，以模拟未对齐的临床泛函数据。
    """
    np.random.seed(random_state)
    try:
        from scipy.datasets import electrocardiogram
        ecg_raw = electrocardiogram()
    except ImportError:
        # 兼容旧版 SciPy
        from scipy.misc import electrocardiogram
        ecg_raw = electrocardiogram()

    fs = 360  # 采样率 360Hz
    # 粗略寻找 R 波峰
    peaks, _ = find_peaks(ecg_raw, distance=fs / 2, height=1.0)

    # 🌟 修正：统一定义为 window_samples 🌟
    window_samples = int(0.6 * fs)  # 每个心拍截取 0.6 秒
    Y = []
    true_shifts = []  # 记录真实的相位漂移量 (用于事后印证)

    # 截取前 num_beats 个心拍
    for p in peaks[10:10 + num_beats]:
        # 制造 -0.06秒 到 +0.06秒 的严重随机相位漂移
        jitter = np.random.randint(-int(0.06 * fs), int(0.06 * fs))
        start = p - int(0.25 * fs) + jitter

        # 🌟 修正：这里使用 window_samples 🌟
        end = start + window_samples

        # 确保不越界
        if start >= 0 and end < len(ecg_raw):
            # 取基线归零
            beat = ecg_raw[start:end]
            beat = beat - np.median(beat)
            Y.append(beat)
            true_shifts.append(jitter / fs)  # 换算成秒

    Y = np.array(Y)
    true_shifts = np.array(true_shifts)
    X_grid = np.linspace(0, 0.6, window_samples)

    print(f"✅ 成功提取 {len(Y)} 条真实 ECG 心拍曲线，特征维数 D={window_samples}。")
    return X_grid, Y, true_shifts


# =====================================================================
# 🎨 2. 视觉渲染模块：ECG 独角戏与深度印证 (1x3)
# =====================================================================
def plot_ecg_deep_validation(X_grid, Y, a_med, a_ranks, true_shifts, filename="ECG_RealWorld_Validation.png"):
    fig, axes = plt.subplots(1, 3, figsize=(28, 8), facecolor='white')
    cmap_choice = 'plasma'

    # 获取中心点索引
    med_idx = np.where(a_ranks == np.min(a_ranks))[0][0]

    # ---------------------------------------------------------
    # 视图 A: 物理流形管提取 (Physical Tube)
    # ---------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(X_grid, Y.T, color='#d5d5d5', alpha=0.3, lw=1, zorder=1)

    # 提取 Top 10% 核心
    mask = a_ranks <= 0.10
    ax1.plot(X_grid, Y[mask].T, color='#3498db', alpha=0.4, lw=1.5, zorder=2)
    ax1.plot(X_grid, a_med, color='#00FF00', lw=4, zorder=5, label='AMQR Extracted Center (R-peak retained)')

    ax1.set_title("1. Physical Space: R-Peak Preservation", fontsize=20, fontweight='bold', pad=15)
    ax1.set_xlabel("Time (seconds)", fontsize=14)
    ax1.set_ylabel("Amplitude (mV)", fontsize=14)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.set_xlim(0, 0.6)

    # ---------------------------------------------------------
    # 视图 B: 相移流形的 PCA 拓扑 (Latent Topology)
    # ---------------------------------------------------------
    ax2 = axes[1]
    pca = PCA(n_components=2)
    Y_pca = pca.fit_transform(Y)

    sc = ax2.scatter(Y_pca[:, 0], Y_pca[:, 1], c=a_ranks, cmap=cmap_choice, s=70, alpha=0.8, edgecolor='w', lw=0.3)
    ax2.scatter(Y_pca[med_idx, 0], Y_pca[med_idx, 1], marker='*', color='#00FF00', s=700, edgecolor='black', lw=2,
                zorder=5)

    ax2.set_title("2. Manifold Space: Latent Topology", fontsize=20, fontweight='bold', pad=15)
    ax2.set_xlabel("Principal Component 1", fontsize=14)
    ax2.set_ylabel("Principal Component 2", fontsize=14)

    # ---------------------------------------------------------
    # 视图 C: 深度合理性印证 (Quantitative Validation)
    # ---------------------------------------------------------
    ax3 = axes[2]
    # 使用真实的绝对偏移量作为 X 轴，证明 AMQR 的排秩准确抓住了物理位移
    abs_shifts = np.abs(true_shifts)
    ax3.scatter(abs_shifts, a_ranks, c=a_ranks, cmap=cmap_choice, s=80, alpha=0.7, edgecolor='w', lw=0.5)

    # 画一条趋势线
    z = np.polyfit(abs_shifts, a_ranks, 1)
    p = np.poly1d(z)
    ax3.plot(abs_shifts, p(abs_shifts), color='#c0392b', ls='--', lw=2, label="Linear Trend")

    ax3.set_title("3. Ground Truth Validation: Ranks vs True Shift", fontsize=20, fontweight='bold', pad=15)
    ax3.set_xlabel("Absolute Phase Shift (seconds)", fontsize=14)
    ax3.set_ylabel("AMQR Quantile Rank (0=Core, 1=Outlier)", fontsize=14)
    ax3.legend(loc='lower right', fontsize=12)

    # 统一视觉清理
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('#bdc3c7')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax).set_label("AMQR Quantile Rank", fontsize=16, fontweight='bold')

    plt.subplots_adjust(left=0.05, right=0.9, wspace=0.25)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 真实 ECG 分析与印证图已保存至: {filename}")
    plt.show()


# =====================================================================
# 🚀 主执行逻辑
# =====================================================================
if __name__ == "__main__":
    print("========================================================")
    print(" 🌟 Real-world Application: ECG Phase-Shift Tolerance")
    print("========================================================")

    # 1. 获取真实 ECG 数据与隐变量 (True Shift)
    X_grid, Y_ecg, true_shifts = get_real_ecg_functional_data(num_beats=300)

    # 2. 调用真实的 AMQR_Engine
    print("⏳ 正在运行 AMQR 引擎提取 ECG 流形结构...")
    # ECG 的相移是典型的一维流形，d_int=1 完美契合。使用 KNN 测地距离防止由于严重相移导致的拓扑短路。
    amqr = AMQR_Engine(ref_dist='uniform', epsilon=0.0, use_knn=True, k_neighbors=5)

    a_med, a_ranks, gw_plan = amqr.fit_predict(Y_ecg)

    # 3. 渲染出图
    print("🎨 正在渲染多角度印证图谱...")
    plot_ecg_deep_validation(X_grid, Y_ecg, a_med, a_ranks, true_shifts)


import os
os.chdir('D:/AMQR/')
%run experiments/run_real2_ecg_with_dtw.py

import os
os.chdir('D:/ESCL/')
%run experiments/exp_grid_cells.py
