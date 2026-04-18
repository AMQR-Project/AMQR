# data/simulations.py
import numpy as np
import scipy.linalg as slinalg
from scipy.stats import ortho_group


def generate_circular_manifold_with_gap(n_points=15000):
    """
    生成带有 135° 拓扑断层的 3D 螺旋/圆柱流形点云。
    用于测试模型在遭遇“真空断层”时的拓扑鲁棒性。
    """
    np.random.seed(42)
    T = np.random.uniform(0, 10, n_points)
    R = np.random.normal(3.0, 0.2, n_points)
    theta_center = T * (np.pi / 2.5)
    theta = np.random.vonmises(mu=theta_center, kappa=1.5, size=n_points)

    # 制造 135° (3π/4) 处的断层
    rel_theta = (theta - theta_center + np.pi) % (2 * np.pi) - np.pi
    gap_center = 3 * np.pi / 4
    gap_width = 0.5
    in_gap = np.abs(rel_theta - gap_center) < gap_width

    # 在 gap 区域只保留 2% 的点，形成真空区
    keep_prob = np.where(in_gap, 0.02, 1.0)
    keep_mask = np.random.rand(n_points) < keep_prob

    T, R, theta = T[keep_mask], R[keep_mask], theta[keep_mask]
    Y1, Y2 = R * np.cos(theta), R * np.sin(theta)
    P_3D = np.column_stack([T, Y1, Y2])

    # 生成绝对真实的骨架 Ground Truth
    T_gt = np.linspace(0, 10, 200)
    theta_gt = T_gt * (np.pi / 2.5)
    GT_3D = np.column_stack([T_gt, 3.0 * np.cos(theta_gt), 3.0 * np.sin(theta_gt)])

    return T, P_3D, GT_3D


def generate_bimodal_crescent(N=1500, bridge_ratio=0.35, thickness=0.4, random_state=42):
    """
    生成一个密度不平衡且具有可控厚度的马蹄形（弯月）流形。
    修复了物理真空与边界截断奇异点问题，确保整体绝对连续。

    参数:
    - N: 总样本数
    - bridge_ratio: 基础连续数据占总数据的比例 (调高它 = 整体底座变密)
    - thickness: 流形法向(径向)的噪声方差 (调高它 = 桥变厚)
    - random_state: 随机种子，设置为 None 可每次生成不同数据
    """
    if random_state is not None:
        np.random.seed(random_state)

    # 1. 精准控制数据分布密度 (Mass Distribution)
    N_bridge = int(N * bridge_ratio)
    N_ends = N - N_bridge
    N_left = N_ends // 2
    N_right = N_ends - N_left

    # 生成角度 theta
    # 基础拓扑底色：确保整个流形连通
    theta_bridge = np.random.uniform(low=0.0, high=np.pi, size=N_bridge)

    # 左右双峰：叠加极度密集的高斯分布
    theta_left = np.random.normal(loc=0.1 * np.pi, scale=0.08, size=N_left)
    theta_right = np.random.normal(loc=0.9 * np.pi, scale=0.08, size=N_right)

    # 拼接所有角度 (🚨 移除了 np.clip，允许月牙尖端自然过渡，避免 k-NN 奇异点)
    theta = np.concatenate([theta_left, theta_right, theta_bridge])

    # 2. 引入径向厚度 (Tubular Thickness)
    R_base = 5.0  # 基础半径
    r_noise = np.random.normal(loc=0, scale=thickness, size=N)
    r = R_base + r_noise

    # 3. 极坐标转笛卡尔坐标
    X = r * np.cos(theta)
    Y = r * np.sin(theta)

    data = np.vstack((X, Y)).T
    return data


def generate_dynamic_functional_data(N=1000, D=50, random_state=42):
    """
    生成带自变量 T 的动态泛函数据 (Dynamic Functional Data)
    真实波峰随时间 T 作正弦漂移，但存在固定在 X=3.0 的高密度离群陷阱。
    """
    np.random.seed(random_state)
    T = np.random.uniform(0, 10, N)
    X_grid = np.linspace(-5, 5, D)
    Y = np.zeros((N, D))

    for i in range(N):
        t = T[i]
        # 【真实波峰动态轨迹】：随时间 T 正弦摆动
        true_center = 2.0 * np.sin(t * np.pi / 5.0)

        rand_val = np.random.rand()
        if rand_val < 0.85:
            # 85% 真实数据，锚定在 true_center，方差较大 (较散)
            shift = np.random.normal(true_center, 0.8)
        else:
            # 15% 离群陷阱，死死固定在右侧 3.0，方差极小 (极密)
            shift = np.random.normal(3.0, 0.1)

        amp = np.random.normal(2.0, 0.2)
        base_curve = amp * np.exp(-((X_grid - shift) ** 2) / 0.5)
        noise = np.random.normal(0, 0.05, D)

        Y[i] = base_curve + noise

    # 生成用于评估的绝对纯净 Ground Truth 表面
    t_eval = np.linspace(0, 10, 50)
    GT_surface = np.zeros((len(t_eval), D))
    for i, t in enumerate(t_eval):
        true_shift = 2.0 * np.sin(t * np.pi / 5.0)
        GT_surface[i] = 2.0 * np.exp(-((X_grid - true_shift) ** 2) / 0.5)

    return T, X_grid, Y, t_eval, GT_surface


def generate_asymmetric_functional_data(N=1500, D=50):
    """
    生成具有不对称相位偏移的泛函数据 (Asymmetric Phase Shifts)
    80% 的曲线向左侧轻微偏移，20% 的曲线向右侧剧烈偏移
    """
    np.random.seed(42)
    T = np.linspace(0, 10, N)
    X_grid = np.linspace(-5, 5, D)

    Y_curves = np.zeros((N, D))
    GT_surface = np.zeros((N, D))

    for i in range(N):
        # 🌟 制造不对称的相位偏移 (Asymmetric Phase Shift)
        # 80% 的概率向左偏 (均值 -1.5，方差小)
        # 20% 的概率向右偏 (均值 +2.5，方差大)
        if np.random.rand() < 0.8:
            phase_shift = np.random.normal(-1.5, 0.5)
        else:
            phase_shift = np.random.normal(2.5, 0.8)

        # 基础波形：标准高斯钟形曲线
        amplitude = 2.0 + np.random.normal(0, 0.1)  # 振幅保持基本一致

        # 叠加波形
        Y_curves[i, :] = amplitude * np.exp(-0.5 * ((X_grid - phase_shift) / 0.8) ** 2)

        # 叠加一点高频白噪声
        Y_curves[i, :] += np.random.normal(0, 0.05, D)

        # 记录完美的无噪声基准形状 (作为参考)
        GT_surface[i, :] = 2.0 * np.exp(-0.5 * (X_grid / 0.8) ** 2)

    return T, X_grid, Y_curves, T, GT_surface


# data/simulations.py (新增函数)

def generate_drifting_bimodal_functional_data(N=1500, D=50, random_state=42):
    """
    生成具有时变双峰混合分布的泛函数据。

    核心特性：
    - 曲线的相位偏移来自两个高斯分布的混合。
    - 混合权重 p(t) 是时间 t 的函数，导致数据重心从左侧平滑漂移至右侧。
    - 这会创造一个非线性的、随时间动态演化的真实条件中位数/均值轨迹。
    """
    if random_state is not None:
        np.random.seed(random_state)

    T = np.linspace(0, 10, N)
    X_grid = np.linspace(-5, 5, D)
    Y_curves = np.zeros((N, D))

    # 定义两个模式的中心
    mean_left = -1.5
    mean_right = 2.5

    for i in range(N):
        t = T[i]

        # 🌟 核心修改：混合概率 p_left(t) 随时间 t 动态变化
        # 使用余弦函数实现从 1 到 0 的平滑过渡
        # t=0  -> p_left=1.0 (完全左模式)
        # t=5  -> p_left=0.5 (50/50 混合)
        # t=10 -> p_left=0.0 (完全右模式)
        p_left = (np.cos(t * np.pi / 10.0) + 1.0) / 2.0

        if np.random.rand() < p_left:
            # 从左侧模式采样
            phase_shift = np.random.normal(mean_left, 0.5)
        else:
            # 从右侧模式采样
            phase_shift = np.random.normal(mean_right, 0.8)

        # 基础波形与噪声（保持不变）
        amplitude = 2.0 + np.random.normal(0, 0.1)
        Y_curves[i, :] = amplitude * np.exp(-0.5 * ((X_grid - phase_shift) / 0.8) ** 2)
        Y_curves[i, :] += np.random.normal(0, 0.05, D)

    # 🌟 核心修改：生成新的、动态变化的 Ground Truth 表面
    # GT 的中心轨迹现在是两个模式中心的期望值，随 p_left(t) 变化
    t_eval = np.linspace(0, 10, 100)
    GT_surface = np.zeros((len(t_eval), D))
    for i, t in enumerate(t_eval):
        p_left_t = (np.cos(t * np.pi / 10.0) + 1.0) / 2.0
        # 真实中心 = 左模式中心 * p_left + 右模式中心 * (1 - p_left)
        true_shift = p_left_t * mean_left + (1 - p_left_t) * mean_right

        GT_surface[i] = 2.0 * np.exp(-0.5 * ((X_grid - true_shift) / 0.8) ** 2)

    # 注意：返回的 t_eval 长度现在是100，以获得更平滑的GT表面
    return T, X_grid, Y_curves, t_eval, GT_surface


def generate_spd_data_with_labels(N=300, dim=2, random_state=42):
    """
    生成任意维度的 SPD 矩阵数据，并返回严格的经验流形中心。
    """
    np.random.seed(random_state)
    Y = np.zeros((N, dim * dim))
    labels = np.zeros(N)

    # 1. 构建绝对的真实基准旋转矩阵 R_true
    np.random.seed(0)
    R_true = ortho_group.rvs(dim) if dim > 2 else np.array(
        [[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    np.random.seed(random_state)

    for i in range(N):
        is_outlier = np.random.rand() < 0.20
        labels[i] = 1 if is_outlier else 0

        # 初始化特征值向量
        l_vals = np.zeros(dim)

        if is_outlier:
            # 【膨胀噪声】：主特征值极大，其余也大
            l_vals[-1] = np.random.normal(12.0, 1.0)
            for j in range(dim - 1):
                l_vals[j] = np.random.normal(2.0, 0.5)

            # 李代数扰动
            A = np.random.normal(0, 0.1, (dim, dim))
            S = A - A.T  # 斜对称矩阵
            R = slinalg.expm(S)
        else:
            # 【核心正常值】：主特征值 3.0，其余较小 0.4
            l_vals[-1] = np.random.lognormal(mean=np.log(3.0), sigma=0.4)
            for j in range(dim - 1):
                l_vals[j] = np.random.lognormal(mean=np.log(0.4), sigma=0.2)

            # 李代数扰动：在 R_true 附近进行微小扰动
            A = np.random.normal(0, 0.3, (dim, dim))
            S = A - A.T
            R = R_true @ slinalg.expm(S)

        # 组合特征分解: M = R * Lambda * R^T
        M = R @ np.diag(l_vals) @ R.T
        Y[i] = M.flatten()

    # =========================================================
    # 🌟 核心修复：计算纯净样本 (labels==0) 的经验 Log-Euclidean Mean
    # 这代表了这批数据在去除了所有异常值后，最绝对、最真实的几何重心
    # =========================================================
    clean_matrices = Y[labels == 0].reshape(-1, dim, dim)
    log_sum = np.zeros((dim, dim))
    for M in clean_matrices:
        log_sum += slinalg.logm(M).real

    log_mean = log_sum / len(clean_matrices)
    SPD_true = slinalg.expm(log_mean).real  # 映射回 SPD 流形

    # 将 SPD_true 作为 Ground Truth 返回 (替代原来错误的 R_true)
    return Y, labels, SPD_true
