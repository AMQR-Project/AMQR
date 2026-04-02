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


def generate_bimodal_crescent():
    """
    生成带有稀疏桥梁的双峰月牙流形 (Bimodal Crescent)。
    特点：两端密度极高，中间连接处极度稀疏。
    用于测试模型抗“局部密度陷阱(Density Mode)”和“欧氏真空(Euclidean Void)”的能力。
    """
    # np.random.seed(42)
    # 高密度集群 1 (右端)
    theta1 = np.random.normal(0, 0.2, 400)
    # 高密度集群 2 (左端)
    theta2 = np.random.normal(np.pi, 0.2, 400)
    # 极度稀疏的中间连接桥梁 (中间部分)
    theta3 = np.random.uniform(0, np.pi, 50)

    theta = np.concatenate([theta1, theta2, theta3])
    R = np.random.normal(5, 0.3, len(theta))

    Y = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
    return Y


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