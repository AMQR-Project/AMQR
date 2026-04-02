# utils/metrics.py
import numpy as np
import pandas as pd
import scipy.linalg as slinalg
from sklearn.metrics import roc_auc_score



def evaluate_spiral_metrics(t_traj, centers, masks, T_sorted, P_sorted):
    """
    评估 3D 穿孔螺旋流形的追踪精度、径向偏离度和核心纯净度。
    """
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

        # 2. 径向偏离度 (越接近 3.0 越好)
        est_radii = np.sqrt(est_centers[:, 1] ** 2 + est_centers[:, 2] ** 2)
        radial_dev = np.mean(np.abs(est_radii - 3.0))

        # 3. 核心纯净度 (提取出的 20% Mask 到真实 GT 中心线的平均距离)
        mask = masks[k]
        P_masked = P_sorted[mask]
        T_masked = P_masked[:, 0]

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


import numpy as np
import pandas as pd


def evaluate_dynamic_functional_metrics(t_traj, X_grid, GT_surface, centers_dict, global_t=None):
    """
    评估动态泛函回归的时空综合误差。
    自动对齐被滑动窗口边缘截断的时间点。

    :param t_traj: 模型实际输出的有效时间点数组 (例如长度 28)
    :param X_grid: 泛函曲线的 50 维空间网格 (D维)
    :param GT_surface: 真实的泛函曲面 (例如长度 50)
    :param centers_dict: 模型输出的中心字典
    :param global_t: 生成 GT_surface 时使用的全局时间网格 (例如长度 50)。用于时间轴对齐。
    """
    metrics = []
    method_names = {'nw': 'NW Mean', 'frechet': 'FRÉCHET', 'kde': 'SW-KDE', 'amqr': 'AMQR (Proposed)'}
    # 兼容你代码里的 key，如果你的 key 是单字母 'f', 'a' 等，请换回你原来的 method_names
    if 'a' in centers_dict:
        method_names = {'nw': 'NW Mean', 'f': 'FRÉCHET', 'kde': 'SW-KDE', 'a': 'AMQR (Proposed)'}

    # ==========================================
    # 🌟 核心修复：自动对齐真实数据与预测数据的时间轴
    # ==========================================
    if len(GT_surface) != len(t_traj):
        if global_t is not None:
            # 根据物理时间 t_traj 去全局网格 global_t 里寻找对应的真实截面
            valid_idx = [np.argmin(np.abs(global_t - t)) for t in t_traj]
            GT_surface_aligned = GT_surface[valid_idx]
        else:
            # 如果没传 global_t，假设截断是对称的 (危险做法，仅作 fallback)
            diff = len(GT_surface) - len(t_traj)
            start_idx = diff // 2
            GT_surface_aligned = GT_surface[start_idx: start_idx + len(t_traj)]
            print(f"⚠️ 警告: 未提供 global_t，已根据长度差强行对称截断 GT (跳过前后各 {start_idx} 个点)。")
    else:
        GT_surface_aligned = GT_surface
    # ==========================================

    for k, name in method_names.items():
        if k not in centers_dict: continue

        est_surface = np.array(centers_dict[k])  # shape: (len(t_traj), D)

        # 1. 平均振幅畸变误差 (Amplitude Error)
        peak_est = np.max(est_surface, axis=1)
        peak_gt = np.max(GT_surface_aligned, axis=1)
        amp_error = np.mean(np.abs(peak_est - peak_gt))

        # 2. 平均相位偏移误差 (Phase Shift Error)
        phase_est = X_grid[np.argmax(est_surface, axis=1)]
        phase_gt = X_grid[np.argmax(GT_surface_aligned, axis=1)]
        phase_error = np.mean(np.abs(phase_est - phase_gt))

        # 3. 时空全局保真度 (Shape RMSE)
        rmse = np.sqrt(np.mean((est_surface - GT_surface_aligned) ** 2))

        metrics.append({
            'Method': name,
            'Amplitude Error ↓': round(amp_error, 3),
            'Phase Shift Error ↓': round(phase_error, 3),
            'Global RMSE ↓': round(rmse, 3)
        })

    return pd.DataFrame(metrics).set_index('Method')

def compute_spd_all_props(Y_flat, dim=2):
    N = len(Y_flat)
    dets = np.zeros(N)
    eigs = np.zeros((N, dim))
    log_Y_flat = np.zeros((N, dim * dim))
    for i in range(N):
        M = Y_flat[i].reshape(dim, dim)
        dets[i] = np.linalg.det(M)
        vals, vecs = np.linalg.eigh(M)  # 默认返回从小到大排序的特征值
        eigs[i] = vals
        log_M = vecs @ np.diag(np.log(np.clip(vals, 1e-10, None))) @ vecs.T
        log_Y_flat[i] = log_M.flatten()
    return dets, eigs, log_Y_flat

def evaluate_spd_metrics(results_dict, true_labels, dim, R_true):
    """自适应维度的定量评估"""
    # 动态构建真实的 M_true
    l_true = np.array([0.4] * (dim - 1) + [3.0])
    M_true = R_true @ np.diag(l_true) @ R_true.T
    log_M_true = slinalg.logm(M_true)
    det_true = 3.0 * (0.4 ** (dim - 1))

    metrics = []
    for model_name, res in results_dict.items():
        M_est = res['med'].reshape(dim, dim)
        ranks = res['ranks']

        # 1. Log-Euclidean Distance (LED)
        log_M_est = slinalg.logm(M_est)
        led_error = np.linalg.norm(log_M_est - log_M_true, ord='fro')

        # 2. Volume Bias
        det_est = np.linalg.det(M_est)
        det_bias = np.abs(det_est - det_true)

        # 3. Spectral Error (比较所有特征值的 L1 误差)
        vals_est = np.linalg.eigvalsh(M_est) # 从小到大
        spectral_error = np.sum(np.abs(vals_est - l_true))

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