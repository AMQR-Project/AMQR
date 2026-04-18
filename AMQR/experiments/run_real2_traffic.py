import sys
import os
import numpy as np
import scipy.linalg
from scipy.interpolate import interp1d
from scipy.stats import rankdata

# 获取项目根目录
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

from models.amqr_engine import AMQR_Engine
from data.real_data import load_pems_traffic_and_locations
from utils.visualization import plot_traffic_tube_validation, plot_spatial_grid, plot_local_matrix_grid


def compute_lem_anomaly_scores(Y_raw_flat, Y_reg_flat, dim=20):
    print("⏳ 正在计算 Log-Euclidean 黎曼流形异常绝对距离...")
    N = Y_raw_flat.shape[0]
    scores = np.zeros(N)
    for i in range(N):
        M_raw = Y_raw_flat[i].reshape(dim, dim) + np.eye(dim) * 1e-5
        M_reg = Y_reg_flat[i].reshape(dim, dim) + np.eye(dim) * 1e-5
        log_raw = scipy.linalg.logm(M_raw).real
        log_reg = scipy.linalg.logm(M_reg).real
        scores[i] = np.linalg.norm(log_raw - log_reg)
    return scores


if __name__ == "__main__":
    print("========================================================")
    print(" 🚗 Pipeline: Urban Traffic Topology Regression & Mapping")
    print("========================================================")

    # 配置文件路径
    h5_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'pems-bay.h5')
    csv_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'graph_sensor_locations_bay.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'graph_sensor_locations.csv')
    save_dir = os.path.join(PROJECT_ROOT, "results", "figures")
    DIM = 20

    # 加载数据
    Y_flat, T, timestamps, coords, sensor_ids = load_pems_traffic_and_locations(h5_path, csv_path, num_nodes=DIM)

    # ========================================================
    # 🌟 修复核心：预计算真正的 Log-Euclidean 黎曼度量矩阵
    # ========================================================
    print("⏳ 正在预计算全局 Log-Euclidean 黎曼距离矩阵 (Metric-Agnostic 注入)...")
    from scipy.spatial.distance import pdist, squareform

    # 1. 提取所有真实的 SPD 矩阵，并映射到切空间 (Log-space)
    Y_matrices = Y_flat.reshape(-1, DIM, DIM)
    Y_log_space = np.array([scipy.linalg.logm(M + np.eye(DIM) * 1e-5).real for M in Y_matrices])

    # 2. 在切空间中计算欧氏距离，这在数学上精确等价于 SPD 流形上的 Log-Euclidean 测地线距离
    # 彻底告别环境欧氏距离！
    lem_dist_matrix = squareform(pdist(Y_log_space.reshape(len(Y_flat), -1), metric='euclidean'))
    # ========================================================

    # ========================================================
    # 🌟 核心修复：全局 MLE 本征维度估计 (Global Intrinsic Dimension)
    # 替代硬编码的 210 维，也避免局部窗口的小样本抖动
    # ========================================================
    print("⏳ 正在全局估计交通流形的有效本征维度 (Global MLE)...")
    k_mle = min(10, len(Y_flat) - 1)
    # 利用已经算好的全局精确黎曼距离矩阵
    dists = np.sort(lem_dist_matrix, axis=1)[:, 1:k_mle + 2]
    dists = np.maximum(dists, 1e-9)
    r_k = dists[:, -1:]
    mle_val = (k_mle - 1) / np.sum(np.log(r_k / dists[:, :-1]), axis=1)
    global_d_int = int(np.round(np.mean(mle_val)))
    global_d_int = max(1, global_d_int)  # 强制封顶

    print(f"🎯 全局 MLE 自动探测到的有效本征维度为: d_int = {global_d_int} (远小于环境维度 210)")

    # 运行 AMQR 引擎，注入全局稳定的 d_int
    print(f"⏳ 正在运行 AMQR 引擎...")
    amqr = AMQR_Engine(ref_dist='uniform', epsilon=0.0, d_int=global_d_int,
                       use_log_squash=False, use_knn=False)

    # 🌟 将黎曼度量矩阵 y_dist_m 完美注入引擎
    t_eval_hours = np.arange(0, 24, 1)
    trajectory, _ = amqr.fit_predict(Y_flat, y_dist_m=lem_dist_matrix, T=T, t_eval=t_eval_hours, window_size=2.0)

    t_sparse = np.array([t for t, _ in trajectory])
    Y_reg_sparse = np.array([med for _, med in trajectory])

    # 补全插值与计算排秩
    print("⏳ 正在通过最近邻映射补全样本期望...")
    interpolator = interp1d(t_sparse, Y_reg_sparse, axis=0, kind='nearest', fill_value="extrapolate")
    Y_reg_flat_full = interpolator(T)

    # 这里的 anomaly_scores 是绝对距离
    anomaly_scores = compute_lem_anomaly_scores(Y_flat, Y_reg_flat_full, dim=DIM)
    a_ranks = rankdata(anomaly_scores) / len(anomaly_scores)

    print("\n🚀 开始全管线可视化渲染...")
    # 1. 1x3 全局管道图 (必须用全局 a_ranks)
    plot_traffic_tube_validation(T, Y_flat, t_sparse, Y_reg_sparse, a_ranks, save_dir, dim=DIM)

    # 2. 3x4 地理空间拓扑图 (传入绝对距离 scores，内部算局部)
    plot_spatial_grid(Y_flat, anomaly_scores, T, coords, save_dir, dim=DIM)

    # 🌟 3. 新增：3x4 微观矩阵热力图 (传入绝对距离 scores，内部算局部)
    plot_local_matrix_grid(Y_flat, anomaly_scores, T, save_dir, dim=DIM)

    print("\n🎉 Pipeline 执行完毕！所有顶刊图表已生成。")
