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

    # 运行 AMQR 引擎
    print("⏳ 正在运行 AMQR 引擎 (构建日内连续流形管道)...")
    amqr = AMQR_Engine(ref_dist='uniform', epsilon=0.0, d_int=None,
                       use_log_squash=True, use_knn=True, k_neighbors=15)

    t_eval_hours = np.arange(0, 24, 1)
    trajectory, _ = amqr.fit_predict(Y_flat, T=T, t_eval=t_eval_hours, window_size=2.0)

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