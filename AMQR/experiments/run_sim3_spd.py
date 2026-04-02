# experiments/run_sim3_spd.py
import sys
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import rankdata

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from data.simulations import generate_spd_data_with_labels
from models.amqr_engine import AMQR_Engine
from models.baselines import get_frechet_tube, get_kde_tube
from utils.metrics import evaluate_spd_metrics
from utils.visualization import plot_spd_3x4_comparison


def extract_spd_sliding_windows(T, Y_spd, t_eval, amqr_params, window_size=1.5):
    """提取动态 SPD 回归轨迹"""
    centers = {k: [] for k in ['nw', 'f', 'kde', 'a']}
    # 记录全局的 ranks (初始化为 1.0 最边缘)
    ranks_dict = {k: np.ones(len(T)) for k in ['nw', 'f', 'kde', 'a']}

    print("⏳ [1/2] 运行基线 SPD 流形回归 (Sliding Windows)...")
    for t_c in tqdm(t_eval, desc="Baselines"):
        idx = np.where(np.abs(T - t_c) <= window_size / 2.0)[0]
        if len(idx) < 5:
            # 兜底：如果没有数据点，返回单位阵
            dummy_med = np.eye(Y_spd.shape[1])
            centers['nw'].append(dummy_med);
            centers['f'].append(dummy_med);
            centers['kde'].append(dummy_med)
            continue

        Y_c = Y_spd[idx]

        # 1. NW Mean (Euclidean) - 会引发更严重的高维膨胀效应
        nw_med = np.mean(Y_c, axis=0)
        centers['nw'].append(nw_med)

        # 2. Frechet (L1 Median)
        f_med, f_ranks_c = get_frechet_tube(Y_c)
        centers['f'].append(f_med)

        # 3. SW-KDE (Mode)
        kde_med, kde_ranks_c = get_kde_tube(Y_c)
        centers['kde'].append(kde_med)

        # 提取中心切片的 ranks 到全局
        inner_condition = (T[idx] >= t_c - 0.2) & (T[idx] < t_c + 0.2)
        inner_idx = idx[np.where(inner_condition)[0]]
        if len(inner_idx) > 0:
            # 简化局部 ranks 拼接，主要用于可视化对比
            ranks_dict['nw'][inner_idx] = (
                        rankdata(np.linalg.norm(Y_spd[inner_idx] - nw_med, axis=1)) / len(inner_idx))
            ranks_dict['f'][inner_idx] = f_ranks_c[:len(inner_idx)]
            ranks_dict['kde'][inner_idx] = kde_ranks_c[:len(inner_idx)]

    print("\n⏳ [2/2] 运行 AMQR 时空流形回归 (Exact OT without OOS)...")

    amqr = AMQR_Engine(**amqr_params)

    # 🌟 传入时间轴 T 进行联合回归
    traj_a, global_ranks_a = amqr.fit_predict(Y_spd, T=T, t_eval=t_eval, window_size=window_size)

    centers['a'] = [med for _, med in traj_a]
    ranks_dict['a'] = global_ranks_a

    return centers, ranks_dict


if __name__ == "__main__":
    print("========================================================")
    print(" 🌟 Sim 3: Dynamic SPD Matrix Regression (Time-Varying)")
    print("========================================================")

    # ========================================================
    # 🎛️ 审稿人复现开关 (Reproducibility Switch)
    # ========================================================
    AUTO_TUNE = False
    DIM = 3
    NUM_FOLDS = 3
    # ========================================================

    print(f"⏳ 1. Generating {DIM}x{DIM} SPD Data with Outliers...")
    Y_spd, true_labels, R_true = generate_spd_data_with_labels(N=400, dim=DIM)

    # 🌟 模拟一个时间协变量 T，让无序的矩阵产生时间维度的演化
    T = np.linspace(0, 10, len(Y_spd))
    t_eval = np.linspace(T.min(), T.max(), 20)

    # 2. 物理与拓扑先验
    fixed_setup = {
        'ref_dist': 'uniform',  # SPD 常伴有极端长尾，用 Gaussian 先验镇压
        'use_log_squash': True,  # 🌟 黎曼对数压缩：防御非紧致流形体积膨胀的绝对核心！
        'use_knn': True,  # 逼近黎曼测地线
        'd_int': None,  # MLE 自动探测维数
        'max_samples': 2000  # 保持纯净 OT，防 OOS 失真
    }

    # 3. 调参分流逻辑
    if AUTO_TUNE:
        print("\n>> 启动 SPD 时空回归 OOS-GW 交叉验证...")
        from utils.tuning import auto_tune_amqr

        # ⚠️ 必须展平为 2D 才能喂给 scikit-learn 和 cdist
        Y_spd_flat = Y_spd.reshape(len(Y_spd), -1)

        param_grid = {
            'epsilon': [0.0, 0.01, 0.03, 0.05],
            'k_neighbors': [5, 10, 15]
        }
        best_hyperparams = auto_tune_amqr(
            Y_spd_flat, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS,
            T=T, t_eval=t_eval, window_size=1.5
        )
    else:
        print("\n>> 极速模式: 使用物理先验最佳参数...")
        best_hyperparams = {'epsilon': 0.0, 'k_neighbors': 5}

    final_amqr_params = {**fixed_setup, **best_hyperparams}
    print(f"\n🚀 使用最终参数进行全量拟合: {final_amqr_params}")

    # 4. 运行模型提取完整时间轨迹
    centers_dict, ranks_dict = extract_spd_sliding_windows(
        T, Y_spd, t_eval, amqr_params=final_amqr_params, window_size=1.5
    )

    # 5. 为了兼容你现有的评估和可视化代码 (要求单一中心点)，我们提取 t=5.0 截面的结果
    print("\n⏳ 正在将动态轨迹截面适配到静态评估接口...")
    mid_idx = len(t_eval) // 2  # 提取最中间的时刻
    results_static = {
        'nw': {'med': centers_dict['nw'][mid_idx], 'ranks': ranks_dict['nw']},
        'frechet': {'med': centers_dict['f'][mid_idx], 'ranks': ranks_dict['f']},
        'kde': {'med': centers_dict['kde'][mid_idx], 'ranks': ranks_dict['kde']},
        'amqr': {'med': centers_dict['a'][mid_idx], 'ranks': ranks_dict['a']}
    }

    # 6. 定量评估
    print("\n=======================================================")
    print(f"📊 Quantitative Evaluation on {DIM}x{DIM} SPD Regression (Cross-Section)")
    print("=======================================================")
    df_metrics = evaluate_spd_metrics(results_static, true_labels, dim=DIM, R_true=R_true)
    print(df_metrics.to_string())
    print("=======================================================\n")

    # 7. 可视化
    print("🎨 Rendering the 3x4 Comparison Plot...")
    save_img_path = os.path.join(PROJECT_ROOT, "results", "figures", f"Fig5_SPD_{DIM}D_Regression.jpg")
    plot_spd_3x4_comparison(Y_spd, results_static, R_true=R_true, dim=DIM, filename=save_img_path)

    print(f"🎉 {DIM}x{DIM} 维动态 SPD 矩阵回归实验圆满完成！")