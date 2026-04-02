# experiments/run_sim2_functional.py
import sys
import os
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from data.simulations import generate_dynamic_functional_data
from models.amqr_engine import AMQR_Engine
from models.baselines import get_frechet_tube, get_kde_tube
from utils.metrics import evaluate_dynamic_functional_metrics
from utils.visualization import plot_dynamic_functional_2x4


# 🌟 1. 修改提取函数：用 AMQR "探路"，确保所有基线模型的时间轴绝对对齐
def extract_functional_sliding_windows(T, Y_curves, t_eval, amqr_params, window_size=1.5):
    centers = {k: [] for k in ['nw', 'f', 'kde', 'a']}

    # 1. 优先运行 AMQR 引擎 (它会自适应跳过拓扑饥饿的危险边缘截面)
    print("\n⏳ 运行 AMQR 时空泛函回归 (Exact OT without OOS distortion)...")
    amqr = AMQR_Engine(**amqr_params)
    traj_a, _ = amqr.fit_predict(Y_curves, T=T, t_eval=t_eval, window_size=window_size)

    # 🌟 提取出 AMQR 认为安全的、成功算出来的实际时间轴 t_traj
    t_traj = np.array([t for t, _ in traj_a])
    centers['a'] = [med for _, med in traj_a]

    # 2. 用这个绝对安全的 t_traj 来运行基线模型！保证四个模型长度 100% 一致！
    print(f"\n⏳ 运行基线泛函回归 (共 {len(t_traj)} 个安全截面)...")
    for t_c in tqdm(t_traj, desc="Baselines"):
        idx = np.where(np.abs(T - t_c) <= window_size / 2.0)[0]
        Y_c = Y_curves[idx]

        centers['nw'].append(np.mean(Y_c, axis=0))
        f_med, _ = get_frechet_tube(Y_c)
        centers['f'].append(f_med)
        kde_med, _ = get_kde_tube(Y_c)
        centers['kde'].append(kde_med)

    return t_traj, centers


if __name__ == "__main__":
    print("========================================================")
    print(" 🌟 Sim 2: Dynamic Functional Regression (50D)")
    print("========================================================")

    # ========================================================
    # 🎛️ 审稿人复现开关 (Reproducibility Switch)
    # ========================================================
    AUTO_TUNE = True  # 设为 True 看看机器的几何直觉！
    NUM_FOLDS = 3
    # ========================================================

    # 1. 生成数据
    T, X_grid, Y_curves, t_eval, GT_surface = generate_dynamic_functional_data(N=1500, D=50)

    # 2. 固定结构先验
    fixed_setup = {
        'ref_dist': 'uniform',
        'd_int': None,  # 曲线随时间演化，流形核心骨架为 1D 轨迹
        'max_samples': 2000  # 🌟 阀门全开，禁止局部窗口发生欧氏 OOS 短路拉扯
    }

    # 3. 调参分流逻辑
    if AUTO_TUNE:
        print("\n>> 启动泛函回归 OOS-GW 交叉验证...")
        from utils.tuning import auto_tune_amqr

        param_grid = {
            'epsilon': [0.0, 0.01, 0.03, 0.05],
            'k_neighbors': [5, 10, 15],
            'use_knn': [True, False]
        }
        best_hyperparams = auto_tune_amqr(
            Y_curves, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS,
            T=T, t_eval=t_eval, window_size=1.5
        )
    else:
        print("\n>> 极速模式: 使用论文汇报的最佳参数...")
        best_hyperparams = {'epsilon': 0.01, 'use_knn': True, 'k_neighbors': 5}

    final_amqr_params = {**fixed_setup, **best_hyperparams}
    print(f"\n🚀 使用最终参数进行全量拟合: {final_amqr_params}")

    # 4. 运行模型 (🚨 千万要把 window_size 改回宽窗口 1.5 甚至 2.0！)
    t_traj, centers_dict = extract_functional_sliding_windows(
        T, Y_curves, t_eval, amqr_params=final_amqr_params, window_size=1.5
    )

    # 5. 定量评估
    print("\n=======================================================")
    print("📊 Quantitative Evaluation on Functional Data")
    print("=======================================================")
    # 🌟 调用上一轮刚写好的带对齐功能的评估函数
    df_metrics = evaluate_dynamic_functional_metrics(
        t_traj=t_traj,
        X_grid=X_grid,
        GT_surface=GT_surface,
        centers_dict=centers_dict,
        global_t=t_eval
    )
    print(df_metrics.to_string())

    # 自动保存表格
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "tables"), exist_ok=True)
    df_metrics.to_csv(os.path.join(PROJECT_ROOT, "results", "tables", "Table_Functional_Metrics.csv"))
    print("=======================================================\n")

    # 6. 可视化 (🌟 画图前，也把真实曲面截取对齐，防止画图函数崩溃)
    print("🎨 渲染 2x4 全局热力与截面对比图...")
    valid_idx = [np.argmin(np.abs(t_eval - t)) for t in t_traj]
    GT_surface_aligned = GT_surface[valid_idx]

    save_img_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig3_Functional_Dynamic.jpg")
    plot_dynamic_functional_2x4(T, X_grid, Y_curves, t_traj, GT_surface_aligned, centers_dict,
                                target_t=7.5, save_path=save_img_path)

    print("🎉 泛函实验圆满结束！")