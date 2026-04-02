import sys
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import rankdata

# 接入项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from data.simulations import generate_circular_manifold_with_gap
from models.amqr_engine import AMQR_Engine
from models.baselines import get_frechet_tube, get_kde_tube
from utils.metrics import evaluate_spiral_metrics
from utils.visualization import plot_2x4_spiral_experiment
from utils.tuning import auto_tune_amqr

AUTO_TUNE = False
NUM_FOLDS = 3


def extract_all_models(T, P_3D, amqr_params, window_size=1.2, step_size=0.1, u_target=0.20):
    n = len(T)
    t_eval = np.arange(T.min(), T.max() + step_size, step_size)

    masks = {k: np.zeros(n, dtype=bool) for k in ['nw', 'f', 'kde', 'a']}
    centers = {k: [] for k in ['nw', 'f', 'kde']}
    t_traj = []

    # ===============================================================
    # 📉 1. 基线模型 (Baselines)：依然需要手动循环，手动拼接 Mask
    # ===============================================================
    print("⏳ [1/2] 正在运行基线模型 (NW, Fréchet, KDE)...")
    for t_c in tqdm(t_eval, desc="Baselines Sliding Window"):
        idx = np.where(np.abs(T - t_c) <= window_size / 2.0)[0]
        if len(idx) < 15: continue

        inner_condition = (T[idx] >= t_c - step_size / 2.0) & (T[idx] < t_c + step_size / 2.0)
        inner_in_idx = np.where(inner_condition)[0]
        P_c = P_3D[idx]

        # A. NW Baseline
        nw_med = np.mean(P_c, axis=0)
        if len(inner_in_idx) > 0:
            masks['nw'][idx[inner_in_idx]] = (rankdata(np.linalg.norm(P_c - nw_med, axis=1)) / len(P_c))[
                                                 inner_in_idx] <= u_target

        # B. Fréchet Baseline
        f_med, f_ranks = get_frechet_tube(P_c)
        if len(inner_in_idx) > 0: masks['f'][idx[inner_in_idx]] = f_ranks[inner_in_idx] <= u_target

        # C. SW-KDE Baseline
        kde_med, kde_ranks = get_kde_tube(P_c)
        if len(inner_in_idx) > 0: masks['kde'][idx[inner_in_idx]] = kde_ranks[inner_in_idx] <= u_target

        t_traj.append(t_c)
        centers['nw'].append(nw_med)
        centers['f'].append(f_med)
        centers['kde'].append(kde_med)

    # ===============================================================
    # 👑 2. AMQR 引擎：一行代码直接接管滑动窗口与平滑计算！
    # ===============================================================
    # 🌟 用外部传进来的最优参数实例化
    print("\n⏳ [2/2] 正在运行 AMQR (调用引擎原生滑动窗口与时空平滑)...")
    amqr = AMQR_Engine(**amqr_params)

    # 🌟 见证奇迹的时刻：只要传入 T 和 t_eval，循环、抽样、补全全在内部完成！
    traj_a, global_ranks_a = amqr.fit_predict(P_3D, T=T, t_eval=t_eval, window_size=window_size)

    # 提取 AMQR 的结果对齐数据结构 (轨迹点和 Mask)
    centers['a'] = [med for _, med in traj_a]
    masks['a'] = global_ranks_a <= u_target

    return np.array(t_traj), {k: np.array(v) for k, v in centers.items()}, masks


if __name__ == "__main__":
    print("========================================================")
    print(" 🌟 Sim 1: 3D Spiral Manifold with Topological Gap")
    print("========================================================")

    # 1. 数据生成 (15000 个点)
    T, P_3D, GT_3D = generate_circular_manifold_with_gap(n_points=15000)
    sort_idx = np.argsort(T)
    T_sorted, P_sorted = T[sort_idx], P_3D[sort_idx]

    t_eval = np.arange(T_sorted.min(), T_sorted.max() + 0.1, 0.1)

    # 2. 固定结构先验
    fixed_setup = {
        'ref_dist': 'uniform',
        'use_knn': True,  # 必须用测地线
        'd_int': None,  # 螺旋核心结构为 1D 线条
        'max_samples': 2500
    }

    # 3. 调参分流逻辑
    if AUTO_TUNE:
        print("\n>> 强行启动回归流形 OOS-GW 交叉验证...")
        from utils.tuning import auto_tune_amqr

        param_grid = {
            'epsilon': [0.0, 0.01, 0.03, 0.05],
            'k_neighbors': [5, 10, 15]
        }
        # 传入 T 和 t_eval 激活双模引擎的时空对齐机制
        best_hyperparams = auto_tune_amqr(
            P_sorted, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS,
            T=T_sorted, t_eval=t_eval, window_size=0.1
        )
    else:
        print("\n>> 极速模式: 使用人工验证的最佳物理先验参数 (From Prior Knowledge)")
        # 填入你手动试出来效果最好的那组参数
        best_hyperparams = {'epsilon': 0.0, 'k_neighbors': 10}

    final_amqr_params = {**fixed_setup, **best_hyperparams}
    print(f"\n🚀 使用最终参数进行全量验证: {final_amqr_params}")

    # 4. 模型执行 (必须确保你的 extract_all_models 已经修改为能接收 amqr_params)
    t_traj, centers, masks = extract_all_models(
        T_sorted, P_sorted, amqr_params=final_amqr_params,
        window_size=0.1, step_size=0.05, u_target=0.20
    )

    # 5. 定量评估与输出表格
    print("\n=======================================================")
    print("📊 Quantitative Evaluation on Spiral Gap")
    print("=======================================================")
    df_metrics = evaluate_spiral_metrics(t_traj, centers, masks, T_sorted, P_sorted)
    print(df_metrics.to_string())

    # 自动保存表格
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "tables"), exist_ok=True)
    df_metrics.to_csv(os.path.join(PROJECT_ROOT, "results", "tables", "Table_Spiral_Metrics.csv"))
    print("=======================================================\n")

    # 4. 可视化渲染大图
    print("🎨 绘制 2x4 穿孔流形对比图中...")
    save_img_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig2_Spiral_Gap.jpg")
    plot_2x4_spiral_experiment(T_sorted, P_sorted, GT_3D, t_traj, centers, masks, target_t=5.0, save_path=save_img_path)

    print("🎉 3D螺旋实验圆满结束！")
