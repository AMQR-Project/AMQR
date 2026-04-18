import sys
import os
import numpy as np
from tqdm import tqdm

# 接入项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from data.simulations import generate_circular_manifold_with_gap
from models.amqr_engine import AMQR_Engine

# 🌟 导入顶刊基线全家桶 (统一的滑动窗口接口)
from models.baselines import (
    get_nw_tube,
    get_frechet_regression_tube,
    get_isotropic_geodesic_tube,
    get_kde_mode_tube
)

from utils.metrics import evaluate_spiral_metrics
from utils.visualization import plot_2x5_spiral_experiment  # 注意：下游绘图脚本需要适配 5 个模型
from utils.tuning import auto_tune_amqr

AUTO_TUNE = False
NUM_FOLDS = 3


def extract_all_models(T, P_3D, amqr_params, window_size=1.2, step_size=0.1, u_target=0.20):
    """
    统一调度 5 大模型，利用封装好的底层引擎，代码极其清爽
    """
    t_eval = np.arange(T.min(), T.max() + step_size, step_size)
    k_neighbors = amqr_params.get('k_neighbors', 15)

    # ===============================================================
    # 📉 1. 基线模型大满贯 (Baselines)
    # ===============================================================
    print("\n⏳ [1/5] Running Baseline 1: NW (环境空间 L2 均值)...")
    nw_traj, nw_ranks = get_nw_tube(T, P_3D, t_eval, window_size=window_size)

    print("\n⏳ [2/5] Running Baseline 2: Fréchet Regression (内蕴测地线 L2 均值 SOTA)...")
    f_l2_traj, f_l2_ranks = get_frechet_regression_tube(T, P_3D, t_eval, window_size=window_size,
                                                        k_neighbors=k_neighbors)

    print("\n⏳ [3/5] Running Baseline 3: Isotropic Geodesic (内蕴测地线 L1 中位数)...")
    f_l1_traj, f_l1_ranks = get_isotropic_geodesic_tube(T, P_3D, t_eval, window_size=window_size,
                                                        k_neighbors=k_neighbors)

    print("\n⏳ [4/5] Running Baseline 4: SW-KDE (滑动窗口密度寻模)...")
    kde_traj, kde_ranks = get_kde_mode_tube(T, P_3D, t_eval, window_size=window_size)

    # ===============================================================
    # 👑 2. AMQR 引擎 (Exact GW)
    # ===============================================================
    print("\n⏳ [5/5] Running AMQR (调用引擎原生滑动窗口与时空同步)...")
    amqr = AMQR_Engine(**amqr_params)
    amqr_traj, amqr_ranks = amqr.fit_predict(P_3D, T=T, t_eval=t_eval, window_size=window_size)

    # ===============================================================
    # 📦 3. 提取结果，组装字典
    # ===============================================================
    t_traj = np.array([t for t, _ in nw_traj])  # 所有模型的 t_eval 是一样的

    centers = {
        'nw': np.array([c for _, c in nw_traj]),
        'f_l2': np.array([c for _, c in f_l2_traj]),
        'f_l1': np.array([c for _, c in f_l1_traj]),
        'kde': np.array([c for _, c in kde_traj]),
        'a': np.array([c for _, c in amqr_traj])
    }

    masks = {
        'nw': nw_ranks <= u_target,
        'f_l2': f_l2_ranks <= u_target,
        'f_l1': f_l1_ranks <= u_target,
        'kde': kde_ranks <= u_target,
        'a': amqr_ranks <= u_target
    }

    return t_traj, centers, masks


if __name__ == "__main__":
    print("========================================================")
    print(" 🌟 Sim 1: 3D Spiral Manifold with Topological Gap (5 Models)")
    print("========================================================")

    # 1. 数据生成 (15000 个点)
    T, P_3D, GT_3D = generate_circular_manifold_with_gap(n_points=15000)
    sort_idx = np.argsort(T)
    T_sorted, P_sorted = T[sort_idx], P_3D[sort_idx]

    t_eval = np.arange(T_sorted.min(), T_sorted.max() + 0.1, 0.1)

    # 2. 固定结构先验
    fixed_setup = {
        'ref_dist': 'uniform',
        'use_knn': True,
        'd_int': None,
        'max_samples': 2500,
        'epsilon': 0.0
    }

    # 3. 调参分流逻辑
    if AUTO_TUNE:
        print("\n>> 强行启动回归流形 OOS-GW 交叉验证...")
        from utils.tuning import auto_tune_amqr

        param_grid = {'k_neighbors': [5, 10, 15]}
        best_hyperparams = auto_tune_amqr(
            P_sorted, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS,
            T=T_sorted, t_eval=t_eval, window_size=0.1
        )
    else:
        print("\n>> 极速模式: 使用人工验证的最佳物理先验参数 (From Prior Knowledge)")
        best_hyperparams = {'k_neighbors': 10}

    final_amqr_params = {**fixed_setup, **best_hyperparams}
    print(f"\n🚀 使用最终参数进行全量验证: {final_amqr_params}")

    # 4. 模型执行
    t_traj, centers, masks = extract_all_models(
        T_sorted, P_sorted, amqr_params=final_amqr_params,
        window_size=0.1, step_size=0.05, u_target=0.20
    )

    # 5. 定量评估与输出表格
    print("\n=======================================================")
    print("📊 Quantitative Evaluation on Spiral Gap (All 5 Models)")
    print("=======================================================")
    # 删除了 GT 传参，直接依靠数据本身计算内蕴误差
    df_metrics = evaluate_spiral_metrics(t_traj, centers, masks, T_sorted, P_sorted, window_size=0.1)
    print(df_metrics.to_string())

    os.makedirs(os.path.join(PROJECT_ROOT, "results", "tables"), exist_ok=True)
    df_metrics.to_csv(os.path.join(PROJECT_ROOT, "results", "tables", "Table_Spiral_Metrics_5Models.csv"))
    print("=======================================================\n")

    # 6. 可视化渲染大图
    print("🎨 绘制 2x5 穿孔流形终极对比图中...")
    save_img_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig2_Spiral_Gap_5Models.pdf")

    # 去掉了 GT_3D 参数
    plot_2x5_spiral_experiment(T_sorted, P_sorted, t_traj, centers, masks, target_t=5.0, save_path=save_img_path)

    print("🎉 3D螺旋实验大满贯运行完毕！")
