# experiments/run_ellipse.py
import sys
import os
import numpy as np
from scipy.stats import rankdata

# 🌟 核心操作：将项目根目录加入系统路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# 1. 导入数据生成器
from data.ellipse import generate_straight_ellipse, generate_bent_ellipse
# 2. 导入核心模型与基线
from models.amqr_engine import AMQR_Engine
# 🌟 修改点：引入强力流形基线，抛弃“稻草人”欧氏版
from models.baselines import _geodesic_l1_median_core
# 3. 导入画图与调参工具
from utils.visualization import plot_combined_motivation_1x4
from utils.tuning import auto_tune_amqr

if __name__ == "__main__":
    print("========================================================")
    print(" 🌟 Running Motivating Example: AMQR vs. Geodesic Fréchet")
    print("========================================================")

    AUTO_TUNE = True
    NUM_FOLDS = 3

    data_results = {'straight': {}, 'bent': {}}

    # ========================================================
    # 📝 核心学术逻辑：分离“物理结构先验”与“数值超参数”
    # ========================================================
    fixed_setup = {
        'ref_dist': 'uniform',
        'use_knn': True,  # 🌟 开启图测地线
        'd_int': 2,
        'max_samples': 500,
        'epsilon': 0.0  # 🚨 Exact GW 保持理论严谨性
    }

    # ========================================================
    # 🟢 阶段一：处理直椭圆 (Straight Ellipse)
    # ========================================================
    print("\n[Step 1] 正在处理直椭圆 (Straight Ellipse)...")
    Y_straight = generate_straight_ellipse(N=1000)

    if AUTO_TUNE:
        print(">> 启动 AMQR 自动调参...")
        param_grid = {'k_neighbors': [5, 8, 12, 15]}
        best_params_s = auto_tune_amqr(Y_straight, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS)
    else:
        best_params_s = {'k_neighbors': 5}

    final_params_s = {**fixed_setup, **best_params_s}
    amqr_s = AMQR_Engine(**final_params_s)
    a_med_s, a_ranks_s = amqr_s.fit_predict(Y_straight)

    # 🌟 升级版 Baseline：使用测地线 Fréchet L1 Median。
    # 🛡️ 学术防御：我们强制赋予基线模型与 AMQR 完全相同的 k_neighbors。
    # 这保证了两者拥有完全相同的底层拓扑图信息，从而将性能差异严格归因于
    # "各向同性距离等高线" vs "各向异性 GW2 拓扑对齐" 的数学架构差异。
    print(">> 正在运行强力基线: Isotropic Geodesic L1...")
    f_med_s, f_dists_s = _geodesic_l1_median_core(Y_straight, k_neighbors=final_params_s['k_neighbors'])

    # 将绝对距离转化为 0~1 的深度秩，以便和 AMQR 放在同一个色带下比较
    f_ranks_s = rankdata(f_dists_s) / len(Y_straight)

    data_results['straight'] = {
        'Y': Y_straight, 'f_med': f_med_s, 'f_ranks': f_ranks_s,
        'a_med': a_med_s, 'a_ranks': a_ranks_s
    }

    # ========================================================
    # 🌙 阶段二：处理弯曲椭圆 (Bent Ellipse)
    # ========================================================
    print("\n[Step 2] 正在处理弯曲椭圆 (Bent Ellipse)...")
    Y_bent = generate_bent_ellipse(N=1000)

    if AUTO_TUNE:
        print(">> 启动 AMQR 自动调参...")
        param_grid = {'k_neighbors': [5, 8, 12, 15]}
        best_params_b = auto_tune_amqr(Y_bent, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS)
    else:
        best_params_b = {'k_neighbors': 15}

    final_params_b = {**fixed_setup, **best_params_b}
    amqr_b = AMQR_Engine(**final_params_b)
    a_med_b, a_ranks_b = amqr_b.fit_predict(Y_bent)

    # 🌟 升级版 Baseline：即便有了测地线，Fréchet 依然无法解决各向异性的分布问题。
    print(">> 正在运行强力基线: Isotropic Geodesic L1...")
    f_med_b, f_dists_b = _geodesic_l1_median_core(Y_bent, k_neighbors=final_params_b['k_neighbors'])

    # 🌟 修复 Bug：分母必须是当前数据集的长度
    f_ranks_b = rankdata(f_dists_b) / len(Y_bent)

    data_results['bent'] = {
        'Y': Y_bent, 'f_med': f_med_b, 'f_ranks': f_ranks_b,
        'a_med': a_med_b, 'a_ranks': a_ranks_b
    }

    # ========================================================
    # 🎨 阶段三：渲染出图
    # ========================================================
    print("\n🎨 正在渲染 1x4 终极对比图 (测地线对决)...")
    save_filepath = os.path.join(PROJECT_ROOT, "results", "figures", "Fig1_Motivation_Geodesic_Comparison.pdf")
    plot_combined_motivation_1x4(data_results, save_path=save_filepath)

    print(f"🎉 动机实验圆满结束！结果已保存至: {save_filepath}")
