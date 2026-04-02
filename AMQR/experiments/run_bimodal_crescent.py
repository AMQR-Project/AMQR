# experiments/run_bimodal_crescent.py
import sys
import os
import numpy as np
from scipy.stats import rankdata

# 接入项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# 导入拆分好的模块
from data.simulations import generate_bimodal_crescent
from models.amqr_engine import AMQR_Engine
from models.baselines import get_frechet_tube, get_kde_tube
from utils.visualization import plot_bimodal_crescent_1x4
from utils.tuning import auto_tune_amqr

if __name__ == "__main__":
    print("========================================================")
    print(" 🌟 Running Sub-Experiment: Bimodal Crescent")
    print("========================================================")

    AUTO_TUNE = False
    NUM_FOLDS = 3

    # 1. 生成数据
    print("⏳ Generating bimodal crescent data...")
    Y = generate_bimodal_crescent()

    # 2. 运行模型
    print("⏳ Running models...")

    # A. NW Mean (全局欧氏均值)
    nw_med = np.mean(Y, axis=0)
    nw_ranks = rankdata(np.linalg.norm(Y - nw_med, axis=1)) / len(Y)

    # B. Fréchet L1 Median
    f_med, f_ranks = get_frechet_tube(Y)

    # C. SW-KDE (局部密度最高点)
    kde_med, kde_ranks = get_kde_tube(Y)

    # D. AMQR (Proposed)
    # 1. 根据数据的物理属性，定死“结构与先验参数”
    fixed_setup = {
        'ref_dist': 'uniform',  # 明确边界，无长尾
        'use_knn': True,  # 存在物理真空
        'd_int': None,
        'max_samples': 500
    }

    # 2. 定义你需要机器自动搜索的“超参数网格”
    param_grid = {
        'epsilon': [0.0, 0.01, 0.03, 0.05],
        'k_neighbors': [5, 10, 15]
    }

    if AUTO_TUNE:
        print(">> 启动 OOS-GW 交叉验证自动调参...")
        param_grid = {'epsilon': [0.0, 0.01, 0.03, 0.05], 'k_neighbors': [5, 10, 15]}
        best_hyperparams = auto_tune_amqr(Y, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS)
    else:
        print(">> 极速模式: 使用预设的最佳参数 (from Paper Table X)")
        best_hyperparams = {'epsilon': 0.05, 'k_neighbors': 5}  # 填入你跑出来的最优值

    # 4. 把机器选出来的最佳参数合并进去，进行最终的拟合与画图！
    final_params = {**fixed_setup, **best_hyperparams}
    print(f"🚀 使用最终参数进行全量拟合: {final_params}")

    # 调用引擎：强制 d_int=1, 自动进行大样本降维与补全
    amqr = AMQR_Engine(**final_params)
    a_med, a_ranks = amqr.fit_predict(Y)

    # 3. 组织数据并画图
    meds = [nw_med, f_med, kde_med, a_med]
    ranks = [nw_ranks, f_ranks, kde_ranks, a_ranks]

    print("🎨 Rendering plots...")
    save_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig_Bimodal_Crescent.jpg")
    plot_bimodal_crescent_1x4(Y, meds, ranks, save_path=save_path)

    print("🎉 双峰月牙小品实验圆满完成！")