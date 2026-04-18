# experiments/run_bimodal_crescent.py
import sys
import os
import numpy as np
from scipy.stats import rankdata

# 接入项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from data.simulations import generate_bimodal_crescent
from models.amqr_engine import AMQR_Engine

# 🌟 修复：直接导入静态流形的底层核心算法 (Core Estimators)
from models.baselines import _frechet_l2_mean_core, _geodesic_l1_median_core, _kde_mode_core

# 假设你的可视化工具箱支持动态接收列表画图
try:
    from utils.visualization import plot_bimodal_crescent_1x5
except ImportError:
    # 如果还没有 1x5 函数，建议你在 utils 里把原来的 1x4 改成动态长度支持，或者直接 rename
    print("⚠️ 提示: 请确保 visualization.py 中存在支持 5 个子图的 plot 函数。")

if __name__ == "__main__":
    print("========================================================")
    print(" 🌟 Running Sub-Experiment: Dumbbell / Bimodal Crescent")
    print("========================================================")

    # 1. 生成数据 (哑铃状：两端密集，中间桥梁稀疏)
    print("⏳ Generating bimodal crescent/dumbbell data...")
    Y = generate_bimodal_crescent(N=2000, bridge_ratio=0.5, thickness=0.5)

    # 2. 运行模型
    print("⏳ Running all baseline models...")

    # ---------------------------------------------------------
    # A. Ambient Euclidean Mean (欧氏均值 - 基线 1)
    # ---------------------------------------------------------
    nw_med = np.mean(Y, axis=0)
    nw_ranks = rankdata(np.linalg.norm(Y - nw_med, axis=1)) / len(Y)

    # ---------------------------------------------------------
    # B. Intrinsic Fréchet L2 Mean (测地线 L2 均值 - 基线 2)
    # ---------------------------------------------------------
    print(">> Running Fréchet L2 Mean...")
    fl2_med, fl2_depths = _frechet_l2_mean_core(Y, k_neighbors=10)
    fl2_ranks = rankdata(fl2_depths) / len(Y)

    # ---------------------------------------------------------
    # C. Intrinsic Fréchet L1 Median (测地线 L1 中位数 - 基线 3)
    # ---------------------------------------------------------
    print(">> Running Fréchet L1 Median...")
    fl1_med, fl1_depths = _geodesic_l1_median_core(Y, k_neighbors=10)
    fl1_ranks = rankdata(fl1_depths) / len(Y)

    # ---------------------------------------------------------
    # D. KDE Mode (核密度寻模 - 基线 4)
    # ---------------------------------------------------------
    print(">> Running KDE Density Mode...")
    kde_med, kde_depths = _kde_mode_core(Y)
    kde_ranks = rankdata(kde_depths) / len(Y)

    # ---------------------------------------------------------
    # E. AMQR (Proposed Method)
    # ---------------------------------------------------------
    fixed_setup = {
        'ref_dist': 'uniform',  # 明确边界，无长尾
        'use_knn': True,  # Pathway B: 依赖图逼近不跨越真空
        'd_int': 2,  # 强制 2D 隐空间，完美适配 2D 哑铃
        'max_samples': 500,
        'epsilon': 0.0  # 🚨 理论铁律：Exact GW 零模糊，防止概率泄漏
    }

    print(">> Running Proposed AMQR...")
    best_hyperparams = {'k_neighbors': 25}  # 保持与 Fréchet 一致的图分辨率，以示公平
    final_params = {**fixed_setup, **best_hyperparams}

    amqr = AMQR_Engine(**final_params)
    a_med, a_ranks = amqr.fit_predict(Y)

    # 3. 组织数据并调用绘图
    meds = [nw_med, fl2_med, fl1_med, kde_med, a_med]
    ranks = [nw_ranks, fl2_ranks, fl1_ranks, kde_ranks, a_ranks]
    method_names = ["Euclidean Mean", "Fréchet L2 Mean", "Fréchet L1 Median", "KDE Mode", "Proposed AMQR"]

    print("🎨 Rendering plots...")
    save_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig_Dumbbell_Comparison.pdf")

    # 这里需要确保你的 plot 函数能处理 5 个输入。
    plot_bimodal_crescent_1x5(Y, meds, ranks, save_path=save_path)

    print(f"🎉 哑铃形状小品实验圆满完成！(请在本地查看 {save_path})")
