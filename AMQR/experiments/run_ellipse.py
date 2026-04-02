# experiments/run_ellipse.py
import sys
import os

# 🌟 核心操作：将项目根目录加入系统路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# 1. 导入数据生成器
from data.ellipse import generate_straight_ellipse, generate_bent_ellipse
# 2. 导入核心模型与基线
from models.amqr_engine import AMQR_Engine
from models.baselines import get_frechet_tube
# 3. 导入画图与调参工具
from utils.visualization import plot_combined_motivation_1x4
from utils.tuning import auto_tune_amqr  # 🌟 引入我们新写的自动调参模块

if __name__ == "__main__":
    print("========================================================")
    print(" 🌟 Running Motivating Example with Auto-Tuning")
    print("========================================================")

    AUTO_TUNE = True
    NUM_FOLDS = 3

    data_results = {'straight': {}, 'bent': {}}

    # ========================================================
    # 📝 核心学术逻辑：分离“物理结构先验”与“数值超参数”
    # ========================================================
    # 1. 固定结构先验 (根据流形物理属性定死，不参与调参)
    fixed_setup = {
        'ref_dist': 'uniform',  # 椭圆边界清晰无长尾
        'use_knn': True,  # 必须开启测地线以适应弯月拓扑
        'd_int': None,  # 强制 2D 隐空间以提取管状厚度
        'max_samples': 500
    }

    # ========================================================
    # 🟢 阶段一：处理直椭圆 (Straight Ellipse)
    # ========================================================
    print("\n[Step 1] 正在生成并处理直椭圆 (Straight Ellipse)...")
    Y_straight = generate_straight_ellipse(N=1000)

    if AUTO_TUNE:
        print(">> 启动 OOS-GW 交叉验证自动调参...")
        param_grid = {'epsilon': [0.0, 0.01, 0.03, 0.05], 'k_neighbors': [5, 10, 15]}
        best_params_s = auto_tune_amqr(Y_straight, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS)
    else:
        print(">> 极速模式: 使用预设的最佳参数 (from Paper Table X)")
        best_params_s = {'epsilon': 0.0, 'k_neighbors': 5}  # 填入你跑出来的最优值

    final_params_s = {**fixed_setup, **best_params_s}
    amqr_s = AMQR_Engine(**final_params_s)
    a_med_s, a_ranks_s = amqr_s.fit_predict(Y_straight)

    # 跑 Baseline
    f_med_s, f_ranks_s = get_frechet_tube(Y_straight)

    data_results['straight'] = {
        'Y': Y_straight, 'f_med': f_med_s, 'f_ranks': f_ranks_s,
        'a_med': a_med_s, 'a_ranks': a_ranks_s
    }

    # ========================================================
    # 🌙 阶段二：处理弯曲椭圆 (Bent Ellipse)
    # ========================================================
    print("\n[Step 2] 正在生成并处理弯曲椭圆 (Bent Ellipse)...")
    Y_bent = generate_bent_ellipse(N=1000)

    if AUTO_TUNE:
        print(">> 启动 OOS-GW 交叉验证自动调参...")
        param_grid = {'epsilon': [0.0, 0.01, 0.03, 0.05], 'k_neighbors': [5, 10, 15]}
        best_params_b = auto_tune_amqr(Y_bent, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS)
    else:
        print(">> 极速模式: 使用预设的最佳参数 (from Paper Table X)")
        best_params_b = {'epsilon': 0.0, 'k_neighbors': 15}  # 填入你跑出来的最优值

    final_params_b = {**fixed_setup, **best_params_b}
    amqr_b = AMQR_Engine(**final_params_b)
    a_med_b, a_ranks_b = amqr_b.fit_predict(Y_bent)

    # 跑 Baseline
    f_med_b, f_ranks_b = get_frechet_tube(Y_bent)

    data_results['bent'] = {
        'Y': Y_bent, 'f_med': f_med_b, 'f_ranks': f_ranks_b,
        'a_med': a_med_b, 'a_ranks': a_ranks_b
    }

    # ========================================================
    # 🎨 阶段三：渲染出图
    # ========================================================
    print("\n🎨 正在渲染 1x4 终极对比图...")
    save_filepath = os.path.join(PROJECT_ROOT, "results", "figures", "Fig1_Motivation_Ellipse_AutoTuned.jpg")
    plot_combined_motivation_1x4(data_results, save_path=save_filepath)

    print("🎉 动机实验 (带自动调参) 圆满结束！")