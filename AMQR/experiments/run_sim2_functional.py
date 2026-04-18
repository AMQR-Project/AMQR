# experiments/run_sim2_functional.py
import sys
import os
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from data.simulations import generate_drifting_bimodal_functional_data, generate_dynamic_functional_data
from models.amqr_engine import AMQR_Engine

# 🌟 引入顶刊流形基线全家桶
from models.baselines import (
    get_nw_tube,
    get_frechet_regression_tube,
    get_isotropic_geodesic_tube,
    get_kde_mode_tube
)

from utils.visualization import plot_dynamic_functional_2x5, plot_functional_depth_coloring


# experiments/run_sim2_functional.py

def extract_functional_sliding_windows(T, Y_curves, t_eval, amqr_params, window_size=1.5):
    """
    在统一的时间网格 t_eval 上，独立运行 AMQR 及所有基线模型。

    此函数确保了所有模型都在完全相同的条件下进行评估，
    遵循了严谨的并行比较实验设计，消除了方法间的依赖性。
    """
    centers = {}
    # 提取 k_neighbors 以确保 Fréchet 基线与 AMQR 使用相同的图分辨率
    k_neighbors = amqr_params.get('k_neighbors', 5)

    # --- 对称化并行执行流程 ---
    # 所有模型现在都直接在统一的 t_eval 网格上进行滑动窗口回归。

    # 1. AMQR (Proposed Method)
    print("\n⏳ [1/5] Running AMQR (Proposed)...")
    amqr = AMQR_Engine(**amqr_params)
    traj_a, _ = amqr.fit_predict(Y_curves, T=T, t_eval=t_eval, window_size=window_size)
    centers['a'] = [med for _, med in traj_a]

    # 2. Nadaraya-Watson Mean (Baseline)
    print("⏳ [2/5] Running NW Mean (Ambient L2)...")
    nw_traj, _ = get_nw_tube(T, Y_curves, t_eval, window_size=window_size)
    centers['nw'] = [med for _, med in nw_traj]

    # 3. Fréchet L2 Mean Regression (Baseline)
    print("⏳ [3/5] Running Fréchet Regression (Geodesic L2)...")
    f_l2_traj, _ = get_frechet_regression_tube(T, Y_curves, t_eval, window_size=window_size, k_neighbors=k_neighbors)
    centers['f_l2'] = [med for _, med in f_l2_traj]

    # 4. Fréchet L1 Median Regression (Baseline)
    print("⏳ [4/5] Running Isotropic Geodesic (Geodesic L1)...")
    f_l1_traj, _ = get_isotropic_geodesic_tube(T, Y_curves, t_eval, window_size=window_size, k_neighbors=k_neighbors)
    centers['f_l1'] = [med for _, med in f_l1_traj]

    # 5. Sliding-Window KDE (Baseline)
    print("⏳ [5/5] Running SW-KDE (Density Mode)...")
    kde_traj, _ = get_kde_mode_tube(T, Y_curves, t_eval, window_size=window_size)
    centers['kde'] = [med for _, med in kde_traj]

    # 从任一模型的输出中提取最终的时间轴。
    # 由于所有模型都在 t_eval 上运行，它们的输出时间戳理论上应该是一致的
    # （除非某个窗口数据过少被跳过）。
    t_traj = np.array([t for t, _ in traj_a])

    return t_traj, centers


if __name__ == "__main__":
    print("========================================================")
    print(" 🌟 Sim 2: Dynamic Functional Regression (50D, 5 Models)")
    print("========================================================")

    AUTO_TUNE = True
    NUM_FOLDS = 3

    # 1. 生成数据
    T, X_grid, Y_curves, t_eval, GT_surface = generate_dynamic_functional_data(N=1500, D=50)

    # 2. 固定结构先验
    fixed_setup = {
        'ref_dist': 'uniform',
        'd_int': 2,
        'max_samples': 2000,
        'epsilon': 0.0,
        'use_knn': True
    }

    # 3. 调参分流
    if AUTO_TUNE:
        print("\n>> 启动泛函回归 OOS-GW 交叉验证...")
        from utils.tuning import auto_tune_amqr

        param_grid = {'k_neighbors': [3, 5, 8, 12]}
        best_hyperparams = auto_tune_amqr(
            Y_curves, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS,
            T=T, t_eval=t_eval, window_size=1.5
        )
    else:
        best_hyperparams = {'k_neighbors': 5}

    final_amqr_params = {**fixed_setup, **best_hyperparams}
    print(f"\n🚀 使用最终参数进行全量拟合: {final_amqr_params}")

    # 4. 运行模型
    t_traj, centers_dict = extract_functional_sliding_windows(
        T, Y_curves, t_eval, amqr_params=final_amqr_params, window_size=1.5
    )

    # 5. 可视化 (彻底删除定量误差表格计算，直接出图！)
    print("\n🎨 渲染 2x5 泛函时空热力与截面对比图...")
    save_img_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig3_Functional_Dynamic_5Models.pdf")

    # 我们不需要传 GT_surface 进去了，因为我们只依靠原始数据说话
    plot_dynamic_functional_2x5(T, X_grid, Y_curves, t_traj, centers_dict, target_t=7.5, save_path=save_img_path)

    print("🎉 泛函实验圆满结束！")

    print("\n🎨 渲染泛函深度分位数染色对比图...")
    target_t = 7.5
    window_size = 1.5
    idx_slice = np.where(np.abs(T - target_t) <= window_size / 2.0)[0]
    Y_slice = Y_curves[idx_slice]

    # 重新在这个截面上计算一次静态的 Rank
    from scipy.stats import rankdata

    # a. NW 欧氏残差深度
    nw_mean = np.mean(Y_slice, axis=0)
    nw_residuals = np.linalg.norm(Y_slice - nw_mean, axis=1)
    nw_ranks = rankdata(nw_residuals) / len(Y_slice)

    # b. AMQR 拓扑深度
    amqr = AMQR_Engine(**final_amqr_params)
    _, amqr_ranks, _ = amqr._run_with_oos_protection(Y_slice)

    save_color_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig3_Functional_Depth_Coloring.pdf")
    plot_functional_depth_coloring(X_grid, Y_slice, nw_ranks, amqr_ranks, top_ratio=0.5, save_path=save_color_path)


    def analyze_peak_fidelity(t_traj, centers_dict, Y_curves, T):
        """
        计算并可视化估计曲线的峰值保真度
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        # 1. 计算原始样本的平均峰值作为基准 (Baseline)
        raw_peaks = np.max(Y_curves, axis=1)
        baseline_peak = np.mean(raw_peaks)

        methods = ['nw', 'f_l2', 'f_l1', 'kde', 'a']
        labels = ['NW Mean', 'Fréchet L2', 'Fréchet L1', 'SW-KDE', 'AMQR (Proposed)']
        colors = ['#34495e', '#e74c3c', '#e67e22', '#9b59b6', '#27ae60']

        results = []
        plt.figure(figsize=(12, 6), facecolor='white')

        for i, m in enumerate(methods):
            # 提取该方法在所有时间点的估计曲线的最大值
            peaks = [np.max(c) for c in centers_dict[m]]

            # 计算统计量
            m_peak = np.mean(peaks)
            v_peak = np.var(peaks)
            bias = m_peak - baseline_peak
            fidelity = (m_peak / baseline_peak) * 100  # 保真度百分比

            results.append({
                'Method': labels[i],
                'Mean Peak': m_peak,
                'Peak Variance': v_peak,
                'Amplitude Fidelity (%)': fidelity
            })

            # 画出峰值随时间的变化曲线
            plt.plot(t_traj, peaks, label=f"{labels[i]} (Fidelity: {fidelity:.1f}%)", color=colors[i], lw=3)

        # 画出基准线
        plt.axhline(baseline_peak, color='black', ls='--', lw=2, label=f'Raw Sample Avg Peak ({baseline_peak:.2f})')

        plt.title("Peak Amplitude Fidelity over Time\n(AMQR vs Baselines)", fontsize=16, fontweight='bold')
        plt.xlabel("Time (T)", fontsize=14)
        plt.ylabel("Maximum Amplitude of Estimated Curve", fontsize=14)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 3.0)

        plt.savefig("results/figures/Fig3_Peak_Fidelity_Analysis.pdf", bbox_inches='tight')
        plt.show()

        # 打印定量表格
        df_peak = pd.DataFrame(results)
        print("\n" + "=" * 50)
        print("📊 AMPLITUDE FIDELITY QUANTITATIVE RESULTS")
        print("=" * 50)
        print(df_peak.to_string(index=False))
        print("=" * 50)

        return df_peak


    analyze_peak_fidelity(t_traj, centers_dict, Y_curves, T)
