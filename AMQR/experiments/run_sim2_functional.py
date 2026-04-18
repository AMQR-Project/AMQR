# experiments/run_sim2_functional.py
import sys
import os
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from data.simulations import generate_drifting_bimodal_functional_data, generate_dynamic_functional_data
from models.amqr_engine import AMQR_Engine

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
    Independently run AMQR and all baseline models on a unified time grid t_eval.

    This function ensures that all models are evaluated under exactly the same conditions,
    following a rigorous parallel comparison experimental design and eliminating inter-method dependencies.
    """
    centers = {}
    # Extract k_neighbors to ensure Fréchet baselines use the same graph resolution as AMQR
    k_neighbors = amqr_params.get('k_neighbors', 5)

    # --- Symmetrical Parallel Execution Flow ---
    # All models now perform sliding window regression directly on the unified t_eval grid.

    # 1. AMQR (Proposed Method)
    print("\n[1/5] Running AMQR (Proposed)...")
    amqr = AMQR_Engine(**amqr_params)
    traj_a, _ = amqr.fit_predict(Y_curves, T=T, t_eval=t_eval, window_size=window_size)
    centers['a'] = [med for _, med in traj_a]

    # 2. Nadaraya-Watson Mean (Baseline)
    print("[2/5] Running NW Mean (Ambient L2)...")
    nw_traj, _ = get_nw_tube(T, Y_curves, t_eval, window_size=window_size)
    centers['nw'] = [med for _, med in nw_traj]

    # 3. Fréchet L2 Mean Regression (Baseline)
    print("[3/5] Running Fréchet Regression (Geodesic L2)...")
    f_l2_traj, _ = get_frechet_regression_tube(T, Y_curves, t_eval, window_size=window_size, k_neighbors=k_neighbors)
    centers['f_l2'] = [med for _, med in f_l2_traj]

    # 4. Fréchet L1 Median Regression (Baseline)
    print("[4/5] Running Isotropic Geodesic (Geodesic L1)...")
    f_l1_traj, _ = get_isotropic_geodesic_tube(T, Y_curves, t_eval, window_size=window_size, k_neighbors=k_neighbors)
    centers['f_l1'] = [med for _, med in f_l1_traj]

    # 5. Sliding-Window KDE (Baseline)
    print("[5/5] Running SW-KDE (Density Mode)...")
    kde_traj, _ = get_kde_mode_tube(T, Y_curves, t_eval, window_size=window_size)
    centers['kde'] = [med for _, med in kde_traj]

    # Extract the final time axis from the output of any model.
    # Since all models run on t_eval, their output timestamps should theoretically be consistent
    # (unless a window with too little data is skipped).
    t_traj = np.array([t for t, _ in traj_a])

    return t_traj, centers


if __name__ == "__main__":
    print("========================================================")
    print(" Sim 2: Dynamic Functional Regression (50D, 5 Models)")
    print("========================================================")

    AUTO_TUNE = True
    NUM_FOLDS = 3

    # 1. Generate data
    T, X_grid, Y_curves, t_eval, GT_surface = generate_dynamic_functional_data(N=1500, D=50)

    # 2. Fixed structural priors
    fixed_setup = {
        'ref_dist': 'uniform',
        'd_int': 2,
        'max_samples': 2000,
        'epsilon': 0.0,
        'use_knn': True
    }

    # 3. Hyperparameter tuning branch
    if AUTO_TUNE:
        print("\n>> Starting functional regression OOS-GW cross-validation...")
        from utils.tuning import auto_tune_amqr

        param_grid = {'k_neighbors': [3, 5, 8, 12]}
        best_hyperparams = auto_tune_amqr(
            Y_curves, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS,
            T=T, t_eval=t_eval, window_size=1.5
        )
    else:
        best_hyperparams = {'k_neighbors': 5}

    final_amqr_params = {**fixed_setup, **best_hyperparams}
    print(f"\nUsing final parameters for full fitting: {final_amqr_params}")

    # 4. Run models
    t_traj, centers_dict = extract_functional_sliding_windows(
        T, Y_curves, t_eval, amqr_params=final_amqr_params, window_size=1.5
    )

    # 5. Visualization (Completely remove quantitative error table calculation, plot directly!)
    print("\nRendering 2x5 functional spatio-temporal heatmap and cross-section comparison plot...")
    save_img_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig3_Functional_Dynamic_5Models.pdf")

    # We don't need to pass in GT_surface anymore, because we rely solely on the raw data
    plot_dynamic_functional_2x5(T, X_grid, Y_curves, t_traj, centers_dict, target_t=7.5, save_path=save_img_path)

    print("Functional data experiment finished")

    print("\nRendering functional depth quantile coloring comparison plot...")
    target_t = 7.5
    window_size = 1.5
    idx_slice = np.where(np.abs(T - target_t) <= window_size / 2.0)[0]
    Y_slice = Y_curves[idx_slice]

    # Re-calculate the static Rank on this cross-section
    from scipy.stats import rankdata

    # a. NW Euclidean residual depth
    nw_mean = np.mean(Y_slice, axis=0)
    nw_residuals = np.linalg.norm(Y_slice - nw_mean, axis=1)
    nw_ranks = rankdata(nw_residuals) / len(Y_slice)

    # b. AMQR topological depth
    amqr = AMQR_Engine(**final_amqr_params)
    _, amqr_ranks, _ = amqr._run_with_oos_protection(Y_slice)

    save_color_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig3_Functional_Depth_Coloring.pdf")
    plot_functional_depth_coloring(X_grid, Y_slice, nw_ranks, amqr_ranks, top_ratio=0.5, save_path=save_color_path)


    def analyze_peak_fidelity(t_traj, centers_dict, Y_curves, T):
        """
        Calculate and visualize the peak fidelity of the estimated curves
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        # 1. Calculate the average peak of the raw samples as a baseline
        raw_peaks = np.max(Y_curves, axis=1)
        baseline_peak = np.mean(raw_peaks)

        methods = ['nw', 'f_l2', 'f_l1', 'kde', 'a']
        labels = ['NW Mean', 'Fréchet L2', 'Fréchet L1', 'SW-KDE', 'AMQR (Proposed)']
        colors = ['#34495e', '#e74c3c', '#e67e22', '#9b59b6', '#27ae60']

        results = []
        plt.figure(figsize=(12, 6), facecolor='white')

        for i, m in enumerate(methods):
            # Extract the maximum value of the estimated curve for this method at all time points
            peaks = [np.max(c) for c in centers_dict[m]]

            # Calculate statistics
            m_peak = np.mean(peaks)
            v_peak = np.var(peaks)
            bias = m_peak - baseline_peak
            fidelity = (m_peak / baseline_peak) * 100  # Fidelity percentage

            results.append({
                'Method': labels[i],
                'Mean Peak': m_peak,
                'Peak Variance': v_peak,
                'Amplitude Fidelity (%)': fidelity
            })

            # Plot the curve of peak value change over time
            plt.plot(t_traj, peaks, label=f"{labels[i]} (Fidelity: {fidelity:.1f}%)", color=colors[i], lw=3)

        # Plot the baseline
        plt.axhline(baseline_peak, color='black', ls='--', lw=2, label=f'Raw Sample Avg Peak ({baseline_peak:.2f})')

        plt.title("Peak Amplitude Fidelity over Time\n(AMQR vs Baselines)", fontsize=16, fontweight='bold')
        plt.xlabel("Time (T)", fontsize=14)
        plt.ylabel("Maximum Amplitude of Estimated Curve", fontsize=14)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 3.0)

        plt.savefig("results/figures/Fig3_Peak_Fidelity_Analysis.pdf", bbox_inches='tight')
        plt.show()

        # Print the quantitative table
        df_peak = pd.DataFrame(results)
        print("\n" + "=" * 50)
        print("AMPLITUDE FIDELITY QUANTITATIVE RESULTS")
        print("=" * 50)
        print(df_peak.to_string(index=False))
        print("=" * 50)

        return df_peak


    analyze_peak_fidelity(t_traj, centers_dict, Y_curves, T)
