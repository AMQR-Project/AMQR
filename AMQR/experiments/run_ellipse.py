# experiments/run_ellipse.py
import sys
import os
import numpy as np
from scipy.stats import rankdata

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from data.ellipse import generate_straight_ellipse, generate_bent_ellipse
from models.amqr_engine import AMQR_Engine
from models.baselines import _geodesic_l1_median_core
from utils.visualization import plot_combined_motivation_1x4
from utils.tuning import auto_tune_amqr

if __name__ == "__main__":
    print("========================================================")
    print("Running Motivating Example: AMQR vs. Geodesic Fréchet")
    print("========================================================")

    AUTO_TUNE = True
    NUM_FOLDS = 3

    data_results = {'straight': {}, 'bent': {}}

    # ========================================================
    # Separate "physical structure priors" from "numerical hyperparameters"
    # ========================================================
    fixed_setup = {
        'ref_dist': 'uniform',
        'use_knn': True,  # Enable graph geodesic
        'd_int': 2,
        'max_samples': 500,
        'epsilon': 0.0  # Exact GW
    }

    # ========================================================
    # Stage 1: Processing the Straight Ellipse
    # ========================================================
    print("\n[Step 1] Processing the Straight Ellipse...")
    Y_straight = generate_straight_ellipse(N=1000)

    if AUTO_TUNE:
        print(">> Starting AMQR auto-tuning...")
        param_grid = {'k_neighbors': [5, 8, 12, 15]}
        best_params_s = auto_tune_amqr(Y_straight, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS)
    else:
        best_params_s = {'k_neighbors': 5}

    final_params_s = {**fixed_setup, **best_params_s}
    amqr_s = AMQR_Engine(**final_params_s)
    a_med_s, a_ranks_s = amqr_s.fit_predict(Y_straight)

    print(">> Running baseline: Isotropic Geodesic L1...")
    f_med_s, f_dists_s = _geodesic_l1_median_core(Y_straight, k_neighbors=final_params_s['k_neighbors'])

    # Convert absolute distances to depth ranks from 0 to 1 for comparison with AMQR under the same color bar
    f_ranks_s = rankdata(f_dists_s) / len(Y_straight)

    data_results['straight'] = {
        'Y': Y_straight, 'f_med': f_med_s, 'f_ranks': f_ranks_s,
        'a_med': a_med_s, 'a_ranks': a_ranks_s
    }

    # ========================================================
    # Stage 2: Processing the Bent Ellipse
    # ========================================================
    print("\n[Step 2] Processing the Bent Ellipse...")
    Y_bent = generate_bent_ellipse(N=1000)

    if AUTO_TUNE:
        print(">> Starting AMQR auto-tuning...")
        param_grid = {'k_neighbors': [5, 8, 12, 15]}
        best_params_b = auto_tune_amqr(Y_bent, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS)
    else:
        best_params_b = {'k_neighbors': 15}

    final_params_b = {**fixed_setup, **best_params_b}
    amqr_b = AMQR_Engine(**final_params_b)
    a_med_b, a_ranks_b = amqr_b.fit_predict(Y_bent)

    # Baseline: Even with geodesics, Fréchet still cannot solve the problem of anisotropic distribution.
    print(">> Running strong baseline: Isotropic Geodesic L1...")
    f_med_b, f_dists_b = _geodesic_l1_median_core(Y_bent, k_neighbors=final_params_b['k_neighbors'])

    f_ranks_b = rankdata(f_dists_b) / len(Y_bent)

    data_results['bent'] = {
        'Y': Y_bent, 'f_med': f_med_b, 'f_ranks': f_ranks_b,
        'a_med': a_med_b, 'a_ranks': a_ranks_b
    }

    # ========================================================
    # Stage 3: Rendering the plot
    # ========================================================
    print("\nRendering comparison plot...")
    save_filepath = os.path.join(PROJECT_ROOT, "results", "figures", "Fig1_Motivation_Geodesic_Comparison.pdf")
    plot_combined_motivation_1x4(data_results, save_path=save_filepath)

    print(f"Motivation experiment completed successfully, results saved to: {save_filepath}")
