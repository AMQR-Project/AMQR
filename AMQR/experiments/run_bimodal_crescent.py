# experiments/run_bimodal_crescent.py
import sys
import os
import numpy as np
from scipy.stats import rankdata

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from data.simulations import generate_bimodal_crescent
from models.amqr_engine import AMQR_Engine

# Directly import the underlying core algorithms for static manifolds (Core Estimators)
from models.baselines import _frechet_l2_mean_core, _geodesic_l1_median_core, _kde_mode_core
from utils.visualization import plot_bimodal_crescent_1x5


if __name__ == "__main__":
    print("========================================================")
    print("Running Sub-Experiment: Dumbbell / Bimodal Crescent")
    print("========================================================")

    # 1. Generate data (dumbbell shape: dense at both ends, sparse in the middle bridge)
    print("Generating bimodal crescent/dumbbell data...")
    Y = generate_bimodal_crescent(N=2000, bridge_ratio=0.5, thickness=0.5)

    # 2. Run models
    print("Running all baseline models...")

    # ---------------------------------------------------------
    # A. Ambient Euclidean Mean (Euclidean Mean - Baseline 1)
    # ---------------------------------------------------------
    nw_med = np.mean(Y, axis=0)
    nw_ranks = rankdata(np.linalg.norm(Y - nw_med, axis=1)) / len(Y)

    # ---------------------------------------------------------
    # B. Intrinsic Fréchet L2 Mean (Geodesic L2 Mean - Baseline 2)
    # ---------------------------------------------------------
    print(">> Running Fréchet L2 Mean...")
    fl2_med, fl2_depths = _frechet_l2_mean_core(Y, k_neighbors=10)
    fl2_ranks = rankdata(fl2_depths) / len(Y)

    # ---------------------------------------------------------
    # C. Intrinsic Fréchet L1 Median (Geodesic L1 Median - Baseline 3)
    # ---------------------------------------------------------
    print(">> Running Fréchet L1 Median...")
    fl1_med, fl1_depths = _geodesic_l1_median_core(Y, k_neighbors=10)
    fl1_ranks = rankdata(fl1_depths) / len(Y)

    # ---------------------------------------------------------
    # D. KDE Mode (Kernel Density Mode - Baseline 4)
    # ---------------------------------------------------------
    print(">> Running KDE Density Mode...")
    kde_med, kde_depths = _kde_mode_core(Y)
    kde_ranks = rankdata(kde_depths) / len(Y)

    # ---------------------------------------------------------
    # E. AMQR (Proposed Method)
    # ---------------------------------------------------------
    fixed_setup = {
        'ref_dist': 'uniform',  # Clear boundaries, no long tail
        'use_knn': True,  # Pathway B: Rely on graph approximation without crossing the vacuum
        'd_int': 2,  # Force 2D latent space, perfectly fits 2D dumbbell
        'max_samples': 500,
        'epsilon': 0.0  # Exact GW zero blur, prevent probability leakage
    }

    print(">> Running Proposed AMQR...")
    best_hyperparams = {'k_neighbors': 25}  # Keep the graph resolution consistent with Fréchet for fairness
    final_params = {**fixed_setup, **best_hyperparams}

    amqr = AMQR_Engine(**final_params)
    a_med, a_ranks = amqr.fit_predict(Y)

    # 3. Organize data and call plotting
    meds =[nw_med, fl2_med, fl1_med, kde_med, a_med]
    ranks =[nw_ranks, fl2_ranks, fl1_ranks, kde_ranks, a_ranks]
    method_names =["Euclidean Mean", "Fréchet L2 Mean", "Fréchet L1 Median", "KDE Mode", "Proposed AMQR"]

    print("Rendering plots...")
    save_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig_Dumbbell_Comparison.pdf")

    plot_bimodal_crescent_1x5(Y, meds, ranks, save_path=save_path)

    print(f"Dumbbell shape experiment completed! (Please check locally at {save_path})")
