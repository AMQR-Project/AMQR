import sys
import os
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from data.simulations import generate_circular_manifold_with_gap
from models.amqr_engine import AMQR_Engine

# Import top-tier baselines (unified sliding window interface)
from models.baselines import (
    get_nw_tube,
    get_frechet_regression_tube,
    get_isotropic_geodesic_tube,
    get_kde_mode_tube
)

from utils.metrics import evaluate_spiral_metrics
from utils.visualization import plot_2x5_spiral_experiment  # Note: The downstream plotting script needs to be adapted for 5 models
from utils.tuning import auto_tune_amqr

AUTO_TUNE = False
NUM_FOLDS = 3


def extract_all_models(T, P_3D, amqr_params, window_size=1.2, step_size=0.1, u_target=0.20):
    """
    Unified scheduling of all models, using the encapsulated underlying engine for extremely clean code
    """
    t_eval = np.arange(T.min(), T.max() + step_size, step_size)
    k_neighbors = amqr_params.get('k_neighbors', 15)

    # ===============================================================
    # 1. Baseline Models
    # ===============================================================
    print("\n[1/5] Running Baseline 1: NW (Ambient Space L2 Mean)...")
    nw_traj, nw_ranks = get_nw_tube(T, P_3D, t_eval, window_size=window_size)

    print("\n[2/5] Running Baseline 2: Fréchet Regression (Intrinsic Geodesic L2 Mean SOTA)...")
    f_l2_traj, f_l2_ranks = get_frechet_regression_tube(T, P_3D, t_eval, window_size=window_size,
                                                        k_neighbors=k_neighbors)

    print("\n[3/5] Running Baseline 3: Isotropic Geodesic (Intrinsic Geodesic L1 Median)...")
    f_l1_traj, f_l1_ranks = get_isotropic_geodesic_tube(T, P_3D, t_eval, window_size=window_size,
                                                        k_neighbors=k_neighbors)

    print("\n[4/5] Running Baseline 4: SW-KDE (Sliding Window Density Mode Seeking)...")
    kde_traj, kde_ranks = get_kde_mode_tube(T, P_3D, t_eval, window_size=window_size)

    # ===============================================================
    # 2. AMQR Engine (Exact GW)
    # ===============================================================
    print("\n[5/5] Running AMQR (Calling engine's native sliding window and spatio-temporal synchronization)...")
    amqr = AMQR_Engine(**amqr_params)
    amqr_traj, amqr_ranks = amqr.fit_predict(P_3D, T=T, t_eval=t_eval, window_size=window_size)

    # ===============================================================
    # 3. Extract results and assemble dictionary
    # ===============================================================
    t_traj = np.array([t for t, _ in nw_traj])  # t_eval is the same for all models

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
    print("Sim 1: 3D Spiral Manifold with Topological Gap (5 Models)")
    print("========================================================")

    # 1. Data Generation (15000 points)
    T, P_3D, GT_3D = generate_circular_manifold_with_gap(n_points=15000)
    sort_idx = np.argsort(T)
    T_sorted, P_sorted = T[sort_idx], P_3D[sort_idx]

    t_eval = np.arange(T_sorted.min(), T_sorted.max() + 0.1, 0.1)

    # 2. Fixed Structural Priors
    fixed_setup = {
        'ref_dist': 'uniform',
        'use_knn': True,
        'd_int': None,
        'max_samples': 2500,
        'epsilon': 0.0
    }

    # 3. Hyperparameter Tuning Logic Branch
    if AUTO_TUNE:
        print("\n>> Starting regression manifold OOS-GW cross-validation...")
        from utils.tuning import auto_tune_amqr

        param_grid = {'k_neighbors': [5, 10, 15]}
        best_hyperparams = auto_tune_amqr(
            P_sorted, param_grid, fixed_kwargs=fixed_setup, cv=NUM_FOLDS,
            T=T_sorted, t_eval=t_eval, window_size=0.1
        )
    else:
        print("\n>> Using manually validated optimal physical prior parameters (From Prior Knowledge)")
        best_hyperparams = {'k_neighbors': 10}

    final_amqr_params = {**fixed_setup, **best_hyperparams}
    print(f"\nPerforming full validation with final parameters: {final_amqr_params}")

    # 4. Model Execution
    t_traj, centers, masks = extract_all_models(
        T_sorted, P_sorted, amqr_params=final_amqr_params,
        window_size=0.1, step_size=0.05, u_target=0.20
    )

    # 5. Quantitative Evaluation and Output Table
    print("\n=======================================================")
    print("Quantitative Evaluation on Spiral Gap (All 5 Models)")
    print("=======================================================")
    # Removed GT parameter, calculating intrinsic error directly from the data itself
    df_metrics = evaluate_spiral_metrics(t_traj, centers, masks, T_sorted, P_sorted, window_size=0.1)
    print(df_metrics.to_string())

    os.makedirs(os.path.join(PROJECT_ROOT, "results", "tables"), exist_ok=True)
    df_metrics.to_csv(os.path.join(PROJECT_ROOT, "results", "tables", "Table_Spiral_Metrics_5Models.csv"))
    print("=======================================================\n")

    # 6. Visualization and Rendering of the Main Figure
    print("Plotting the final 2x5 comparison figure for the perforated manifold...")
    save_img_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig2_Spiral_Gap_5Models.pdf")

    # Removed GT_3D parameter
    plot_2x5_spiral_experiment(T_sorted, P_sorted, t_traj, centers, masks, target_t=5.0, save_path=save_img_path)

    print("3D spiral experiment finished!")
