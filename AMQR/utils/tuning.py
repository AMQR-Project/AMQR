# utils/tuning.py

import numpy as np
import itertools
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial.distance import cdist

# Ensure AMQR_Engine is correctly imported
from models.amqr_engine import AMQR_Engine


def auto_tune_amqr(Y, param_grid, fixed_kwargs, cv=5, T=None, t_eval=None, window_size=None):
    """
    Unsupervised auto-tuning (supports both global static mode & time regression mode)
    """
    N_original = len(Y)

    # =========================================================
    # Mechanism 1: Large-scale data downsampling protection
    # =========================================================
    if N_original > 3000:
        print(f"Dataset size ({N_original}) is large, automatically enabling random downsampling (N=3000) for high-speed tuning...")
        np.random.seed(42)
        sub_idx = np.random.choice(N_original, 3000, replace=False)
        Y = Y[sub_idx]
        if T is not None:
            T = T[sub_idx]

    # Expand the hyperparameter grid
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_params = None
    best_score = float('inf')
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # =========================================================
    # Mechanism 2: Sparse time axis acceleration (only in regression mode)
    # =========================================================
    t_eval_tune = None
    if t_eval is not None:
        step = max(1, len(t_eval) // 10)  # Sampling density is adjustable, set to 10% here
        t_eval_tune = t_eval[::step]
        print(f"Enabling sparse time tuning: Compressing {len(t_eval)} windows into {len(t_eval_tune)} to accelerate computation")

    print(f"Starting {cv}-Fold cross-validation (total of {len(param_combinations)} parameter sets)...")

    for params in param_combinations:
        fold_scores = []
        for train_idx, val_idx in kf.split(Y):
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            # Instantiate the engine
            engine = AMQR_Engine(**params, **fixed_kwargs)

            # --- Dynamic Regression Mode ---
            if T is not None:
                T_train, T_val = T[train_idx], T[val_idx]

                # 1. Run sliding window regression on the training set to get the final "global" rank for each training point
                _, ranks_train = engine.fit_predict(Y_train, T=T_train, t_eval=t_eval_tune, window_size=window_size)

                # 2. Out-of-sample extension: Use spatio-temporal coordinates to predict the ranks of the validation set
                T_scale = np.mean(np.std(Y_train, axis=0)) / (np.std(T_train) + 1e-9)
                TY_train = np.column_stack([T_train * T_scale, Y_train.reshape(len(Y_train), -1)])
                TY_val = np.column_stack([T_val * T_scale, Y_val.reshape(len(Y_val), -1)])

                knn_ext = KNeighborsRegressor(n_neighbors=5).fit(TY_train, ranks_train)
                ranks_val = knn_ext.predict(TY_val)

                # 3. Calculate loss: Convert ranks to one-dimensional latent coordinates and compute their squared L2 norm
                # For a uniform reference distribution, the mapping from rank u to coordinate z is z = 2u - 1
                z_val = (2 * ranks_val - 1).reshape(-1, 1)
                score = np.mean(np.sum(z_val ** 2, axis=1))
                fold_scores.append(score)

            # --- Static Global Mode ---
            else:
                # 1. Fit on the training set to obtain its coordinates z_train in the latent space
                #    To get the z coordinates, the internal core method _fit_predict_core needs to be called
                try:
                    # _fit_predict_core returns (med, ranks, z_coords)
                    _, _, z_train = engine._fit_predict_core(Y_train)
                except Exception as e:
                    # If the engine API changes, provide a fallback solution
                    print(f"Cannot call _fit_predict_core directly, error: {e}")
                    print("Falling back to using ranks as the loss function. It is recommended to check the internal methods of AMQR_Engine.")
                    _, ranks_train = engine.fit_predict(Y_train)
                    z_train = (2 * ranks_train - 1).reshape(-1, 1)  # Assuming d=1

                # 2. Out-of-sample extension: Use a KNN regressor to map from the Y_train space to the Z_train space
                #    This is an efficient implementation of "Smooth Intrinsic Projection" from the paper
                knn_regressor = KNeighborsRegressor(n_neighbors=5, weights='distance')
                knn_regressor.fit(Y_train.reshape(len(Y_train), -1), z_train)
                z_val = knn_regressor.predict(Y_val.reshape(len(Y_val), -1))

                # 3. Calculate loss: According to formula (6) in the paper, minimize the expected squared norm of the projected coordinates of the validation set
                score = np.mean(np.sum(z_val ** 2, axis=1))
                fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        print(f"  Config {params} | Latent Fréchet Variance: {mean_score:.6f}")

        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    print(f"Best parameter combination: {best_params} (Score: {best_score:.6f})")
    return best_params
