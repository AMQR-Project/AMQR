# data/simulations.py
import numpy as np
import scipy.linalg as slinalg
from scipy.stats import ortho_group


def generate_circular_manifold_with_gap(n_points=15000):
    """
    Generate a 3D spiral/cylindrical manifold point cloud with a 135° topological gap.
    Used to test the model's topological robustness when encountering a "vacuum gap".
    """
    np.random.seed(42)
    T = np.random.uniform(0, 10, n_points)
    R = np.random.normal(3.0, 0.2, n_points)
    theta_center = T * (np.pi / 2.5)
    theta = np.random.vonmises(mu=theta_center, kappa=1.5, size=n_points)

    # Create a gap at 135° (3π/4)
    rel_theta = (theta - theta_center + np.pi) % (2 * np.pi) - np.pi
    gap_center = 3 * np.pi / 4
    gap_width = 0.5
    in_gap = np.abs(rel_theta - gap_center) < gap_width

    # Keep only 2% of the points in the gap region to form a vacuum zone
    keep_prob = np.where(in_gap, 0.02, 1.0)
    keep_mask = np.random.rand(n_points) < keep_prob

    T, R, theta = T[keep_mask], R[keep_mask], theta[keep_mask]
    Y1, Y2 = R * np.cos(theta), R * np.sin(theta)
    P_3D = np.column_stack([T, Y1, Y2])

    # Generate the absolute real skeleton Ground Truth
    T_gt = np.linspace(0, 10, 200)
    theta_gt = T_gt * (np.pi / 2.5)
    GT_3D = np.column_stack([T_gt, 3.0 * np.cos(theta_gt), 3.0 * np.sin(theta_gt)])

    return T, P_3D, GT_3D


def generate_bimodal_crescent():
    """
    Generate a Bimodal Crescent manifold with a sparse bridge.
    Characteristics: Extremely high density at both ends, and an extremely sparse connection in the middle.
    Used to test the model's ability to resist "Density Mode" traps and "Euclidean Voids".
    """
    # np.random.seed(42)
    # High-density cluster 1 (right end)
    theta1 = np.random.normal(0, 0.2, 400)
    # High-density cluster 2 (left end)
    theta2 = np.random.normal(np.pi, 0.2, 400)
    # Extremely sparse middle connecting bridge (middle part)
    theta3 = np.random.uniform(0, np.pi, 50)

    theta = np.concatenate([theta1, theta2, theta3])
    R = np.random.normal(5, 0.3, len(theta))

    Y = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
    return Y


def generate_dynamic_functional_data(N=1000, D=50, random_state=42):
    """
    Generate Dynamic Functional Data with independent variable T.
    The true peak drifts sinusoidally with time T, but there is a high-density outlier trap fixed at X=3.0.
    """
    np.random.seed(random_state)
    T = np.random.uniform(0, 10, N)
    X_grid = np.linspace(-5, 5, D)
    Y = np.zeros((N, D))

    for i in range(N):
        t = T[i]
        #[Dynamic trajectory of the true peak]: Oscillates sinusoidally with time T
        true_center = 2.0 * np.sin(t * np.pi / 5.0)

        rand_val = np.random.rand()
        if rand_val < 0.85:
            # 85% true data, anchored at true_center, with a larger variance (more scattered)
            shift = np.random.normal(true_center, 0.8)
        else:
            # 15% outlier trap, firmly fixed at 3.0 on the right, with an extremely small variance (extremely dense)
            shift = np.random.normal(3.0, 0.1)

        amp = np.random.normal(2.0, 0.2)
        base_curve = amp * np.exp(-((X_grid - shift) ** 2) / 0.5)
        noise = np.random.normal(0, 0.05, D)

        Y[i] = base_curve + noise

    # Generate an absolutely pure Ground Truth surface for evaluation
    t_eval = np.linspace(0, 10, 50)
    GT_surface = np.zeros((len(t_eval), D))
    for i, t in enumerate(t_eval):
        true_shift = 2.0 * np.sin(t * np.pi / 5.0)
        GT_surface[i] = 2.0 * np.exp(-((X_grid - true_shift) ** 2) / 0.5)

    return T, X_grid, Y, t_eval, GT_surface


def generate_spd_data_with_labels(N=300, dim=2, random_state=42):
    """
    Generate SPD matrix data of arbitrary dimensions, and return the strict empirical manifold center.
    """
    np.random.seed(random_state)
    Y = np.zeros((N, dim * dim))
    labels = np.zeros(N)

    # 1. Construct the absolute true baseline rotation matrix R_true
    np.random.seed(0)
    R_true = ortho_group.rvs(dim) if dim > 2 else np.array(
        [[np.cos(np.pi / 4), -np.sin(np.pi / 4)],[np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    np.random.seed(random_state)

    for i in range(N):
        is_outlier = np.random.rand() < 0.20
        labels[i] = 1 if is_outlier else 0

        # Initialize the eigenvalue vector
        l_vals = np.zeros(dim)

        if is_outlier:
            # [Inflation noise]: The principal eigenvalue is extremely large, and the rest are also large
            l_vals[-1] = np.random.normal(12.0, 1.0)
            for j in range(dim - 1):
                l_vals[j] = np.random.normal(2.0, 0.5)

            # Lie algebra perturbation
            A = np.random.normal(0, 0.1, (dim, dim))
            S = A - A.T  # Skew-symmetric matrix
            R = slinalg.expm(S)
        else:
            # [Core normal values]: The principal eigenvalue is 3.0, and the rest are smaller at 0.4
            l_vals[-1] = np.random.lognormal(mean=np.log(3.0), sigma=0.4)
            for j in range(dim - 1):
                l_vals[j] = np.random.lognormal(mean=np.log(0.4), sigma=0.2)

            # Lie algebra perturbation: Apply a minor perturbation around R_true
            A = np.random.normal(0, 0.3, (dim, dim))
            S = A - A.T
            R = R_true @ slinalg.expm(S)

        # Combined eigendecomposition: M = R * Lambda * R^T
        M = R @ np.diag(l_vals) @ R.T
        Y[i] = M.flatten()

    # =========================================================
    # Calculate the empirical Log-Euclidean Mean of clean samples (labels==0)
    # This represents the most absolute and true geometric center of gravity of this batch of data after removing all outliers
    # =========================================================
    clean_matrices = Y[labels == 0].reshape(-1, dim, dim)
    log_sum = np.zeros((dim, dim))
    for M in clean_matrices:
        log_sum += slinalg.logm(M).real

    log_mean = log_sum / len(clean_matrices)
    SPD_true = slinalg.expm(log_mean).real  # Map back to the SPD manifold

    # Return SPD_true as the Ground Truth (replacing the originally incorrect R_true)
    return Y, labels, SPD_true
