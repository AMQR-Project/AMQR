# data/simulations.py
import numpy as np
import scipy.linalg as slinalg
from scipy.stats import ortho_group


def generate_circular_manifold_with_gap(n_points=15000):
    """
    Generate a 3D spiral/cylindrical manifold point cloud with a 135° topological gap.
    Used to test the topological robustness of the model when encountering a "vacuum gap".
    """
    np.random.seed(42)
    T = np.random.uniform(0, 10, n_points)
    R = np.random.normal(3.0, 0.2, n_points)
    theta_center = T * (np.pi / 2.5)
    theta = np.random.vonmises(mu=theta_center, kappa=1.5, size=n_points)

    # Create a gap in the 135° (3π/4) direction
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

    # Generate the absolute ground truth skeleton 
    T_gt = np.linspace(0, 10, 200)
    theta_gt = T_gt * (np.pi / 2.5)
    GT_3D = np.column_stack([T_gt, 3.0 * np.cos(theta_gt), 3.0 * np.sin(theta_gt)])

    return T, P_3D, GT_3D


def generate_bimodal_crescent(N=1500, bridge_ratio=0.35, thickness=0.4, random_state=42):
    """
    Generate a horseshoe (crescent) manifold with unbalanced density and controllable thickness.
    Fixed the physical vacuum and boundary truncation singularity issues to ensure absolute overall continuity.

    Parameters:
    - N: Total number of samples
    - bridge_ratio: Proportion of basic continuous data to total data (increase it = denser overall base)
    - thickness: Noise variance in the normal (radial) direction of the manifold (increase it = thicker bridge)
    - random_state: Random seed, set to None to generate different data each time
    """
    if random_state is not None:
        np.random.seed(random_state)

    # 1. Precisely control data distribution density (Mass Distribution)
    N_bridge = int(N * bridge_ratio)
    N_ends = N - N_bridge
    N_left = N_ends // 2
    N_right = N_ends - N_left

    # Generate angle theta
    # Basic topological background: ensure the entire manifold is connected
    theta_bridge = np.random.uniform(low=0.0, high=np.pi, size=N_bridge)

    # Left and right bimodal: superimpose extremely dense Gaussian distributions
    theta_left = np.random.normal(loc=0.1 * np.pi, scale=0.08, size=N_left)
    theta_right = np.random.normal(loc=0.9 * np.pi, scale=0.08, size=N_right)

    # Concatenate all angles (removed np.clip, allowing natural transition at the crescent tips to avoid k-NN singularities)
    theta = np.concatenate([theta_left, theta_right, theta_bridge])

    # 2. Introduce radial thickness (Tubular Thickness)
    R_base = 5.0  # Base radius
    r_noise = np.random.normal(loc=0, scale=thickness, size=N)
    r = R_base + r_noise

    # 3. Polar to Cartesian coordinates
    X = r * np.cos(theta)
    Y = r * np.sin(theta)

    data = np.vstack((X, Y)).T
    return data


def generate_dynamic_functional_data(N=1000, D=50, random_state=42):
    """
    Generate Dynamic Functional Data with independent variable T
    The true peak drifts sinusoidally with time T, but there is a high-density outlier trap fixed at X=3.0.
    """
    np.random.seed(random_state)
    T = np.random.uniform(0, 10, N)
    X_grid = np.linspace(-5, 5, D)
    Y = np.zeros((N, D))

    for i in range(N):
        t = T[i]
        # [True peak dynamic trajectory]: Sinusoidal swing with time T
        true_center = 2.0 * np.sin(t * np.pi / 5.0)

        rand_val = np.random.rand()
        if rand_val < 0.85:
            # 85% true data, anchored at true_center, with larger variance (more scattered)
            shift = np.random.normal(true_center, 0.8)
        else:
            # 15% outlier trap, firmly fixed at 3.0 on the right, with extremely small variance (extremely dense)
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


def generate_asymmetric_functional_data(N=1500, D=50):
    """
    Generate functional data with Asymmetric Phase Shifts
    80% of the curves shift slightly to the left, and 20% shift drastically to the right
    """
    np.random.seed(42)
    T = np.linspace(0, 10, N)
    X_grid = np.linspace(-5, 5, D)

    Y_curves = np.zeros((N, D))
    GT_surface = np.zeros((N, D))

    for i in range(N):
        # Create Asymmetric Phase Shift
        # 80% probability of shifting left (mean -1.5, small variance)
        # 20% probability of shifting right (mean +2.5, large variance)
        if np.random.rand() < 0.8:
            phase_shift = np.random.normal(-1.5, 0.5)
        else:
            phase_shift = np.random.normal(2.5, 0.8)

        # Base waveform: standard Gaussian bell curve
        amplitude = 2.0 + np.random.normal(0, 0.1)  # Amplitude remains basically consistent

        # Superimpose waveform
        Y_curves[i, :] = amplitude * np.exp(-0.5 * ((X_grid - phase_shift) / 0.8) ** 2)

        # Superimpose a little high-frequency white noise
        Y_curves[i, :] += np.random.normal(0, 0.05, D)

        # Record the perfect noise-free reference shape (as a reference)
        GT_surface[i, :] = 2.0 * np.exp(-0.5 * (X_grid / 0.8) ** 2)

    return T, X_grid, Y_curves, T, GT_surface


# data/simulations.py (New function)

def generate_drifting_bimodal_functional_data(N=1500, D=50, random_state=42):
    """
    Generate functional data with a time-varying bimodal mixture distribution.

    Core features:
    - The phase shift of the curve comes from a mixture of two Gaussian distributions.
    - The mixture weight p(t) is a function of time t, causing the data center of gravity to drift smoothly from left to right.
    - This creates a non-linear, dynamically evolving true conditional median/mean trajectory over time.
    """
    if random_state is not None:
        np.random.seed(random_state)

    T = np.linspace(0, 10, N)
    X_grid = np.linspace(-5, 5, D)
    Y_curves = np.zeros((N, D))

    # Define the centers of the two modes
    mean_left = -1.5
    mean_right = 2.5

    for i in range(N):
        t = T[i]

        # Mixture probability p_left(t) changes dynamically with time t
        # Use a cosine function to achieve a smooth transition from 1 to 0
        # t=0  -> p_left=1.0 (completely left mode)
        # t=5  -> p_left=0.5 (50/50 mixture)
        # t=10 -> p_left=0.0 (completely right mode)
        p_left = (np.cos(t * np.pi / 10.0) + 1.0) / 2.0

        if np.random.rand() < p_left:
            # Sample from the left mode
            phase_shift = np.random.normal(mean_left, 0.5)
        else:
            # Sample from the right mode
            phase_shift = np.random.normal(mean_right, 0.8)

        # Base waveform and noise (remain unchanged)
        amplitude = 2.0 + np.random.normal(0, 0.1)
        Y_curves[i, :] = amplitude * np.exp(-0.5 * ((X_grid - phase_shift) / 0.8) ** 2)
        Y_curves[i, :] += np.random.normal(0, 0.05, D)

    # Generate dynamically changing Ground Truth surface
    # The center trajectory of GT is now the expected value of the two mode centers, changing with p_left(t)
    t_eval = np.linspace(0, 10, 100)
    GT_surface = np.zeros((len(t_eval), D))
    for i, t in enumerate(t_eval):
        p_left_t = (np.cos(t * np.pi / 10.0) + 1.0) / 2.0
        # True center = left mode center * p_left + right mode center * (1 - p_left)
        true_shift = p_left_t * mean_left + (1 - p_left_t) * mean_right

        GT_surface[i] = 2.0 * np.exp(-0.5 * ((X_grid - true_shift) / 0.8) ** 2)

    # Note: The length of the returned t_eval is now 100 to obtain a smoother GT surface
    return T, X_grid, Y_curves, t_eval, GT_surface


def generate_spd_data_with_labels(N=300, dim=2, random_state=42):
    """
    Generate SPD matrix data of arbitrary dimensions and return the strict empirical manifold center.
    """
    np.random.seed(random_state)
    Y = np.zeros((N, dim * dim))
    labels = np.zeros(N)

    # 1. Construct the absolute true reference rotation matrix R_true
    np.random.seed(0)
    R_true = ortho_group.rvs(dim) if dim > 2 else np.array(
        [[np.cos(np.pi / 4), -np.sin(np.pi / 4)],[np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    np.random.seed(random_state)

    for i in range(N):
        is_outlier = np.random.rand() < 0.20
        labels[i] = 1 if is_outlier else 0

        # Initialize eigenvalue vector
        l_vals = np.zeros(dim)

        if is_outlier:
            # [Inflation noise]: Principal eigenvalue is extremely large, others are also large
            l_vals[-1] = np.random.normal(12.0, 1.0)
            for j in range(dim - 1):
                l_vals[j] = np.random.normal(2.0, 0.5)

            # Lie algebra perturbation
            A = np.random.normal(0, 0.1, (dim, dim))
            S = A - A.T  # Skew-symmetric matrix
            R = slinalg.expm(S)
        else:
            # [Core normal values]: Principal eigenvalue 3.0, others are smaller 0.4
            l_vals[-1] = np.random.lognormal(mean=np.log(3.0), sigma=0.4)
            for j in range(dim - 1):
                l_vals[j] = np.random.lognormal(mean=np.log(0.4), sigma=0.2)

            # Lie algebra perturbation: minor perturbation around R_true
            A = np.random.normal(0, 0.3, (dim, dim))
            S = A - A.T
            R = R_true @ slinalg.expm(S)

        # Combine eigendecomposition: M = R * Lambda * R^T
        M = R @ np.diag(l_vals) @ R.T
        Y[i] = M.flatten()

    # =========================================================
    # Calculate the empirical Log-Euclidean Mean of pure samples (labels==0)
    # This represents the most absolute and true geometric center of gravity of this batch of data after removing all outliers
    # =========================================================
    clean_matrices = Y[labels == 0].reshape(-1, dim, dim)
    log_sum = np.zeros((dim, dim))
    for M in clean_matrices:
        log_sum += slinalg.logm(M).real

    log_mean = log_sum / len(clean_matrices)
    SPD_true = slinalg.expm(log_mean).real  # Map back to the SPD manifold

    # Return SPD_true as Ground Truth (replacing the originally incorrect R_true)
    return Y, labels, SPD_true
