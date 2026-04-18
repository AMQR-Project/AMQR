# data/ellipse.py
import numpy as np

def generate_straight_ellipse(N=1000, random_state=42):
    """
    Generate a 2D linear, highly anisotropic point cloud (a cigar-shaped straight ellipse). 
    Used to test the baseline model's performance under extremely stretched covariance.
    """
    np.random.seed(random_state)
    mean_Y = [0, 0]
    cov_Y = [[8.0, 7.5],
             [7.5, 8.0]]
    Y = np.random.multivariate_normal(mean_Y, cov_Y, N)
    return Y

def generate_bent_ellipse(N=1000, random_state=42):
    """
    Generate a 2D non-linear curved manifold point cloud (crescent-shaped ellipse).
    Used to test the model's ability to capture the non-linear manifold skeleton.
    """
    np.random.seed(random_state)
    mean_Y = [0, 0]
    cov_Y = [[8.0, 7.5],
             [7.5, 8.0]]
    Y_raw = np.random.multivariate_normal(mean_Y, cov_Y, N)

    u = (Y_raw[:, 0] + Y_raw[:, 1]) / np.sqrt(2)
    v = (Y_raw[:, 1] - Y_raw[:, 0]) / np.sqrt(2)

    bend_factor = 0.08
    v_bent = v + bend_factor * (u ** 2) - bend_factor * np.mean(u ** 2)

    X_new = (u - v_bent) / np.sqrt(2)
    Y_new = (u + v_bent) / np.sqrt(2)

    return np.column_stack([X_new, Y_new])
