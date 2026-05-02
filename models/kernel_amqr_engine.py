import ot
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.metrics.pairwise import pairwise_kernels

class Kernel_AMQR_Engine:
    """
    Kernel Auto-conditioned Manifold Quantile Regression (Kernel-AMQR)
    Utilizes the Kernel Trick to implicitly map data into an infinite-dimensional RKHS space,
    extracts manifold distances within a perfectly flat space, and then maps them to a target disk via GW.
    """

    def __init__(self, ref_dist='uniform', epsilon=0.0, d_int=2,
                 kernel='rbf', gamma=None, degree=3, coef0=1,
                 use_log_squash=False):
        """
        :param kernel: 'rbf', 'poly', 'linear', or 'precomputed' (directly pass the kernel matrix)
        :param gamma: Bandwidth parameter for RBF or Poly kernels (if None, it is automatically inferred as 1/n_features)
        """
        self.ref_dist = ref_dist.lower()
        self.epsilon = epsilon
        self.d_int = d_int  # The kernel space is often infinite-dimensional, so it is recommended to explicitly specify the target space d_int (e.g., use 2 for plotting a bullseye chart)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.use_log_squash = use_log_squash

    def _generate_latent_qmc(self, d, N):
        """Generate QMC uniform disk points in the target space (adopting the excellent design from your original engine)"""
        from scipy.stats import qmc
        sampler = qmc.Halton(d=d, scramble=True)
        sample = sampler.random(n=N)
        if self.ref_dist == 'uniform':
            if d == 2:
                r = np.sqrt(sample[:, 0])
                theta = 2 * np.pi * sample[:, 1]
                return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            return sample * 2 - 1
        else:
            from scipy.stats import norm
            return norm.ppf(sample)

    def fit_predict(self, Y):
        """
        Directly input the original features Y, or input the kernel matrix K when kernel='precomputed'
        """
        N = len(Y)
        Y_flat = Y.reshape(N, -1) if self.kernel != 'precomputed' else Y

        # =========================================================
        # 1. Compute the Kernel Matrix K
        # =========================================================
        if self.kernel == 'precomputed':
            K = Y_flat.copy()
        else:
            # Dynamically assemble the parameter dictionary supported by the current kernel function
            kernel_kwargs = {}
            if self.gamma is not None:
                kernel_kwargs['gamma'] = self.gamma
            if self.kernel == 'poly':
                kernel_kwargs['degree'] = self.degree
                kernel_kwargs['coef0'] = self.coef0
            elif self.kernel == 'sigmoid':
                kernel_kwargs['coef0'] = self.coef0

            K = pairwise_kernels(Y_flat, Y_flat, metric=self.kernel, **kernel_kwargs)

        # =========================================================
        # 2. Extract the pure Euclidean distance Cy in the RKHS space from the kernel matrix
        # D^2(x, y) = K(x,x) + K(y,y) - 2K(x,y)
        # =========================================================
        K_diag = np.diag(K)

        Cy_sq = K_diag[:, None] + K_diag[None, :] - 2 * K
        Cy_sq = np.maximum(Cy_sq, 0)  # Prevent minor negative numbers due to floating-point errors
        Cy = np.sqrt(Cy_sq)

        # =========================================================
        # 3. Target disk construction and GW mapping (following the AMQR logic)
        # =========================================================
        Z_ref = self._generate_latent_qmc(self.d_int, N)
        Cz = cdist(Z_ref, Z_ref, metric='euclidean')

        Cz_norm = Cz / (Cz.max() + 1e-9)
        Cy_proc = np.log1p(Cy) if self.use_log_squash else Cy
        Cy_norm = Cy_proc / (np.nanmax(Cy_proc) + 1e-9)

        # Inject symmetry-breaking noise
        noise_z = np.random.uniform(0, 1e-8, size=Cz_norm.shape)
        Cz_norm += (noise_z + noise_z.T) / 2.0
        np.fill_diagonal(Cz_norm, 0)

        noise_y = np.random.uniform(0, 1e-8, size=Cy_norm.shape)
        Cy_norm += (noise_y + noise_y.T) / 2.0
        np.fill_diagonal(Cy_norm, 0)

        # Solve GW
        py, pz = ot.unif(N), ot.unif(N)
        if self.epsilon > 0:
            gw_plan = ot.gromov.entropic_gromov_wasserstein(
                Cz_norm, Cy_norm, pz, py, 'square_loss', epsilon=self.epsilon, max_iter=100)
        else:
            gw_plan = ot.gromov.gromov_wasserstein(
                Cz_norm, Cy_norm, pz, py, 'square_loss', max_iter=50, tol=1e-4)

        # =========================================================
        # 4. Extract depth and ranks
        # =========================================================
        Y_mapped_to_Z = np.dot(gw_plan.T, Z_ref) * N
        amqr_depths = np.linalg.norm(Y_mapped_to_Z, axis=1)
        med_idx = np.argmin(amqr_depths)
        ranks = rankdata(amqr_depths) / N

        # Note: If the input is a precomputed kernel matrix, Y[med_idx] returns a row of the kernel matrix
        return Y[med_idx], ranks
