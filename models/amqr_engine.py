import ot
import warnings
import numpy as np
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cdist
from scipy.stats import qmc, norm, laplace, rankdata
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph, NearestNeighbors, KNeighborsRegressor

warnings.filterwarnings("ignore")


# =====================================================================
# Core Class: AMQR Manifold Quantile Regression Engine
# =====================================================================
class AMQR_Engine:
    """
    Auto-conditioned Manifold Quantile Regression (AMQR)
    Features:
    1. Dynamic/Forced Intrinsic Dimension Detection (MLE)
    2. Exact Optimal Transport Immune to Symmetry-Breaking Noise (Exact GW with Symmetry Breaking)
    3. Automatic Landmark Sampling and Out-of-Sample Extension
    4. Automatic Conditional Sliding Window Regression
    """

    def __init__(self, ref_dist='uniform', epsilon=0.0, d_int=None,
                 use_knn=True, k_neighbors=15, max_samples=2500, use_log_squash=False):
        self.ref_dist = ref_dist.lower()
        self.epsilon = epsilon
        self.d_int = d_int
        self.use_knn = use_knn
        self.k_neighbors = k_neighbors
        self.max_samples = max_samples  # The core valve to control O(N^3) complexity
        self.use_log_squash = use_log_squash

    def _run_with_oos_protection(self, Y, y_dist_m=None):
        """
        Unified computational firewall: Triggers landmark isolation and Kernel-regularized Intrinsic Projection
        """
        N = len(Y)
        if N <= self.max_samples:
            return self._fit_predict_core(Y, y_dist_m=y_dist_m)

        # Triggering OOS protection mechanism
        core_idx = np.random.choice(N, size=self.max_samples, replace=False)
        oos_idx = np.setdiff1d(np.arange(N), core_idx)

        Y_core = Y[core_idx]
        y_dist_m_core = y_dist_m[np.ix_(core_idx, core_idx)] if y_dist_m is not None else None

        # 1. Run exact GW only on core landmarks
        med_core, ranks_core, z_core, S_x = self._fit_predict_core(Y_core, y_dist_m=y_dist_m_core, return_scale=True)

        # 2. Extract pure intrinsic distances from OOS points to Core points
        if y_dist_m is not None:
            dist_oos_to_core = y_dist_m[np.ix_(oos_idx, core_idx)]
        else:
            dist_oos_to_core = cdist(Y[oos_idx].reshape(len(oos_idx), -1), Y_core.reshape(len(core_idx), -1))

        # 3. Kernel-regularized smooth intrinsic projection (Strictly aligned with Paper Step 3)
        k_nn = min(int(np.log(len(core_idx)) * 2), len(core_idx)) # k \asymp \log N_x
        nearest_core_indices = np.argsort(dist_oos_to_core, axis=1)[:, :k_nn]
        nearest_dists = np.take_along_axis(dist_oos_to_core, nearest_core_indices, axis=1)

        # Use Gaussian kernel instead of original IDW to ensure equicontinuity
        sigma = np.median(nearest_dists) + 1e-8
        weights = np.exp(-(nearest_dists ** 2) / (2 * sigma ** 2))
        weights /= np.sum(weights, axis=1, keepdims=True)

        z_oos = np.zeros((len(oos_idx), z_core.shape[1]))
        for i in range(len(oos_idx)):
            z_oos[i] = np.average(z_core[nearest_core_indices[i]], axis=0, weights=weights[i])

        # 4. Assemble the global Z coordinates
        z_full = np.zeros((N, z_core.shape[1]))
        z_full[core_idx] = z_core
        z_full[oos_idx] = z_oos

        # Local scaling factor S(x)
        raw_depths = S_x * np.linalg.norm(z_full, axis=1)

        # Dynamically calculate the expected norm of the reference space
        expected_norm = np.sqrt(np.mean(np.linalg.norm(z_core, axis=1)**2))
        depths = raw_depths / (expected_norm + 1e-9)

        ranks_full = rankdata(depths) / N

        return med_core, ranks_full, z_full

    def _generate_latent_qmc(self, d, N):
        sampler = qmc.Halton(d=d+1, scramble=False)
        u_grid = np.clip(sampler.random(n=N), 1e-4, 1 - 1e-4)
        Z = np.zeros((N, d))

        if self.ref_dist == 'uniform':
            if d == 1:
                Z = (u_grid[:, 0:1] * 2) - 1.0
            elif d == 2:
                r = np.sqrt(u_grid[:, 0])
                theta = 2 * np.pi * u_grid[:, 1]
                Z = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            else:
                Z_dir = norm.ppf(u_grid[:, :d])
                Z_dir /= (np.linalg.norm(Z_dir, axis=1, keepdims=True) + 1e-9)
                radius = u_grid[:, d] ** (1.0 / d)
                Z = Z_dir * radius[:, None]
        elif self.ref_dist == 'gaussian':
            Z = norm.ppf(u_grid[:, :d], loc=0, scale=1.0)
        elif self.ref_dist == 'laplace':
            Z = laplace.ppf(u_grid[:, :d], loc=0, scale=1.0)

        Z[0] = np.zeros(d)
        return Z

    def _fit_predict_core(self, Y, y_dist_m=None, return_scale=False):
        """Underlying pure mathematical alignment engine (supports direct input of pre-computed distance matrix y_dist_m)"""
        N = len(Y)
        Y_flat = Y.reshape(N, -1)

        # =========================================================
        # 1. Calculate or directly use the manifold geodesic distance Cy
        # =========================================================
        if y_dist_m is not None:
            Cy = y_dist_m.copy()  # Directly use the externally provided true metric (e.g., Wasserstein distance)
        else:
            if self.use_knn:
                k = min(self.k_neighbors, N - 1)
                A = kneighbors_graph(Y_flat, n_neighbors=k, mode='distance', include_self=False)
                Cy = shortest_path(A, method='D', directed=False)
                if np.isinf(Cy).any():
                    Cy[np.isinf(Cy)] = np.nanmax(Cy[Cy != np.inf]) * 2.0
            else:
                Cy = cdist(Y_flat, Y_flat, metric='euclidean')

        # =========================================================
        # 2. MLE Intrinsic Dimension Detection (supports direct inference from the distance matrix)
        # =========================================================
        if self.d_int is not None:
            final_d = self.d_int
        else:
            k_mle = min(10, N - 1)
            if y_dist_m is not None:
                # If a distance matrix is provided, directly sort it to get the top k nearest neighbors
                dists = np.sort(Cy, axis=1)[:, 1:k_mle + 2]
            else:
                nn = NearestNeighbors(n_neighbors=k_mle + 1).fit(Y_flat)
                dists, _ = nn.kneighbors(Y_flat)
                dists = dists[:, 1:]

            dists = np.maximum(dists, 1e-9)
            r_k = dists[:, -1:]
            mle_val = (k_mle - 1) / np.sum(np.log(r_k / dists[:, :-1]), axis=1)
            final_d = int(np.round(np.mean(mle_val)))
            final_d = max(1, final_d)  # Force capping

        # =========================================================
        # 3. Construct the target space and break symmetry
        # =========================================================
        Z_ref = self._generate_latent_qmc(final_d, N)
        Cz = cdist(Z_ref, Z_ref, metric='euclidean')

        Cz_norm = Cz / (Cz.max() + 1e-9)
        if self.use_log_squash:
            Cy_proc = np.log1p(Cy)
        else:
            Cy_proc = Cy

        # Extract local scaling factor S(x)
        S_x = np.nanmax(Cy_proc)
        Cy_norm = Cy_proc / (S_x + 1e-9)

        noise_z = np.random.uniform(0, 1e-8, size=Cz_norm.shape)
        Cz_norm += (noise_z + noise_z.T) / 2.0  # Force symmetrization
        np.fill_diagonal(Cz_norm, 0)

        noise_y = np.random.uniform(0, 1e-8, size=Cy_norm.shape)
        Cy_norm += (noise_y + noise_y.T) / 2.0
        np.fill_diagonal(Cy_norm, 0)

        # =========================================================
        # 4. Solve GW at high speed
        # =========================================================
        py, pz = ot.unif(N), ot.unif(N)
        if self.epsilon > 0:
            gw_plan = ot.gromov.entropic_gromov_wasserstein(
                Cz_norm, Cy_norm, pz, py, 'square_loss', epsilon=self.epsilon, max_iter=100)
        else:
            gw_plan = ot.gromov.gromov_wasserstein(
                Cz_norm, Cy_norm, pz, py, 'square_loss', max_iter=50, tol=1e-4)

        # =========================================================
        # 5. Extract depth and ranks
        # =========================================================
        Y_mapped_to_Z = np.dot(gw_plan.T, Z_ref) * N

        raw_depths = S_x * np.linalg.norm(Y_mapped_to_Z, axis=1)

        expected_norm = np.sqrt(np.mean(np.linalg.norm(Z_ref, axis=1) ** 2))
        amqr_depths = raw_depths / (expected_norm + 1e-9)

        # Find the minimum depth and all candidate points within the tolerance range
        min_depth = np.min(amqr_depths)
        tolerance = 1e-9
        candidates = np.where(np.abs(amqr_depths - min_depth) < tolerance)[0]

        if len(candidates) == 1:
            med_idx = candidates[0]
        else:
            # Trigger Secondary Fréchet Refinement
            # Utilize the previously calculated manifold distance matrix Cy (N x N)
            print(f"Triggering geometric tie-breaking mechanism! Number of candidates: {len(candidates)}")
            # Vectorized calculation of the sum of distances from each candidate to all points in the current neighborhood
            frechet_sums = np.sum(Cy[candidates, :], axis=1)
            best_local_idx = np.argmin(frechet_sums)
            med_idx = candidates[best_local_idx]

        ranks = rankdata(amqr_depths) / N

        if return_scale:
            return Y[med_idx], ranks, Y_mapped_to_Z, S_x

        return Y[med_idx], ranks, Y_mapped_to_Z

    def fit_predict(self, Y, y_dist_m=None, T=None, t_eval=None, window_size=1.0):
        N = len(Y)

        # ==========================================
        # Branch A: Static global manifold (no time series)
        # ==========================================
        if T is None:
            # Directly call the engine with OOS protection
            med, ranks, _ = self._run_with_oos_protection(Y, y_dist_m)
            return med, ranks

        # ==========================================
        # Branch B: Conditional sliding window (dynamic spatio-temporal assembly)
        # ==========================================
        if t_eval is None:
            t_eval = np.linspace(T.min(), T.max(), 50)

        step_size = t_eval[1] - t_eval[0] if len(t_eval) > 1 else window_size / 5.0

        trajectory_med = []
        final_ranks = np.ones(N)

        # State variables for Procrustes synchronization
        prev_z = None
        prev_idx = None

        for t_c in tqdm(t_eval):
            idx = np.where(np.abs(T - t_c) <= window_size / 2.0)[0]
            if len(idx) < 15:
                continue

            Y_c = Y[idx]
            y_dist_m_c = y_dist_m[np.ix_(idx, idx)] if y_dist_m is not None else None

            med_c, ranks_c, z_c = self._run_with_oos_protection(Y=Y_c, y_dist_m=y_dist_m_c)

            if prev_z is not None:
                common_global_idx, curr_local_idx, prev_local_idx = np.intersect1d(
                    idx, prev_idx, return_indices=True
                )
                if len(common_global_idx) > z_c.shape[1]:
                    R, _ = orthogonal_procrustes(z_c[curr_local_idx], prev_z[prev_local_idx])
                    z_c = z_c @ R

            prev_z = z_c
            prev_idx = idx

            trajectory_med.append((t_c, med_c))

            # Inner-core slice assignment logic
            inner_condition = (T[idx] >= t_c - step_size / 2.0) & (T[idx] < t_c + step_size / 2.0)
            inner_in_idx = np.where(inner_condition)[0]
            if len(inner_in_idx) > 0:
                final_ranks[idx[inner_in_idx]] = ranks_c[inner_in_idx]

        return trajectory_med, final_ranks
