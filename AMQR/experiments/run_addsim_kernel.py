import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from models.kernel_amqr_engine import Kernel_AMQR_Engine


def plot_3x5_grid(images, ranks, filename="Quantile_Images_3x5.png"):
    print("Rendering 3x5 quantile matrix slice plot...")
    fig, axes = plt.subplots(3, 5, figsize=(15, 9), facecolor='white')

    target_quantiles = [0.1, 0.5, 0.9]
    row_titles =['u ≈ 0.1\n(Core / Healthy)', 'u ≈ 0.5\n(Normal Edge)', 'u ≈ 0.9\n(Anomaly)']

    for i, q in enumerate(target_quantiles):
        # Find the 5 samples closest to the target quantile
        idx_sorted = np.argsort(np.abs(ranks - q))
        selected_idx = idx_sorted[:5]

        axes[i, 0].set_ylabel(row_titles[i], fontsize=16, fontweight='bold',
                              rotation=0, labelpad=80, ha='center', va='center')

        for j, idx in enumerate(selected_idx):
            ax = axes[i, j]
            img_inverted = 1.0 - images[idx]

            ax.imshow(img_inverted, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_color('gray')
                spine.set_linewidth(1)

            ax.text(0.5, -0.15, f"u = {ranks[idx]:.3f}",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, color='darkred', fontweight='bold')

    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    plt.suptitle("Kernel-AMQR: 5 Nearest Samples per Target Quantile",
                 fontsize=22, fontweight='bold', y=0.98)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Rendering complete! Please check the image: {filename}")


# =====================================================================
# Main execution flow
# =====================================================================
if __name__ == "__main__":
    from scipy.spatial.distance import pdist

    print("========================================================")
    print(" Appendix: Implicit RKHS Geometry via Kernel-AMQR")
    print("========================================================")

    # Lock random seed
    np.random.seed(42)

    print("Loading MNIST data...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    mask = (mnist.target == '8')
    X_digit = mnist.data[mask][:400]

    # Normalization
    X_digit_scaled = X_digit / 255.0
    images = X_digit_scaled.reshape(-1, 28, 28)

    # =====================================================================
    # Use Median Heuristic to accurately anchor the RKHS manifold scale
    # =====================================================================
    print("Calculating Median Heuristic for RBF kernel...")
    pairwise_sq_dists = pdist(X_digit_scaled, metric='sqeuclidean')
    median_sq_dist = np.median(pairwise_sq_dists)
    # Ensure no division by 0
    gamma_val = 1.0 / (median_sq_dist + 1e-8)
    print(f"   -> Dynamically calculated optimal Gamma: {gamma_val:.6f}")

    print("Running unified Kernel AMQR Engine (Pathway C)...")

    # =====================================================================
    # Increase intrinsic dimension d_int to avoid squashing high-dimensional features in RKHS
    # =====================================================================
    engine = Kernel_AMQR_Engine(
        kernel='rbf',
        gamma=gamma_val,
        d_int=2,
        epsilon=0.0,  # Exact GW zero entropy blur
        use_log_squash=False
    )

    med, ranks = engine.fit_predict(X_digit_scaled)

    # Output image to the specified unified results directory
    os.makedirs(os.path.join(PROJECT_ROOT, "results", "figures"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "results", "figures", "Fig_Appendix_Kernel_MNIST.pdf")
    plot_3x5_grid(images, ranks, save_path)
