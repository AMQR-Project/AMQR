# AMQR: Adaptive Manifold Quantile Regression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official Python implementation of the paper:
> **"Distance-Driven Conditional Quantiles on Manifolds via Gromov-Wasserstein Alignment"**  

## 📖 Overview

**Adaptive Manifold Quantile Regression (AMQR)** is a fully data-driven, metric-agnostic framework designed to estimate geometric medians and structurally faithful center-outward quantile regions on complex, non-linear manifolds. 

By reformulating discrete topological extraction as a continuous optimal transport problem, AMQR utilizes exact, unregularized Gromov-Wasserstein ($\mathcal{GW}_2$) alignment to map potentially anisotropic empirical geometries onto a perfectly symmetric latent space. 

### ✨ Core Highlights
* **Metric-Agnostic Universal Interface:** Seamlessly accommodates empirical $k$-NN graph geodesics (Pathway B), exact analytical metrics (Pathway A), and implicit RKHS distances (Pathway C).
* **Dimensionality Decoupling:** Successfully circumvents the ambient curse of dimensionality. The convergence rate strictly depends on the predictor dimension $p$ under a rigorously defined geometric bottleneck ($d < p+2$).
* **Topological Orthogonalization:** Decouples heavy-tailed structural anomalies (e.g., Euclidean swelling, phase shifts) from the healthy geometric backbone, yielding shape-adaptive quantile contours free from rigid isotropic constraints.

---

## 🗂️ Repository Structure

```text
.
├── data/                   # Data generation and loading modules
│   ├── ellipse.py          # 2D linear/non-linear ellipse generators
│   ├── simulations.py      # 3D spiral, bimodal crescent, functional, and SPD data
│   └── real_data.py        # Loaders for PeMS-BAY traffic and CHB-MIT EEG datasets
├── models/                 # Core algorithm implementations
│   ├── amqr_engine.py      # The main AMQR framework (Exact GW, OOS projection)
│   ├── kernel_amqr_engine.py # Kernel-AMQR for infinite-dimensional RKHS
│   └── baselines.py        # SOTA baselines (Fréchet L1/L2, NW Mean, SW-KDE)
├── utils/                  # Helper functions
│   ├── tuning.py           # Unsupervised OOS-GW cross-validation
│   ├── visualization.py    # Plotting scripts for all paper figures
│   └── metrics.py          # Quantitative evaluation metrics
└── experiments/            # Scripts to reproduce all experiments in the paper
```

---

## ⚙️ Installation & Dependencies

The code is written in Python 3.8+. To install the required dependencies, run:

```bash
pip install numpy scipy pandas scikit-learn matplotlib seaborn pot tslearn mne tables
```
*Note: `pot` is the Python Optimal Transport library used for the core $\mathcal{GW}_2$ alignment. `tables` is required for reading `.h5` traffic data.*

---

## 🚀 Reproducing the Experiments

We provide standalone scripts in the `experiments/` directory to reproduce all figures and tables presented in the paper. All generated figures will be automatically saved to the `results/figures/` directory.

### 1. Motivating Example (Fig. 1)
Demonstrates the "Topological Orthogonalization" of AMQR vs. the rigid isotropic contours of the Fréchet $L_1$ median on linear and non-linear manifolds.
```bash
python experiments/run_ellipse.py
```

### 2. 3D Spiral with Topological Gap (Fig. 2)
Tests the estimators' ability to bridge a 135° topological vacuum without collapsing into the ambient space.
```bash
python experiments/run_sim1_spiral.py
```

### 3. The Density-Inversion Fallacy (Fig. 3)
Evaluates robustness against severe density biases on a bimodal crescent manifold.
```bash
python experiments/run_bimodal_crescent.py
```

### 4. Dynamic Functional Regression (Fig. 4, 5, 6)
Tests the framework on high-dimensional functional curves with time-varying bimodal phase shifts, demonstrating AMQR's immunity to *destructive interference*.
```bash
python experiments/run_sim2_functional.py
```

### 5. Dynamic Regression on Riemannian Cones (Fig. 7)
Evaluates Pathway A (Analytical Metric) on $3 \times 3$ SPD matrices contaminated by volumetric outliers, demonstrating resistance to *Euclidean swelling*.
```bash
python experiments/run_sim3_spd.py
```

### 6. Real Application I: ECG Phase Alignment (Fig. 8)
Extracts morphological prototypes from raw ECG signals, comparing AMQR's intrinsic topological retrieval against DTW.
```bash
python experiments/run_real1_ecg_with_dtw.py
```

### 7. Real Application II: Traffic Network Topology (Fig. 9, 10)
Monitors the heteroscedastic spatiotemporal evolution of the PeMS-BAY traffic network using $20 \times 20$ SPD covariance matrices.
*(Note: Requires `pems-bay.h5` and `graph_sensor_locations.csv` in `data/raw/`)*
```bash
python experiments/run_real2_traffic.py
```

### 8. Appendix: Kernel-AMQR on MNIST (Fig. 11)
Demonstrates Pathway C (Implicit RKHS Metric) by performing unsupervised morphological ranking on raw 784-D pixel data using a 2D topological bottleneck.
```bash
python experiments/run_addsim_kernel.py
```

---

## 💻 Quick Start: Using AMQR on Your Own Data

The `AMQR_Engine` is designed to be highly modular and easy to integrate into your own pipelines.

```python
import numpy as np
from models.amqr_engine import AMQR_Engine

# 1. Prepare your data (N samples, D dimensions)
Y = np.random.randn(500, 10) 

# 2. Initialize the AMQR Engine
# Pathway B (Data-driven k-NN metric) is used by default when y_dist_m is None
amqr = AMQR_Engine(
    ref_dist='uniform',   # Latent reference distribution
    epsilon=0.0,          # Exact unregularized GW alignment
    d_int=None,           # Auto-detect intrinsic dimension via MLE
    use_knn=True,         # Use k-NN graph for geodesic approximation
    k_neighbors=15
)

# 3. Fit and Predict
# Returns the geometric median, quantile ranks (0 to 1), and latent coordinates
median, ranks, latent_z = amqr._run_with_oos_protection(Y)

print(f"Topological Median Shape: {median.shape}")
print(f"Anomaly Quantile Ranks: {ranks[:5]}")
```

To use **Pathway A (Analytical Metric)**, simply precompute your exact distance matrix (e.g., Log-Euclidean, Wasserstein) and pass it to the engine:

```python
# y_dist_m must be a condensed or square distance matrix
median, ranks = amqr.fit_predict(Y, y_dist_m=exact_distance_matrix)
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
