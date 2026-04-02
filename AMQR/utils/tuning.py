import numpy as np
import itertools
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph

from models.amqr_engine import AMQR_Engine


def auto_tune_amqr(Y, param_grid, fixed_kwargs, cv=5, T=None, t_eval=None, window_size=None):
    """
    无监督自动调参 (支持全局静态模式 & 时间回归模式)
    """
    N_original = len(Y)

    # =========================================================
    # 🛡️ 机制 1：大规模数据降采样保护 (防止内存与算力爆炸)
    # 调参不需要全量数据，3000 个点足以完美刻画流形的拓扑骨架！
    # =========================================================
    if N_original > 3000:
        print(f"⚠️ 数据量 ({N_original}) 较大，自动启用随机降采样 (N=3000) 进行极速调参...")
        np.random.seed(42)
        sub_idx = np.random.choice(N_original, 3000, replace=False)
        Y = Y[sub_idx]
        if T is not None:
            T = T[sub_idx]

    N = len(Y)
    print("⏳ 准备全局度量结构 (Global Metric Structure)...")
    use_knn = fixed_kwargs.get('use_knn', False)
    use_log_squash = fixed_kwargs.get('use_log_squash', False)

    # 建立底层度量矩阵
    if use_knn:
        k = min(15, N - 1)
        A = kneighbors_graph(Y, n_neighbors=k, mode='distance', include_self=False)
        Cy_full = shortest_path(A, method='auto', directed=False)
        Cy_full[np.isinf(Cy_full)] = np.nanmax(Cy_full[Cy_full != np.inf]) * 2.0
    else:
        Cy_full = cdist(Y, Y, metric='euclidean')

    if use_log_squash:
        Cy_full = np.log1p(Cy_full)

    # 展开超参数网格
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_params = None
    best_score = float('inf')
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # =========================================================
    # ⚡ 机制 2：稀疏时间轴加速
    # 回归模式下，在调参时只抽样 20% 的时间窗口，可提速 5 倍！
    # =========================================================
    t_eval_tune = None
    if t_eval is not None:
        step = max(1, len(t_eval) // 5)
        t_eval_tune = t_eval[::step]
        print(f"⚡ 启用稀疏时间调参: 将 {len(t_eval)} 个窗口压缩为 {len(t_eval_tune)} 个以加速计算")

    print(f"🔍 开始 {cv}-Fold 交叉验证 (共 {len(param_combinations)} 组参数)...")

    for params in param_combinations:
        fold_scores = []
        for train_idx, val_idx in kf.split(Y):
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            engine = AMQR_Engine(**params, **fixed_kwargs)

            # --- 动态回归模式 ---
            if T is not None:
                T_train, T_val = T[train_idx], T[val_idx]
                _, ranks_train = engine.fit_predict(Y_train, T=T_train, t_eval=t_eval_tune, window_size=window_size)

                # 🛡️ 机制 3：时空量纲对齐
                # 样本外扩展必须同时参考时间 T 和空间 Y，为防止 T 的尺度过大或过小，将其方差对齐到 Y
                T_scale = np.mean(np.std(Y_train, axis=0)) / (np.std(T_train) + 1e-9)
                TY_train = np.column_stack([T_train * T_scale, Y_train])
                TY_val = np.column_stack([T_val * T_scale, Y_val])

                knn_ext = KNeighborsRegressor(n_neighbors=5).fit(TY_train, ranks_train)
                ranks_val = knn_ext.predict(TY_val)

            # --- 静态全局模式 ---
            else:
                _, ranks_train = engine.fit_predict(Y_train)
                knn_ext = KNeighborsRegressor(n_neighbors=5).fit(Y_train, ranks_train)
                ranks_val = knn_ext.predict(Y_val)

            # 计算 OOS-Stress 误差
            Cy_val = Cy_full[np.ix_(val_idx, val_idx)]
            Cy_val_norm = Cy_val / (np.max(Cy_val) + 1e-9)

            Cz_val = np.abs(ranks_val[:, None] - ranks_val[None, :])
            Cz_val_norm = Cz_val / (np.max(Cz_val) + 1e-9)

            stress = np.mean((Cy_val_norm - Cz_val_norm) ** 2)
            fold_scores.append(stress)

        mean_score = np.mean(fold_scores)
        print(f"  配置 {params} | OOS-Stress: {mean_score:.5f}")

        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    print(f"🏆 最佳参数组合: {best_params} (Score: {best_score:.5f})")
    return best_params