# utils/tuning.py

import numpy as np
import itertools
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial.distance import cdist

# 确保 AMQR_Engine 被正确导入
from models.amqr_engine import AMQR_Engine


def auto_tune_amqr(Y, param_grid, fixed_kwargs, cv=5, T=None, t_eval=None, window_size=None):
    """
    无监督自动调参 (支持全局静态模式 & 时间回归模式)

    🌟 核心更新：损失函数已从 "OOS-Stress" (结构应力)
    切换为 "Latent Fréchet Variance" (潜在空间弗雷歇方差),
    严格遵循论文公式 (6) 的理论指导。
    """
    N_original = len(Y)

    # =========================================================
    # 机制 1：大规模数据降采样保护
    # =========================================================
    if N_original > 3000:
        print(f"⚠️ 数据量 ({N_original}) 较大，自动启用随机降采样 (N=3000) 进行极速调参...")
        np.random.seed(42)
        sub_idx = np.random.choice(N_original, 3000, replace=False)
        Y = Y[sub_idx]
        if T is not None:
            T = T[sub_idx]

    # 展开超参数网格
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_params = None
    best_score = float('inf')
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # =========================================================
    # ⚡ 机制 2：稀疏时间轴加速 (仅在回归模式下)
    # =========================================================
    t_eval_tune = None
    if t_eval is not None:
        step = max(1, len(t_eval) // 10)  # 抽样密度可调，这里设为10%
        t_eval_tune = t_eval[::step]
        print(f"⚡ 启用稀疏时间调参: 将 {len(t_eval)} 个窗口压缩为 {len(t_eval_tune)} 个以加速计算")

    print(f"🔍 开始 {cv}-Fold 交叉验证 (共 {len(param_combinations)} 组参数)...")

    for params in param_combinations:
        fold_scores = []
        for train_idx, val_idx in kf.split(Y):
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            # 实例化引擎
            engine = AMQR_Engine(**params, **fixed_kwargs)

            # --- 动态回归模式 ---
            if T is not None:
                T_train, T_val = T[train_idx], T[val_idx]

                # 1. 在训练集上运行滑动窗口回归，得到每个训练点的最终“全局”秩
                _, ranks_train = engine.fit_predict(Y_train, T=T_train, t_eval=t_eval_tune, window_size=window_size)

                # 2. 样本外扩展：使用时空坐标来预测验证集的秩
                # 论文中未明确定义回归模式下的CV，这里沿用一种合理的时空KNN扩展方式
                T_scale = np.mean(np.std(Y_train, axis=0)) / (np.std(T_train) + 1e-9)
                TY_train = np.column_stack([T_train * T_scale, Y_train.reshape(len(Y_train), -1)])
                TY_val = np.column_stack([T_val * T_scale, Y_val.reshape(len(Y_val), -1)])

                knn_ext = KNeighborsRegressor(n_neighbors=5).fit(TY_train, ranks_train)
                ranks_val = knn_ext.predict(TY_val)

                # 3. 计算损失：将秩转换为一维潜在坐标，并计算其二范数的平方
                # 对于uniform参考分布，秩u到坐标z的映射为 z = 2u - 1
                z_val = (2 * ranks_val - 1).reshape(-1, 1)
                score = np.mean(np.sum(z_val ** 2, axis=1))
                fold_scores.append(score)

            # --- 静态全局模式 ---
            else:
                # 1. 在训练集上拟合，获取其在潜在空间中的坐标 z_train
                #    为获取z坐标，需调用内部核心方法 _fit_predict_core
                try:
                    # _fit_predict_core 返回 (med, ranks, z_coords)
                    _, _, z_train = engine._fit_predict_core(Y_train)
                except Exception as e:
                    # 如果引擎API变动，提供备用方案
                    print(f"无法直接调用 _fit_predict_core, 错误: {e}")
                    print("将退回使用秩作为损失函数。建议检查 AMQR_Engine 内部方法。")
                    _, ranks_train = engine.fit_predict(Y_train)
                    z_train = (2 * ranks_train - 1).reshape(-1, 1)  # 假设d=1

                # 2. 样本外扩展：使用KNN回归器，从Y_train空间映射到Z_train空间
                #    这是论文中 "Smooth Intrinsic Projection" 的一个高效实现
                knn_regressor = KNeighborsRegressor(n_neighbors=5, weights='distance')
                knn_regressor.fit(Y_train.reshape(len(Y_train), -1), z_train)
                z_val = knn_regressor.predict(Y_val.reshape(len(Y_val), -1))

                # 3. 计算损失：根据论文公式(6)，最小化验证集投影坐标的期望平方范数
                score = np.mean(np.sum(z_val ** 2, axis=1))
                fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        print(f"  配置 {params} | Latent Fréchet Variance: {mean_score:.6f}")

        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    print(f"🏆 最佳参数组合: {best_params} (Score: {best_score:.6f})")
    return best_params
