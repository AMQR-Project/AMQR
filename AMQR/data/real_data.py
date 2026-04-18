import os
import numpy as np
import pandas as pd
import urllib.request
import mne  # 强大的神经科学 EEG 处理库


def load_pems_traffic_and_locations(h5_path, csv_path, num_nodes=20):
    """
    加载 PEMS-BAY 交通速度数据与传感器空间经纬度
    返回: Y_flat (矩阵), T (小时), timestamps (时间戳), coords (经纬度), sensor_ids
    """
    print(f"📥 正在加载 PEMS-BAY 交通数据集: {h5_path} ...")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"找不到交通数据文件 {h5_path}")

    df_traffic = pd.read_hdf(h5_path)
    sensor_ids = df_traffic.columns[:num_nodes].astype(str).tolist()

    print(f"📥 正在加载传感器经纬度信息: {csv_path} ...")
    df_loc = pd.read_csv(csv_path)
    df_loc['sensor_id'] = df_loc['sensor_id'].astype(str)
    loc_dict = dict(zip(df_loc['sensor_id'], zip(df_loc['longitude'], df_loc['latitude'])))

    # 获取坐标，处理可能缺失的节点
    coords = []
    for sid in sensor_ids:
        if sid in loc_dict:
            coords.append(loc_dict[sid])
        else:
            coords.append((np.nan, np.nan))

    coords = np.array(coords)
    valid_mask = ~np.isnan(coords[:, 0])
    coords[~valid_mask] = np.nanmean(coords[valid_mask], axis=0)  # 均值填补缺失

    # 提取时间与矩阵特征
    df_traffic['date'] = df_traffic.index.date
    df_traffic['hour'] = df_traffic.index.hour

    cov_matrices, time_of_day, timestamps = [], [], []
    grouped = df_traffic.groupby(['date', 'hour'])

    print("🧠 正在按‘小时’构建空间拓扑协方差矩阵...")
    for (date, hour), group in grouped:
        if len(group) < 10:  # 剔除无效时间段
            continue
        vals = group.iloc[:, :num_nodes].values
        cov = np.cov(vals, rowvar=False) + np.eye(num_nodes) * 1e-3
        cov_matrices.append(cov.flatten())
        time_of_day.append(hour)
        timestamps.append(f"{date} {hour:02d}:00")

    Y_flat = np.array(cov_matrices)
    T = np.array(time_of_day)
    timestamps = np.array(timestamps)

    print(f"✅ 成功生成 {len(Y_flat)} 个交通拓扑样本！")
    return Y_flat, T, timestamps, coords, sensor_ids


def load_chbmit_eeg_topology(save_dir="data/raw"):
    """
    自动下载并处理 CHB-MIT 脑电波数据，转化为 SPD 流形协方差矩阵
    """
    os.makedirs(save_dir, exist_ok=True)
    edf_path = os.path.join(save_dir, "chb01_03.edf")

    # 1. 自动从麻省理工 PhysioNet 官方直链下载 (约 40MB，只需下载一次)
    url = "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf?download"
    if not os.path.exists(edf_path):
        print(f"📥 正在从 MIT 官方下载脑电波原始数据 (Patient 01, Session 03)...")
        urllib.request.urlretrieve(url, edf_path)
        print("✅ 下载完成！")

    # 2. 读取原始 EDF 医学文件
    print("🧠 正在解析 EDF 脑电极信号...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # 提取信号: 23个电极通道，采样率为 256 Hz
    data, times = raw[:]
    sfreq = raw.info['sfreq']
    num_nodes = data.shape[0]  # 23 个电极节点

    # 3. 核心物理转换：滑动窗口切片 -> 协方差拓扑矩阵
    window_sec = 2.0  # 每 2 秒算一个拓扑状态 (类似交通的每1小时)
    window_samples = int(window_sec * sfreq)
    n_windows = data.shape[1] // window_samples

    print(f"✂️ 正在以 {window_sec}秒 为窗口滑动提取动态网络拓扑...")
    cov_matrices = []
    time_centers = []
    is_seizure = []

    # 👨‍⚕️ 医生手工标注的 Ground Truth (发作时间: 2996秒 ~ 3036秒)
    seizure_start = 2996
    seizure_end = 3036

    for i in range(n_windows):
        start_idx = i * window_samples
        end_idx = start_idx + window_samples
        window_data = data[:, start_idx:end_idx]

        # 计算 23x23 的脑网络功能连接协方差矩阵
        cov = np.cov(window_data) + np.eye(num_nodes) * 1e-5
        cov_matrices.append(cov.flatten())

        # 记录当前窗口的中心时间 (秒)
        t_center = (start_idx + window_samples / 2) / sfreq
        time_centers.append(t_center)

        # 打上医学金标准标签
        if seizure_start <= t_center <= seizure_end:
            is_seizure.append(1)  # 癫痫发作中
        else:
            is_seizure.append(0)  # 健康平稳期

    Y_flat = np.array(cov_matrices)
    T_sec = np.array(time_centers)
    labels = np.array(is_seizure)

    print(f"✅ 成功提取 {len(Y_flat)} 个连续的脑电 SPD 矩阵样本！")
    return Y_flat, T_sec, labels, num_nodes
