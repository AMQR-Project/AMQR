import os
import numpy as np
import pandas as pd
import urllib.request
import mne 


def load_pems_traffic_and_locations(h5_path, csv_path, num_nodes=20):
    """
    Load PEMS-BAY traffic speed data and sensor spatial latitude and longitude
    Returns: Y_flat (matrix), T (hour), timestamps (timestamp), coords (latitude and longitude), sensor_ids
    """
    print(f"Loading PEMS-BAY traffic dataset: {h5_path} ...")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Traffic data file not found {h5_path}")

    df_traffic = pd.read_hdf(h5_path)
    sensor_ids = df_traffic.columns[:num_nodes].astype(str).tolist()

    print(f"Loading sensor latitude and longitude information: {csv_path} ...")
    df_loc = pd.read_csv(csv_path)
    df_loc['sensor_id'] = df_loc['sensor_id'].astype(str)
    loc_dict = dict(zip(df_loc['sensor_id'], zip(df_loc['longitude'], df_loc['latitude'])))

    # Get coordinates, handle potentially missing nodes
    coords =[]
    for sid in sensor_ids:
        if sid in loc_dict:
            coords.append(loc_dict[sid])
        else:
            coords.append((np.nan, np.nan))

    coords = np.array(coords)
    valid_mask = ~np.isnan(coords[:, 0])
    coords[~valid_mask] = np.nanmean(coords[valid_mask], axis=0)  # Mean imputation for missing values

    # Extract time and matrix features
    df_traffic['date'] = df_traffic.index.date
    df_traffic['hour'] = df_traffic.index.hour

    cov_matrices, time_of_day, timestamps = [], [], []
    grouped = df_traffic.groupby(['date', 'hour'])

    print("🧠 Building spatial topology covariance matrix by 'hour'...")
    for (date, hour), group in grouped:
        if len(group) < 10:  # Remove invalid time periods
            continue
        vals = group.iloc[:, :num_nodes].values
        cov = np.cov(vals, rowvar=False) + np.eye(num_nodes) * 1e-3
        cov_matrices.append(cov.flatten())
        time_of_day.append(hour)
        timestamps.append(f"{date} {hour:02d}:00")

    Y_flat = np.array(cov_matrices)
    T = np.array(time_of_day)
    timestamps = np.array(timestamps)

    print(f"Successfully generated {len(Y_flat)} traffic topology samples!")
    return Y_flat, T, timestamps, coords, sensor_ids


def load_chbmit_eeg_topology(save_dir="data/raw"):
    """
    Automatically download and process CHB-MIT EEG data, convert to SPD manifold covariance matrix
    """
    os.makedirs(save_dir, exist_ok=True)
    edf_path = os.path.join(save_dir, "chb01_03.edf")

    # 1. Automatically download from MIT PhysioNet official direct link (approx. 40MB, only needs to be downloaded once)
    url = "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf?download"
    if not os.path.exists(edf_path):
        print(f"Downloading raw EEG data from MIT official (Patient 01, Session 03)...")
        urllib.request.urlretrieve(url, edf_path)
        print("Download complete!")

    # 2. Read raw EDF medical file
    print("Parsing EDF EEG electrode signals...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Extract signals: 23 electrode channels, sampling rate is 256 Hz
    data, times = raw[:]
    sfreq = raw.info['sfreq']
    num_nodes = data.shape[0]  # 23 electrode nodes

    # 3. Core physical conversion: sliding window slicing -> covariance topology matrix
    window_sec = 2.0  # Calculate a topology state every 2 seconds (similar to every 1 hour for traffic)
    window_samples = int(window_sec * sfreq)
    n_windows = data.shape[1] // window_samples

    print(f"Extracting dynamic network topology with a sliding window of {window_sec} seconds...")
    cov_matrices =[]
    time_centers = []
    is_seizure =[]

    # Ground Truth manually annotated by doctors (Seizure time: 2996s ~ 3036s)
    seizure_start = 2996
    seizure_end = 3036

    for i in range(n_windows):
        start_idx = i * window_samples
        end_idx = start_idx + window_samples
        window_data = data[:, start_idx:end_idx]

        # Calculate 23x23 brain network functional connectivity covariance matrix
        cov = np.cov(window_data) + np.eye(num_nodes) * 1e-5
        cov_matrices.append(cov.flatten())

        # Record the center time of the current window (seconds)
        t_center = (start_idx + window_samples / 2) / sfreq
        time_centers.append(t_center)

        # Apply medical gold standard labels
        if seizure_start <= t_center <= seizure_end:
            is_seizure.append(1)  # Seizure in progress
        else:
            is_seizure.append(0)  # Healthy stable period

    Y_flat = np.array(cov_matrices)
    T_sec = np.array(time_centers)
    labels = np.array(is_seizure)

    print(f"Successfully extracted {len(Y_flat)} continuous EEG SPD matrix samples!")
    return Y_flat, T_sec, labels, num_nodes
