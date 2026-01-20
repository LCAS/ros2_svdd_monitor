"""
Feature Extraction for SVDD Anomaly Detection

This module extracts features from sliding windows of cmd_vel and IMU data.
Features capture the relationship between commanded velocity and expected IMU response.
"""

import numpy as np


def extract_window_features(cmd_vel_window, imu_window):
    """
    Extract statistical features from cmd_vel and IMU sliding windows.
    
    This function computes features that capture:
    - cmd_vel statistics (linear and angular velocities)
    - IMU statistics (linear acceleration and angular velocity)
    - Cross-features mapping cmd_vel to expected IMU response
    
    Args:
        cmd_vel_window: List of cmd_vel messages (linear.x, linear.y, linear.z, 
                       angular.x, angular.y, angular.z)
        imu_window: List of IMU messages (linear_acceleration.x/y/z, 
                   angular_velocity.x/y/z)
    
    Returns:
        numpy array of extracted features
    """
    features = []
    
    # Extract cmd_vel data
    if len(cmd_vel_window) > 0:
        cmd_vel_array = np.array(cmd_vel_window)
        
        # Statistics for linear velocity
        features.extend([
            np.mean(cmd_vel_array[:, 0]),  # mean linear.x
            np.std(cmd_vel_array[:, 0]),   # std linear.x
            np.min(cmd_vel_array[:, 0]),   # min linear.x
            np.max(cmd_vel_array[:, 0]),   # max linear.x
        ])
        
        # Statistics for angular velocity (z-axis, typical for differential drive)
        features.extend([
            np.mean(cmd_vel_array[:, 5]),  # mean angular.z
            np.std(cmd_vel_array[:, 5]),   # std angular.z
            np.min(cmd_vel_array[:, 5]),   # min angular.z
            np.max(cmd_vel_array[:, 5]),   # max angular.z
        ])
    else:
        # If no cmd_vel data, use zeros
        features.extend([0.0] * 8)
    
    # Extract IMU data
    if len(imu_window) > 0:
        imu_array = np.array(imu_window)
        
        # Statistics for linear acceleration (x-axis, forward acceleration)
        features.extend([
            np.mean(imu_array[:, 0]),  # mean accel.x
            np.std(imu_array[:, 0]),   # std accel.x
            np.min(imu_array[:, 0]),   # min accel.x
            np.max(imu_array[:, 0]),   # max accel.x
        ])
        
        # Statistics for angular velocity (z-axis gyroscope)
        features.extend([
            np.mean(imu_array[:, 5]),  # mean gyro.z
            np.std(imu_array[:, 5]),   # std gyro.z
            np.min(imu_array[:, 5]),   # min gyro.z
            np.max(imu_array[:, 5]),   # max gyro.z
        ])
    else:
        # If no IMU data, use zeros
        features.extend([0.0] * 8)
    
    # Cross-features: relationship between cmd_vel and IMU
    if len(cmd_vel_window) > 0 and len(imu_window) > 0:
        cmd_vel_array = np.array(cmd_vel_window)
        imu_array = np.array(imu_window)
        
        # Align sequences: use most recent overlapping samples when lengths differ
        cmd_linear = cmd_vel_array[:, 0]
        imu_accel = imu_array[:, 0]

        min_len = min(len(cmd_linear), len(imu_accel))
        if min_len >= 2:
            cmd_seg = cmd_linear[-min_len:]
            imu_seg = imu_accel[-min_len:]
            if np.std(cmd_seg) > 1e-6 and np.std(imu_seg) > 1e-6:
                correlation = np.corrcoef(cmd_seg, imu_seg)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
        else:
            correlation = 0.0

        features.append(correlation)

        # Angular correlation (z)
        cmd_angular = cmd_vel_array[:, 5]
        imu_gyro = imu_array[:, 5]

        min_len = min(len(cmd_angular), len(imu_gyro))
        if min_len >= 2:
            cmd_seg = cmd_angular[-min_len:]
            imu_seg = imu_gyro[-min_len:]
            if np.std(cmd_seg) > 1e-6 and np.std(imu_seg) > 1e-6:
                correlation = np.corrcoef(cmd_seg, imu_seg)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
        else:
            correlation = 0.0

        features.append(correlation)

        # Magnitude ratios (should be relatively stable for normal operation)
        # Use overlapping segments consistent with correlation computation
        min_len = min(len(cmd_linear), len(imu_accel))
        if min_len >= 1:
            cmd_seg = np.abs(cmd_linear[-min_len:])
            imu_seg = np.abs(imu_accel[-min_len:])
            mean_cmd_linear = np.mean(cmd_seg)
            mean_imu_accel = np.mean(imu_seg)
        else:
            mean_cmd_linear = 0.0
            mean_imu_accel = 0.0

        if mean_cmd_linear > 1e-6:
            accel_ratio = mean_imu_accel / mean_cmd_linear
        else:
            accel_ratio = 0.0

        features.append(accel_ratio)

        min_len = min(len(cmd_angular), len(imu_gyro))
        if min_len >= 1:
            cmd_seg = np.abs(cmd_angular[-min_len:])
            imu_seg = np.abs(imu_gyro[-min_len:])
            mean_cmd_angular = np.mean(cmd_seg)
            mean_imu_gyro = np.mean(imu_seg)
        else:
            mean_cmd_angular = 0.0
            mean_imu_gyro = 0.0

        if mean_cmd_angular > 1e-6:
            gyro_ratio = mean_imu_gyro / mean_cmd_angular
        else:
            gyro_ratio = 0.0

        features.append(gyro_ratio)
    else:
        # If missing data, use zeros for cross-features
        features.extend([0.0] * 4)
    
    return np.array(features)


def extract_features_from_dataframe(df, window_size=10):
    """
    Extract features from a pandas DataFrame containing synchronized cmd_vel and IMU data.
    
    Args:
        df: DataFrame with columns for cmd_vel (linear_x, angular_z) and 
            IMU (accel_x, gyro_z, etc.)
        window_size: Size of the sliding window
    
    Returns:
        numpy array of features, shape (n_samples, n_features)
    """
    features_list = []
    
    # Expected columns for cmd_vel
    cmd_vel_cols = ['cmd_vel_linear_x', 'cmd_vel_linear_y', 'cmd_vel_linear_z',
                    'cmd_vel_angular_x', 'cmd_vel_angular_y', 'cmd_vel_angular_z']
    
    # Expected columns for IMU
    imu_cols = ['imu_accel_x', 'imu_accel_y', 'imu_accel_z',
                'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z']
    
    # Check if columns exist, if not use simplified names
    if not all(col in df.columns for col in cmd_vel_cols):
        # Try simplified column names
        cmd_vel_cols = ['linear_x', 'linear_y', 'linear_z',
                       'angular_x', 'angular_y', 'angular_z']
    
    if not all(col in df.columns for col in imu_cols):
        # Try simplified column names
        imu_cols = ['accel_x', 'accel_y', 'accel_z',
                   'gyro_x', 'gyro_y', 'gyro_z']
    
    # Extract data
    try:
        cmd_vel_data = df[cmd_vel_cols].values
        imu_data = df[imu_cols].values
    except KeyError as e:
        raise ValueError(f"Missing required columns in DataFrame: {e}")
    
    # Sliding window feature extraction
    for i in range(len(df)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        
        cmd_vel_window = cmd_vel_data[start_idx:end_idx]
        imu_window = imu_data[start_idx:end_idx]
        
        features = extract_window_features(cmd_vel_window, imu_window)
        features_list.append(features)
    
    return np.array(features_list)


def get_feature_names():
    """
    Get names of all features for interpretability.
    
    Returns:
        List of feature names
    """
    feature_names = [
        # cmd_vel linear.x statistics
        'cmd_vel_linear_x_mean',
        'cmd_vel_linear_x_std',
        'cmd_vel_linear_x_min',
        'cmd_vel_linear_x_max',
        # cmd_vel angular.z statistics
        'cmd_vel_angular_z_mean',
        'cmd_vel_angular_z_std',
        'cmd_vel_angular_z_min',
        'cmd_vel_angular_z_max',
        # IMU accel.x statistics
        'imu_accel_x_mean',
        'imu_accel_x_std',
        'imu_accel_x_min',
        'imu_accel_x_max',
        # IMU gyro.z statistics
        'imu_gyro_z_mean',
        'imu_gyro_z_std',
        'imu_gyro_z_min',
        'imu_gyro_z_max',
        # Cross-features
        'cmd_linear_to_imu_accel_correlation',
        'cmd_angular_to_imu_gyro_correlation',
        'imu_accel_to_cmd_linear_ratio',
        'imu_gyro_to_cmd_angular_ratio',
    ]
    return feature_names
