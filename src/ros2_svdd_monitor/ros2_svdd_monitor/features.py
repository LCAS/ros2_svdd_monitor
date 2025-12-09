"""Feature extraction module for IMU and cmd_vel data."""

import numpy as np
from collections import deque


class FeatureExtractor:
    """Extract features from IMU and cmd_vel messages for SVDD."""
    
    def __init__(self, window_size=10):
        """
        Initialize feature extractor.
        
        Args:
            window_size: Number of samples for feature computation
        """
        self.window_size = window_size
        self.imu_buffer = deque(maxlen=window_size)
        self.cmd_vel_buffer = deque(maxlen=window_size)
        
    def add_imu_data(self, imu_msg):
        """
        Add IMU data to buffer.
        
        Args:
            imu_msg: sensor_msgs/Imu message
        """
        # Extract linear acceleration and angular velocity
        imu_data = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z,
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])
        self.imu_buffer.append(imu_data)
        
    def add_cmd_vel_data(self, cmd_vel_msg):
        """
        Add cmd_vel data to buffer.
        
        Args:
            cmd_vel_msg: geometry_msgs/Twist message
        """
        # Extract linear and angular velocity commands
        cmd_vel_data = np.array([
            cmd_vel_msg.linear.x,
            cmd_vel_msg.linear.y,
            cmd_vel_msg.linear.z,
            cmd_vel_msg.angular.x,
            cmd_vel_msg.angular.y,
            cmd_vel_msg.angular.z
        ])
        self.cmd_vel_buffer.append(cmd_vel_data)
        
    def extract_features(self):
        """
        Extract features from buffered data.
        
        Returns:
            np.ndarray: Feature vector, or None if insufficient data
        """
        if len(self.imu_buffer) < self.window_size or len(self.cmd_vel_buffer) < self.window_size:
            return None
            
        # Convert buffers to numpy arrays
        imu_array = np.array(self.imu_buffer)
        cmd_vel_array = np.array(self.cmd_vel_buffer)
        
        # Extract statistical features
        features = []
        
        # IMU features (mean, std, min, max for each dimension)
        features.extend(np.mean(imu_array, axis=0))
        features.extend(np.std(imu_array, axis=0))
        features.extend(np.min(imu_array, axis=0))
        features.extend(np.max(imu_array, axis=0))
        
        # cmd_vel features (mean, std, min, max for each dimension)
        features.extend(np.mean(cmd_vel_array, axis=0))
        features.extend(np.std(cmd_vel_array, axis=0))
        features.extend(np.min(cmd_vel_array, axis=0))
        features.extend(np.max(cmd_vel_array, axis=0))
        
        return np.array(features)
        
    def reset(self):
        """Reset the feature buffers."""
        self.imu_buffer.clear()
        self.cmd_vel_buffer.clear()
