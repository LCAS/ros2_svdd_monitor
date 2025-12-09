"""Configuration module for SVDD monitor."""

import os


class SVDDConfig:
    """Configuration for SVDD anomaly detection."""
    
    def __init__(self):
        """Initialize SVDD configuration."""
        # Topics
        self.imu_topic = '/imu'
        self.cmd_vel_topic = '/cmd_vel'
        
        # Model parameters
        self.nu = 0.1  # Outlier fraction for OneClassSVM
        self.kernel = 'rbf'
        self.gamma = 'auto'
        
        # Feature extraction parameters
        self.window_size = 10  # Number of samples for feature computation
        self.feature_buffer_size = 100  # Buffer size for streaming features
        
        # Model path
        self.model_path = os.path.expanduser('~/svdd_model.pkl')
        
        # Training parameters
        self.training_duration = 60.0  # Duration in seconds for training data collection
        
        # Monitoring parameters
        self.anomaly_threshold = 0.0  # Decision threshold (positive = anomaly)
        self.publish_rate = 1.0  # Rate for publishing anomaly detection results (Hz)
