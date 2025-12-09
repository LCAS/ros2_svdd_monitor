"""
Unit tests for feature extraction module.
"""

import pytest
import numpy as np
import pandas as pd
from ros2_svdd_monitor.features import (
    extract_window_features,
    extract_features_from_dataframe,
    get_feature_names
)


class TestFeatureExtraction:
    """Test suite for feature extraction functions."""

    def test_extract_window_features_basic(self):
        """Test basic feature extraction with simple data."""
        # Create simple cmd_vel data (linear.x=1.0, angular.z=0.5)
        cmd_vel_window = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        ]
        
        # Create simple IMU data (accel.x=2.0, gyro.z=1.0)
        imu_window = [
            [2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
        
        features = extract_window_features(cmd_vel_window, imu_window)
        
        # Check feature vector length
        assert len(features) == 20, "Should have 20 features"
        
        # Check cmd_vel linear.x statistics
        assert features[0] == 1.0, "Mean linear.x should be 1.0"
        assert features[1] == 0.0, "Std linear.x should be 0.0"
        assert features[2] == 1.0, "Min linear.x should be 1.0"
        assert features[3] == 1.0, "Max linear.x should be 1.0"
        
        # Check cmd_vel angular.z statistics
        assert features[4] == 0.5, "Mean angular.z should be 0.5"
        
        # Check IMU accel.x statistics
        assert features[8] == 2.0, "Mean accel.x should be 2.0"

    def test_extract_window_features_empty(self):
        """Test feature extraction with empty windows."""
        cmd_vel_window = []
        imu_window = []
        
        features = extract_window_features(cmd_vel_window, imu_window)
        
        # Should return zeros for all features
        assert len(features) == 20, "Should have 20 features"
        assert np.all(features == 0.0), "All features should be zero for empty windows"

    def test_extract_window_features_varying_data(self):
        """Test feature extraction with varying data."""
        # Create varying cmd_vel data
        cmd_vel_window = [
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.2],
            [1.5, 0.0, 0.0, 0.0, 0.0, 0.4],
        ]
        
        # Create varying IMU data
        imu_window = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.1],
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.2],
        ]
        
        features = extract_window_features(cmd_vel_window, imu_window)
        
        # Check that statistics are computed correctly
        assert features[0] == 1.0, "Mean linear.x should be 1.0"
        assert features[1] > 0.0, "Std linear.x should be > 0"
        assert features[2] == 0.5, "Min linear.x should be 0.5"
        assert features[3] == 1.5, "Max linear.x should be 1.5"

    def test_extract_features_from_dataframe(self):
        """Test feature extraction from a DataFrame."""
        # Create sample DataFrame
        data = {
            'timestamp': [1.0, 2.0, 3.0, 4.0, 5.0],
            'linear_x': [1.0, 1.0, 1.0, 1.0, 1.0],
            'linear_y': [0.0, 0.0, 0.0, 0.0, 0.0],
            'linear_z': [0.0, 0.0, 0.0, 0.0, 0.0],
            'angular_x': [0.0, 0.0, 0.0, 0.0, 0.0],
            'angular_y': [0.0, 0.0, 0.0, 0.0, 0.0],
            'angular_z': [0.5, 0.5, 0.5, 0.5, 0.5],
            'accel_x': [2.0, 2.0, 2.0, 2.0, 2.0],
            'accel_y': [0.0, 0.0, 0.0, 0.0, 0.0],
            'accel_z': [0.0, 0.0, 0.0, 0.0, 0.0],
            'gyro_x': [0.0, 0.0, 0.0, 0.0, 0.0],
            'gyro_y': [0.0, 0.0, 0.0, 0.0, 0.0],
            'gyro_z': [1.0, 1.0, 1.0, 1.0, 1.0],
        }
        df = pd.DataFrame(data)
        
        features = extract_features_from_dataframe(df, window_size=3)
        
        # Check shape
        assert features.shape[0] == 5, "Should have 5 feature vectors"
        assert features.shape[1] == 20, "Should have 20 features per vector"
        
        # Check that features are computed
        assert np.all(np.isfinite(features)), "All features should be finite"

    def test_get_feature_names(self):
        """Test that feature names are returned correctly."""
        feature_names = get_feature_names()
        
        assert len(feature_names) == 20, "Should have 20 feature names"
        assert 'cmd_vel_linear_x_mean' in feature_names
        assert 'imu_accel_x_mean' in feature_names
        assert 'cmd_linear_to_imu_accel_correlation' in feature_names

    def test_cross_features_correlation(self):
        """Test that cross-feature correlations are computed."""
        # Create perfectly correlated data
        cmd_vel_window = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 0.0, 0.0, 1.5],
        ]
        
        imu_window = [
            [2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [4.0, 0.0, 0.0, 0.0, 0.0, 2.0],
            [6.0, 0.0, 0.0, 0.0, 0.0, 3.0],
        ]
        
        features = extract_window_features(cmd_vel_window, imu_window)
        
        # Check correlation features (indices 16 and 17)
        # Should be close to 1.0 for perfectly correlated data
        assert features[16] > 0.99, "Linear correlation should be high"
        assert features[17] > 0.99, "Angular correlation should be high"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
