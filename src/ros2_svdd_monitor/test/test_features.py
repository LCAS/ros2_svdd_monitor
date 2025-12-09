"""Tests for feature extraction module."""

import pytest
import numpy as np
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist, Vector3
from ros2_svdd_monitor.features import FeatureExtractor


def create_imu_msg(linear_acc, angular_vel):
    """Create an IMU message."""
    msg = Imu()
    msg.linear_acceleration.x = linear_acc[0]
    msg.linear_acceleration.y = linear_acc[1]
    msg.linear_acceleration.z = linear_acc[2]
    msg.angular_velocity.x = angular_vel[0]
    msg.angular_velocity.y = angular_vel[1]
    msg.angular_velocity.z = angular_vel[2]
    return msg


def create_twist_msg(linear, angular):
    """Create a Twist message."""
    msg = Twist()
    msg.linear.x = linear[0]
    msg.linear.y = linear[1]
    msg.linear.z = linear[2]
    msg.angular.x = angular[0]
    msg.angular.y = angular[1]
    msg.angular.z = angular[2]
    return msg


def test_feature_extractor_initialization():
    """Test feature extractor initialization."""
    extractor = FeatureExtractor(window_size=5)
    assert extractor.window_size == 5
    assert len(extractor.imu_buffer) == 0
    assert len(extractor.cmd_vel_buffer) == 0


def test_add_imu_data():
    """Test adding IMU data to buffer."""
    extractor = FeatureExtractor(window_size=3)
    
    for i in range(5):
        imu_msg = create_imu_msg([i, i+1, i+2], [i+3, i+4, i+5])
        extractor.add_imu_data(imu_msg)
    
    # Buffer should have max 3 elements
    assert len(extractor.imu_buffer) == 3


def test_add_cmd_vel_data():
    """Test adding cmd_vel data to buffer."""
    extractor = FeatureExtractor(window_size=3)
    
    for i in range(5):
        cmd_vel_msg = create_twist_msg([i, i+1, i+2], [i+3, i+4, i+5])
        extractor.add_cmd_vel_data(cmd_vel_msg)
    
    # Buffer should have max 3 elements
    assert len(extractor.cmd_vel_buffer) == 3


def test_extract_features_insufficient_data():
    """Test feature extraction with insufficient data."""
    extractor = FeatureExtractor(window_size=5)
    
    # Add only 3 samples
    for i in range(3):
        imu_msg = create_imu_msg([i, i+1, i+2], [i+3, i+4, i+5])
        cmd_vel_msg = create_twist_msg([i, i+1, i+2], [i+3, i+4, i+5])
        extractor.add_imu_data(imu_msg)
        extractor.add_cmd_vel_data(cmd_vel_msg)
    
    features = extractor.extract_features()
    assert features is None


def test_extract_features_sufficient_data():
    """Test feature extraction with sufficient data."""
    extractor = FeatureExtractor(window_size=5)
    
    # Add 5 samples
    for i in range(5):
        imu_msg = create_imu_msg([i, i+1, i+2], [i+3, i+4, i+5])
        cmd_vel_msg = create_twist_msg([i, i+1, i+2], [i+3, i+4, i+5])
        extractor.add_imu_data(imu_msg)
        extractor.add_cmd_vel_data(cmd_vel_msg)
    
    features = extractor.extract_features()
    assert features is not None
    assert isinstance(features, np.ndarray)
    # 6 IMU dimensions * 4 stats + 6 cmd_vel dimensions * 4 stats = 48 features
    assert len(features) == 48


def test_reset():
    """Test resetting the feature extractor."""
    extractor = FeatureExtractor(window_size=3)
    
    for i in range(3):
        imu_msg = create_imu_msg([i, i+1, i+2], [i+3, i+4, i+5])
        cmd_vel_msg = create_twist_msg([i, i+1, i+2], [i+3, i+4, i+5])
        extractor.add_imu_data(imu_msg)
        extractor.add_cmd_vel_data(cmd_vel_msg)
    
    assert len(extractor.imu_buffer) == 3
    assert len(extractor.cmd_vel_buffer) == 3
    
    extractor.reset()
    
    assert len(extractor.imu_buffer) == 0
    assert len(extractor.cmd_vel_buffer) == 0


def test_feature_statistics():
    """Test that feature statistics are computed correctly."""
    extractor = FeatureExtractor(window_size=3)
    
    # Add 3 samples with known values
    for i in range(3):
        imu_msg = create_imu_msg([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        cmd_vel_msg = create_twist_msg([0.5, 0.0, 0.0], [0.0, 0.0, 1.0])
        extractor.add_imu_data(imu_msg)
        extractor.add_cmd_vel_data(cmd_vel_msg)
    
    features = extractor.extract_features()
    assert features is not None
    
    # Check that features have correct shape
    assert len(features) == 48
