"""Tests for configuration module."""

import pytest
from ros2_svdd_monitor.config import SVDDConfig


def test_config_initialization():
    """Test configuration initialization."""
    config = SVDDConfig()
    
    # Test topic names
    assert config.imu_topic == '/imu'
    assert config.cmd_vel_topic == '/cmd_vel'
    
    # Test model parameters
    assert config.nu == 0.1
    assert config.kernel == 'rbf'
    assert config.gamma == 'auto'
    
    # Test feature parameters
    assert config.window_size == 10
    assert config.feature_buffer_size == 100
    
    # Test paths
    assert config.model_path is not None
    assert 'svdd_model.pkl' in config.model_path
    
    # Test training parameters
    assert config.training_duration == 60.0
    
    # Test monitoring parameters
    assert config.anomaly_threshold == 0.0
    assert config.publish_rate == 1.0


def test_config_model_path():
    """Test that model path is properly expanded."""
    config = SVDDConfig()
    
    # Path should be expanded (no ~ in path)
    assert '~' not in config.model_path
