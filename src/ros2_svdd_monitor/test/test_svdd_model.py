"""Tests for SVDD model module."""

import pytest
import numpy as np
import tempfile
import os
from ros2_svdd_monitor.svdd_model import SVDDModel


def test_svdd_model_initialization():
    """Test SVDD model initialization."""
    model = SVDDModel(nu=0.1, kernel='rbf', gamma='auto')
    assert model.nu == 0.1
    assert model.kernel == 'rbf'
    assert model.gamma == 'auto'
    assert not model.is_trained


def test_train_with_valid_data():
    """Test training with valid data."""
    model = SVDDModel()
    
    # Generate some training data
    np.random.seed(42)
    features = np.random.randn(100, 10)
    
    success = model.train(features)
    assert success
    assert model.is_trained


def test_train_with_empty_data():
    """Test training with empty data."""
    model = SVDDModel()
    
    features = np.array([])
    success = model.train(features)
    assert not success
    assert not model.is_trained


def test_train_with_none():
    """Test training with None data."""
    model = SVDDModel()
    
    success = model.train(None)
    assert not success
    assert not model.is_trained


def test_predict_before_training():
    """Test prediction before training raises error."""
    model = SVDDModel()
    
    features = np.random.randn(1, 10)
    
    with pytest.raises(RuntimeError):
        model.predict(features)


def test_predict_after_training():
    """Test prediction after training."""
    model = SVDDModel()
    
    # Train the model
    np.random.seed(42)
    train_features = np.random.randn(100, 10)
    model.train(train_features)
    
    # Test prediction with single sample
    test_features = np.random.randn(10)
    predictions, scores = model.predict(test_features)
    
    assert len(predictions) == 1
    assert len(scores) == 1
    assert predictions[0] in [-1, 1]
    
    # Test prediction with multiple samples
    test_features_multi = np.random.randn(5, 10)
    predictions, scores = model.predict(test_features_multi)
    
    assert len(predictions) == 5
    assert len(scores) == 5


def test_save_and_load_model():
    """Test saving and loading model."""
    model = SVDDModel()
    
    # Train the model
    np.random.seed(42)
    train_features = np.random.randn(100, 10)
    model.train(train_features)
    
    # Save the model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        tmp_path = tmp.name
    
    try:
        success = model.save(tmp_path)
        assert success
        assert os.path.exists(tmp_path)
        
        # Load the model
        new_model = SVDDModel()
        success = new_model.load(tmp_path)
        assert success
        assert new_model.is_trained
        assert new_model.nu == model.nu
        assert new_model.kernel == model.kernel
        
        # Test prediction with loaded model
        test_features = np.random.randn(10)
        predictions, scores = new_model.predict(test_features)
        assert len(predictions) == 1
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_load_nonexistent_file():
    """Test loading from nonexistent file."""
    model = SVDDModel()
    success = model.load('/nonexistent/path/model.pkl')
    assert not success
    assert not model.is_trained


def test_model_consistency():
    """Test that model produces consistent results."""
    model = SVDDModel()
    
    # Train the model
    np.random.seed(42)
    train_features = np.random.randn(100, 10)
    model.train(train_features)
    
    # Test prediction consistency
    test_features = np.random.randn(10)
    predictions1, scores1 = model.predict(test_features)
    predictions2, scores2 = model.predict(test_features)
    
    assert np.array_equal(predictions1, predictions2)
    assert np.array_equal(scores1, scores2)
