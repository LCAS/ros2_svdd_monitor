"""
Unit tests for SVDD model wrapper.
"""

import pytest
import numpy as np
import tempfile
import os
from ros2_svdd_monitor.svdd_model import SVDDModel


class TestSVDDModel:
    """Test suite for SVDD model wrapper."""

    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = SVDDModel()
        
        assert model.nu == 0.1
        assert model.gamma == 'scale'
        assert model.kernel == 'rbf'
        assert not model.is_fitted

    def test_model_initialization_custom(self):
        """Test model initialization with custom parameters."""
        model = SVDDModel(nu=0.2, gamma=0.5, kernel='rbf')
        
        assert model.nu == 0.2
        assert model.gamma == 0.5
        assert model.kernel == 'rbf'

    def test_model_fit(self):
        """Test model fitting."""
        # Generate simple training data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        
        model = SVDDModel(nu=0.1)
        model.fit(X_train, scale=True)
        
        assert model.is_fitted
        assert model.scaler is not None

    def test_model_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        model = SVDDModel()
        X_test = np.random.randn(10, 5)
        
        with pytest.raises(RuntimeError):
            model.predict(X_test)

    def test_model_predict(self):
        """Test model prediction."""
        # Generate training data (normal operation)
        np.random.seed(42)
        X_train = np.random.randn(100, 5) * 0.5  # Small variance
        
        # Train model
        model = SVDDModel(nu=0.1)
        model.fit(X_train, scale=True)
        
        # Test on normal data
        X_test_normal = np.random.randn(10, 5) * 0.5
        predictions = model.predict(X_test_normal, scale=True)
        
        assert len(predictions) == 10
        assert all(p in [-1, 1] for p in predictions)

    def test_model_decision_function(self):
        """Test model decision function."""
        # Generate training data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        
        # Train model
        model = SVDDModel(nu=0.1)
        model.fit(X_train, scale=True)
        
        # Test decision function
        X_test = np.random.randn(10, 5)
        scores = model.decision_function(X_test, scale=True)
        
        assert len(scores) == 10
        assert all(np.isfinite(s) for s in scores)

    def test_model_save_load(self):
        """Test model save and load functionality."""
        # Generate training data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        
        # Train model
        model = SVDDModel(nu=0.15, gamma=0.5)
        model.fit(X_train, scale=True)
        
        # Get prediction before saving
        X_test = np.random.randn(10, 5)
        predictions_before = model.predict(X_test, scale=True)
        scores_before = model.decision_function(X_test, scale=True)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            scaler_path = os.path.join(tmpdir, 'test_scaler.pkl')
            
            model.save(model_path, scaler_path)
            
            # Load model
            model_loaded = SVDDModel()
            model_loaded.load(model_path, scaler_path)
            
            # Get prediction after loading
            predictions_after = model_loaded.predict(X_test, scale=True)
            scores_after = model_loaded.decision_function(X_test, scale=True)
            
            # Check that predictions are the same
            assert np.array_equal(predictions_before, predictions_after)
            assert np.allclose(scores_before, scores_after)
            
            # Check that model is marked as fitted
            assert model_loaded.is_fitted

    def test_model_save_before_fit(self):
        """Test that saving fails before fitting."""
        model = SVDDModel()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            
            with pytest.raises(RuntimeError):
                model.save(model_path)

    def test_model_get_params(self):
        """Test getting model parameters."""
        model = SVDDModel(nu=0.2, gamma=0.5)
        params = model.get_params()
        
        assert params['nu'] == 0.2
        assert params['gamma'] == 0.5
        assert params['kernel'] == 'rbf'
        assert 'is_fitted' in params

    def test_model_outlier_detection(self):
        """Test that model can detect outliers."""
        # Generate normal data (clustered around origin)
        np.random.seed(42)
        X_train = np.random.randn(100, 5) * 0.5
        
        # Train model
        model = SVDDModel(nu=0.1)
        model.fit(X_train, scale=True)
        
        # Test on normal data
        X_test_normal = np.random.randn(10, 5) * 0.5
        predictions_normal = model.predict(X_test_normal, scale=True)
        
        # Test on outlier data (far from origin)
        X_test_outlier = np.random.randn(10, 5) * 10.0
        predictions_outlier = model.predict(X_test_outlier, scale=True)
        
        # Check that more outliers are detected in the outlier set
        normal_outliers = np.sum(predictions_normal == -1)
        outlier_outliers = np.sum(predictions_outlier == -1)
        
        # We expect more outliers in the outlier set
        # (This is a probabilistic test, so we use a lenient threshold)
        assert outlier_outliers >= normal_outliers

    def test_model_no_scaling(self):
        """Test model without feature scaling."""
        # Generate training data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        
        # Train model without scaling
        model = SVDDModel(nu=0.1)
        model.fit(X_train, scale=False)
        
        # Predict without scaling
        X_test = np.random.randn(10, 5)
        predictions = model.predict(X_test, scale=False)
        
        assert len(predictions) == 10
        assert model.is_fitted


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
