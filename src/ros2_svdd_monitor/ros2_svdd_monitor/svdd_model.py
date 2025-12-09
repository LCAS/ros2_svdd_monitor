"""SVDD model using scikit-learn's OneClassSVM."""

import pickle
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class SVDDModel:
    """Support Vector Data Description model for anomaly detection."""
    
    def __init__(self, nu=0.1, kernel='rbf', gamma='auto'):
        """
        Initialize SVDD model.
        
        Args:
            nu: Upper bound on fraction of outliers (0 < nu <= 1)
            kernel: Kernel type for SVM
            gamma: Kernel coefficient
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, features):
        """
        Train the SVDD model.
        
        Args:
            features: numpy array of shape (n_samples, n_features)
            
        Returns:
            bool: True if training successful, False otherwise
        """
        if features is None or len(features) == 0:
            return False
            
        try:
            # Normalize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train the model
            self.model.fit(features_scaled)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error training SVDD model: {e}")
            return False
            
    def predict(self, features):
        """
        Predict if features are anomalous.
        
        Args:
            features: numpy array of shape (n_samples, n_features) or (n_features,)
            
        Returns:
            tuple: (predictions, decision_scores)
                predictions: 1 for inliers, -1 for outliers
                decision_scores: signed distance to boundary (negative = outlier)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
            
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Normalize features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        predictions = self.model.predict(features_scaled)
        decision_scores = self.model.decision_function(features_scaled)
        
        return predictions, decision_scores
        
    def save(self, filepath):
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'nu': self.nu,
                'kernel': self.kernel,
                'gamma': self.gamma
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
            
    def load(self, filepath):
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            bool: True if load successful, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.nu = model_data['nu']
            self.kernel = model_data['kernel']
            self.gamma = model_data['gamma']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
