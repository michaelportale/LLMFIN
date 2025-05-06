import numpy as np
import pandas as pd
import os
from sklearn.ensemble import VotingRegressor, StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from keras.models import load_model
from stable_baselines3 import PPO, A2C, SAC, TD3
import joblib
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEnsemble:
    """
    Ensemble class for combining multiple models to improve prediction accuracy
    and robustness. Supports voting, stacking, and weighted ensemble methods.
    """
    
    def __init__(self, models=None, weights=None, method='voting'):
        """
        Initialize the ensemble model.
        
        Args:
            models (list): List of trained model objects or model paths
            weights (list): List of weights for each model (for weighted ensemble)
            method (str): Ensemble method - 'voting', 'stacking', or 'weighted'
        """
        self.models = models or []
        self.weights = weights or []
        self.method = method
        self.ensemble_model = None
        self.model_types = []
        self.trained = False
    
    def add_model(self, model, weight=1.0, model_type=None):
        """
        Add a model to the ensemble.
        
        Args:
            model: Trained model object or path to saved model
            weight (float): Weight for this model in weighted ensemble
            model_type (str): Type of model ('ppo', 'a2c', 'lstm', etc.)
        """
        self.models.append(model)
        self.weights.append(weight)
        self.model_types.append(model_type)
        self.trained = False
    
    def load_models(self, model_paths, model_types):
        """
        Load models from saved files.
        
        Args:
            model_paths (list): List of paths to saved models
            model_types (list): List of model types corresponding to paths
        """
        self.models = []
        self.model_types = model_types
        
        for path, model_type in zip(model_paths, model_types):
            if model_type in ['ppo', 'a2c', 'sac', 'td3']:
                if model_type == 'ppo':
                    model = PPO.load(path)
                elif model_type == 'a2c':
                    model = A2C.load(path)
                elif model_type == 'sac':
                    model = SAC.load(path)
                elif model_type == 'td3':
                    model = TD3.load(path)
            elif model_type == 'lstm':
                model = load_model(path)
            elif model_type.startswith('sklearn'):
                model = joblib.load(path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.models.append(model)
            
    def build_ensemble(self):
        """
        Build the ensemble model based on the specified method.
        """
        if len(self.models) == 0:
            raise ValueError("No models added to ensemble")
            
        if self.method == 'voting':
            # Create estimators for VotingRegressor
            estimators = [(f"model_{i}", model) 
                         for i, model in enumerate(self.models) 
                         if "sklearn" in self.model_types[i]]
            
            if estimators:
                self.ensemble_model = VotingRegressor(estimators)
            else:
                # If no sklearn models, use weighted ensemble method
                self.method = 'weighted'
                print("No sklearn models provided for voting. Using weighted ensemble instead.")
                
        elif self.method == 'stacking':
            # Create estimators for StackingRegressor
            estimators = [(f"model_{i}", model) 
                         for i, model in enumerate(self.models) 
                         if "sklearn" in self.model_types[i]]
            
            if estimators:
                # Use linear regression as final estimator
                self.ensemble_model = StackingRegressor(
                    estimators=estimators,
                    final_estimator=LinearRegression()
                )
            else:
                # If no sklearn models, use weighted ensemble method
                self.method = 'weighted'
                print("No sklearn models provided for stacking. Using weighted ensemble instead.")
        
        # For weighted method, no need to build a specific model - we'll use 
        # the predict method for weighted averaging
        
        self.trained = False
    
    def fit(self, X, y):
        """
        Train the ensemble model.
        
        Args:
            X: Training features
            y: Target values
        """
        # Check if ensemble needs to be built
        if self.method in ['voting', 'stacking'] and self.ensemble_model is None:
            self.build_ensemble()
            
        # Train the ensemble for voting and stacking methods
        if self.method in ['voting', 'stacking']:
            # Only fit sklearn models within the ensemble
            self.ensemble_model.fit(X, y)
            
        # For weighted method, models should already be trained
        self.trained = True
        
    def predict(self, X, uncertainty=False):
        """
        Make predictions using the ensemble.
        
        Args:
            X: Input features for prediction
            uncertainty (bool): Whether to return uncertainty estimates
            
        Returns:
            np.array: Predictions and optionally uncertainty estimates
        """
        if not self.trained and self.method in ['voting', 'stacking']:
            raise ValueError("Ensemble model not trained. Call fit() first.")
        
        if self.method in ['voting', 'stacking']:
            predictions = self.ensemble_model.predict(X)
            if uncertainty:
                # Generate individual model predictions to estimate uncertainty
                individual_preds = np.array([est.predict(X) for _, est in self.ensemble_model.estimators_])
                uncertainty_est = np.std(individual_preds, axis=0)
                return predictions, uncertainty_est
            return predictions
            
        elif self.method == 'weighted':
            # Make predictions with each model and apply weighted average
            all_predictions = []
            
            for i, (model, model_type) in enumerate(zip(self.models, self.model_types)):
                if model_type in ['ppo', 'a2c', 'sac', 'td3']:
                    # RL model prediction
                    obs = self._prepare_rl_observation(X)
                    action, _ = model.predict(obs)
                    all_predictions.append(action)
                elif model_type == 'lstm':
                    # LSTM model prediction
                    pred = model.predict(self._prepare_lstm_input(X))
                    all_predictions.append(pred.flatten())
                elif model_type.startswith('sklearn'):
                    # Sklearn model prediction
                    pred = model.predict(X)
                    all_predictions.append(pred)
            
            # Convert to numpy arrays
            all_predictions = [np.array(pred).flatten() for pred in all_predictions]
            
            # Make sure all predictions have the same shape
            min_length = min(len(p) for p in all_predictions)
            all_predictions = [p[:min_length] for p in all_predictions]
            
            # Apply weights
            normalized_weights = np.array(self.weights) / sum(self.weights)
            weighted_preds = np.zeros(min_length)
            
            for i, pred in enumerate(all_predictions):
                weighted_preds += pred * normalized_weights[i]
            
            if uncertainty:
                # Estimate uncertainty as weighted standard deviation of predictions
                uncertainty_est = np.sqrt(
                    sum(normalized_weights[i] * (pred - weighted_preds)**2 
                        for i, pred in enumerate(all_predictions))
                )
                return weighted_preds, uncertainty_est
            
            return weighted_preds
    
    def _prepare_rl_observation(self, X):
        """
        Prepare observations for RL models.
        """
        # This should be customized based on how your RL models expect input
        return X
    
    def _prepare_lstm_input(self, X):
        """
        Prepare input for LSTM models.
        """
        # This should be customized based on how your LSTM models expect input
        # For example, reshaping to 3D: [samples, time steps, features]
        if len(X.shape) == 2:
            return np.expand_dims(X, axis=1)
        return X
    
    def evaluate(self, X, y_true):
        """
        Evaluate the ensemble model.
        
        Args:
            X: Input features
            y_true: True target values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics
    
    def save(self, path):
        """
        Save the ensemble configuration and model.
        
        Args:
            path (str): Path to save the ensemble
        """
        ensemble_config = {
            'method': self.method,
            'weights': self.weights,
            'model_types': self.model_types,
            'trained': self.trained
        }
        
        # Save ensemble configuration
        joblib.dump(ensemble_config, f"{path}_config.joblib")
        
        # For voting and stacking methods, save the ensemble model
        if self.method in ['voting', 'stacking'] and self.ensemble_model is not None:
            joblib.dump(self.ensemble_model, f"{path}_ensemble.joblib")
    
    @classmethod
    def load(cls, path, model_paths=None):
        """
        Load an ensemble from saved configuration.
        
        Args:
            path (str): Path to the saved ensemble configuration
            model_paths (list): Optional list of paths to individual models
            
        Returns:
            ModelEnsemble: Loaded ensemble model
        """
        # Load ensemble configuration
        ensemble_config = joblib.load(f"{path}_config.joblib")
        
        # Create a new ensemble with the loaded configuration
        ensemble = cls(method=ensemble_config['method'])
        ensemble.weights = ensemble_config['weights']
        ensemble.model_types = ensemble_config['model_types']
        ensemble.trained = ensemble_config['trained']
        
        # Load the ensemble model for voting and stacking methods
        if ensemble.method in ['voting', 'stacking'] and os.path.exists(f"{path}_ensemble.joblib"):
            ensemble.ensemble_model = joblib.load(f"{path}_ensemble.joblib")
        
        # Load individual models if paths provided
        if model_paths:
            ensemble.load_models(model_paths, ensemble.model_types)
        
        return ensemble


def combine_predictions_with_uncertainty(predictions, uncertainties, method='weighted'):
    """
    Combine multiple predictions with their uncertainty estimates.
    
    Args:
        predictions (list): List of prediction arrays
        uncertainties (list): List of uncertainty estimates for each prediction
        method (str): Method to combine predictions ('weighted', 'bayesian')
        
    Returns:
        tuple: (combined_prediction, combined_uncertainty)
    """
    if method == 'weighted':
        # Calculate weights as inverse of uncertainty
        # Higher uncertainty = lower weight
        weights = 1.0 / (np.array(uncertainties) + 1e-8)
        weights = weights / np.sum(weights)
        
        # Apply weights to predictions
        combined_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            combined_pred += pred * weights[i]
        
        # Combined uncertainty (approximation)
        combined_uncertainty = np.sqrt(np.sum([(uncertainties[i] * weights[i])**2 
                                             for i in range(len(weights))]))
        
        return combined_pred, combined_uncertainty
    
    elif method == 'bayesian':
        # Implement Bayesian model averaging
        # Assuming Gaussian distribution of predictions
        precisions = 1.0 / (np.array(uncertainties)**2 + 1e-8)
        weighted_preds = np.array([predictions[i] * precisions[i] 
                                 for i in range(len(predictions))])
        
        # Combined prediction and uncertainty
        combined_precision = np.sum(precisions)
        combined_pred = np.sum(weighted_preds, axis=0) / combined_precision
        combined_uncertainty = np.sqrt(1.0 / combined_precision)
        
        return combined_pred, combined_uncertainty
    
    else:
        raise ValueError(f"Unknown combination method: {method}")


def create_model_snapshots(base_model, save_path, n_snapshots=5, train_kwargs=None):
    """
    Create snapshots of a model during training for ensembling.
    
    Args:
        base_model: Base model to train
        save_path (str): Path to save model snapshots
        n_snapshots (int): Number of snapshots to create
        train_kwargs (dict): Arguments for training the model
        
    Returns:
        list: Paths to model snapshots
    """
    os.makedirs(save_path, exist_ok=True)
    train_kwargs = train_kwargs or {}
    snapshot_paths = []
    
    # Train model and save snapshots at different points
    for i in range(n_snapshots):
        # Clone the base model or create a new one
        if hasattr(base_model, 'clone'):
            model = base_model.clone()
        else:
            # Implement custom cloning if needed
            model = base_model
        
        # Train the model with different seed or data subset
        train_kwargs['seed'] = i + 1  # Different seed for each snapshot
        
        # Training logic depends on model type
        # Should be customized based on your specific models
        
        # Save the snapshot
        snapshot_path = os.path.join(save_path, f"snapshot_{i+1}")
        
        # Save logic depends on model type
        if hasattr(model, 'save'):
            model.save(snapshot_path)
        else:
            joblib.dump(model, f"{snapshot_path}.joblib")
        
        snapshot_paths.append(snapshot_path)
    
    return snapshot_paths 