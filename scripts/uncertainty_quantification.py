import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout, Input, Lambda
from keras.models import Model
import matplotlib.pyplot as plt
from scipy.stats import norm
import joblib
from sklearn.metrics import mean_squared_error
import os

class BayesianDropoutModel:
    """
    Neural network model with Bayesian dropout for uncertainty estimation.
    This approach uses MC Dropout for approximate Bayesian inference.
    """
    
    def __init__(self, input_dim, hidden_units=None, dropout_rate=0.1):
        """
        Initialize the Bayesian dropout model.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_units (list): List of hidden layer units
            dropout_rate (float): Dropout rate for uncertainty estimation
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units or [64, 32]
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self):
        """
        Build the Bayesian neural network model with dropout layers.
        
        Returns:
            tf.keras.Model: Compiled model
        """
        inputs = Input(shape=(self.input_dim,))
        
        # Build hidden layers with dropout
        x = inputs
        for units in self.hidden_units:
            x = Dense(units, activation='relu')(x)
            # Apply dropout in both training and inference
            x = Dropout(self.dropout_rate)(x, training=True)
        
        # Output layer
        outputs = Dense(1)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        
        self.model = model
        return model
    
    def fit(self, X, y, **kwargs):
        """
        Train the model.
        
        Args:
            X: Features
            y: Target values
            **kwargs: Additional arguments for model.fit
        """
        if self.model is None:
            self.build_model()
            
        return self.model.fit(X, y, **kwargs)
    
    def predict_with_uncertainty(self, X, n_samples=100):
        """
        Make predictions with uncertainty estimates using MC Dropout.
        
        Args:
            X: Features
            n_samples (int): Number of Monte Carlo samples
            
        Returns:
            tuple: (mean_prediction, uncertainty)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Make multiple predictions with dropout enabled
        predictions = []
        for _ in range(n_samples):
            preds = self.model.predict(X)
            predictions.append(preds)
            
        # Convert to numpy array
        predictions = np.array(predictions)
        
        # Calculate mean and standard deviation
        mean_prediction = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_prediction, uncertainty
    
    def save(self, path):
        """
        Save the model.
        
        Args:
            path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model architecture and weights
        self.model.save(path)
        
        # Save model hyperparameters
        model_params = {
            'input_dim': self.input_dim,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate
        }
        
        joblib.dump(model_params, f"{path}_params.joblib")
    
    @classmethod
    def load(cls, path):
        """
        Load a saved model.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            BayesianDropoutModel: Loaded model
        """
        # Load model parameters
        model_params = joblib.load(f"{path}_params.joblib")
        
        # Create model with saved parameters
        model = cls(
            input_dim=model_params['input_dim'],
            hidden_units=model_params['hidden_units'],
            dropout_rate=model_params['dropout_rate']
        )
        
        # Load model weights
        model.model = tf.keras.models.load_model(path)
        
        return model


class BootstrapEnsembleModel:
    """
    Bootstrap ensemble model for uncertainty estimation.
    Trains multiple models on bootstrapped samples of the data.
    """
    
    def __init__(self, base_model_class, n_models=5, **model_kwargs):
        """
        Initialize the bootstrap ensemble model.
        
        Args:
            base_model_class: Class of base model to use
            n_models (int): Number of models in the ensemble
            **model_kwargs: Arguments for the base model
        """
        self.base_model_class = base_model_class
        self.n_models = n_models
        self.model_kwargs = model_kwargs
        self.models = []
        
    def fit(self, X, y, **fit_kwargs):
        """
        Train the ensemble on bootstrapped samples.
        
        Args:
            X: Features
            y: Target values
            **fit_kwargs: Additional arguments for model.fit
        """
        n_samples = len(X)
        
        for i in range(self.n_models):
            print(f"Training model {i+1}/{self.n_models}")
            
            # Create bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # Create and train model
            model = self.base_model_class(**self.model_kwargs)
            
            # Check if model has build_model method
            if hasattr(model, 'build_model'):
                model.build_model()
                
            model.fit(X_bootstrap, y_bootstrap, **fit_kwargs)
            self.models.append(model)
    
    def predict_with_uncertainty(self, X):
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Features
            
        Returns:
            tuple: (mean_prediction, uncertainty)
        """
        if not self.models:
            raise ValueError("Models not trained. Call fit() first.")
            
        # Check if models have predict_with_uncertainty method
        if hasattr(self.models[0], 'predict_with_uncertainty'):
            # For models that support uncertainty estimation
            all_means = []
            all_uncertainties = []
            
            for model in self.models:
                mean, uncertainty = model.predict_with_uncertainty(X)
                all_means.append(mean)
                all_uncertainties.append(uncertainty)
                
            # Combine predictions
            all_means = np.array(all_means)
            mean_prediction = np.mean(all_means, axis=0)
            
            # Total uncertainty: model uncertainty + data uncertainty
            model_uncertainty = np.std(all_means, axis=0)
            data_uncertainty = np.sqrt(np.mean(np.array(all_uncertainties)**2, axis=0))
            
            total_uncertainty = np.sqrt(model_uncertainty**2 + data_uncertainty**2)
            
            return mean_prediction, total_uncertainty
        else:
            # For models that don't support uncertainty estimation
            predictions = []
            
            for model in self.models:
                if hasattr(model, 'predict'):
                    preds = model.predict(X)
                else:
                    # Assume model is a scikit-learn model
                    preds = model.predict(X)
                    
                predictions.append(preds)
                
            # Convert to numpy array and ensure consistent shape
            predictions = np.array([p.flatten() for p in predictions])
            
            # Calculate mean and standard deviation
            mean_prediction = np.mean(predictions, axis=0)
            uncertainty = np.std(predictions, axis=0)
            
            return mean_prediction, uncertainty
    
    def save(self, path_prefix):
        """
        Save all models in the ensemble.
        
        Args:
            path_prefix (str): Prefix for model save paths
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        
        model_paths = []
        
        # Save each model
        for i, model in enumerate(self.models):
            model_path = f"{path_prefix}_model_{i}"
            
            if hasattr(model, 'save'):
                model.save(model_path)
            else:
                # Assume model is a scikit-learn model
                joblib.dump(model, f"{model_path}.joblib")
                
            model_paths.append(model_path)
            
        # Save ensemble metadata
        ensemble_data = {
            'base_model_class': self.base_model_class.__name__,
            'n_models': self.n_models,
            'model_kwargs': self.model_kwargs,
            'model_paths': model_paths
        }
        
        joblib.dump(ensemble_data, f"{path_prefix}_ensemble.joblib")
    
    @classmethod
    def load(cls, path_prefix, base_model_class):
        """
        Load a saved ensemble.
        
        Args:
            path_prefix (str): Prefix for model save paths
            base_model_class: Class of base model
            
        Returns:
            BootstrapEnsembleModel: Loaded ensemble
        """
        # Load ensemble metadata
        ensemble_data = joblib.load(f"{path_prefix}_ensemble.joblib")
        
        # Create ensemble
        ensemble = cls(
            base_model_class=base_model_class,
            n_models=ensemble_data['n_models'],
            **ensemble_data['model_kwargs']
        )
        
        # Load each model
        for model_path in ensemble_data['model_paths']:
            if hasattr(base_model_class, 'load'):
                model = base_model_class.load(model_path)
            else:
                # Assume model is a scikit-learn model
                model = joblib.load(f"{model_path}.joblib")
                
            ensemble.models.append(model)
            
        return ensemble


class QuantileRegressionModel:
    """
    Model for quantile regression to estimate prediction intervals.
    """
    
    def __init__(self, input_dim, hidden_units=None, quantiles=[0.1, 0.5, 0.9]):
        """
        Initialize the quantile regression model.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_units (list): List of hidden layer units
            quantiles (list): List of quantiles to predict
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units or [64, 32]
        self.quantiles = quantiles
        self.model = None
        
    def quantile_loss(self, q, y_true, y_pred):
        """
        Quantile loss function.
        
        Args:
            q (float): Quantile value
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            tf.Tensor: Quantile loss
        """
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
    
    def build_model(self):
        """
        Build the quantile regression model.
        
        Returns:
            tf.keras.Model: Compiled model
        """
        inputs = Input(shape=(self.input_dim,))
        
        # Build shared layers
        x = inputs
        for units in self.hidden_units:
            x = Dense(units, activation='relu')(x)
        
        # Separate output for each quantile
        outputs = []
        for q in self.quantiles:
            output = Dense(1, name=f'quantile_{q}')(x)
            outputs.append(output)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model with quantile loss
        losses = {f'quantile_{q}': lambda y_true, y_pred: self.quantile_loss(q, y_true, y_pred) 
                 for q in self.quantiles}
        
        model.compile(optimizer='adam', loss=losses)
        
        self.model = model
        return model
    
    def fit(self, X, y, **kwargs):
        """
        Train the model.
        
        Args:
            X: Features
            y: Target values
            **kwargs: Additional arguments for model.fit
        """
        if self.model is None:
            self.build_model()
            
        # Prepare target data for each quantile output
        y_dict = {f'quantile_{q}': y for q in self.quantiles}
        
        return self.model.fit(X, y_dict, **kwargs)
    
    def predict(self, X):
        """
        Make predictions for all quantiles.
        
        Args:
            X: Features
            
        Returns:
            dict: Predictions for each quantile
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        predictions = self.model.predict(X)
        
        # Organize predictions by quantile
        result = {}
        for i, q in enumerate(self.quantiles):
            result[q] = predictions[i]
            
        return result
    
    def predict_with_uncertainty(self, X):
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Features
            
        Returns:
            tuple: (mean_prediction, lower_bound, upper_bound)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        predictions = self.predict(X)
        
        # Use median as mean prediction
        if 0.5 in predictions:
            mean_prediction = predictions[0.5]
        else:
            # If no median quantile, use mean of available quantiles
            mean_prediction = np.mean([predictions[q] for q in predictions], axis=0)
            
        # Get lower and upper bounds
        lower_bound = predictions[min(self.quantiles)]
        upper_bound = predictions[max(self.quantiles)]
        
        # Calculate uncertainty as half the prediction interval width
        uncertainty = (upper_bound - lower_bound) / 2
        
        return mean_prediction, uncertainty
    
    def save(self, path):
        """
        Save the model.
        
        Args:
            path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model architecture and weights
        self.model.save(path)
        
        # Save model hyperparameters
        model_params = {
            'input_dim': self.input_dim,
            'hidden_units': self.hidden_units,
            'quantiles': self.quantiles
        }
        
        joblib.dump(model_params, f"{path}_params.joblib")
    
    @classmethod
    def load(cls, path):
        """
        Load a saved model.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            QuantileRegressionModel: Loaded model
        """
        # Load model parameters
        model_params = joblib.load(f"{path}_params.joblib")
        
        # Create model with saved parameters
        model = cls(
            input_dim=model_params['input_dim'],
            hidden_units=model_params['hidden_units'],
            quantiles=model_params['quantiles']
        )
        
        # Custom objects for loading the model with the quantile loss
        custom_objects = {}
        for q in model_params['quantiles']:
            custom_objects[f'quantile_{q}_loss'] = lambda y_true, y_pred, q=q: model.quantile_loss(q, y_true, y_pred)
        
        # Load model weights
        model.model = tf.keras.models.load_model(
            path, 
            custom_objects=custom_objects
        )
        
        return model


def plot_prediction_with_uncertainty(x, y_true, y_pred, uncertainty, title=None, save_path=None):
    """
    Plot predictions with uncertainty intervals.
    
    Args:
        x: x-values for plotting
        y_true: True values
        y_pred: Predicted values
        uncertainty: Uncertainty estimates
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot true values
    plt.plot(x, y_true, 'k-', label='True Values')
    
    # Plot predicted values with uncertainty intervals
    plt.plot(x, y_pred, 'b-', label='Predictions')
    plt.fill_between(x, y_pred - 2*uncertainty, y_pred + 2*uncertainty, 
                    color='blue', alpha=0.2, label='95% Confidence Interval')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title or 'Predictions with Uncertainty')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def evaluate_uncertainty_calibration(y_true, y_pred, uncertainty):
    """
    Evaluate the calibration of uncertainty estimates.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        uncertainty: Uncertainty estimates
        
    Returns:
        dict: Calibration metrics
    """
    # Calculate standardized errors
    z_scores = (y_true - y_pred) / uncertainty
    
    # Calculate proportion of true values within different confidence intervals
    confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
    observed_proportions = {}
    
    for level in confidence_levels:
        z_critical = norm.ppf(0.5 + level / 2)
        within_interval = np.abs(z_scores) < z_critical
        observed_proportions[level] = np.mean(within_interval)
    
    # Calculate calibration error: difference between expected and observed proportions
    calibration_error = np.mean([np.abs(observed_proportions[level] - level) 
                              for level in confidence_levels])
    
    # Calculate sharpness: average width of prediction intervals
    sharpness = np.mean(uncertainty)
    
    return {
        'observed_proportions': observed_proportions,
        'expected_proportions': confidence_levels,
        'calibration_error': calibration_error,
        'sharpness': sharpness,
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }


def plot_calibration_curve(y_true, y_pred, uncertainty, save_path=None):
    """
    Plot calibration curve for uncertainty estimates.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        uncertainty: Uncertainty estimates
        save_path (str): Path to save the plot
    """
    # Calculate standardized errors
    z_scores = (y_true - y_pred) / uncertainty
    
    # Create confidence levels
    confidence_levels = np.linspace(0.01, 0.99, 20)
    observed_proportions = []
    
    for level in confidence_levels:
        z_critical = norm.ppf(0.5 + level / 2)
        within_interval = np.abs(z_scores) < z_critical
        observed_proportions.append(np.mean(within_interval))
    
    # Plot calibration curve
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(confidence_levels, observed_proportions, 'bo-', label='Model Calibration')
    
    plt.xlabel('Expected Proportion')
    plt.ylabel('Observed Proportion')
    plt.title('Uncertainty Calibration Curve')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 