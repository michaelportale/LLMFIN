"""
Deep Learning Models for FinPort

This module implements LSTM and sentiment analysis models for stock prediction.
"""

import numpy as np
import pandas as pd
import os
import json
import pickle
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a directory for saving/loading models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'deep_learning')
os.makedirs(MODEL_DIR, exist_ok=True)

# Create a directory for sentiment data
SENTIMENT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sentiment')
os.makedirs(SENTIMENT_DIR, exist_ok=True)

class LSTMModel:
    """LSTM model for time series forecasting"""
    
    def __init__(self, ticker, layers=2, sequence_length=20, prediction_horizon=1):
        """
        Initialize LSTM model for stock prediction
        
        Args:
            ticker (str): Stock ticker symbol
            layers (int): Number of LSTM layers
            sequence_length (int): Number of time steps to look back
            prediction_horizon (int): Number of days to predict ahead
        """
        self.ticker = ticker
        self.layers = layers
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = None
        self.history = None
        
        # Model parameters
        self.batch_size = 64
        self.epochs = 50
        self.early_stopping = True
        self.patience = 5
        
        # File paths for saving/loading
        self.model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm_l{layers}_s{sequence_length}_h{prediction_horizon}")
        
        logger.info(f"Initialized LSTM model for {ticker} with {layers} layers")
    
    def preprocess_data(self, data):
        """
        Preprocess data for LSTM model
        
        Args:
            data (DataFrame): Stock price data with at least 'Close' column
            
        Returns:
            tuple: Preprocessed X and y data for training
        """
        try:
            # Use price, volume, and technical indicators
            features = ['Close', 'Volume', 'MA_5', 'MA_20', 'RSI', 'MACD']
            available_features = [f for f in features if f in data.columns]
            
            if len(available_features) < 2:
                # Fallback to just using Close prices if other features aren't available
                logger.warning(f"Limited features available for {self.ticker}, using Close prices only")
                X = data[['Close']].values
            else:
                X = data[available_features].values
            
            # Scale the data
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Create sequences
            X_sequences, y = self._create_sequences(X_scaled, self.sequence_length, self.prediction_horizon)
            
            return X_sequences, y
        
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def _create_sequences(self, data, seq_length, horizon):
        """
        Create input sequences and target values
        
        Args:
            data (ndarray): Scaled feature data
            seq_length (int): Sequence length (lookback period)
            horizon (int): Prediction horizon
            
        Returns:
            tuple: X sequences and y targets
        """
        X, y = [], []
        for i in range(len(data) - seq_length - horizon + 1):
            X.append(data[i:(i + seq_length)])
            # Target is the closing price 'horizon' days ahead
            y.append(data[i + seq_length + horizon - 1][0])  # First column is Close price
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        
        Args:
            input_shape (tuple): Shape of input data (seq_length, features)
            
        Returns:
            Model: Compiled Keras model
        """
        try:
            # Import TensorFlow/Keras here to avoid loading it unnecessarily
            import tensorflow as tf
            from keras.models import Sequential
            from keras.layers import LSTM, Dense, Dropout
            from keras.optimizers import Adam
            
            # Set memory growth to avoid GPU memory issues
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                logger.warning(f"Error setting GPU memory growth: {e}")
            
            # Build model
            model = Sequential()
            
            # Add LSTM layers
            if self.layers == 1:
                model.add(LSTM(50, input_shape=input_shape, activation='relu'))
            else:
                model.add(LSTM(50, input_shape=input_shape, activation='relu', return_sequences=True))
                
                # Add intermediate layers
                for _ in range(self.layers - 2):
                    model.add(LSTM(50, activation='relu', return_sequences=True))
                
                # Add final LSTM layer
                model.add(LSTM(50, activation='relu'))
            
            # Add Dense layers
            model.add(Dropout(0.2))
            model.add(Dense(25, activation='relu'))
            model.add(Dense(1))
            
            # Compile model
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            return model
        
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise
    
    def train(self, data, batch_size=None, epochs=None, verbose=1):
        """
        Train the LSTM model
        
        Args:
            data (DataFrame): Stock price data
            batch_size (int, optional): Batch size for training
            epochs (int, optional): Number of epochs to train
            verbose (int, optional): Verbosity level
            
        Returns:
            dict: Training history
        """
        try:
            # Set batch size and epochs if provided
            if batch_size:
                self.batch_size = batch_size
            if epochs:
                self.epochs = epochs
            
            # Preprocess data
            X, y = self.preprocess_data(data)
            
            # Build model
            input_shape = (X.shape[1], X.shape[2])
            self.model = self.build_model(input_shape)
            
            # Create callbacks
            callbacks = []
            
            # Add early stopping if enabled
            if self.early_stopping:
                from keras.callbacks import EarlyStopping
                callbacks.append(EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True))
            
            # Split data into train and validation sets (80/20)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=verbose
            )
            
            self.history = history.history
            
            # Save model
            self.save_model()
            
            return self.history
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, data):
        """
        Make predictions using the trained model
        
        Args:
            data (DataFrame): Stock price data
            
        Returns:
            ndarray: Predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first or load a saved model.")
            
            # Preprocess data
            X, _ = self.preprocess_data(data)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Convert to original scale
            if hasattr(self, 'scaler') and self.scaler is not None:
                # Create a dummy array with zeros except for the prediction value
                dummy = np.zeros((len(predictions), self.scaler.n_features_in_))
                dummy[:, 0] = predictions.flatten()
                
                # Inverse transform
                predictions = self.scaler.inverse_transform(dummy)[:, 0]
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def save_model(self):
        """Save the trained model and metadata"""
        try:
            if self.model is None:
                raise ValueError("No model to save. Train a model first.")
            
            # Save Keras model
            self.model.save(f"{self.model_path}_model")
            
            # Save scaler and metadata
            metadata = {
                'ticker': self.ticker,
                'layers': self.layers,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'history': self.history,
                'created_at': datetime.now().isoformat()
            }
            
            with open(f"{self.model_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            with open(f"{self.model_path}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self):
        """Load the trained model and metadata"""
        try:
            # Import TensorFlow/Keras here
            from keras.models import load_model
            
            # Load Keras model
            model_path = f"{self.model_path}_model"
            if os.path.exists(model_path):
                self.model = load_model(model_path)
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            # Load scaler
            scaler_path = f"{self.model_path}_scaler.pkl"
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            # Load metadata
            metadata_path = f"{self.model_path}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.history = metadata.get('history')
            
            logger.info(f"Model loaded from {self.model_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def evaluate(self, data):
        """
        Evaluate model performance on test data
        
        Args:
            data (DataFrame): Test data
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first or load a saved model.")
            
            # Preprocess data
            X, y_true = self.preprocess_data(data)
            
            # Make predictions
            y_pred = self.model.predict(X).flatten()
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            metrics = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise


class SentimentAnalysisModel:
    """Sentiment analysis model for financial text data"""
    
    def __init__(self, ticker, sources=None):
        """
        Initialize sentiment analysis model
        
        Args:
            ticker (str): Stock ticker symbol
            sources (list, optional): List of source types to use (e.g., 'news', 'twitter')
        """
        self.ticker = ticker
        self.sources = sources or ['news']
        self.sentiment_data = {}
        
        # File paths
        self.data_path = os.path.join(SENTIMENT_DIR, f"{ticker}_sentiment.json")
        
        logger.info(f"Initialized sentiment analysis for {ticker} with sources: {', '.join(self.sources)}")
    
    def fetch_data(self, start_date=None, end_date=None):
        """
        Fetch sentiment data from various sources
        
        Args:
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame: Sentiment data with dates and scores
        """
        # Convert dates
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Initialize data dictionary
        sentiment_data = {
            'date': [],
            'compound': [],
            'positive': [],
            'negative': [],
            'neutral': [],
            'source': []
        }
        
        # In a real implementation, we would fetch data from APIs
        # For demonstration, we'll create mock data
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for source in self.sources:
            for date in date_range:
                date_str = date.strftime('%Y-%m-%d')
                
                # Generate mock sentiment scores
                # In a real implementation, these would come from API calls
                positive = np.random.beta(2, 5) if source == 'twitter' else np.random.beta(5, 5)
                negative = np.random.beta(5, 5) if source == 'twitter' else np.random.beta(2, 5)
                neutral = max(0, 1 - positive - negative)
                
                # Generate compound score between -1 and 1
                compound = positive - negative
                
                sentiment_data['date'].append(date_str)
                sentiment_data['compound'].append(compound)
                sentiment_data['positive'].append(positive)
                sentiment_data['negative'].append(negative)
                sentiment_data['neutral'].append(neutral)
                sentiment_data['source'].append(source)
        
        # Create DataFrame
        df = pd.DataFrame(sentiment_data)
        
        # Save data
        self.sentiment_data = df
        self._save_data()
        
        return df
    
    def _save_data(self):
        """Save sentiment data to file"""
        try:
            self.sentiment_data.to_json(self.data_path, orient='records')
            logger.info(f"Sentiment data saved to {self.data_path}")
        except Exception as e:
            logger.error(f"Error saving sentiment data: {e}")
    
    def _load_data(self):
        """Load sentiment data from file"""
        try:
            if os.path.exists(self.data_path):
                self.sentiment_data = pd.read_json(self.data_path, orient='records')
                logger.info(f"Sentiment data loaded from {self.data_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading sentiment data: {e}")
            return False
    
    def get_daily_sentiment(self, start_date=None, end_date=None):
        """
        Get daily aggregated sentiment scores
        
        Args:
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame: Daily sentiment scores
        """
        # Load data if not already loaded
        if self.sentiment_data.empty and not self._load_data():
            self.fetch_data(start_date, end_date)
        
        # Filter by date if provided
        data = self.sentiment_data.copy()
        if start_date:
            data = data[data['date'] >= start_date]
        if end_date:
            data = data[data['date'] <= end_date]
        
        # Group by date and calculate daily sentiment
        daily = data.groupby('date').agg({
            'compound': 'mean',
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean'
        }).reset_index()
        
        return daily
    
    def analyze_recent_sentiment(self, days=30):
        """
        Analyze recent sentiment trends
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        daily = self.get_daily_sentiment(start_date, end_date)
        
        # Calculate average sentiment
        avg_sentiment = daily['compound'].mean()
        
        # Calculate trend (positive or negative)
        trend = 'neutral'
        if len(daily) > 5:
            recent = daily['compound'].iloc[-5:].mean()
            earlier = daily['compound'].iloc[:-5].mean()
            
            if recent > earlier + 0.1:
                trend = 'improving'
            elif recent < earlier - 0.1:
                trend = 'worsening'
        
        # Calculate volatility
        volatility = daily['compound'].std()
        
        return {
            'ticker': self.ticker,
            'avg_sentiment': float(avg_sentiment),
            'trend': trend,
            'volatility': float(volatility),
            'days_analyzed': days,
            'positive_days': int((daily['compound'] > 0.2).sum()),
            'negative_days': int((daily['compound'] < -0.2).sum()),
            'neutral_days': int(((daily['compound'] >= -0.2) & (daily['compound'] <= 0.2)).sum())
        }
    
    def generate_summary(self):
        """
        Generate a natural language summary of sentiment analysis
        
        Returns:
            str: Summary text
        """
        analysis = self.analyze_recent_sentiment()
        
        # Create summary text
        summary = f"Sentiment Analysis for {self.ticker}:\n\n"
        
        if analysis['avg_sentiment'] > 0.3:
            sentiment_desc = "very positive"
        elif analysis['avg_sentiment'] > 0.1:
            sentiment_desc = "positive"
        elif analysis['avg_sentiment'] < -0.3:
            sentiment_desc = "very negative"
        elif analysis['avg_sentiment'] < -0.1:
            sentiment_desc = "negative"
        else:
            sentiment_desc = "neutral"
        
        summary += f"• Overall sentiment is {sentiment_desc} with an average score of {analysis['avg_sentiment']:.2f}.\n"
        summary += f"• The sentiment trend is {analysis['trend']}.\n"
        summary += f"• Out of {analysis['days_analyzed']} days analyzed, sentiment was positive for {analysis['positive_days']} days, negative for {analysis['negative_days']} days, and neutral for {analysis['neutral_days']} days.\n"
        
        # Add source-specific insights
        if 'news' in self.sources:
            summary += f"• News coverage shows a {sentiment_desc} tone in recent articles.\n"
        if 'twitter' in self.sources:
            summary += f"• Social media sentiment appears to be {sentiment_desc} with moderate volatility.\n"
        if 'reddit' in self.sources:
            summary += f"• Discussion forums show {sentiment_desc} investor sentiment.\n"
        
        return summary


class ExplainableAIModel:
    """Explainable AI wrapper for FinPort models"""
    
    def __init__(self, base_model, model_type='LSTM'):
        """
        Initialize explainable AI wrapper
        
        Args:
            base_model: The underlying model to explain
            model_type (str): Type of the base model
        """
        self.base_model = base_model
        self.model_type = model_type
        self.explanations = {}
        
        logger.info(f"Initialized explainable AI wrapper for {model_type} model")
    
    def explain_prediction(self, data):
        """
        Generate explanation for a model prediction
        
        Args:
            data (DataFrame): Input data
            
        Returns:
            dict: Explanation with feature importances
        """
        try:
            if self.model_type == 'LSTM':
                return self._explain_lstm_prediction(data)
            else:
                return {'error': 'Unsupported model type for explanation'}
        
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {'error': str(e)}
    
    def _explain_lstm_prediction(self, data):
        """
        Explain LSTM prediction using integrated gradients or SHAP
        
        Args:
            data (DataFrame): Input data
            
        Returns:
            dict: Feature importances and explanation
        """
        # This is a simplified example
        # In a real implementation, use SHAP or similar libraries
        
        # Get features
        features = list(data.columns)
        
        # Calculate mock feature importances (random for demonstration)
        importances = np.random.random(len(features))
        importances = importances / importances.sum()
        
        # Create explanation
        explanation = {
            'features': features,
            'importances': importances.tolist(),
            'top_features': [features[i] for i in np.argsort(importances)[-3:][::-1]],
            'explanation_text': self._generate_explanation_text(features, importances)
        }
        
        self.explanations = explanation
        return explanation
    
    def _generate_explanation_text(self, features, importances):
        """Generate natural language explanation"""
        sorted_indices = np.argsort(importances)[::-1]
        top_features = [features[i] for i in sorted_indices[:3]]
        top_importances = [importances[i] for i in sorted_indices[:3]]
        
        explanation = "Prediction explanation:\n\n"
        explanation += f"The model's prediction is most influenced by {top_features[0]} ({top_importances[0]:.1%} importance), "
        explanation += f"followed by {top_features[1]} ({top_importances[1]:.1%}) and {top_features[2]} ({top_importances[2]:.1%}).\n\n"
        
        if 'Close' in top_features:
            explanation += "Recent closing prices have a significant impact on the prediction.\n"
        if 'Volume' in top_features:
            explanation += "Trading volume patterns are influencing the forecast.\n"
        if 'RSI' in top_features:
            explanation += "The Relative Strength Index (RSI) is indicating important momentum patterns.\n"
        if 'MACD' in top_features:
            explanation += "The MACD indicator is signaling potential trend changes.\n"
        
        return explanation
    
    def visualize_explanation(self):
        """
        Generate a visualization of feature importances
        
        Returns:
            dict: Visualization data for rendering charts
        """
        if not self.explanations:
            return {'error': 'No explanations available. Run explain_prediction() first.'}
        
        # Create visualization data
        visualization = {
            'type': 'bar_chart',
            'data': {
                'labels': self.explanations['features'],
                'values': self.explanations['importances'],
                'colors': self._generate_colors(len(self.explanations['features']))
            },
            'title': 'Feature Importance in Prediction',
            'x_label': 'Importance (%)',
            'y_label': 'Features'
        }
        
        return visualization
    
    def _generate_colors(self, n):
        """Generate colors for visualization"""
        import matplotlib.cm as cm
        
        colors = cm.viridis(np.linspace(0, 1, n))
        return [f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})" 
                for r, g, b, a in colors] 