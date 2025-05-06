"""
Custom Metrics Module

This module handles the definition, calculation, and tracking of custom metrics
for model evaluation and comparison.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Define global constants
METRICS_DIR = os.path.join("static", "metrics", "custom")
os.makedirs(METRICS_DIR, exist_ok=True)

# Initialize logging
logger = logging.getLogger("custom_metrics")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Standard metrics functions
def calculate_annualized_return(returns, days=252):
    """Calculate annualized return from a series of returns"""
    total_return = returns[-1]
    n_days = len(returns)
    years = n_days / days
    return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

def calculate_volatility(returns, days=252):
    """Calculate annualized volatility"""
    if len(returns) <= 1:
        return 0
    return np.std(returns) * np.sqrt(days)

def calculate_sharpe_ratio(returns, risk_free_rate=0, days=252):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate
    volatility = calculate_volatility(returns, days)
    return excess_returns.mean() / volatility if volatility > 0 else 0

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative_returns = (1 + returns).cumprod() - 1
    peaks = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - peaks) / (1 + peaks)
    return drawdowns.min()

def calculate_sortino_ratio(returns, risk_free_rate=0, days=252):
    """Calculate Sortino ratio"""
    excess_returns = returns - risk_free_rate
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns) * np.sqrt(days) if len(downside_returns) > 0 else 1e-6
    return excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0

def calculate_calmar_ratio(returns, days=252):
    """Calculate Calmar ratio (annual return / max drawdown)"""
    ann_return = calculate_annualized_return(returns, days)
    max_dd = abs(calculate_max_drawdown(returns))
    return ann_return / max_dd if max_dd > 0 else 0

def calculate_win_rate(trades):
    """Calculate win rate from trades list"""
    if not trades:
        return 0
    profitable_trades = sum(1 for trade in trades if trade.get("profit", 0) > 0)
    return profitable_trades / len(trades)

def calculate_profit_ratio(trades):
    """Calculate profit ratio (average win / average loss)"""
    if not trades:
        return 0
    
    wins = [trade.get("profit", 0) for trade in trades if trade.get("profit", 0) > 0]
    losses = [abs(trade.get("profit", 0)) for trade in trades if trade.get("profit", 0) < 0]
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 1
    
    return avg_win / avg_loss if avg_loss > 0 else 0

# Custom metric registry
custom_metrics = {
    "annualized_return": {
        "name": "Annualized Return",
        "description": "The return expressed as an annual percentage",
        "function": calculate_annualized_return,
        "higher_is_better": True,
        "format": "percentage"
    },
    "volatility": {
        "name": "Volatility",
        "description": "The annualized standard deviation of returns",
        "function": calculate_volatility,
        "higher_is_better": False,
        "format": "percentage"
    },
    "sharpe_ratio": {
        "name": "Sharpe Ratio",
        "description": "Excess return per unit of risk",
        "function": calculate_sharpe_ratio,
        "higher_is_better": True,
        "format": "decimal"
    },
    "sortino_ratio": {
        "name": "Sortino Ratio",
        "description": "Return per unit of downside risk",
        "function": calculate_sortino_ratio,
        "higher_is_better": True,
        "format": "decimal"
    },
    "max_drawdown": {
        "name": "Maximum Drawdown",
        "description": "The maximum peak-to-trough decline",
        "function": calculate_max_drawdown,
        "higher_is_better": False,
        "format": "percentage"
    },
    "calmar_ratio": {
        "name": "Calmar Ratio",
        "description": "Annual return divided by maximum drawdown",
        "function": calculate_calmar_ratio,
        "higher_is_better": True,
        "format": "decimal"
    },
    "win_rate": {
        "name": "Win Rate",
        "description": "Percentage of trades that are profitable",
        "function": calculate_win_rate,
        "higher_is_better": True,
        "format": "percentage"
    },
    "profit_ratio": {
        "name": "Profit Ratio",
        "description": "Average win / average loss",
        "function": calculate_profit_ratio,
        "higher_is_better": True,
        "format": "decimal"
    }
}

class CustomMetric:
    """Custom metric definition class"""
    
    def __init__(self, metric_id, name, description, formula=None, higher_is_better=True, format="decimal"):
        """
        Initialize a custom metric
        
        Args:
            metric_id (str): Unique identifier for the metric
            name (str): Display name of the metric
            description (str): Description of what the metric measures
            formula (str, optional): Python code formula as string
            higher_is_better (bool): Whether higher values are better
            format (str): Formatting type (decimal, percentage, currency)
        """
        self.metric_id = metric_id
        self.name = name
        self.description = description
        self.formula = formula
        self.higher_is_better = higher_is_better
        self.format = format
        
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "description": self.description,
            "formula": self.formula,
            "higher_is_better": self.higher_is_better,
            "format": self.format
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary representation"""
        return cls(
            metric_id=data.get("metric_id"),
            name=data.get("name"),
            description=data.get("description"),
            formula=data.get("formula"),
            higher_is_better=data.get("higher_is_better", True),
            format=data.get("format", "decimal")
        )
    
    def calculate(self, returns=None, trades=None, **kwargs):
        """
        Calculate metric value using the formula or a predefined function
        
        Args:
            returns (pandas.Series, optional): Series of returns
            trades (list, optional): List of trade dictionaries
            **kwargs: Additional parameters for calculation
            
        Returns:
            float: Calculated metric value
        """
        if self.metric_id in custom_metrics:
            # Use predefined function
            metric_def = custom_metrics[self.metric_id]
            metric_func = metric_def.get("function")
            
            if callable(metric_func):
                if "returns" in metric_func.__code__.co_varnames and returns is not None:
                    return metric_func(returns, **kwargs)
                elif "trades" in metric_func.__code__.co_varnames and trades is not None:
                    return metric_func(trades, **kwargs)
                elif "returns" in metric_func.__code__.co_varnames:
                    return 0  # No returns provided
                elif "trades" in metric_func.__code__.co_varnames:
                    return 0  # No trades provided
                else:
                    return metric_func(**kwargs)
            else:
                logger.warning(f"No valid function found for metric {self.metric_id}")
                return 0
        
        elif self.formula:
            # Execute custom formula
            try:
                # Create a safe execution environment
                local_vars = {
                    "returns": returns,
                    "trades": trades,
                    "np": np,
                    "pd": pd,
                    "len": len,
                    "sum": sum,
                    "abs": abs,
                    "min": min,
                    "max": max
                }
                # Update with additional kwargs
                local_vars.update(kwargs)
                
                # Execute formula in safe environment
                result = eval(self.formula, {"__builtins__": {}}, local_vars)
                return float(result)
            
            except Exception as e:
                logger.error(f"Error calculating custom metric {self.metric_id}: {str(e)}")
                return 0
        
        else:
            logger.warning(f"No calculation method defined for metric {self.metric_id}")
            return 0

def register_custom_metric(custom_metric):
    """
    Register a new custom metric
    
    Args:
        custom_metric (CustomMetric): The custom metric to register
        
    Returns:
        bool: True if successfully registered
    """
    try:
        metric_id = custom_metric.metric_id
        
        # Check if metric already exists
        if metric_id in custom_metrics:
            logger.warning(f"Metric {metric_id} already exists in the registry")
            return False
        
        # Get all custom metric definitions
        metrics_file = os.path.join(METRICS_DIR, "custom_metrics.json")
        custom_metrics_data = []
        
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                custom_metrics_data = json.load(f)
        
        # Check for duplicate
        for existing_metric in custom_metrics_data:
            if existing_metric.get("metric_id") == metric_id:
                logger.warning(f"Metric {metric_id} already exists in the file")
                return False
        
        # Add the new metric
        custom_metrics_data.append(custom_metric.to_dict())
        
        # Save the updated metrics
        with open(metrics_file, "w") as f:
            json.dump(custom_metrics_data, f, indent=2)
        
        logger.info(f"Successfully registered custom metric: {metric_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error registering custom metric: {str(e)}")
        return False

def get_custom_metrics():
    """
    Get all registered custom metrics
    
    Returns:
        list: List of CustomMetric objects
    """
    metrics_file = os.path.join(METRICS_DIR, "custom_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            custom_metrics_data = json.load(f)
        
        return [CustomMetric.from_dict(data) for data in custom_metrics_data]
    else:
        return []

def calculate_and_store_metrics(ticker, returns, trades=None, algorithm=None):
    """
    Calculate and store all custom metrics for a model
    
    Args:
        ticker (str): Ticker symbol
        returns (pandas.Series): Series of returns
        trades (list, optional): List of trade dictionaries
        algorithm (str, optional): Algorithm name
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    # Convert returns to pandas Series if it's not already
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    
    # Get all custom metrics
    all_metrics = get_custom_metrics()
    
    # Add standard metrics if not in custom metrics
    for metric_id, metric_def in custom_metrics.items():
        if not any(m.metric_id == metric_id for m in all_metrics):
            all_metrics.append(CustomMetric(
                metric_id=metric_id,
                name=metric_def.get("name"),
                description=metric_def.get("description"),
                higher_is_better=metric_def.get("higher_is_better"),
                format=metric_def.get("format")
            ))
    
    # Calculate each metric
    results = {}
    for metric in all_metrics:
        value = metric.calculate(returns=returns, trades=trades)
        results[metric.metric_id] = {
            "name": metric.name,
            "value": value,
            "is_good": (value > 0) if metric.higher_is_better else (value < 0),
            "format": metric.format
        }
    
    # Store results
    model_metrics_dir = os.path.join(METRICS_DIR, ticker)
    os.makedirs(model_metrics_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"{ticker}_{algorithm or 'model'}_{timestamp}.json"
    metrics_path = os.path.join(model_metrics_dir, metrics_filename)
    
    with open(metrics_path, "w") as f:
        json.dump({
            "ticker": ticker,
            "algorithm": algorithm,
            "timestamp": timestamp,
            "metrics": results
        }, f, indent=2)
    
    logger.info(f"Stored custom metrics for {ticker} at {metrics_path}")
    return results

def get_metric_history(ticker, metric_id, limit=10):
    """
    Get historical values of a specific metric for a ticker
    
    Args:
        ticker (str): Ticker symbol
        metric_id (str): Metric identifier
        limit (int): Maximum number of historical values to return
        
    Returns:
        list: List of dictionaries with timestamp and value
    """
    model_metrics_dir = os.path.join(METRICS_DIR, ticker)
    
    if not os.path.exists(model_metrics_dir):
        return []
    
    # Get all metric files for this ticker
    metric_files = [f for f in os.listdir(model_metrics_dir) if f.endswith(".json")]
    history = []
    
    for filename in metric_files:
        try:
            file_path = os.path.join(model_metrics_dir, filename)
            with open(file_path, "r") as f:
                data = json.load(f)
            
            if "metrics" in data and metric_id in data["metrics"]:
                metric_data = data["metrics"][metric_id]
                history.append({
                    "timestamp": data.get("timestamp"),
                    "algorithm": data.get("algorithm"),
                    "value": metric_data.get("value")
                })
        except Exception as e:
            logger.error(f"Error reading metric file {filename}: {str(e)}")
    
    # Sort by timestamp (newest first) and limit
    history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return history[:limit]

def compare_models(tickers, algorithms=None, metrics=None):
    """
    Compare multiple models based on selected metrics
    
    Args:
        tickers (list): List of ticker symbols
        algorithms (list, optional): List of algorithms to filter by
        metrics (list, optional): List of metric IDs to include
        
    Returns:
        dict: Comparison data grouped by metric
    """
    if not metrics:
        # Use default metrics if none provided
        metrics = ["sharpe_ratio", "max_drawdown", "annualized_return", "win_rate"]
    
    comparison = {metric: [] for metric in metrics}
    
    for ticker in tickers:
        model_metrics_dir = os.path.join(METRICS_DIR, ticker)
        if not os.path.exists(model_metrics_dir):
            continue
        
        # Get most recent metrics file for each algorithm
        latest_metrics = {}
        
        for filename in os.listdir(model_metrics_dir):
            if not filename.endswith(".json"):
                continue
            
            try:
                parts = filename.split("_")
                if len(parts) >= 3:
                    algo = parts[1]
                    
                    # Skip if filtering by algorithms
                    if algorithms and algo not in algorithms:
                        continue
                    
                    timestamp = parts[2].replace(".json", "")
                    
                    if algo not in latest_metrics or timestamp > latest_metrics[algo]["timestamp"]:
                        file_path = os.path.join(model_metrics_dir, filename)
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        
                        latest_metrics[algo] = {
                            "timestamp": timestamp,
                            "data": data
                        }
            except Exception as e:
                logger.error(f"Error processing metrics file {filename}: {str(e)}")
        
        # Add to comparison
        for algo, metrics_data in latest_metrics.items():
            data = metrics_data["data"]
            if "metrics" not in data:
                continue
            
            model_name = f"{ticker} ({algo})"
            
            for metric_id in metrics:
                if metric_id in data["metrics"]:
                    metric_value = data["metrics"][metric_id]["value"]
                    comparison[metric_id].append({
                        "model": model_name,
                        "ticker": ticker,
                        "algorithm": algo,
                        "value": metric_value
                    })
    
    # Sort each metric by value (appropriate for each metric)
    for metric_id in comparison:
        if metric_id in custom_metrics:
            higher_is_better = custom_metrics[metric_id].get("higher_is_better", True)
            comparison[metric_id].sort(key=lambda x: x["value"], reverse=higher_is_better)
    
    return comparison

def track_metric_threshold(ticker, metric_id, threshold, is_above_threshold=True):
    """
    Check if a metric has crossed a threshold
    
    Args:
        ticker (str): Ticker symbol
        metric_id (str): Metric identifier
        threshold (float): Threshold value
        is_above_threshold (bool): True to check if value exceeds threshold, False for below
        
    Returns:
        dict: Result with fields: crossed (bool), current_value (float)
    """
    # Get latest metric value
    history = get_metric_history(ticker, metric_id, limit=1)
    
    if not history:
        return {"crossed": False, "current_value": None}
    
    current_value = history[0]["value"]
    
    # Check if threshold is crossed
    if is_above_threshold:
        crossed = current_value > threshold
    else:
        crossed = current_value < threshold
    
    return {
        "crossed": crossed,
        "current_value": current_value
    } 