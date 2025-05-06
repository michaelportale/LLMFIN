# scripts/train_r1.py

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from envs.portfolio import PortfolioEnv
from scripts.sentiment_analysis import get_dummy_sentiment


def create_benchmark(csv_path, initial_cash=10000):
    """Create a buy-and-hold benchmark for comparison"""
    df = pd.read_csv(csv_path)
    
    # Ensure Date column is properly formatted
    if df.columns[0] not in ["Date", "date"]:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Calculate how many shares we could buy on day 1
    initial_price = df["Close"].iloc[0]
    shares = initial_cash // initial_price
    remaining_cash = initial_cash - (shares * initial_price)
    
    # Calculate portfolio value over time
    df["Benchmark"] = (df["Close"] * shares) + remaining_cash
    
    # Calculate daily returns
    df["Benchmark_Return"] = df["Benchmark"].pct_change().fillna(0)
    
    # Calculate benchmark metrics
    initial_value = df["Benchmark"].iloc[0]
    final_value = df["Benchmark"].iloc[-1]
    total_return = (final_value - initial_value) / initial_value
    
    # Calculate max drawdown
    df["Benchmark_Peak"] = df["Benchmark"].cummax()
    df["Benchmark_Drawdown"] = (df["Benchmark_Peak"] - df["Benchmark"]) / df["Benchmark_Peak"]
    max_drawdown = df["Benchmark_Drawdown"].max()
    
    # Calculate Sharpe ratio
    daily_returns = df["Benchmark_Return"].values
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
    annualized_sharpe = sharpe_ratio * np.sqrt(252)  # Annualize assuming 252 trading days
    
    benchmark_metrics = {
        "initial_value": float(initial_value),
        "final_value": float(final_value),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(annualized_sharpe)
    }
    
    return df["Benchmark"].values, benchmark_metrics


def evaluate_model(model, env, num_episodes=10):
    """Evaluate the trained model performance"""

    all_rewards = []
    all_portfolio_values = []
    all_trades = []

    for episode in range(num_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result  # Unpack if it returns a tuple (newer Gymnasium API)
        else:
            obs = reset_result     # Use directly if it returns just the observation (older Gym API)
            
        done = False
        truncated = False
        episode_rewards = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Handle both Gym and Gymnasium API formats for step()
            step_result = env.step(action)
            
            if len(step_result) == 5:  # Gymnasium API (obs, reward, terminated, truncated, info)
                obs, reward, done, truncated, info = step_result
                done = done or truncated
            else:  # Gym API (obs, reward, done, info)
                obs, reward, done, info = step_result
                truncated = False
                
            episode_rewards.append(reward)
            
        # After episode finishes, collect metrics
        all_rewards.append(sum(episode_rewards))
        
        # Access the unwrapped environment to get portfolio values and trades
        if hasattr(env, 'envs'):  # If it's a DummyVecEnv
            base_env = env.envs[0].unwrapped
        else:
            base_env = env.unwrapped
            
        all_portfolio_values.append(base_env.portfolio_values)
        all_trades.append(base_env.trade_history)
    
    # Average portfolio performance across episodes
    avg_portfolio = np.mean([values[-1] for values in all_portfolio_values])
    
    # Get detailed metrics from the last run
    if hasattr(env, 'envs'):  # If it's a DummyVecEnv
        performance_metrics = env.envs[0].unwrapped.get_performance_metrics()
    else:
        performance_metrics = env.unwrapped.get_performance_metrics()
    
    return {
        "avg_return": float(np.mean(all_rewards)),
        "avg_final_value": float(avg_portfolio),
        **performance_metrics,
        "trade_examples": all_trades[-1][:5]  # Include sample of trades from last episode
    }


# Create a custom callback for early stopping
class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback that stops training if a metric doesn't improve
    for a specified number of evaluations
    """
    def __init__(self, eval_env, eval_freq=5000, patience=5, min_improvement=0.01, verbose=1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.patience_counter = 0
        
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            # Evaluate the current model
            mean_reward, _ = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=5,
                deterministic=True
            )
            
            if self.verbose > 0:
                print(f"Evaluation at timestep {self.num_timesteps}: Mean reward = {mean_reward:.2f}")
            
            # Check if there's significant improvement
            if mean_reward > self.best_mean_reward * (1 + self.min_improvement):
                self.best_mean_reward = mean_reward
                self.patience_counter = 0
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f}")
            else:
                self.patience_counter += 1
                if self.verbose > 0:
                    print(f"No significant improvement for {self.patience_counter} evaluations")
                
                if self.patience_counter >= self.patience:
                    if self.verbose > 0:
                        print(f"Early stopping at timestep {self.num_timesteps}")
                    return False  # Stop training
        
        return True  # Continue training
        

def train_model(
    csv_path, 
    total_timesteps=50000, 
    model_save_path="models/ppo_portfolio",
    eval_freq=5000,
    tensorboard_log="logs",
    resume_training=False,
    early_stopping=True,
    patience=5,
    min_improvement=0.01,
    learning_rate=1e-4,
    batch_size=64,
    n_steps=2048,
    n_epochs=10,
    gamma=0.99,
    algorithm="PPO"
):
    """
    Train a reinforcement learning model for portfolio management
    
    Parameters:
    -----------
    algorithm : str
        The RL algorithm to use: 'PPO', 'A2C', 'SAC', or 'TD3'
    """
    # Extract ticker name from file path for logs
    ticker = os.path.basename(csv_path).split('.')[0]
    logs_dir = os.path.join(tensorboard_log, f"{algorithm}_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create logs directory for checkpoints
    checkpoint_dir = os.path.join(logs_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Log hyperparameters
    print(f"Training with algorithm: {algorithm}, learning_rate={learning_rate}, batch_size={batch_size}, n_steps={n_steps}")
    
    def make_env():
        env = PortfolioEnv(
            csv_path=csv_path,
            sentiment_func=get_dummy_sentiment,
            initial_cash=10000,
            transaction_cost=0.001,  # 0.1% transaction cost
            lookback_window=10,
            position_size=0.2  # Use 20% of cash per trade
        )

        return env

    # Create environment
    base_env = make_env()
    env = DummyVecEnv([lambda: Monitor(base_env, logs_dir)])
    eval_env = DummyVecEnv([lambda: Monitor(make_env(), logs_dir)])
    
    # Set up callbacks for evaluation and checkpoints
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=logs_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=checkpoint_dir,
        name_prefix=f"{algorithm.lower()}_{ticker}"
    )
    
    # Create early stopping callback if enabled
    callbacks = [eval_callback, checkpoint_callback]
    if early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            eval_env=eval_env,
            eval_freq=eval_freq,
            patience=patience,
            min_improvement=min_improvement
        )
        callbacks.append(early_stopping_callback)
    
    # Update model path to include algorithm name
    model_path = f"{model_save_path.replace('ppo', algorithm.lower())}_{ticker}"
    
    # Create or load model based on algorithm
    if resume_training and os.path.exists(model_path + ".zip"):
        print(f"Resuming training from existing model: {model_path}")
        try:
            # Load model based on algorithm
            if algorithm == "PPO":
                model = PPO.load(model_path, env=env)
            elif algorithm == "A2C":
                model = A2C.load(model_path, env=env)
            elif algorithm == "SAC":
                model = SAC.load(model_path, env=env)
            elif algorithm == "TD3":
                model = TD3.load(model_path, env=env)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
            print("Successfully loaded model for continued training")
            
            # Update model parameters if they've changed
            model.learning_rate = learning_rate
            if hasattr(model, "batch_size"):
                model.batch_size = batch_size
            if hasattr(model, "n_steps"):
                model.n_steps = n_steps
            if hasattr(model, "n_epochs"):
                model.n_epochs = n_epochs
            model.gamma = gamma
            
        except Exception as e:
            print(f"Error loading model: {e}. Creating new model instead.")
            model = create_new_model(algorithm, env, logs_dir, learning_rate, n_steps, 
                                    batch_size, n_epochs, gamma)
    else:
        # Create new model
        model = create_new_model(algorithm, env, logs_dir, learning_rate, n_steps, 
                                batch_size, n_epochs, gamma)
    
    print(f"Training {algorithm} model for {ticker} with {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps, 
        callback=callbacks,
        reset_num_timesteps=not resume_training  # Don't reset timesteps if resuming
    )
    
    # Save the final model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Create and evaluate a benchmark
    print("Evaluating against buy-and-hold benchmark...")
    benchmark_values, benchmark_metrics = create_benchmark(csv_path)
    
    # Evaluate the trained model
    print("Evaluating trained model...")
    eval_metrics = evaluate_model(model, env, num_episodes=5)
    
    # Compare model vs benchmark
    performance_comparison = {
        "model": eval_metrics,
        "benchmark": benchmark_metrics,
        "outperformance": eval_metrics["total_return"] - benchmark_metrics["total_return"],
        "algorithm": algorithm,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_steps": n_steps if algorithm in ["PPO", "A2C"] else None,
            "n_epochs": n_epochs if algorithm == "PPO" else None,
            "gamma": gamma,
            "early_stopping": early_stopping,
            "patience": patience if early_stopping else None,
            "min_improvement": min_improvement if early_stopping else None
        }
    }
    
    # Save metrics to file
    metrics_dir = os.path.join("static", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"{os.path.basename(csv_path).split('.')[0]}.json")
    
    with open(metrics_path, "w") as f:
        json.dump(performance_comparison, f, indent=2)
    
    # Create performance visualization
    create_performance_plot(csv_path, model, benchmark_values)
    
    return performance_comparison


def create_new_model(algorithm, env, logs_dir, learning_rate, n_steps, batch_size, n_epochs, gamma):
    """Create a new model based on the algorithm"""
    if algorithm == "PPO":
        return PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log=logs_dir,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma
        )
    elif algorithm == "A2C":
        return A2C(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log=logs_dir,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma
        )
    elif algorithm == "SAC":
        return SAC(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log=logs_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma
        )
    elif algorithm == "TD3":
        return TD3(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log=logs_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def create_performance_plot(csv_path, model, benchmark_values):
    """Create a plot comparing model performance vs benchmark"""
    ticker = os.path.basename(csv_path).split('.')[0]
    plot_dir = os.path.join("static", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{ticker}.png")
    
    # Create a fresh environment instance for plotting (not wrapped)
    plot_env = PortfolioEnv(
        csv_path=csv_path,
        sentiment_func=get_dummy_sentiment,
        initial_cash=10000,
        transaction_cost=0.001,
        lookback_window=10,
        position_size=0.2
    )
    
    reset_result = plot_env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result  # Unpack if it returns a tuple (newer Gymnasium API)
    else:
        obs = reset_result     # Use directly if it returns just the observation (older Gym API)
        
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # Handle both Gym and Gymnasium API formats for step()
        step_result = plot_env.step(action)
        
        if len(step_result) == 5:  # Gymnasium API (obs, reward, terminated, truncated, info)
            obs, reward, done, truncated, info = step_result
            done = done or truncated
        else:  # Gym API (obs, reward, done, info)
            obs, reward, done, info = step_result

    # Get the dates for the x-axis
    df = pd.read_csv(csv_path)
    if df.columns[0] not in ["Date", "date"]:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    dates = df["Date"].iloc[plot_env.lookback_window:plot_env.lookback_window+len(plot_env.portfolio_values)]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot model performance
    plt.plot(dates, plot_env.portfolio_values, label="RL Model", color="blue", linewidth=2)
    
    # Plot benchmark performance
    benchmark_subset = benchmark_values[plot_env.lookback_window:plot_env.lookback_window+len(plot_env.portfolio_values)]
    plt.plot(dates, benchmark_subset, label="Buy & Hold", color="green", linestyle="--", linewidth=2)
    
    # Plot buy/sell actions
    dates_list = dates.tolist()  # Convert to list for integer indexing
    
    for i, action in enumerate(plot_env.actions_taken):
        if i >= len(dates_list):
            break
            
        if action == 1:  # BUY
            plt.scatter(dates_list[i], plot_env.portfolio_values[i], color="green", marker="^", s=100)
        elif action == 2:  # SELL
            plt.scatter(dates_list[i], plot_env.portfolio_values[i], color="red", marker="v", s=100)
    
    # Add chart details
    plt.title(f"Portfolio Performance: RL Model vs Buy & Hold ({ticker})", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Performance plot saved to {plot_path}")


if __name__ == "__main__":
    csv_path = os.path.join("data", "AAPL.csv")  # Change ticker if needed
    train_model(csv_path, total_timesteps=50000)