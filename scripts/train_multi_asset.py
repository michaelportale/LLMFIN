# scripts/train_multi_asset.py

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from envs.multi_asset_portfolio import MultiAssetPortfolioEnv
from scripts.sentiment_analysis import get_dummy_sentiment


def create_equal_weight_benchmark(csv_paths, initial_cash=10000):
    """Create an equal-weight buy-and-hold benchmark for comparison"""
    # Load all dataframes
    dfs = []
    tickers = []
    
    for path in csv_paths:
        df = pd.read_csv(path)
        
        # Ensure Date column is properly formatted
        if df.columns[0] not in ["Date", "date"]:
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Add ticker column
        ticker = os.path.basename(path).split('.')[0]
        df["Ticker"] = ticker
        tickers.append(ticker)
        dfs.append(df)
    
    # Find common dates
    common_dates = set(dfs[0]["Date"])
    for df in dfs[1:]:
        common_dates &= set(df["Date"])
    
    # Sort dates chronologically
    common_dates = sorted(list(common_dates))
    
    # Filter DataFrames to only include common dates
    filtered_dfs = []
    for df in dfs:
        filtered_df = df[df["Date"].isin(common_dates)].copy()
        filtered_df.reset_index(drop=True, inplace=True)
        filtered_dfs.append(filtered_df)
    
    # Create a consolidated DataFrame
    benchmark_data = {
        "Date": common_dates,
        "Portfolio_Value": np.zeros(len(common_dates))
    }
    
    # Initialize with equal weight allocation
    allocation_per_asset = initial_cash / len(tickers)
    shares_per_asset = {}
    
    # Calculate initial shares for each asset
    for i, ticker in enumerate(tickers):
        initial_price = filtered_dfs[i].loc[0, "Close"]
        shares = allocation_per_asset / initial_price
        shares_per_asset[ticker] = shares
        benchmark_data[f"{ticker}_Value"] = np.zeros(len(common_dates))
    
    # Calculate portfolio value over time
    for i in range(len(common_dates)):
        total_value = 0
        
        for j, ticker in enumerate(tickers):
            asset_price = filtered_dfs[j].loc[i, "Close"]
            asset_value = shares_per_asset[ticker] * asset_price
            benchmark_data[f"{ticker}_Value"][i] = asset_value
            total_value += asset_value
        
        benchmark_data["Portfolio_Value"][i] = total_value
    
    benchmark_df = pd.DataFrame(benchmark_data)
    
    # Calculate daily returns
    benchmark_df["Daily_Return"] = benchmark_df["Portfolio_Value"].pct_change().fillna(0)
    
    # Calculate benchmark metrics
    initial_value = benchmark_df["Portfolio_Value"].iloc[0]
    final_value = benchmark_df["Portfolio_Value"].iloc[-1]
    total_return = (final_value - initial_value) / initial_value
    
    # Calculate max drawdown
    benchmark_df["Peak"] = benchmark_df["Portfolio_Value"].cummax()
    benchmark_df["Drawdown"] = (benchmark_df["Peak"] - benchmark_df["Portfolio_Value"]) / benchmark_df["Peak"]
    max_drawdown = benchmark_df["Drawdown"].max()
    
    # Calculate Sharpe ratio
    daily_returns = benchmark_df["Daily_Return"].values
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
    annualized_sharpe = sharpe_ratio * np.sqrt(252)  # Annualize assuming 252 trading days
    
    benchmark_metrics = {
        "initial_value": float(initial_value),
        "final_value": float(final_value),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(annualized_sharpe)
    }
    
    return benchmark_df["Portfolio_Value"].values, benchmark_metrics, benchmark_df


def evaluate_model(model, env, num_episodes=5):
    """Evaluate the trained model performance"""

    all_rewards = []
    all_portfolio_values = []
    all_trades = []
    all_allocations = []

    for episode in range(num_episodes):
        # Make the reset() call more robust to handle both API versions
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result  # Unpack if it returns a tuple (newer Gymnasium API)
        else:
            obs = reset_result     # Use directly if it returns just the observation (older Gym API)
            
        done = False
        truncated = False
        episode_rewards = []
        episode_allocations = []
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            
            # Handle both Gym and Gymnasium API formats for step()
            step_result = env.step(action)
            
            if len(step_result) == 5:  # Gymnasium API (obs, reward, terminated, truncated, info)
                obs, reward, done, truncated, info = step_result
            else:  # Gym API (obs, reward, done, info)
                obs, reward, done, info = step_result
                truncated = False
                
            episode_rewards.append(reward)
            episode_allocations.append(info.get("allocations", {}))
            
        # After episode finishes, collect metrics
        all_rewards.append(sum(episode_rewards))
        
        # Access the unwrapped environment to get portfolio values and trades
        if hasattr(env, 'envs'):  # If it's a DummyVecEnv
            base_env = env.envs[0].unwrapped
        else:
            base_env = env.unwrapped
            
        all_portfolio_values.append(base_env.portfolio_values)
        all_trades.append(base_env.trade_history)
        all_allocations.append(episode_allocations)
    
    # Average portfolio performance across episodes
    avg_portfolio = np.mean([values[-1] for values in all_portfolio_values])
    
    # Get detailed metrics from the last run
    if hasattr(env, 'envs'):  # If it's a DummyVecEnv
        performance_metrics = env.envs[0].unwrapped.get_performance_metrics()
        tickers = env.envs[0].unwrapped.tickers
    else:
        performance_metrics = env.unwrapped.get_performance_metrics()
        tickers = env.unwrapped.tickers
    
    # Add average allocation information
    avg_final_allocations = {}
    for ticker in tickers:
        allocations = [episode_allocations[-1].get(ticker, 0) for episode_allocations in all_allocations]
        avg_final_allocations[ticker] = float(np.mean(allocations))
    
    performance_metrics["avg_final_allocations"] = avg_final_allocations
    
    return {
        "avg_return": float(np.mean(all_rewards)),
        "avg_final_value": float(avg_portfolio),
        **performance_metrics,
        "trade_examples": all_trades[-1][:10]  # Include sample of trades from last episode
    }


# Use the early stopping callback from train_r1.py
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
    csv_paths, 
    total_timesteps=50000, 
    model_save_path="models/ppo_multi",
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
    algorithm="PPO",
    position_size=0.2,
    max_allocation=0.4,  # Max 40% in any single asset by default
    support_advanced_orders=True,  # Support limit and stop-loss orders
    max_order_expiration=10  # Maximum days until order expiration
):
    """
    Train a reinforcement learning model for multi-asset portfolio management
    
    Parameters:
    -----------
    csv_paths : list of str
        Paths to CSV files for each asset
    algorithm : str
        The RL algorithm to use: 'PPO', 'A2C', 'SAC', or 'TD3'
    support_advanced_orders : bool
        Whether to support advanced order types (limit and stop-loss)
    max_order_expiration : int
        Maximum number of days until limit/stop orders expire
    """
    # Extract tickers from file paths for naming
    tickers = [os.path.basename(path).split('.')[0] for path in csv_paths]
    model_name = "_".join(tickers)
    if len(model_name) > 50:  # Truncate if too long
        model_name = model_name[:47] + "..."
    
    # Create logs directory
    logs_dir = os.path.join(tensorboard_log, f"{algorithm}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(logs_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Log hyperparameters
    print(f"Training multi-asset portfolio with {len(csv_paths)} assets: {', '.join(tickers)}")
    print(f"Algorithm: {algorithm}, learning_rate={learning_rate}, batch_size={batch_size}, n_steps={n_steps}")
    print(f"Max allocation per asset: {max_allocation * 100:.1f}%, Position size per trade: {position_size * 100:.1f}%")
    
    if support_advanced_orders:
        print(f"Advanced orders: ENABLED (limit and stop-loss orders with {max_order_expiration} day expiration)")
    else:
        print("Advanced orders: DISABLED (using market orders only)")
    
    def make_env():
        env = MultiAssetPortfolioEnv(
            csv_paths=csv_paths,
            sentiment_func=get_dummy_sentiment,
            initial_cash=10000,
            transaction_cost=0.001,
            lookback_window=10,
            position_size=position_size,
            max_allocation=max_allocation,
            support_advanced_orders=support_advanced_orders,
            max_order_expiration=max_order_expiration
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
        name_prefix=f"{algorithm.lower()}_{model_name}"
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
    
    # Define model path
    model_dir = os.path.join(model_save_path, algorithm.lower())
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{algorithm.lower()}_{model_name}")
    
    # Create or load model
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
    
    print(f"Training {algorithm} model for multi-asset portfolio with {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps, 
        callback=callbacks,
        reset_num_timesteps=not resume_training
    )
    
    # Save the final model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Create and evaluate benchmark
    print("Evaluating against equal-weight benchmark...")
    benchmark_values, benchmark_metrics, benchmark_df = create_equal_weight_benchmark(csv_paths)
    
    # Evaluate the trained model
    print("Evaluating trained model...")
    eval_metrics = evaluate_model(model, make_env(), num_episodes=5)
    
    # Compare model vs benchmark
    performance_comparison = {
        "portfolio": {
            "tickers": tickers,
            "num_assets": len(tickers)
        },
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
            "position_size": position_size,
            "max_allocation": max_allocation,
            "early_stopping": early_stopping,
            "patience": patience if early_stopping else None,
            "min_improvement": min_improvement if early_stopping else None
        }
    }
    
    # Create directory for metrics and plots
    metrics_dir = os.path.join("static", "metrics", "multi")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Generate a unique identifier for this portfolio
    portfolio_id = f"{'_'.join(tickers[:3])}"
    if len(tickers) > 3:
        portfolio_id += f"_plus_{len(tickers) - 3}"
    
    # Save metrics to file
    metrics_path = os.path.join(metrics_dir, f"{portfolio_id}.json")
    with open(metrics_path, "w") as f:
        json.dump(performance_comparison, f, indent=2)
    
    # Create performance visualization
    create_performance_plots(csv_paths, model, benchmark_values, benchmark_df, portfolio_id)
    
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


def create_performance_plots(csv_paths, model, benchmark_values, benchmark_df, portfolio_id):
    """Create performance visualizations for multi-asset portfolio"""
    # Directory for plots
    plot_dir = os.path.join("static", "plots", "multi")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create a new environment instance for plotting (not wrapped)
    plot_env = MultiAssetPortfolioEnv(
        csv_paths=csv_paths,
        sentiment_func=get_dummy_sentiment,
        initial_cash=10000,
        transaction_cost=0.001,
        lookback_window=10,
        position_size=0.2,
        max_allocation=0.4
    )
    
    # Get tickers
    tickers = plot_env.tickers
    
    # Run model for a single episode
    reset_result = plot_env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result  # Unpack if it returns a tuple (newer Gymnasium API)
    else:
        obs = reset_result     # Use directly if it returns just the observation (older Gym API)
        
    done = False
    truncated = False
    
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        
        # Handle both Gym and Gymnasium API formats for step()
        step_result = plot_env.step(action)
        
        if len(step_result) == 5:  # Gymnasium API (obs, reward, terminated, truncated, info)
            obs, reward, done, truncated, info = step_result
        else:  # Gym API (obs, reward, done, info)
            obs, reward, done, info = step_result
            truncated = False
    
    # Performance plot
    plt.figure(figsize=(14, 7))
    
    # Plot model performance vs benchmark
    dates = plot_env.aligned_dates[plot_env.lookback_window:plot_env.lookback_window + len(plot_env.portfolio_values)]
    dates_list = list(dates)  # Convert to list for safer indexing
    plt.plot(dates_list, plot_env.portfolio_values, label="RL Portfolio", color="blue", linewidth=2)
    
    # Plot benchmark performance
    benchmark_subset = benchmark_values[plot_env.lookback_window:plot_env.lookback_window + len(plot_env.portfolio_values)]
    plt.plot(dates_list, benchmark_subset, label="Equal Weight", color="green", linestyle="--", linewidth=2)
    
    # Add buy/sell markers
    for trade in plot_env.trade_history:
        trade_date = trade['date']
        try:
            # Try to find the date index in a safer way
            trade_index = dates_list.index(trade_date)
            if trade['action'] == 'BUY':
                plt.scatter(dates_list[trade_index], plot_env.portfolio_values[trade_index], 
                           color="green", marker="^", s=80, alpha=0.7)
            else:  # SELL
                plt.scatter(dates_list[trade_index], plot_env.portfolio_values[trade_index], 
                           color="red", marker="v", s=80, alpha=0.7)
        except (ValueError, IndexError):
            # Skip if date not found or index out of range
            continue
    
    # Add chart details
    plt.title(f"Portfolio Performance: RL Model vs Equal Weight ({portfolio_id})", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the performance plot
    performance_plot_path = os.path.join(plot_dir, f"{portfolio_id}_performance.png")
    plt.savefig(performance_plot_path)
    plt.close()
    
    # Create allocation history plot
    plt.figure(figsize=(14, 7))
    
    # Plot allocation over time
    allocation_df = pd.DataFrame(plot_env.asset_allocations)
    # Ensure dates are properly aligned with allocation data
    allocation_dates = dates_list[:len(allocation_df)]
    allocation_df.index = allocation_dates
    
    # Stacked area chart of allocations
    ax = allocation_df.plot.area(stacked=True, figsize=(14, 7), alpha=0.7, colormap='viridis')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Allocation", fontsize=12)
    ax.set_title(f"Portfolio Allocation Over Time ({portfolio_id})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Assets", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the allocation plot
    allocation_plot_path = os.path.join(plot_dir, f"{portfolio_id}_allocation.png")
    plt.savefig(allocation_plot_path)
    plt.close()
    
    # Create final allocation pie chart
    plt.figure(figsize=(10, 10))
    
    # Get final allocations including cash
    final_allocations = plot_env.asset_allocations[-1].copy()
    total_value = plot_env.portfolio_values[-1]
    cash_ratio = plot_env.cash / total_value if total_value > 0 else 0
    final_allocations["Cash"] = cash_ratio
    
    # Plot pie chart
    plt.pie(
        final_allocations.values(), 
        labels=final_allocations.keys(), 
        autopct='%1.1f%%',
        startangle=90, 
        shadow=False,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 12}
    )
    plt.axis('equal')
    plt.title(f"Final Portfolio Allocation ({portfolio_id})", fontsize=14)
    
    # Save the pie chart
    pie_plot_path = os.path.join(plot_dir, f"{portfolio_id}_final_allocation.png")
    plt.savefig(pie_plot_path)
    plt.close()
    
    print(f"Performance plots saved to {plot_dir}")
    

if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    sample_tickers = ["AAPL", "MSFT", "AMZN"]
    csv_paths = [os.path.join(data_dir, f"{ticker}.csv") for ticker in sample_tickers]
    
    # Check if files exist
    if all(os.path.exists(path) for path in csv_paths):
        train_model(csv_paths, total_timesteps=50000, algorithm="PPO")
    else:
        print("One or more data files not found. Please fetch data for these tickers first.") 