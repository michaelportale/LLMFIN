"""
Monte Carlo simulations for financial risk assessment.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import scipy.stats as stats
import os
from tqdm import tqdm


def simulate_gbm(initial_price: float, mu: float, sigma: float, 
                days: int, n_simulations: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Simulate asset prices using Geometric Brownian Motion.
    
    Args:
        initial_price: Initial asset price
        mu: Expected annual return (drift)
        sigma: Annual volatility
        days: Number of days to simulate
        n_simulations: Number of simulation paths
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (days, n_simulations) with simulated prices
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert annual parameters to daily
    daily_returns = np.exp(
        (mu - 0.5 * sigma ** 2) / 252 + 
        sigma * np.sqrt(1 / 252) * np.random.normal(size=(days, n_simulations))
    )
    
    # Calculate price paths
    price_paths = np.zeros((days, n_simulations))
    price_paths[0] = initial_price
    
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
    
    return price_paths


def simulate_portfolio(initial_weights: Dict[str, float], 
                      historical_data: Dict[str, pd.DataFrame],
                      forecast_days: int = 252, 
                      n_simulations: int = 1000,
                      return_column: str = 'Close',
                      random_state: Optional[int] = None) -> Dict[str, Any]:
    """
    Simulate portfolio performance using Monte Carlo.
    
    Args:
        initial_weights: Dictionary mapping asset symbols to their weights
        historical_data: Dictionary mapping asset symbols to their historical price DataFrames
        forecast_days: Number of days to forecast
        n_simulations: Number of simulation paths
        return_column: Column to use for return calculations
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with simulation results
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Validate inputs
    if not sum(initial_weights.values()) == 1.0:
        raise ValueError("Portfolio weights must sum to 1.0")
    
    # Extract asset returns from historical data
    returns = {}
    for symbol, df in historical_data.items():
        if symbol in initial_weights:
            returns[symbol] = df[return_column].pct_change().dropna()
    
    # Calculate mean and std of returns for each asset
    mu_sigma = {}
    for symbol, ret in returns.items():
        annual_return = np.mean(ret) * 252
        annual_vol = np.std(ret) * np.sqrt(252)
        mu_sigma[symbol] = (annual_return, annual_vol)
    
    # Simulate price paths for each asset
    price_paths = {}
    for symbol, weight in initial_weights.items():
        if symbol in historical_data:
            initial_price = historical_data[symbol][return_column].iloc[-1]
            mu, sigma = mu_sigma[symbol]
            
            price_paths[symbol] = simulate_gbm(
                initial_price=initial_price,
                mu=mu,
                sigma=sigma,
                days=forecast_days,
                n_simulations=n_simulations,
                seed=random_state
            )
    
    # Calculate portfolio value paths
    portfolio_paths = np.zeros((forecast_days, n_simulations))
    initial_portfolio_value = 10000  # Starting with $10,000 for simplicity
    
    for symbol, weight in initial_weights.items():
        initial_asset_value = initial_portfolio_value * weight
        initial_shares = initial_asset_value / historical_data[symbol][return_column].iloc[-1]
        
        asset_value_paths = price_paths[symbol] * initial_shares
        portfolio_paths += asset_value_paths
    
    # Calculate statistics from simulations
    final_values = portfolio_paths[-1, :]
    
    # Percentiles of final portfolio value
    percentiles = {
        "1%": np.percentile(final_values, 1),
        "5%": np.percentile(final_values, 5),
        "10%": np.percentile(final_values, 10),
        "50%": np.percentile(final_values, 50),
        "90%": np.percentile(final_values, 90),
        "95%": np.percentile(final_values, 95),
        "99%": np.percentile(final_values, 99)
    }
    
    # Calculate VaR and CVaR
    var_95 = np.percentile(final_values - initial_portfolio_value, 5)
    cvar_95 = np.mean((final_values - initial_portfolio_value)[final_values - initial_portfolio_value <= var_95])
    
    return {
        "portfolio_paths": portfolio_paths,
        "initial_value": initial_portfolio_value,
        "final_values": final_values,
        "percentiles": percentiles,
        "mean_final_value": np.mean(final_values),
        "median_final_value": np.median(final_values),
        "std_final_value": np.std(final_values),
        "var_95": var_95,
        "cvar_95": cvar_95
    }


def plot_monte_carlo_simulation(simulation_result: Dict[str, Any],
                              confidence_interval: float = 0.9,
                              title: str = "Monte Carlo Simulation",
                              figsize: Tuple[int, int] = (12, 6),
                              output_file: Optional[str] = None) -> None:
    """
    Plot Monte Carlo simulation results.
    
    Args:
        simulation_result: Dictionary with simulation results
        confidence_interval: Confidence interval to highlight
        title: Plot title
        figsize: Figure size
        output_file: Path to save the output file (if None, display the plot)
    """
    portfolio_paths = simulation_result["portfolio_paths"]
    initial_value = simulation_result["initial_value"]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Calculate time axis (assuming daily)
    days = portfolio_paths.shape[0]
    
    # Plot all paths with low alpha
    for i in range(min(portfolio_paths.shape[1], 100)):  # Plot up to 100 paths
        plt.plot(range(days), portfolio_paths[:, i], color='blue', alpha=0.05)
    
    # Calculate and plot confidence intervals
    lower_bound = (1 - confidence_interval) / 2
    upper_bound = 1 - lower_bound
    
    lower_path = np.percentile(portfolio_paths, lower_bound * 100, axis=1)
    upper_path = np.percentile(portfolio_paths, upper_bound * 100, axis=1)
    median_path = np.percentile(portfolio_paths, 50, axis=1)
    
    plt.plot(range(days), median_path, color='blue', linewidth=2, label='Median')
    plt.plot(range(days), lower_path, color='red', linewidth=2, 
            label=f'{lower_bound*100:.1f}th Percentile')
    plt.plot(range(days), upper_path, color='green', linewidth=2, 
            label=f'{upper_bound*100:.1f}th Percentile')
    
    # Fill between confidence interval
    plt.fill_between(range(days), lower_path, upper_path, color='blue', alpha=0.1)
    
    # Add initial value as horizontal line
    plt.axhline(y=initial_value, color='black', linestyle='--', label='Initial Value')
    
    # Add labels and title
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value ($)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text with statistics
    stats_text = f"""
    Mean Final Value: ${simulation_result['mean_final_value']:.2f}
    Median Final Value: ${simulation_result['median_final_value']:.2f}
    95% VaR: ${-simulation_result['var_95']:.2f}
    95% CVaR: ${-simulation_result['cvar_95']:.2f}
    """
    
    plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def calculate_var_cvar(returns: pd.Series, 
                     confidence_level: float = 0.95, 
                     initial_investment: float = 10000) -> Dict[str, float]:
    """
    Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95% confidence)
        initial_investment: Initial investment amount
        
    Returns:
        Dictionary with VaR and CVaR values
    """
    # Sort returns
    sorted_returns = returns.sort_values()
    
    # Calculate VaR
    var_percentile = 1 - confidence_level
    var = sorted_returns.quantile(var_percentile)
    var_amount = initial_investment * var
    
    # Calculate CVaR (Expected Shortfall)
    cvar = sorted_returns[sorted_returns <= var].mean()
    cvar_amount = initial_investment * cvar
    
    return {
        'var_percent': var,
        'var_amount': var_amount,
        'cvar_percent': cvar,
        'cvar_amount': cvar_amount
    }


def simulate_gbm_with_jumps(initial_price: float, mu: float, sigma: float,
                          jump_intensity: float, jump_mean: float, jump_std: float,
                          days: int, n_simulations: int, 
                          seed: Optional[int] = None) -> np.ndarray:
    """
    Simulate asset prices using Geometric Brownian Motion with Jumps (Merton Jump-Diffusion).
    
    Args:
        initial_price: Initial asset price
        mu: Expected annual return (drift)
        sigma: Annual volatility
        jump_intensity: Average number of jumps per year
        jump_mean: Mean jump size
        jump_std: Standard deviation of jump size
        days: Number of days to simulate
        n_simulations: Number of simulation paths
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (days, n_simulations) with simulated prices
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert annual parameters to daily
    daily_drift = (mu - 0.5 * sigma ** 2) / 252
    daily_diffusion = sigma * np.sqrt(1 / 252)
    daily_jump_intensity = jump_intensity / 252
    
    # Generate random Brownian motion
    brownian_motion = np.random.normal(0, 1, size=(days, n_simulations))
    
    # Generate jump process for each path
    jumps = np.zeros((days, n_simulations))
    
    for sim in range(n_simulations):
        # Poisson process for jump times
        n_jumps = np.random.poisson(daily_jump_intensity * days)
        
        if n_jumps > 0:
            # Randomly distribute jumps across days
            jump_days = np.random.randint(0, days, size=n_jumps)
            
            # Generate jump sizes
            jump_sizes = np.random.normal(jump_mean, jump_std, size=n_jumps)
            
            # Add jumps to the process
            for i, day in enumerate(jump_days):
                jumps[day, sim] += jump_sizes[i]
    
    # Calculate daily returns
    daily_returns = np.exp(daily_drift + daily_diffusion * brownian_motion + jumps)
    
    # Calculate price paths
    price_paths = np.zeros((days, n_simulations))
    price_paths[0] = initial_price
    
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
    
    return price_paths


def run_stress_test(portfolio_weights: Dict[str, float], 
                   historical_data: Dict[str, pd.DataFrame],
                   scenarios: Dict[str, Dict[str, float]],
                   n_simulations: int = 1000,
                   forecast_days: int = 63,  # About 3 months
                   return_column: str = 'Close') -> Dict[str, Dict[str, Any]]:
    """
    Run stress tests on a portfolio under different scenarios.
    
    Args:
        portfolio_weights: Dictionary mapping asset symbols to their weights
        historical_data: Dictionary mapping asset symbols to their historical price DataFrames
        scenarios: Dictionary of scenarios with adjusted parameters
                  Example: {'market_crash': {'mu_factor': 0.5, 'sigma_factor': 2.0}}
        n_simulations: Number of simulation paths
        forecast_days: Number of days to forecast
        return_column: Column to use for return calculations
        
    Returns:
        Dictionary mapping scenario names to simulation results
    """
    # Extract asset returns from historical data
    returns = {}
    for symbol, df in historical_data.items():
        if symbol in portfolio_weights:
            returns[symbol] = df[return_column].pct_change().dropna()
    
    # Calculate baseline mean and std of returns for each asset
    baseline_mu_sigma = {}
    for symbol, ret in returns.items():
        annual_return = np.mean(ret) * 252
        annual_vol = np.std(ret) * np.sqrt(252)
        baseline_mu_sigma[symbol] = (annual_return, annual_vol)
    
    # Run simulations for each scenario
    results = {}
    for scenario_name, scenario_params in scenarios.items():
        print(f"Running stress test for scenario: {scenario_name}")
        
        # Adjust parameters based on scenario
        adjusted_mu_sigma = {}
        for symbol, (mu, sigma) in baseline_mu_sigma.items():
            # Apply adjustments
            adj_mu = mu * scenario_params.get('mu_factor', 1.0)
            adj_sigma = sigma * scenario_params.get('sigma_factor', 1.0)
            adjusted_mu_sigma[symbol] = (adj_mu, adj_sigma)
        
        # Simulate price paths for each asset
        price_paths = {}
        for symbol, weight in portfolio_weights.items():
            if symbol in historical_data:
                initial_price = historical_data[symbol][return_column].iloc[-1]
                mu, sigma = adjusted_mu_sigma[symbol]
                
                # Add jumps if specified
                if 'jump_intensity' in scenario_params:
                    price_paths[symbol] = simulate_gbm_with_jumps(
                        initial_price=initial_price,
                        mu=mu,
                        sigma=sigma,
                        jump_intensity=scenario_params.get('jump_intensity', 0),
                        jump_mean=scenario_params.get('jump_mean', 0),
                        jump_std=scenario_params.get('jump_std', 0.1),
                        days=forecast_days,
                        n_simulations=n_simulations,
                        seed=scenario_params.get('seed', None)
                    )
                else:
                    price_paths[symbol] = simulate_gbm(
                        initial_price=initial_price,
                        mu=mu,
                        sigma=sigma,
                        days=forecast_days,
                        n_simulations=n_simulations,
                        seed=scenario_params.get('seed', None)
                    )
        
        # Calculate portfolio value paths
        portfolio_paths = np.zeros((forecast_days, n_simulations))
        initial_portfolio_value = 10000  # Starting with $10,000 for simplicity
        
        for symbol, weight in portfolio_weights.items():
            initial_asset_value = initial_portfolio_value * weight
            initial_shares = initial_asset_value / historical_data[symbol][return_column].iloc[-1]
            
            asset_value_paths = price_paths[symbol] * initial_shares
            portfolio_paths += asset_value_paths
        
        # Calculate statistics from simulations
        final_values = portfolio_paths[-1, :]
        
        # Percentiles of final portfolio value
        percentiles = {
            "1%": np.percentile(final_values, 1),
            "5%": np.percentile(final_values, 5),
            "10%": np.percentile(final_values, 10),
            "50%": np.percentile(final_values, 50),
            "90%": np.percentile(final_values, 90),
            "95%": np.percentile(final_values, 95),
            "99%": np.percentile(final_values, 99)
        }
        
        # Calculate VaR and CVaR
        var_95 = np.percentile(final_values - initial_portfolio_value, 5)
        cvar_95 = np.mean((final_values - initial_portfolio_value)[final_values - initial_portfolio_value <= var_95])
        
        results[scenario_name] = {
            "portfolio_paths": portfolio_paths,
            "initial_value": initial_portfolio_value,
            "final_values": final_values,
            "percentiles": percentiles,
            "mean_final_value": np.mean(final_values),
            "median_final_value": np.median(final_values),
            "std_final_value": np.std(final_values),
            "var_95": var_95,
            "cvar_95": cvar_95,
            "scenario_params": scenario_params
        }
    
    return results


def compare_stress_test_results(stress_test_results: Dict[str, Dict[str, Any]],
                               figsize: Tuple[int, int] = (15, 10),
                               output_file: Optional[str] = None) -> None:
    """
    Compare and visualize stress test results.
    
    Args:
        stress_test_results: Dictionary mapping scenario names to simulation results
        figsize: Figure size
        output_file: Path to save the output file (if None, display the plot)
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # 1. Plot median portfolio paths for each scenario
    for scenario, results in stress_test_results.items():
        portfolio_paths = results["portfolio_paths"]
        median_path = np.percentile(portfolio_paths, 50, axis=1)
        days = len(median_path)
        
        ax1.plot(range(days), median_path, linewidth=2, label=scenario)
    
    # Add initial value as horizontal line
    initial_value = next(iter(stress_test_results.values()))["initial_value"]
    ax1.axhline(y=initial_value, color='black', linestyle='--', label='Initial Value')
    
    ax1.set_title('Median Portfolio Value by Scenario')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Plot VaR and CVaR comparison
    scenarios = list(stress_test_results.keys())
    var_values = [-results["var_95"] for _, results in stress_test_results.items()]
    cvar_values = [-results["cvar_95"] for _, results in stress_test_results.items()]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax2.bar(x - width/2, var_values, width, label='95% VaR')
    ax2.bar(x + width/2, cvar_values, width, label='95% CVaR')
    
    ax2.set_title('Risk Metrics by Scenario')
    ax2.set_ylabel('Value at Risk ($)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add percentage loss labels on top of bars
    for i, v in enumerate(var_values):
        ax2.text(i - width/2, v + 100, f"{v/initial_value*100:.1f}%", 
                ha='center', va='bottom', fontsize=8)
    
    for i, v in enumerate(cvar_values):
        ax2.text(i + width/2, v + 100, f"{v/initial_value*100:.1f}%", 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def monte_carlo_var(returns: pd.Series, 
                  confidence_level: float = 0.95, 
                  initial_investment: float = 10000,
                  n_simulations: int = 10000,
                  time_horizon: int = 1,
                  seed: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate Value at Risk (VaR) using Monte Carlo simulation.
    
    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.95 for 95% confidence)
        initial_investment: Initial investment amount
        n_simulations: Number of simulations
        time_horizon: Time horizon in days
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with VaR and CVaR values
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate mean and standard deviation of returns
    mu = returns.mean()
    sigma = returns.std()
    
    # Simulate returns
    simulated_returns = np.random.normal(
        mu * time_horizon, 
        sigma * np.sqrt(time_horizon), 
        n_simulations
    )
    
    # Calculate portfolio values
    simulated_values = initial_investment * (1 + simulated_returns)
    
    # Calculate VaR
    var_percentile = 1 - confidence_level
    var_percent = -np.percentile(simulated_returns, var_percentile * 100)
    var_amount = initial_investment * var_percent
    
    # Calculate CVaR
    cvar_percent = -np.mean(simulated_returns[simulated_returns <= -var_percent])
    cvar_amount = initial_investment * cvar_percent
    
    return {
        'var_percent': var_percent,
        'var_amount': var_amount,
        'cvar_percent': cvar_percent,
        'cvar_amount': cvar_amount,
        'simulated_values': simulated_values,
        'simulated_returns': simulated_returns
    }


if __name__ == "__main__":
    # Example usage
    pass 