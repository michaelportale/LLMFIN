#!/usr/bin/env python3
"""
Example script demonstrating the advanced analytics capabilities.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime, timedelta
import argparse

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import data sources
from scripts.data_sources import YahooFinanceDataSource

# Import advanced analytics modules
from scripts.advanced_analytics.monte_carlo import (
    simulate_gbm, simulate_portfolio, plot_monte_carlo_simulation, 
    run_stress_test, compare_stress_test_results
)

from scripts.advanced_analytics.portfolio_optimization import (
    optimize_portfolio, generate_efficient_frontier, plot_efficient_frontier,
    plot_optimal_portfolio_weights, calculate_portfolio_statistics
)

from scripts.advanced_analytics.sector_analysis import (
    fetch_sector_data, plot_sector_performance, plot_sector_correlation,
    calculate_sector_metrics, plot_sector_metrics, analyze_sector_rotation,
    plot_sector_rotation
)

from scripts.advanced_analytics.performance_attribution import (
    calculate_returns, calculate_attribution_brinson, plot_attribution,
    calculate_factor_attribution, plot_factor_attribution
)


def demo_monte_carlo(output_dir: str = 'data/advanced_analytics_demo'):
    """
    Demonstrate Monte Carlo simulations for risk assessment.
    
    Args:
        output_dir: Directory to save output files
    """
    print("\n" + "="*80)
    print("MONTE CARLO SIMULATION DEMO")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data source
    data_source = YahooFinanceDataSource(save_path=os.path.join(output_dir, 'price_data'))
    
    # Fetch data for a set of stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching price data for {', '.join(symbols)}...")
    data = data_source.fetch_historical_data(symbols, start_date=start_date, end_date=end_date)
    
    # Define portfolio weights
    portfolio_weights = {
        'AAPL': 0.25,
        'MSFT': 0.25,
        'GOOGL': 0.20,
        'AMZN': 0.15,
        'META': 0.15
    }
    
    # Run Monte Carlo simulation
    print("Running Monte Carlo simulation for portfolio...")
    simulation_result = simulate_portfolio(
        initial_weights=portfolio_weights,
        historical_data={symbol: data[symbol] for symbol in symbols if symbol in data},
        forecast_days=252,  # 1 year forecast
        n_simulations=1000,
        random_state=42
    )
    
    # Plot simulation results
    print("Plotting Monte Carlo simulation results...")
    plot_monte_carlo_simulation(
        simulation_result,
        title=f"Monte Carlo Simulation: {', '.join(symbols)} Portfolio",
        output_file=os.path.join(output_dir, "monte_carlo_simulation.png")
    )
    
    # Run stress tests
    print("Running stress tests for different scenarios...")
    
    # Define stress test scenarios
    scenarios = {
        "Base Case": {
            "mu_factor": 1.0,
            "sigma_factor": 1.0
        },
        "Market Crash": {
            "mu_factor": -0.5,
            "sigma_factor": 2.0,
            "jump_intensity": 5,
            "jump_mean": -0.05,  # Negative jumps
            "jump_std": 0.02
        },
        "Bull Market": {
            "mu_factor": 1.5,
            "sigma_factor": 0.8
        },
        "High Volatility": {
            "mu_factor": 1.0,
            "sigma_factor": 1.8
        },
        "Stagflation": {
            "mu_factor": 0.3,
            "sigma_factor": 1.3,
            "jump_intensity": 3,
            "jump_mean": -0.02,
            "jump_std": 0.015
        }
    }
    
    stress_test_results = run_stress_test(
        portfolio_weights=portfolio_weights,
        historical_data={symbol: data[symbol] for symbol in symbols if symbol in data},
        scenarios=scenarios,
        n_simulations=500,
        forecast_days=126  # ~6 months
    )
    
    # Compare stress test results
    print("Comparing stress test results...")
    compare_stress_test_results(
        stress_test_results,
        title="Portfolio Stress Test Comparison",
        output_file=os.path.join(output_dir, "stress_test_comparison.png")
    )
    
    # Report key metrics
    print("\nKey Risk Metrics:")
    for scenario, result in stress_test_results.items():
        print(f"  {scenario}:")
        print(f"    Expected Final Value: ${result['mean_final_value']:.2f}")
        print(f"    95% VaR: ${-result['var_95']:.2f} ({-result['var_95']/result['initial_value']*100:.1f}%)")
        print(f"    95% CVaR: ${-result['cvar_95']:.2f} ({-result['cvar_95']/result['initial_value']*100:.1f}%)")
    
    return simulation_result, stress_test_results


def demo_portfolio_optimization(output_dir: str = 'data/advanced_analytics_demo'):
    """
    Demonstrate portfolio optimization using Modern Portfolio Theory.
    
    Args:
        output_dir: Directory to save output files
    """
    print("\n" + "="*80)
    print("PORTFOLIO OPTIMIZATION DEMO")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data source
    data_source = YahooFinanceDataSource(save_path=os.path.join(output_dir, 'price_data'))
    
    # Fetch data for a set of stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'JNJ', 'PG', 'XOM', 'KO']
    start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching price data for {len(symbols)} stocks...")
    data = data_source.fetch_historical_data(symbols, start_date=start_date, end_date=end_date)
    
    # Extract Close prices
    prices = {}
    for symbol in symbols:
        if symbol in data:
            prices[symbol] = data[symbol]['Close']
    
    # Create a DataFrame with prices
    price_df = pd.DataFrame(prices)
    
    # Calculate returns
    returns_df = price_df.pct_change().dropna()
    
    print("Optimizing portfolio with different objectives...")
    
    # Run optimization with different objectives
    portfolios = {}
    
    objectives = ['sharpe', 'min_risk', 'max_return']
    for objective in objectives:
        print(f"  Optimizing for {objective}...")
        result = optimize_portfolio(returns_df, optimization_goal=objective)
        portfolios[objective] = result
        
        weights = result['weights']
        print(f"    Expected Return: {result['expected_return']*100:.2f}%")
        print(f"    Volatility: {result['volatility']*100:.2f}%")
        print(f"    Sharpe Ratio: {result['sharpe_ratio']:.4f}")
        print(f"    Top 3 allocations: " + 
              ", ".join([f"{asset}: {weight*100:.1f}%" 
                        for asset, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]]))
    
    # Generate efficient frontier
    print("Generating efficient frontier...")
    efficient_frontier = generate_efficient_frontier(
        returns_df,
        n_points=100,
        risk_free_rate=0.0,  # Assuming 0% risk-free rate for simplicity
        include_assets=True,
        include_monte_carlo=True,
        monte_carlo_n=500
    )
    
    # Plot efficient frontier
    print("Plotting efficient frontier...")
    plot_efficient_frontier(
        efficient_frontier,
        title="Efficient Frontier",
        output_file=os.path.join(output_dir, "efficient_frontier.png")
    )
    
    # Plot optimal portfolio weights
    print("Plotting optimal portfolio weights...")
    plot_optimal_portfolio_weights(
        portfolios['sharpe']['weights'],
        title="Optimal Portfolio Weights (Max Sharpe)",
        output_file=os.path.join(output_dir, "optimal_portfolio_weights.png")
    )
    
    # Calculate portfolio statistics
    print("Calculating portfolio statistics...")
    stats = calculate_portfolio_statistics(
        portfolios['sharpe']['weights'],
        returns_df
    )
    
    print("\nOptimal Portfolio Statistics:")
    print(f"  Annual Return: {stats['expected_annual_return']:.2f}%")
    print(f"  Annual Volatility: {stats['annual_volatility']:.2f}%")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
    print(f"  Maximum Drawdown: {stats['max_drawdown']:.2f}%")
    print(f"  Value at Risk (95%): {stats['var_95']*100:.2f}%")
    print(f"  Positive Days: {stats['positive_days_pct']:.1f}%")
    
    return portfolios, efficient_frontier, stats


def demo_sector_analysis(output_dir: str = 'data/advanced_analytics_demo'):
    """
    Demonstrate sector and industry analysis tools.
    
    Args:
        output_dir: Directory to save output files
    """
    print("\n" + "="*80)
    print("SECTOR ANALYSIS DEMO")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define tickers from different sectors
    tickers = [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV',
        # Financial
        'JPM', 'BAC', 'WFC', 'GS', 'MS',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',
        # Consumer Staples
        'PG', 'KO', 'PEP', 'WMT', 'COST'
    ]
    
    print(f"Fetching sector data for {len(tickers)} stocks...")
    
    # Fetch sector data
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    sector_data = fetch_sector_data(tickers, start_date=start_date, end_date=end_date)
    
    # Extract sector returns
    sector_returns = sector_data['sector_returns']
    
    # Plot sector performance
    print("Plotting sector performance...")
    plot_sector_performance(
        sector_returns,
        period='all',
        title="Sector Performance",
        output_file=os.path.join(output_dir, "sector_performance.png")
    )
    
    # Plot sector correlation
    print("Plotting sector correlation...")
    plot_sector_correlation(
        sector_returns,
        title="Sector Return Correlation",
        output_file=os.path.join(output_dir, "sector_correlation.png")
    )
    
    # Calculate sector metrics
    print("Calculating sector metrics...")
    sector_metrics = calculate_sector_metrics(sector_returns)
    
    # Plot sector metrics
    print("Plotting sector metrics...")
    plot_sector_metrics(
        sector_metrics,
        title="Sector Performance Metrics",
        output_file=os.path.join(output_dir, "sector_metrics.png")
    )
    
    # Analyze sector rotation
    print("Analyzing sector rotation...")
    sector_rotation = analyze_sector_rotation(
        sector_returns,
        window=63,  # ~3 months
        top_n=3  # Track top 3 sectors
    )
    
    # Plot sector rotation
    print("Plotting sector rotation...")
    plot_sector_rotation(
        sector_rotation,
        title="Sector Rotation Analysis (Rolling 3-Month Returns)",
        output_file=os.path.join(output_dir, "sector_rotation.png")
    )
    
    # Print sector metrics
    print("\nSector Performance Metrics:")
    print(sector_metrics[['annualized_return', 'annualized_volatility', 'sharpe_ratio', 'max_drawdown']].round(2))
    
    return sector_data, sector_metrics, sector_rotation


def demo_performance_attribution(output_dir: str = 'data/advanced_analytics_demo'):
    """
    Demonstrate performance attribution analysis.
    
    Args:
        output_dir: Directory to save output files
    """
    print("\n" + "="*80)
    print("PERFORMANCE ATTRIBUTION DEMO")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data source
    data_source = YahooFinanceDataSource(save_path=os.path.join(output_dir, 'price_data'))
    
    # Fetch data for portfolio assets
    portfolio_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'JNJ', 'PG', 'XOM', 'KO']
    
    # Fetch data for benchmark - using S&P 500 ETF (SPY) as benchmark
    benchmark_symbols = ['SPY']
    
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching price data for {len(portfolio_symbols)} portfolio assets and benchmark...")
    
    portfolio_data = data_source.fetch_historical_data(portfolio_symbols, start_date=start_date, end_date=end_date)
    benchmark_data = data_source.fetch_historical_data(benchmark_symbols, start_date=start_date, end_date=end_date)
    
    # Extract asset prices
    asset_prices = {}
    for symbol in portfolio_symbols:
        if symbol in portfolio_data:
            asset_prices[symbol] = portfolio_data[symbol]['Close']
    
    # Create DataFrame with prices
    prices_df = pd.DataFrame(asset_prices)
    
    # Calculate returns
    returns_df = calculate_returns(prices_df)
    
    # Calculate benchmark returns
    benchmark_returns = calculate_returns(pd.DataFrame({
        'SPY': benchmark_data['SPY']['Close']
    }))['SPY']
    
    # Define portfolio weights over time (using a simplified approach with static weights)
    # In a real scenario, these would change based on price movements and rebalancing
    portfolio_weights = pd.DataFrame({
        'AAPL': [0.15] * len(returns_df),
        'MSFT': [0.15] * len(returns_df),
        'GOOGL': [0.10] * len(returns_df),
        'AMZN': [0.10] * len(returns_df),
        'META': [0.10] * len(returns_df),
        'JPM': [0.08] * len(returns_df),
        'JNJ': [0.08] * len(returns_df),
        'PG': [0.08] * len(returns_df),
        'XOM': [0.08] * len(returns_df),
        'KO': [0.08] * len(returns_df)
    }, index=returns_df.index)
    
    # Define benchmark weights (simplified example - in real S&P 500, these would be based on market cap)
    benchmark_weights = pd.DataFrame({
        'AAPL': [0.06] * len(returns_df),
        'MSFT': [0.06] * len(returns_df),
        'GOOGL': [0.04] * len(returns_df),
        'AMZN': [0.03] * len(returns_df),
        'META': [0.02] * len(returns_df),
        'JPM': [0.01] * len(returns_df),
        'JNJ': [0.01] * len(returns_df),
        'PG': [0.01] * len(returns_df),
        'XOM': [0.01] * len(returns_df),
        'KO': [0.01] * len(returns_df)
    }, index=returns_df.index)
    
    # Normalize benchmark weights to sum to 1 (the rest would be other S&P 500 stocks)
    benchmark_weight_sum = benchmark_weights.sum(axis=1).iloc[0]
    for col in benchmark_weights.columns:
        benchmark_weights[col] = benchmark_weights[col] / benchmark_weight_sum
    
    # Calculate Brinson attribution
    print("Calculating Brinson performance attribution...")
    attribution_data = calculate_attribution_brinson(
        portfolio_weights, benchmark_weights, returns_df
    )
    
    # Plot attribution
    print("Plotting attribution results...")
    plot_attribution(
        attribution_data,
        title="Performance Attribution Analysis",
        output_file=os.path.join(output_dir, "performance_attribution.png")
    )
    
    # Define some market factors for factor attribution
    print("Creating market factors for factor attribution...")
    
    # Create simple market factor (SPY returns)
    market_factor = benchmark_returns
    
    # Create size factor (small minus big) - simplified example
    size_factor = pd.Series(np.random.normal(0, 0.01, len(returns_df)), index=returns_df.index)
    
    # Create value factor (value minus growth) - simplified example
    value_factor = pd.Series(np.random.normal(0, 0.01, len(returns_df)), index=returns_df.index)
    
    # Create momentum factor - simplified example
    momentum_factor = pd.Series(np.random.normal(0, 0.01, len(returns_df)), index=returns_df.index)
    
    # Combine factors into a DataFrame
    factors_df = pd.DataFrame({
        'Market': market_factor,
        'Size': size_factor,
        'Value': value_factor,
        'Momentum': momentum_factor
    })
    
    # Calculate portfolio returns
    portfolio_returns = (returns_df * portfolio_weights).sum(axis=1)
    
    # Calculate factor attribution
    print("Calculating factor attribution...")
    factor_attribution = calculate_factor_attribution(
        portfolio_returns, factors_df
    )
    
    # Plot factor attribution
    print("Plotting factor attribution...")
    plot_factor_attribution(
        factor_attribution,
        title="Factor Attribution Analysis",
        output_file=os.path.join(output_dir, "factor_attribution.png")
    )
    
    # Generate a comprehensive attribution report
    print("Generating attribution summary report...")
    attribution_summary = attribution_summary_report(
        attribution_data,
        factor_attribution,
        output_file=os.path.join(output_dir, "attribution_summary.csv")
    )
    
    # Print attribution results
    print("\nAttribution Summary:")
    summary = attribution_data['summary']
    print(f"  Total Active Return: {summary['Total_Active_Return'].sum()*100:.2f}%")
    print(f"  Allocation Effect: {summary['Allocation_Effect'].sum()*100:.2f}%")
    print(f"  Selection Effect: {summary['Selection_Effect'].sum()*100:.2f}%")
    print(f"  Interaction Effect: {summary['Interaction_Effect'].sum()*100:.2f}%")
    
    print("\nFactor Attribution:")
    print(f"  R-squared: {factor_attribution['r_squared']:.4f}")
    print(f"  Information Ratio: {factor_attribution['information_ratio']:.4f}")
    print("  Factor Exposures:")
    for factor, exposure in factor_attribution['factor_exposures'].items():
        print(f"    {factor}: {exposure:.4f}")
    
    return attribution_data, factor_attribution, attribution_summary


def main():
    """
    Main function to run the advanced analytics demo.
    """
    parser = argparse.ArgumentParser(description="Advanced Analytics demo script")
    parser.add_argument("--demo", 
                       choices=["monte_carlo", "portfolio", "sector", "attribution", "all"], 
                       default="all", 
                       help="Which demo to run")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = 'data/advanced_analytics_demo'
    os.makedirs(output_dir, exist_ok=True)
    
    if args.demo in ["monte_carlo", "all"]:
        demo_monte_carlo(output_dir)
    
    if args.demo in ["portfolio", "all"]:
        demo_portfolio_optimization(output_dir)
    
    if args.demo in ["sector", "all"]:
        demo_sector_analysis(output_dir)
    
    if args.demo in ["attribution", "all"]:
        demo_performance_attribution(output_dir)
    
    if args.demo == "all":
        print("\n" + "="*80)
        print("ADVANCED ANALYTICS DEMOS COMPLETED")
        print("="*80)
        print(f"All output files have been saved to {output_dir}")


if __name__ == "__main__":
    main() 