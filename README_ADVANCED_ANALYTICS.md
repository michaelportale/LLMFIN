# Advanced Analytics for Financial Data

This module provides comprehensive advanced analytics capabilities for financial data, including Monte Carlo simulations, portfolio optimization, sector analysis, and performance attribution.

## Features

### 1. Monte Carlo Simulations

Risk assessment tools using Monte Carlo simulation techniques:

- **Geometric Brownian Motion (GBM)**: Simulate future asset price paths
- **Portfolio Simulations**: Run multi-asset simulations with correlation effects
- **VaR & CVaR Calculations**: Calculate Value at Risk and Conditional Value at Risk
- **Stress Testing**: Run simulations under different market scenarios
- **Jump Diffusion Models**: Account for market shocks in simulations

### 2. Portfolio Optimization

Modern Portfolio Theory implementation for optimal asset allocation:

- **Mean-Variance Optimization**: Construct efficient portfolios
- **Efficient Frontier Generation**: Visualize the risk/return trade-off
- **Different Optimization Goals**: Maximize Sharpe, minimize risk, target return
- **Portfolio Statistics**: Calculate comprehensive performance metrics
- **Constrained Optimization**: Add constraints like min/max weights and risk budgeting

### 3. Sector and Industry Analysis

Tools for analyzing sector performance and rotation:

- **Sector Data Aggregation**: Group stocks by sector and industry
- **Sector Performance**: Track and visualize sector performance
- **Correlation Analysis**: Analyze inter-sector relationships
- **Sector Metrics**: Calculate key performance metrics by sector
- **Sector Rotation**: Track changing leadership among sectors

### 4. Performance Attribution

Methods for attributing portfolio performance:

- **Brinson Attribution**: Allocation, selection, and interaction effects
- **Factor Attribution**: Regression-based attribution to market factors
- **Rolling Attribution**: Analyze performance attribution over time
- **Visual Reports**: Comprehensive visualization of attribution results

## Installation

The Advanced Analytics module is part of the Financial Analytics package. Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Usage

### Monte Carlo Simulations

```python
from scripts.advanced_analytics import simulate_portfolio, plot_monte_carlo_simulation

# Define portfolio weights
portfolio_weights = {'AAPL': 0.3, 'MSFT': 0.3, 'GOOGL': 0.4}

# Run simulation (historical_data should be a dict of DataFrames with price data)
simulation_result = simulate_portfolio(
    initial_weights=portfolio_weights,
    historical_data=historical_data,
    forecast_days=252,  # 1 year forecast
    n_simulations=1000
)

# Plot results
plot_monte_carlo_simulation(simulation_result)

# Run stress tests
from scripts.advanced_analytics import run_stress_test, compare_stress_test_results

# Define scenarios
scenarios = {
    "Base Case": {"mu_factor": 1.0, "sigma_factor": 1.0},
    "Market Crash": {"mu_factor": -0.5, "sigma_factor": 2.0}
}

# Run stress tests
stress_results = run_stress_test(portfolio_weights, historical_data, scenarios)

# Compare results
compare_stress_test_results(stress_results)
```

### Portfolio Optimization

```python
from scripts.advanced_analytics import optimize_portfolio, generate_efficient_frontier

# Optimize portfolio (returns_df should be a DataFrame with asset returns)
optimal_portfolio = optimize_portfolio(
    returns_df, 
    optimization_goal='sharpe'  # 'sharpe', 'min_risk', 'max_return', etc.
)

# Generate and plot efficient frontier
from scripts.advanced_analytics import plot_efficient_frontier

frontier_data = generate_efficient_frontier(returns_df)
plot_efficient_frontier(frontier_data)

# Plot optimal weights
from scripts.advanced_analytics import plot_optimal_portfolio_weights

plot_optimal_portfolio_weights(optimal_portfolio['weights'])
```

### Sector Analysis

```python
from scripts.advanced_analytics import fetch_sector_data, plot_sector_performance

# Fetch sector data
tickers = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'XOM', 'PG']
sector_data = fetch_sector_data(tickers)

# Plot sector performance
plot_sector_performance(sector_data['sector_returns'])

# Calculate and plot sector metrics
from scripts.advanced_analytics import calculate_sector_metrics, plot_sector_metrics

metrics = calculate_sector_metrics(sector_data['sector_returns'])
plot_sector_metrics(metrics)

# Analyze sector rotation
from scripts.advanced_analytics import analyze_sector_rotation, plot_sector_rotation

rotation = analyze_sector_rotation(sector_data['sector_returns'])
plot_sector_rotation(rotation)
```

### Performance Attribution

```python
from scripts.advanced_analytics import calculate_attribution_brinson, plot_attribution

# Calculate attribution (portfolio_weights, benchmark_weights, and returns should be DataFrames)
attribution = calculate_attribution_brinson(
    portfolio_weights,
    benchmark_weights,
    returns
)

# Plot attribution
plot_attribution(attribution)

# Factor attribution
from scripts.advanced_analytics import calculate_factor_attribution, plot_factor_attribution

# factor_returns should be a DataFrame with factor returns
factor_attribution = calculate_factor_attribution(portfolio_returns, factor_returns)
plot_factor_attribution(factor_attribution)
```

## Example Script

See the comprehensive example script at `examples/advanced_analytics_demo.py`, which demonstrates all features:

```bash
# Run all demos
python examples/advanced_analytics_demo.py

# Run specific demo
python examples/advanced_analytics_demo.py --demo monte_carlo
python examples/advanced_analytics_demo.py --demo portfolio
python examples/advanced_analytics_demo.py --demo sector
python examples/advanced_analytics_demo.py --demo attribution
```

## Dependencies

- pandas, numpy: Data manipulation
- matplotlib, seaborn: Visualization
- scipy, scikit-learn: Statistical and machine learning methods
- yfinance: Optional for fetching example financial data

See `requirements.txt` for full dependencies list.

## References

- Brinson, G. P., Hood, L. R., & Beebower, G. L. (1986). Determinants of Portfolio Performance. Financial Analysts Journal, 42(4), 39-44.
- Markowitz, H. (1952). Portfolio Selection. The Journal of Finance, 7(1), 77-91.
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. Journal of Financial Economics, 3(1-2), 125-144.
- Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk. The Journal of Finance, 19(3), 425-442. 