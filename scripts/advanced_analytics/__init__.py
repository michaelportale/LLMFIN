"""
Advanced analytics package for financial data analysis.
"""
from .monte_carlo import (
    simulate_gbm,
    simulate_portfolio,
    plot_monte_carlo_simulation,
    calculate_var_cvar,
    simulate_gbm_with_jumps,
    run_stress_test,
    compare_stress_test_results,
    monte_carlo_var
)

from .portfolio_optimization import (
    calculate_portfolio_performance,
    optimize_portfolio,
    generate_efficient_frontier,
    plot_efficient_frontier,
    plot_optimal_portfolio_weights,
    calculate_portfolio_statistics,
    calculate_max_drawdown,
    optimal_portfolio_with_constraints
)

from .sector_analysis import (
    fetch_sector_data,
    calculate_sector_returns,
    calculate_industry_returns,
    plot_sector_performance,
    plot_sector_correlation,
    plot_sector_volatility,
    calculate_sector_metrics,
    plot_sector_metrics,
    analyze_sector_rotation,
    plot_sector_rotation
)

from .performance_attribution import (
    calculate_returns,
    calculate_active_return,
    calculate_attribution_brinson,
    plot_attribution,
    calculate_factor_attribution,
    plot_factor_attribution,
    attribution_summary_report,
    calculate_rolling_attribution,
    plot_rolling_attribution
)

__all__ = [
    # Monte Carlo simulations
    'simulate_gbm',
    'simulate_portfolio',
    'plot_monte_carlo_simulation',
    'calculate_var_cvar',
    'simulate_gbm_with_jumps',
    'run_stress_test',
    'compare_stress_test_results',
    'monte_carlo_var',
    
    # Portfolio optimization
    'calculate_portfolio_performance',
    'optimize_portfolio',
    'generate_efficient_frontier',
    'plot_efficient_frontier',
    'plot_optimal_portfolio_weights',
    'calculate_portfolio_statistics',
    'calculate_max_drawdown',
    'optimal_portfolio_with_constraints',
    
    # Sector analysis
    'fetch_sector_data',
    'calculate_sector_returns',
    'calculate_industry_returns',
    'plot_sector_performance',
    'plot_sector_correlation',
    'plot_sector_volatility',
    'calculate_sector_metrics',
    'plot_sector_metrics',
    'analyze_sector_rotation',
    'plot_sector_rotation',
    
    # Performance attribution
    'calculate_returns',
    'calculate_active_return',
    'calculate_attribution_brinson',
    'plot_attribution',
    'calculate_factor_attribution',
    'plot_factor_attribution',
    'attribution_summary_report',
    'calculate_rolling_attribution',
    'plot_rolling_attribution'
] 