"""
Feature engineering package for financial data analysis.
"""
from .technical_indicators import (
    add_moving_averages,
    add_exponential_moving_averages,
    add_macd,
    add_rsi,
    add_bollinger_bands,
    add_stochastic_oscillator,
    add_average_true_range,
    add_fibonacci_retracement_levels,
    add_ichimoku_cloud,
    add_on_balance_volume,
    add_accumulation_distribution,
    add_average_directional_index,
    add_vwap,
    add_money_flow_index,
    add_all_indicators,
    calculate_price_returns,
    generate_technical_features
)

from .feature_importance import (
    calculate_correlation_importance,
    calculate_mutual_information,
    calculate_random_forest_importance,
    calculate_permutation_importance,
    combined_feature_importance,
    plot_feature_importance,
    get_top_n_features,
    feature_correlation_heatmap,
    feature_target_correlation_analysis,
    run_feature_importance_analysis
)

from .fundamental_data import (
    FundamentalDataFetcher,
    calculate_financial_ratios,
    combine_price_and_fundamental_data,
    calculate_fundamental_features,
    get_sector_industry_data,
    add_sector_industry_features,
    add_fundamental_features
)

from .feature_visualization import (
    plot_time_series,
    plot_technical_indicators,
    plot_feature_distributions,
    plot_feature_correlations,
    plot_feature_importance,
    plot_pairplot,
    visualize_features,
    plot_seasonal_decomposition,
    create_feature_dashboard
)

__all__ = [
    # Technical indicators
    'add_moving_averages',
    'add_exponential_moving_averages',
    'add_macd',
    'add_rsi',
    'add_bollinger_bands',
    'add_stochastic_oscillator',
    'add_average_true_range',
    'add_fibonacci_retracement_levels',
    'add_ichimoku_cloud',
    'add_on_balance_volume',
    'add_accumulation_distribution',
    'add_average_directional_index',
    'add_vwap',
    'add_money_flow_index',
    'add_all_indicators',
    'calculate_price_returns',
    'generate_technical_features',
    
    # Feature importance
    'calculate_correlation_importance',
    'calculate_mutual_information',
    'calculate_random_forest_importance',
    'calculate_permutation_importance',
    'combined_feature_importance',
    'plot_feature_importance',
    'get_top_n_features',
    'feature_correlation_heatmap',
    'feature_target_correlation_analysis',
    'run_feature_importance_analysis',
    
    # Fundamental data
    'FundamentalDataFetcher',
    'calculate_financial_ratios',
    'combine_price_and_fundamental_data',
    'calculate_fundamental_features',
    'get_sector_industry_data',
    'add_sector_industry_features',
    'add_fundamental_features',
    
    # Visualization
    'plot_time_series',
    'plot_technical_indicators',
    'plot_feature_distributions',
    'plot_feature_correlations',
    'plot_feature_importance',
    'plot_pairplot',
    'visualize_features',
    'plot_seasonal_decomposition',
    'create_feature_dashboard'
] 