#!/usr/bin/env python3
"""
Example script demonstrating the feature engineering capabilities.
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

# Import feature engineering modules
from scripts.feature_engineering.technical_indicators import (
    add_moving_averages, add_exponential_moving_averages, add_rsi, add_macd, 
    add_bollinger_bands, add_stochastic_oscillator, add_average_true_range,
    add_on_balance_volume, add_average_directional_index, add_all_indicators,
    calculate_price_returns, generate_technical_features
)

from scripts.feature_engineering.feature_importance import (
    calculate_correlation_importance, calculate_mutual_information,
    calculate_random_forest_importance, combined_feature_importance,
    plot_feature_importance, feature_correlation_heatmap,
    feature_target_correlation_analysis, run_feature_importance_analysis
)

from scripts.feature_engineering.fundamental_data import (
    FundamentalDataFetcher, calculate_financial_ratios, 
    combine_price_and_fundamental_data, calculate_fundamental_features,
    get_sector_industry_data, add_sector_industry_features
)

from scripts.feature_engineering.feature_visualization import (
    plot_time_series, plot_technical_indicators, plot_feature_distributions,
    plot_feature_correlations, plot_feature_importance, plot_pairplot,
    visualize_features, plot_seasonal_decomposition, create_feature_dashboard
)


def demo_technical_indicators(output_dir: str = 'data/feature_engineering_demo'):
    """
    Demonstrate technical indicators feature engineering.
    
    Args:
        output_dir: Directory to save output files
    """
    print("\n" + "="*80)
    print("TECHNICAL INDICATORS DEMO")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data source
    data_source = YahooFinanceDataSource(save_path=os.path.join(output_dir, 'price_data'))
    
    # Fetch data for a stock
    symbols = ['AAPL']
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching price data for {symbols[0]}...")
    data = data_source.fetch_historical_data(symbols, start_date=start_date, end_date=end_date)
    
    if symbols[0] not in data:
        print(f"Error: Could not fetch data for {symbols[0]}")
        return
    
    df = data[symbols[0]]
    
    # Add technical indicators
    print("Adding technical indicators...")
    
    # 1. Add moving averages
    df = add_moving_averages(df, periods=[10, 20, 50, 200])
    
    # 2. Add exponential moving averages
    df = add_exponential_moving_averages(df, periods=[10, 20, 50, 200])
    
    # 3. Add RSI
    df = add_rsi(df)
    
    # 4. Add MACD
    df = add_macd(df)
    
    # 5. Add Bollinger Bands
    df = add_bollinger_bands(df)
    
    # 6. Add Stochastic Oscillator
    df = add_stochastic_oscillator(df)
    
    # 7. Add ATR
    df = add_average_true_range(df)
    
    # 8. Add OBV
    df = add_on_balance_volume(df)
    
    # 9. Add ADX
    df = add_average_directional_index(df)
    
    # 10. Calculate price returns
    df = calculate_price_returns(df, periods=[1, 5, 10, 20])
    
    # Save the data with indicators
    indicators_file = os.path.join(output_dir, f"{symbols[0]}_with_indicators.csv")
    df.to_csv(indicators_file)
    print(f"Saved data with indicators to {indicators_file}")
    
    # Plot technical indicators
    print("Plotting technical indicators...")
    plot_technical_indicators(
        df, 
        output_file=os.path.join(output_dir, f"{symbols[0]}_technical_indicators.png")
    )
    
    return df


def demo_feature_importance(df: pd.DataFrame, target_column: str = 'Return_5d', 
                          output_dir: str = 'data/feature_engineering_demo'):
    """
    Demonstrate feature importance analysis.
    
    Args:
        df: DataFrame with technical indicators
        target_column: Target column for feature importance analysis
        output_dir: Directory to save output files
    """
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS DEMO")
    print("="*80)
    
    # Create output directory
    feature_importance_dir = os.path.join(output_dir, 'feature_importance')
    os.makedirs(feature_importance_dir, exist_ok=True)
    
    # Ensure we have a clean dataset for analysis
    analysis_df = df.copy().dropna()
    
    # Calculate feature importance using different methods
    print("Calculating correlation importance...")
    corr_importance = calculate_correlation_importance(analysis_df, target_column)
    print(f"Top 5 features by correlation: \n{corr_importance.head(5)}")
    
    print("\nCalculating mutual information...")
    mi_importance = calculate_mutual_information(analysis_df, target_column)
    print(f"Top 5 features by mutual information: \n{mi_importance.head(5)}")
    
    print("\nCalculating random forest importance...")
    rf_importance = calculate_random_forest_importance(analysis_df, target_column)
    print(f"Top 5 features by random forest: \n{rf_importance.head(5)}")
    
    print("\nCalculating combined feature importance...")
    combined = combined_feature_importance(analysis_df, target_column)
    print(f"Top 5 features by combined score: \n{combined.head(5)}")
    
    # Visualize feature importance
    print("Visualizing feature importance...")
    plot_feature_importance(
        combined, 
        score_column='combined_score',
        title=f'Combined Feature Importance (Target: {target_column})',
        output_file=os.path.join(feature_importance_dir, 'combined_importance.png')
    )
    
    # Create correlation heatmap
    print("Creating correlation heatmap...")
    top_features = combined.head(10)['feature'].tolist()
    feature_correlation_heatmap(
        analysis_df,
        features=top_features + [target_column],
        output_file=os.path.join(feature_importance_dir, 'correlation_heatmap.png')
    )
    
    # Analyze target correlations
    print("Analyzing feature-target correlations...")
    feature_target_correlation_analysis(
        analysis_df,
        target_column=target_column,
        top_n=8,
        output_file=os.path.join(feature_importance_dir, 'target_correlations.png')
    )
    
    # Run comprehensive feature importance analysis
    print("\nRunning comprehensive feature importance analysis...")
    results = run_feature_importance_analysis(
        analysis_df,
        target_column=target_column,
        output_dir=feature_importance_dir,
        n_features=15
    )
    
    return results


def demo_fundamental_data(symbols: List[str] = ['AAPL', 'MSFT', 'GOOG'], 
                         output_dir: str = 'data/feature_engineering_demo'):
    """
    Demonstrate fundamental data integration.
    
    Args:
        symbols: List of stock symbols to analyze
        output_dir: Directory to save output files
    """
    print("\n" + "="*80)
    print("FUNDAMENTAL DATA INTEGRATION DEMO")
    print("="*80)
    
    # Create output directory
    fundamental_dir = os.path.join(output_dir, 'fundamental_data')
    os.makedirs(fundamental_dir, exist_ok=True)
    
    # Initialize fundamental data fetcher
    fetcher = FundamentalDataFetcher(save_path=fundamental_dir)
    
    # Fetch key metrics
    print("Fetching key metrics...")
    key_metrics = fetcher.fetch_key_metrics_yf(symbols)
    
    if not key_metrics.empty:
        print(f"Key metrics: \n{key_metrics.head()}")
    
    # Fetch sector and industry data
    print("\nFetching sector and industry data...")
    sector_data = get_sector_industry_data(symbols)
    
    if not sector_data.empty:
        print(f"Sector data: \n{sector_data}")
    
    # Fetch price data
    print("\nFetching price data...")
    data_source = YahooFinanceDataSource(save_path=os.path.join(output_dir, 'price_data'))
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    price_data = data_source.fetch_historical_data(symbols, start_date=start_date, end_date=end_date)
    
    # Create price data dictionary
    price_dict = {}
    for symbol in symbols:
        if symbol in price_data:
            price_dict[symbol] = price_data[symbol]
    
    # Create fundamental data dictionary
    fundamental_data = {'key_metrics': key_metrics}
    
    # Combine price and fundamental data
    print("\nCombining price and fundamental data...")
    combined_data = combine_price_and_fundamental_data(price_dict, fundamental_data, resample='M')
    
    # Calculate additional fundamental features for one symbol
    if symbols[0] in combined_data:
        print(f"\nCalculating fundamental features for {symbols[0]}...")
        fundamental_features = calculate_fundamental_features(combined_data[symbols[0]])
        
        # Save fundamental features
        features_file = os.path.join(fundamental_dir, f"{symbols[0]}_fundamental_features.csv")
        fundamental_features.to_csv(features_file)
        print(f"Saved fundamental features to {features_file}")
        
        return fundamental_features
    
    return None


def demo_feature_visualization(df: pd.DataFrame, target_column: str = 'Return_5d',
                             output_dir: str = 'data/feature_engineering_demo'):
    """
    Demonstrate feature visualization tools.
    
    Args:
        df: DataFrame with features
        target_column: Target column for supervised learning
        output_dir: Directory to save output files
    """
    print("\n" + "="*80)
    print("FEATURE VISUALIZATION DEMO")
    print("="*80)
    
    # Create output directory
    visualization_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Ensure we have a clean dataset for visualization
    viz_df = df.copy().dropna()
    
    # Calculate correlation importance to get top features
    importance = calculate_correlation_importance(viz_df, target_column)
    top_features = importance.head(10)['feature'].tolist()
    
    # 1. Plot time series
    print("Plotting time series...")
    plot_time_series(
        viz_df,
        columns=['Close'] + top_features[:3],
        title='Price and Top Features',
        output_file=os.path.join(visualization_dir, 'time_series.png')
    )
    
    # 2. Plot technical indicators (already done in demo_technical_indicators)
    
    # 3. Plot feature distributions
    print("Plotting feature distributions...")
    plot_feature_distributions(
        viz_df,
        features=top_features[:9],
        output_file=os.path.join(visualization_dir, 'distributions.png')
    )
    
    # 4. Plot feature correlations
    print("Plotting feature correlations...")
    plot_feature_correlations(
        viz_df,
        target_column=target_column,
        top_n=10,
        output_file=os.path.join(visualization_dir, 'correlations.png')
    )
    
    # 5. Plot pairplot
    print("Plotting feature pairplot...")
    plot_pairplot(
        viz_df,
        features=top_features[:5] + [target_column],
        hue=None,
        output_file=os.path.join(visualization_dir, 'pairplot.png')
    )
    
    # 6. Plot seasonal decomposition
    print("Plotting seasonal decomposition...")
    try:
        plot_seasonal_decomposition(
            viz_df,
            column='Close',
            period=252,  # Approximately 1 trading year
            output_file=os.path.join(visualization_dir, 'seasonal_decomposition.png')
        )
    except Exception as e:
        print(f"Error in seasonal decomposition: {str(e)}")
    
    # 7. Create interactive dashboard
    print("Creating interactive dashboard...")
    try:
        create_feature_dashboard(
            viz_df,
            target_column=target_column,
            output_file=os.path.join(visualization_dir, 'dashboard.html')
        )
        print(f"Interactive dashboard saved to {os.path.join(visualization_dir, 'dashboard.html')}")
    except ImportError:
        print("Plotly is required for interactive dashboards. Install with: pip install plotly")
    
    # 8. Run comprehensive feature visualization
    print("\nRunning comprehensive feature visualization...")
    visualize_features(
        viz_df,
        target_column=target_column,
        top_n_features=10,
        output_dir=visualization_dir,
        prefix='comprehensive_'
    )


def demo_combined_features(output_dir: str = 'data/feature_engineering_demo'):
    """
    Demonstrate combined feature engineering workflow.
    
    Args:
        output_dir: Directory to save output files
    """
    print("\n" + "="*80)
    print("COMBINED FEATURE ENGINEERING WORKFLOW DEMO")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Fetch price data
    symbols = ['AAPL']
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching price data for {symbols[0]}...")
    data_source = YahooFinanceDataSource(save_path=os.path.join(output_dir, 'price_data'))
    data = data_source.fetch_historical_data(symbols, start_date=start_date, end_date=end_date)
    
    if symbols[0] not in data:
        print(f"Error: Could not fetch data for {symbols[0]}")
        return
    
    df = data[symbols[0]]
    
    # 2. Add all technical indicators
    print("Adding all technical indicators...")
    df_with_indicators = add_all_indicators(df)
    
    # 3. Calculate price returns
    print("Calculating price returns...")
    df_with_returns = calculate_price_returns(df_with_indicators)
    
    # 4. Integrate fundamental data
    print("Integrating fundamental data...")
    fetcher = FundamentalDataFetcher(save_path=os.path.join(output_dir, 'fundamental_data'))
    try:
        # This will fetch fundamental data and combine it with the price data
        # Note: This step may be slow and hit API rate limits
        df_with_fundamental = add_fundamental_features(
            df_with_returns,
            fetcher,
            symbols=symbols
        )
    except Exception as e:
        print(f"Error integrating fundamental data: {str(e)}")
        df_with_fundamental = df_with_returns
    
    # 5. Analyze feature importance
    print("Analyzing feature importance...")
    # Choose a target (return over 5 days)
    target_column = 'Return_5d'
    
    # Clean data for analysis
    analysis_df = df_with_fundamental.copy().dropna()
    
    # Run feature importance analysis
    feature_importance_dir = os.path.join(output_dir, 'combined_feature_analysis')
    os.makedirs(feature_importance_dir, exist_ok=True)
    
    importance_results = run_feature_importance_analysis(
        analysis_df,
        target_column=target_column,
        output_dir=feature_importance_dir,
        n_features=20
    )
    
    # 6. Save the processed dataset
    final_file = os.path.join(output_dir, f"{symbols[0]}_full_features.csv")
    analysis_df.to_csv(final_file)
    print(f"Saved full feature dataset to {final_file}")
    
    return {
        'data': analysis_df,
        'importance_results': importance_results,
        'top_features': importance_results['top_features'] if 'top_features' in importance_results else []
    }


def main():
    """
    Main function to run the feature engineering demo.
    """
    parser = argparse.ArgumentParser(description="Feature engineering demo script")
    parser.add_argument("--demo", 
                      choices=["technical", "importance", "fundamental", "visualization", "combined", "all"], 
                      default="all", 
                      help="Which demo to run")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = 'data/feature_engineering_demo'
    os.makedirs(output_dir, exist_ok=True)
    
    if args.demo in ["technical", "all"]:
        df_with_indicators = demo_technical_indicators(output_dir)
    
    if args.demo in ["importance", "all"] and 'df_with_indicators' in locals():
        demo_feature_importance(df_with_indicators, output_dir=output_dir)
    
    if args.demo in ["fundamental", "all"]:
        fundamental_features = demo_fundamental_data(output_dir=output_dir)
    
    if args.demo in ["visualization", "all"] and 'df_with_indicators' in locals():
        demo_feature_visualization(df_with_indicators, output_dir=output_dir)
    
    if args.demo in ["combined", "all"]:
        results = demo_combined_features(output_dir)
        
        if results and 'top_features' in results:
            print("\nTop features for predicting 5-day returns:")
            for i, feature in enumerate(results['top_features'][:10], 1):
                print(f"{i}. {feature}")


if __name__ == "__main__":
    main() 