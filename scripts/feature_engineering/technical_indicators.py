"""
Technical indicators for financial data analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple


def add_moving_averages(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50, 200], 
                       column: str = 'Close') -> pd.DataFrame:
    """
    Add Simple Moving Averages (SMA) to a DataFrame.
    
    Args:
        df: DataFrame with price data
        periods: List of periods for calculating moving averages
        column: Column to calculate moving averages on
        
    Returns:
        DataFrame with added moving averages
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    result = df.copy()
    
    for period in periods:
        result[f'SMA_{period}'] = df[column].rolling(window=period).mean()
    
    return result


def add_exponential_moving_averages(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50, 200], 
                                   column: str = 'Close') -> pd.DataFrame:
    """
    Add Exponential Moving Averages (EMA) to a DataFrame.
    
    Args:
        df: DataFrame with price data
        periods: List of periods for calculating EMAs
        column: Column to calculate EMAs on
        
    Returns:
        DataFrame with added EMAs
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    result = df.copy()
    
    for period in periods:
        result[f'EMA_{period}'] = df[column].ewm(span=period, adjust=False).mean()
    
    return result


def add_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
            signal_period: int = 9, column: str = 'Close') -> pd.DataFrame:
    """
    Add Moving Average Convergence Divergence (MACD) to a DataFrame.
    
    Args:
        df: DataFrame with price data
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line
        column: Column to calculate MACD on
        
    Returns:
        DataFrame with added MACD indicators
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    result = df.copy()
    
    # Calculate MACD components
    fast_ema = df[column].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df[column].ewm(span=slow_period, adjust=False).mean()
    
    result['MACD'] = fast_ema - slow_ema
    result['MACD_Signal'] = result['MACD'].ewm(span=signal_period, adjust=False).mean()
    result['MACD_Hist'] = result['MACD'] - result['MACD_Signal']
    
    return result


def add_rsi(df: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI) to a DataFrame.
    
    Args:
        df: DataFrame with price data
        period: Period for RSI calculation
        column: Column to calculate RSI on
        
    Returns:
        DataFrame with added RSI
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    result = df.copy()
    
    # Calculate price changes
    delta = df[column].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RSI
    rs = avg_gain / avg_loss
    result['RSI'] = 100 - (100 / (1 + rs))
    
    return result


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, 
                       column: str = 'Close') -> pd.DataFrame:
    """
    Add Bollinger Bands to a DataFrame.
    
    Args:
        df: DataFrame with price data
        period: Period for moving average
        std_dev: Number of standard deviations for bands
        column: Column to calculate Bollinger Bands on
        
    Returns:
        DataFrame with added Bollinger Bands
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    result = df.copy()
    
    # Calculate middle band (SMA)
    result['BB_Middle'] = df[column].rolling(window=period).mean()
    
    # Calculate standard deviation
    rolling_std = df[column].rolling(window=period).std()
    
    # Calculate upper and lower bands
    result['BB_Upper'] = result['BB_Middle'] + (rolling_std * std_dev)
    result['BB_Lower'] = result['BB_Middle'] - (rolling_std * std_dev)
    
    # Calculate bandwidth and %B
    result['BB_Width'] = (result['BB_Upper'] - result['BB_Lower']) / result['BB_Middle']
    result['BB_%B'] = (df[column] - result['BB_Lower']) / (result['BB_Upper'] - result['BB_Lower'])
    
    return result


def add_stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Add Stochastic Oscillator to a DataFrame.
    
    Args:
        df: DataFrame with OHLC data
        k_period: Period for %K
        d_period: Period for %D
        
    Returns:
        DataFrame with added Stochastic Oscillator
    """
    required_columns = ['High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    result = df.copy()
    
    # Calculate %K
    lowest_low = df['Low'].rolling(window=k_period).min()
    highest_high = df['High'].rolling(window=k_period).max()
    
    result['Stoch_%K'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    
    # Calculate %D (SMA of %K)
    result['Stoch_%D'] = result['Stoch_%K'].rolling(window=d_period).mean()
    
    return result


def add_average_true_range(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Average True Range (ATR) to a DataFrame.
    
    Args:
        df: DataFrame with OHLC data
        period: Period for ATR calculation
        
    Returns:
        DataFrame with added ATR
    """
    required_columns = ['High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    result = df.copy()
    
    # Calculate True Range
    high_low = df['High'] - df['Low']
    high_close_prev = abs(df['High'] - df['Close'].shift(1))
    low_close_prev = abs(df['Low'] - df['Close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate ATR
    result['ATR'] = true_range.rolling(window=period).mean()
    
    return result


def add_fibonacci_retracement_levels(df: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    """
    Add Fibonacci Retracement Levels to a DataFrame.
    
    Args:
        df: DataFrame with OHLC data
        window: Lookback window to find high and low
        
    Returns:
        DataFrame with added Fibonacci Retracement Levels
    """
    required_columns = ['High', 'Low']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    result = df.copy()
    
    # Calculate rolling high and low
    rolling_high = df['High'].rolling(window=window).max()
    rolling_low = df['Low'].rolling(window=window).min()
    
    # Calculate Fibonacci levels
    range_diff = rolling_high - rolling_low
    result['Fib_0.0'] = rolling_low
    result['Fib_0.236'] = rolling_low + 0.236 * range_diff
    result['Fib_0.382'] = rolling_low + 0.382 * range_diff
    result['Fib_0.5'] = rolling_low + 0.5 * range_diff
    result['Fib_0.618'] = rolling_low + 0.618 * range_diff
    result['Fib_0.786'] = rolling_low + 0.786 * range_diff
    result['Fib_1.0'] = rolling_high
    
    return result


def add_ichimoku_cloud(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, 
                      senkou_b_period: int = 52, displacement: int = 26) -> pd.DataFrame:
    """
    Add Ichimoku Cloud to a DataFrame.
    
    Args:
        df: DataFrame with OHLC data
        tenkan_period: Period for Tenkan-sen (Conversion Line)
        kijun_period: Period for Kijun-sen (Base Line)
        senkou_b_period: Period for Senkou Span B
        displacement: Displacement period for Senkou Span A and B
        
    Returns:
        DataFrame with added Ichimoku Cloud components
    """
    required_columns = ['High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    result = df.copy()
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past 9 periods
    tenkan_sen_high = df['High'].rolling(window=tenkan_period).max()
    tenkan_sen_low = df['Low'].rolling(window=tenkan_period).min()
    result['Ichimoku_Tenkan_Sen'] = (tenkan_sen_high + tenkan_sen_low) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past 26 periods
    kijun_sen_high = df['High'].rolling(window=kijun_period).max()
    kijun_sen_low = df['Low'].rolling(window=kijun_period).min()
    result['Ichimoku_Kijun_Sen'] = (kijun_sen_high + kijun_sen_low) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2 plotted 26 periods ahead
    result['Ichimoku_Senkou_Span_A'] = ((result['Ichimoku_Tenkan_Sen'] + result['Ichimoku_Kijun_Sen']) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past 52 periods, plotted 26 periods ahead
    senkou_span_b_high = df['High'].rolling(window=senkou_b_period).max()
    senkou_span_b_low = df['Low'].rolling(window=senkou_b_period).min()
    result['Ichimoku_Senkou_Span_B'] = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span): Close price plotted 26 periods behind
    result['Ichimoku_Chikou_Span'] = df['Close'].shift(-displacement)
    
    return result


def add_on_balance_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add On-Balance Volume (OBV) to a DataFrame.
    
    Args:
        df: DataFrame with OHLC and Volume data
        
    Returns:
        DataFrame with added OBV
    """
    required_columns = ['Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    result = df.copy()
    
    # Calculate price changes
    price_change = df['Close'].diff()
    
    # Initialize OBV
    obv = pd.Series(0, index=df.index)
    
    # Calculate OBV
    obv.iloc[1:] = np.where(
        price_change.iloc[1:] > 0, 
        df['Volume'].iloc[1:], 
        np.where(
            price_change.iloc[1:] < 0, 
            -df['Volume'].iloc[1:], 
            0
        )
    )
    
    result['OBV'] = obv.cumsum()
    
    return result


def add_accumulation_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Accumulation/Distribution (A/D) to a DataFrame.
    
    Args:
        df: DataFrame with OHLC and Volume data
        
    Returns:
        DataFrame with added A/D
    """
    required_columns = ['High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    result = df.copy()
    
    # Calculate Money Flow Multiplier
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfm = mfm.replace([np.inf, -np.inf], 0)  # Handle division by zero
    
    # Calculate Money Flow Volume
    mfv = mfm * df['Volume']
    
    # Calculate A/D Line
    result['AD_Line'] = mfv.cumsum()
    
    return result


def add_average_directional_index(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Average Directional Index (ADX) to a DataFrame.
    
    Args:
        df: DataFrame with OHLC data
        period: Period for ADX calculation
        
    Returns:
        DataFrame with added ADX, +DI, and -DI
    """
    required_columns = ['High', 'Low', 'Close']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    result = df.copy()
    
    # Calculate True Range
    high_low = df['High'] - df['Low']
    high_close_prev = abs(df['High'] - df['Close'].shift(1))
    low_close_prev = abs(df['Low'] - df['Close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate smoothed True Range and DM
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
    
    # Calculate Directional Index
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], 0)  # Handle division by zero
    
    # Calculate ADX
    result['+DI'] = plus_di
    result['-DI'] = minus_di
    result['ADX'] = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return result


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Volume Weighted Average Price (VWAP) to a DataFrame.
    
    Args:
        df: DataFrame with OHLC and Volume data
        
    Returns:
        DataFrame with added VWAP
    """
    required_columns = ['High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    result = df.copy()
    
    # Calculate typical price
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Calculate VWAP
    result['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    return result


def add_money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add Money Flow Index (MFI) to a DataFrame.
    
    Args:
        df: DataFrame with OHLC and Volume data
        period: Period for MFI calculation
        
    Returns:
        DataFrame with added MFI
    """
    required_columns = ['High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    result = df.copy()
    
    # Calculate typical price
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Calculate raw money flow
    raw_money_flow = typical_price * df['Volume']
    
    # Determine money flow direction
    positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    
    # Sum positive and negative money flows over the period
    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
    
    # Calculate money flow ratio
    money_flow_ratio = positive_mf / negative_mf
    money_flow_ratio = money_flow_ratio.replace([np.inf, -np.inf], 100)  # Handle division by zero
    
    # Calculate MFI
    result['MFI'] = 100 - (100 / (1 + money_flow_ratio))
    
    return result


def add_all_indicators(df: pd.DataFrame, periods: Dict[str, Union[int, List[int]]] = None) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLC and Volume data
        periods: Dictionary of periods for different indicators
            Default: {
                'sma': [5, 20, 50, 200],
                'ema': [5, 20, 50, 200],
                'rsi': 14,
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger': {'period': 20, 'std_dev': 2.0},
                'stochastic': {'k': 14, 'd': 3},
                'atr': 14,
                'adx': 14,
                'mfi': 14
            }
        
    Returns:
        DataFrame with all indicators added
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
    
    # Set default periods if not provided
    if periods is None:
        periods = {
            'sma': [5, 20, 50, 200],
            'ema': [5, 20, 50, 200],
            'rsi': 14,
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'period': 20, 'std_dev': 2.0},
            'stochastic': {'k': 14, 'd': 3},
            'atr': 14,
            'adx': 14,
            'mfi': 14
        }
    
    result = df.copy()
    
    # Add SMA
    result = add_moving_averages(result, periods=periods.get('sma', [5, 20, 50, 200]))
    
    # Add EMA
    result = add_exponential_moving_averages(result, periods=periods.get('ema', [5, 20, 50, 200]))
    
    # Add RSI
    result = add_rsi(result, period=periods.get('rsi', 14))
    
    # Add MACD
    macd_params = periods.get('macd', {'fast': 12, 'slow': 26, 'signal': 9})
    result = add_macd(result, 
                      fast_period=macd_params.get('fast', 12),
                      slow_period=macd_params.get('slow', 26),
                      signal_period=macd_params.get('signal', 9))
    
    # Add Bollinger Bands
    bb_params = periods.get('bollinger', {'period': 20, 'std_dev': 2.0})
    result = add_bollinger_bands(result, 
                                period=bb_params.get('period', 20),
                                std_dev=bb_params.get('std_dev', 2.0))
    
    # Add Stochastic Oscillator
    stoch_params = periods.get('stochastic', {'k': 14, 'd': 3})
    result = add_stochastic_oscillator(result,
                                      k_period=stoch_params.get('k', 14),
                                      d_period=stoch_params.get('d', 3))
    
    # Add ATR
    result = add_average_true_range(result, period=periods.get('atr', 14))
    
    # Add OBV
    result = add_on_balance_volume(result)
    
    # Add A/D Line
    result = add_accumulation_distribution(result)
    
    # Add ADX
    result = add_average_directional_index(result, period=periods.get('adx', 14))
    
    # Add VWAP
    result = add_vwap(result)
    
    # Add MFI
    result = add_money_flow_index(result, period=periods.get('mfi', 14))
    
    return result


def calculate_price_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20], 
                           column: str = 'Close') -> pd.DataFrame:
    """
    Calculate price returns over various periods.
    
    Args:
        df: DataFrame with price data
        periods: List of periods for calculating returns
        column: Column to calculate returns on
        
    Returns:
        DataFrame with added return columns
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    result = df.copy()
    
    for period in periods:
        # Calculate percentage change
        result[f'Return_{period}d'] = df[column].pct_change(periods=period)
        
        # Calculate log returns
        result[f'LogReturn_{period}d'] = np.log(df[column] / df[column].shift(period))
    
    return result


def generate_technical_features(df: pd.DataFrame, include_returns: bool = True, 
                               include_all: bool = False) -> pd.DataFrame:
    """
    Generate technical features for a DataFrame.
    
    Args:
        df: DataFrame with OHLC and Volume data
        include_returns: Whether to include price returns
        include_all: Whether to include all technical indicators
        
    Returns:
        DataFrame with added technical features
    """
    result = df.copy()
    
    # Add standard indicators
    result = add_moving_averages(result, periods=[10, 50, 200])
    result = add_exponential_moving_averages(result, periods=[10, 50, 200])
    result = add_rsi(result)
    result = add_macd(result)
    result = add_bollinger_bands(result)
    
    if include_all:
        # Add additional indicators
        result = add_stochastic_oscillator(result)
        result = add_average_true_range(result)
        result = add_on_balance_volume(result)
        result = add_accumulation_distribution(result)
        result = add_average_directional_index(result)
        result = add_vwap(result)
        result = add_money_flow_index(result)
    
    if include_returns:
        # Add returns
        result = calculate_price_returns(result)
    
    return result 