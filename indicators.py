"""
Technical indicators calculation functions for stock analysis.
All functions take a pandas DataFrame with OHLCV data and return calculated indicators.
"""

import pandas as pd
import numpy as np


def calculate_sma(df, periods):
    """Calculate Simple Moving Averages for given periods."""
    for period in periods:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    return df


def calculate_ema(df, periods):
    """Calculate Exponential Moving Averages for given periods."""
    for period in periods:
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    return df


def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    df['macd_line'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd_line'].ewm(span=signal).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']
    return df


def calculate_bollinger_bands(df, periods, std_dev=2):
    """Calculate Bollinger Bands for given periods."""
    for period in periods:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df[f'bb_upper_{period}'] = sma + (std * std_dev)
        df[f'bb_lower_{period}'] = sma - (std * std_dev)
        df[f'bb_middle_{period}'] = sma
        # Calculate Bollinger Band position (0-1 scale)
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
    return df


def calculate_stochastic(df, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator."""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    df[f'stoch_k_{k_period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df[f'stoch_d_{d_period}'] = df[f'stoch_k_{k_period}'].rolling(window=d_period).mean()
    return df


def calculate_atr(df, periods):
    """Calculate Average True Range for given periods."""
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    
    for period in periods:
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()
    return df


def calculate_obv(df):
    """Calculate On-Balance Volume."""
    obv = []
    obv_value = 0
    
    for i in range(len(df)):
        if i == 0:
            obv_value = df['volume'].iloc[i]
        else:
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv_value += df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv_value -= df['volume'].iloc[i]
            # If close is same as previous, OBV stays the same
        obv.append(obv_value)
    
    df['obv'] = obv
    return df


def calculate_volume_sma(df, periods):
    """Calculate Simple Moving Averages for volume."""
    for period in periods:
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
    return df


def calculate_price_change(df):
    """Calculate price change metrics."""
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = df['close'].diff()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    return df


def calculate_volatility(df, periods=[5, 10, 20]):
    """Calculate price volatility (rolling standard deviation of returns)."""
    returns = df['close'].pct_change()
    for period in periods:
        df[f'volatility_{period}'] = returns.rolling(window=period).std()
    return df


def calculate_cci(df, period=20):
    """Calculate Commodity Channel Index using vectorized operations for better performance."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(window=period, min_periods=period).mean()
    
    # Calculate mean deviation using vectorized operations
    rolling_mean = typical_price.rolling(window=period, min_periods=period).mean()
    abs_diff = (typical_price - rolling_mean).abs()
    mean_deviation = abs_diff.rolling(window=period, min_periods=period).mean()
    
    # Calculate CCI with a small epsilon to avoid division by zero
    epsilon = 1e-10
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation + epsilon)
    
    # Only assign values where we have enough data
    df[f'cci_{period}'] = cci
    return df


def calculate_adx(df, period=14):
    """Calculate Average Directional Index."""
    high_diff = df['high'].diff()
    low_diff = df['low'].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff < 0), 0)
    
    tr = np.maximum(df['high'] - df['low'], 
                   np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                             np.abs(df['low'] - df['close'].shift(1))))
    
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    df[f'adx_{period}'] = dx.rolling(window=period).mean()
    df[f'plus_di_{period}'] = plus_di
    df[f'minus_di_{period}'] = minus_di
    
    return df


def calculate_roc(df, periods=[5, 10, 20]):
    """Calculate Rate of Change."""
    for period in periods:
        df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
    return df


def calculate_williams_r(df, period=14):
    """Calculate Williams %R."""
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    df[f'williams_r_{period}'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
    return df


def calculate_aroon(df, period=25):
    """Calculate Aroon Up and Aroon Down."""
    aroon_up = []
    aroon_down = []
    
    for i in range(len(df)):
        if i < period - 1:
            aroon_up.append(np.nan)
            aroon_down.append(np.nan)
        else:
            high_slice = df['high'].iloc[i-period+1:i+1]
            low_slice = df['low'].iloc[i-period+1:i+1]
            
            periods_since_high = period - 1 - high_slice.idxmax() + (i - period + 1)
            periods_since_low = period - 1 - low_slice.idxmin() + (i - period + 1)
            
            aroon_up.append(((period - periods_since_high) / period) * 100)
            aroon_down.append(((period - periods_since_low) / period) * 100)
    
    df[f'aroon_up_{period}'] = aroon_up
    df[f'aroon_down_{period}'] = aroon_down
    df[f'aroon_oscillator_{period}'] = np.array(aroon_up) - np.array(aroon_down)
    
    return df


def calculate_mfi(df, period=14):
    """Calculate Money Flow Index."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = []
    negative_flow = []
    
    for i in range(1, len(df)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.append(money_flow.iloc[i])
            negative_flow.append(0)
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i])
        else:
            positive_flow.append(0)
            negative_flow.append(0)
    
    positive_flow = [0] + positive_flow
    negative_flow = [0] + negative_flow
    
    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    df[f'mfi_{period}'] = mfi
    
    return df


def calculate_tsi(df, long_period=25, short_period=13):
    """Calculate True Strength Index."""
    price_change = df['close'].diff()
    
    # Double smoothed price change
    first_smooth = price_change.ewm(span=long_period).mean()
    double_smooth = first_smooth.ewm(span=short_period).mean()
    
    # Double smoothed absolute price change
    abs_price_change = np.abs(price_change)
    first_smooth_abs = abs_price_change.ewm(span=long_period).mean()
    double_smooth_abs = first_smooth_abs.ewm(span=short_period).mean()
    
    df[f'tsi_{long_period}_{short_period}'] = 100 * (double_smooth / double_smooth_abs)
    
    return df


def calculate_ultimate_oscillator(df, period1=7, period2=14, period3=28):
    """Calculate Ultimate Oscillator."""
    min_low_close = np.minimum(df['low'], df['close'].shift(1))
    max_high_close = np.maximum(df['high'], df['close'].shift(1))
    
    bp = df['close'] - min_low_close
    tr = max_high_close - min_low_close
    
    avg1 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
    avg2 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
    avg3 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()
    
    uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / (4 + 2 + 1)
    df[f'uo_{period1}_{period2}_{period3}'] = uo
    
    return df


def calculate_keltner_channels(df, period=20, multiplier=2):
    """Calculate Keltner Channels."""
    ema = df['close'].ewm(span=period).mean()
    atr = calculate_atr(df, [period])[f'atr_{period}']
    
    df[f'keltner_upper_{period}'] = ema + (multiplier * atr)
    df[f'keltner_lower_{period}'] = ema - (multiplier * atr)
    df[f'keltner_middle_{period}'] = ema
    
    return df


def calculate_donchian_channels(df, period=20):
    """Calculate Donchian Channels."""
    df[f'donchian_upper_{period}'] = df['high'].rolling(window=period).max()
    df[f'donchian_lower_{period}'] = df['low'].rolling(window=period).min()
    df[f'donchian_middle_{period}'] = (df[f'donchian_upper_{period}'] + df[f'donchian_lower_{period}']) / 2
    
    return df


def calculate_all_indicators(df, config):
    """Calculate all indicators based on configuration."""
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    indicators_config = config['indicators']
    
    # Calculate SMAs
    if 'sma_periods' in indicators_config:
        df = calculate_sma(df, indicators_config['sma_periods'])
    
    # Calculate EMAs
    if 'ema_periods' in indicators_config:
        df = calculate_ema(df, indicators_config['ema_periods'])
    
    # Calculate RSI
    if 'rsi_period' in indicators_config:
        df = calculate_rsi(df, indicators_config['rsi_period'])
    
    # Calculate MACD
    if all(k in indicators_config for k in ['macd_fast', 'macd_slow', 'macd_signal']):
        df = calculate_macd(df, 
                          indicators_config['macd_fast'],
                          indicators_config['macd_slow'], 
                          indicators_config['macd_signal'])
    
    # Calculate Bollinger Bands
    if 'bollinger_periods' in indicators_config:
        std_dev = indicators_config.get('bollinger_std', 2)
        df = calculate_bollinger_bands(df, indicators_config['bollinger_periods'], std_dev)
    
    # Calculate Stochastic
    if 'stochastic_k' in indicators_config and 'stochastic_d' in indicators_config:
        df = calculate_stochastic(df, 
                                indicators_config['stochastic_k'],
                                indicators_config['stochastic_d'])
    
    # Calculate ATR
    if 'atr_periods' in indicators_config:
        df = calculate_atr(df, indicators_config['atr_periods'])
    
    # Calculate OBV
    if indicators_config.get('obv_enabled', False):
        df = calculate_obv(df)
    
    # Calculate Volume SMAs
    if 'volume_sma_periods' in indicators_config:
        df = calculate_volume_sma(df, indicators_config['volume_sma_periods'])
    
    # Calculate additional metrics
    df = calculate_price_change(df)
    df = calculate_volatility(df)
    
    # Calculate advanced indicators based on config
    if 'cci_period' in indicators_config:
        df = calculate_cci(df, indicators_config['cci_period'])
    
    if 'adx_period' in indicators_config:
        df = calculate_adx(df, indicators_config['adx_period'])
    
    if 'roc_periods' in indicators_config:
        df = calculate_roc(df, indicators_config['roc_periods'])
    else:
        df = calculate_roc(df, [5, 10, 20])  # Default periods
    
    if 'williams_r_period' in indicators_config:
        df = calculate_williams_r(df, indicators_config['williams_r_period'])
    
    if 'aroon_period' in indicators_config:
        df = calculate_aroon(df, indicators_config['aroon_period'])
    else:
        df = calculate_aroon(df, 25)  # Default period
    
    if 'mfi_period' in indicators_config:
        df = calculate_mfi(df, indicators_config['mfi_period'])
    
    if 'tsi_long' in indicators_config and 'tsi_short' in indicators_config:
        df = calculate_tsi(df, indicators_config['tsi_long'], indicators_config['tsi_short'])
    else:
        df = calculate_tsi(df, 25, 13)  # Default periods
    
    if 'uo_periods' in indicators_config and len(indicators_config['uo_periods']) >= 3:
        periods = indicators_config['uo_periods']
        df = calculate_ultimate_oscillator(df, periods[0], periods[1], periods[2])
    else:
        df = calculate_ultimate_oscillator(df, 7, 14, 28)  # Default periods
    
    if 'keltner_period' in indicators_config:
        multiplier = indicators_config.get('keltner_multiplier', 2)
        df = calculate_keltner_channels(df, indicators_config['keltner_period'], multiplier)
    else:
        df = calculate_keltner_channels(df, 20, 2)  # Default
    
    if 'donchian_period' in indicators_config:
        df = calculate_donchian_channels(df, indicators_config['donchian_period'])
    else:
        df = calculate_donchian_channels(df, 20)  # Default
    
    return df
