"""
Expression functions for stock analysis.
All expressions return 1 if condition is met, 0 otherwise.
Expressions are relative comparisons, no absolute values.
"""

import pandas as pd
import numpy as np


def sma_crossover(df, fast_period, slow_period, operator='gt'):
    """SMA crossover: fast SMA vs slow SMA"""
    fast_col = f'sma_{fast_period}'
    slow_col = f'sma_{slow_period}'
    
    if fast_col not in df.columns or slow_col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    
    if operator == 'gt':
        return (df[fast_col] > df[slow_col]).astype(int)
    elif operator == 'lt':
        return (df[fast_col] < df[slow_col]).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def price_vs_sma(df, sma_period, threshold, operator='gt'):
    """Price relative to SMA with threshold"""
    sma_col = f'sma_{sma_period}'
    
    if sma_col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    
    ratio = df['close'] / df[sma_col]
    
    if operator == 'gt':
        return (ratio > threshold).astype(int)
    elif operator == 'lt':
        return (ratio < threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def volume_vs_average(df, avg_period, threshold, operator='gt'):
    """Volume relative to its average"""
    vol_avg_col = f'volume_sma_{avg_period}'
    
    if vol_avg_col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    
    ratio = df['volume'] / df[vol_avg_col]
    
    if operator == 'gt':
        return (ratio > threshold).astype(int)
    elif operator == 'lt':
        return (ratio < threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def close_vs_previous_high(df, threshold, operator='lt'):
    """Close price relative to previous day's high"""
    prev_high = df['high'].shift(1)
    ratio = df['close'] / prev_high
    
    if operator == 'lt':
        return (ratio < threshold).astype(int)
    elif operator == 'gt':
        return (ratio > threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def rsi_threshold(df, rsi_period, threshold, operator='lt'):
    """RSI threshold comparison"""
    rsi_col = f'rsi_{rsi_period}'
    
    if rsi_col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    
    if operator == 'lt':
        return (df[rsi_col] < threshold).astype(int)
    elif operator == 'gt':
        return (df[rsi_col] > threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def macd_signal_cross(df, operator='gt'):
    """MACD line vs signal line"""
    if 'macd_line' not in df.columns or 'macd_signal' not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    
    if operator == 'gt':
        return (df['macd_line'] > df['macd_signal']).astype(int)
    elif operator == 'lt':
        return (df['macd_line'] < df['macd_signal']).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def bollinger_position(df, bb_period, threshold, operator='lt'):
    """Position within Bollinger Bands"""
    bb_pos_col = f'bb_position_{bb_period}'
    
    if bb_pos_col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    
    if operator == 'lt':
        return (df[bb_pos_col] < threshold).astype(int)
    elif operator == 'gt':
        return (df[bb_pos_col] > threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def stochastic_threshold(df, k_period, d_period, threshold, operator='lt'):
    """Stochastic oscillator threshold"""
    stoch_k_col = f'stoch_k_{k_period}'
    
    if stoch_k_col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    
    if operator == 'lt':
        return (df[stoch_k_col] < threshold).astype(int)
    elif operator == 'gt':
        return (df[stoch_k_col] > threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def price_momentum(df, periods, threshold, operator='gt'):
    """Price momentum over N periods"""
    if len(df) < periods:
        return pd.Series([0] * len(df), index=df.index)
    
    momentum = df['close'] / df['close'].shift(periods)
    
    if operator == 'gt':
        return (momentum > threshold).astype(int)
    elif operator == 'lt':
        return (momentum < threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def volatility_threshold(df, vol_period, threshold, operator='lt'):
    """Volatility threshold comparison"""
    vol_col = f'volatility_{vol_period}'
    
    if vol_col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    
    if operator == 'lt':
        return (df[vol_col] < threshold).astype(int)
    elif operator == 'gt':
        return (df[vol_col] > threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def atr_relative(df, atr_period, threshold, operator='lt'):
    """ATR relative to price"""
    atr_col = f'atr_{atr_period}'
    
    if atr_col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    
    atr_ratio = df[atr_col] / df['close']
    
    if operator == 'lt':
        return (atr_ratio < threshold).astype(int)
    elif operator == 'gt':
        return (atr_ratio > threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def ema_crossover(df, fast_period, slow_period, operator='gt'):
    """EMA crossover: fast EMA vs slow EMA"""
    fast_col = f'ema_{fast_period}'
    slow_col = f'ema_{slow_period}'
    
    if fast_col not in df.columns or slow_col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    
    if operator == 'gt':
        return (df[fast_col] > df[slow_col]).astype(int)
    elif operator == 'lt':
        return (df[fast_col] < df[slow_col]).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def price_vs_ema(df, ema_period, threshold, operator='gt'):
    """Price relative to EMA with threshold"""
    ema_col = f'ema_{ema_period}'
    
    if ema_col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    
    ratio = df['close'] / df[ema_col]
    
    if operator == 'gt':
        return (ratio > threshold).astype(int)
    elif operator == 'lt':
        return (ratio < threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def close_vs_open(df, threshold, operator='gt'):
    """Close vs Open ratio"""
    ratio = df['close'] / df['open']
    
    if operator == 'gt':
        return (ratio > threshold).astype(int)
    elif operator == 'lt':
        return (ratio < threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def high_low_range(df, threshold, operator='gt'):
    """High-Low range relative to close price"""
    range_ratio = (df['high'] - df['low']) / df['close']
    
    if operator == 'gt':
        return (range_ratio > threshold).astype(int)
    elif operator == 'lt':
        return (range_ratio < threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def open_close_gap(df, threshold, operator='gt'):
    """Gap between open and previous close"""
    prev_close = df['close'].shift(1)
    gap_ratio = (df['open'] - prev_close) / prev_close
    
    if operator == 'gt':
        return (gap_ratio > threshold).astype(int)
    elif operator == 'lt':
        return (gap_ratio < threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def volume_breakout(df, periods, threshold):
    """Volume breakout: current volume vs average volume"""
    if len(df) < periods:
        return pd.Series([0] * len(df), index=df.index)
    
    avg_volume = df['volume'].rolling(window=periods).mean()
    volume_ratio = df['volume'] / avg_volume
    
    return (volume_ratio > threshold).astype(int)


def consecutive_moves(df, periods, direction='up'):
    """Consecutive price moves in same direction"""
    if direction == 'up':
        moves = (df['close'] > df['close'].shift(1)).astype(int)
    else:
        moves = (df['close'] < df['close'].shift(1)).astype(int)
    
    consecutive = moves.rolling(window=periods).sum()
    return (consecutive == periods).astype(int)


def price_position_in_range(df, periods, threshold, operator='gt'):
    """Current price position within N-period high-low range"""
    if len(df) < periods:
        return pd.Series([0] * len(df), index=df.index)
    
    period_high = df['high'].rolling(window=periods).max()
    period_low = df['low'].rolling(window=periods).min()
    
    position = (df['close'] - period_low) / (period_high - period_low)
    
    if operator == 'gt':
        return (position > threshold).astype(int)
    elif operator == 'lt':
        return (position < threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def stochastic_crossover(df, k_period, d_period, operator='gt'):
    """Stochastic %K vs %D crossover"""
    k_col = f'stoch_k_{k_period}'
    d_col = f'stoch_d_{d_period}'
    
    if k_col not in df.columns or d_col not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    
    if operator == 'gt':
        return (df[k_col] > df[d_col]).astype(int)
    elif operator == 'lt':
        return (df[k_col] < df[d_col]).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def hammer_pattern(df):
    """Hammer candlestick pattern: small body, long lower wick"""
    body = abs(df['close'] - df['open'])
    lower_wick = df['open'].where(df['open'] < df['close'], df['close']) - df['low']
    upper_wick = df['high'] - df['open'].where(df['open'] > df['close'], df['close'])
    total_range = df['high'] - df['low']
    
    # Hammer: body < 30% of range, lower wick > 60% of range, upper wick < 10% of range
    body_ratio = body / total_range
    lower_ratio = lower_wick / total_range
    upper_ratio = upper_wick / total_range
    
    hammer = (body_ratio < 0.3) & (lower_ratio > 0.6) & (upper_ratio < 0.1)
    return hammer.astype(int)


def doji_pattern(df):
    """Doji candlestick pattern: open and close very close"""
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    
    # Doji: body < 5% of total range
    body_ratio = body / total_range
    doji = body_ratio < 0.05
    return doji.astype(int)


def engulfing_bullish(df):
    """Bullish engulfing pattern: current candle engulfs previous bearish candle"""
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    
    # Previous candle was bearish, current is bullish and engulfs it
    prev_bearish = prev_close < prev_open
    curr_bullish = df['close'] > df['open']
    engulfs = (df['open'] < prev_close) & (df['close'] > prev_open)
    
    pattern = prev_bearish & curr_bullish & engulfs
    return pattern.astype(int)


def engulfing_bearish(df):
    """Bearish engulfing pattern: current candle engulfs previous bullish candle"""
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    
    # Previous candle was bullish, current is bearish and engulfs it
    prev_bullish = prev_close > prev_open
    curr_bearish = df['close'] < df['open']
    engulfs = (df['open'] > prev_close) & (df['close'] < prev_open)
    
    pattern = prev_bullish & curr_bearish & engulfs
    return pattern.astype(int)


def piercing_line(df):
    """Piercing line pattern: bullish reversal after bearish candle"""
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    
    # Previous candle bearish, current opens below prev low, closes above prev midpoint
    prev_bearish = prev_close < prev_open
    prev_midpoint = (prev_open + prev_close) / 2
    
    opens_below = df['open'] < prev_close
    closes_above_mid = df['close'] > prev_midpoint
    current_bullish = df['close'] > df['open']
    
    pattern = prev_bearish & opens_below & closes_above_mid & current_bullish
    return pattern.astype(int)


def shooting_star(df):
    """Shooting star pattern: small body, long upper wick"""
    body = abs(df['close'] - df['open'])
    upper_wick = df['high'] - df['open'].where(df['open'] > df['close'], df['close'])
    lower_wick = df['open'].where(df['open'] < df['close'], df['close']) - df['low']
    total_range = df['high'] - df['low']
    
    # Shooting star: body < 30% of range, upper wick > 60% of range, lower wick < 10% of range
    body_ratio = body / total_range
    upper_ratio = upper_wick / total_range
    lower_ratio = lower_wick / total_range
    
    shooting_star = (body_ratio < 0.3) & (upper_ratio > 0.6) & (lower_ratio < 0.1)
    return shooting_star.astype(int)


def golden_cross(df, short_period=50, long_period=200):
    """Golden cross: short MA crosses above long MA"""
    short_ma = df['close'].rolling(window=short_period).mean()
    long_ma = df['close'].rolling(window=long_period).mean()
    
    # Current: short > long, Previous: short <= long
    current_above = short_ma > long_ma
    prev_below = short_ma.shift(1) <= long_ma.shift(1)
    
    golden_cross = current_above & prev_below
    return golden_cross.astype(int)


def death_cross(df, short_period=50, long_period=200):
    """Death cross: short MA crosses below long MA"""
    short_ma = df['close'].rolling(window=short_period).mean()
    long_ma = df['close'].rolling(window=long_period).mean()
    
    # Current: short < long, Previous: short >= long
    current_below = short_ma < long_ma
    prev_above = short_ma.shift(1) >= long_ma.shift(1)
    
    death_cross = current_below & prev_above
    return death_cross.astype(int)


def breakout_resistance(df, periods=20, threshold=0.02):
    """Price breaks above resistance level"""
    if len(df) < periods:
        return pd.Series([0] * len(df), index=df.index)
    
    # Resistance = highest high in last N periods
    resistance = df['high'].rolling(window=periods).max().shift(1)
    
    # Breakout: close > resistance * (1 + threshold)
    breakout = df['close'] > resistance * (1 + threshold)
    return breakout.astype(int)


def breakdown_support(df, periods=20, threshold=0.02):
    """Price breaks below support level"""
    if len(df) < periods:
        return pd.Series([0] * len(df), index=df.index)
    
    # Support = lowest low in last N periods
    support = df['low'].rolling(window=periods).min().shift(1)
    
    # Breakdown: close < support * (1 - threshold)
    breakdown = df['close'] < support * (1 - threshold)
    return breakdown.astype(int)


def inside_bar(df):
    """Inside bar: current bar's range is within previous bar's range"""
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    
    inside = (df['high'] <= prev_high) & (df['low'] >= prev_low)
    return inside.astype(int)


def outside_bar(df):
    """Outside bar: current bar's range engulfs previous bar's range"""
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    
    outside = (df['high'] > prev_high) & (df['low'] < prev_low)
    return outside.astype(int)


def price_vs_previous_close(df, periods, threshold, operator='gt'):
    """Current price vs close N periods ago"""
    if len(df) < periods:
        return pd.Series([0] * len(df), index=df.index)
    
    prev_close = df['close'].shift(periods)
    ratio = df['close'] / prev_close
    
    if operator == 'gt':
        return (ratio > threshold).astype(int)
    elif operator == 'lt':
        return (ratio < threshold).astype(int)
    else:
        return pd.Series([0] * len(df), index=df.index)


def volume_price_confirmation(df, periods=5):
    """Volume confirms price movement"""
    if len(df) < periods:
        return pd.Series([0] * len(df), index=df.index)
    
    price_up = df['close'] > df['close'].shift(1)
    volume_up = df['volume'] > df['volume'].rolling(window=periods).mean()
    
    confirmation = price_up & volume_up
    return confirmation.astype(int)


def three_white_soldiers(df):
    """Three consecutive bullish candles with higher closes"""
    # Three consecutive bullish candles
    bull1 = df['close'] > df['open']
    bull2 = df['close'].shift(1) > df['open'].shift(1)
    bull3 = df['close'].shift(2) > df['open'].shift(2)
    
    # Each close higher than previous
    higher1 = df['close'] > df['close'].shift(1)
    higher2 = df['close'].shift(1) > df['close'].shift(2)
    
    pattern = bull1 & bull2 & bull3 & higher1 & higher2
    return pattern.astype(int)


def three_black_crows(df):
    """Three consecutive bearish candles with lower closes"""
    # Three consecutive bearish candles
    bear1 = df['close'] < df['open']
    bear2 = df['close'].shift(1) < df['open'].shift(1)
    bear3 = df['close'].shift(2) < df['open'].shift(2)
    
    # Each close lower than previous
    lower1 = df['close'] < df['close'].shift(1)
    lower2 = df['close'].shift(1) < df['close'].shift(2)
    
    pattern = bear1 & bear2 & bear3 & lower1 & lower2
    return pattern.astype(int)


def morning_star(df):
    """Morning star pattern: bearish, small body, bullish"""
    # Three candles: bearish, small body, bullish
    bear_candle = df['close'].shift(2) < df['open'].shift(2)
    
    # Middle candle has small body
    middle_body = abs(df['close'].shift(1) - df['open'].shift(1))
    middle_range = df['high'].shift(1) - df['low'].shift(1)
    small_body = middle_body / middle_range < 0.3
    
    # Third candle is bullish
    bull_candle = df['close'] > df['open']
    
    # Gaps between candles
    gap_down = df['high'].shift(1) < df['close'].shift(2)
    gap_up = df['low'] > df['high'].shift(1)
    
    pattern = bear_candle & small_body & bull_candle & gap_down & gap_up
    return pattern.astype(int)


def evening_star(df):
    """Evening star pattern: bullish, small body, bearish"""
    # Three candles: bullish, small body, bearish
    bull_candle = df['close'].shift(2) > df['open'].shift(2)
    
    # Middle candle has small body
    middle_body = abs(df['close'].shift(1) - df['open'].shift(1))
    middle_range = df['high'].shift(1) - df['low'].shift(1)
    small_body = middle_body / middle_range < 0.3
    
    # Third candle is bearish
    bear_candle = df['close'] < df['open']
    
    # Gaps between candles
    gap_up = df['low'].shift(1) > df['close'].shift(2)
    gap_down = df['high'] < df['low'].shift(1)
    
    pattern = bull_candle & small_body & bear_candle & gap_up & gap_down
    return pattern.astype(int)


# Expression registry - maps string identifiers to functions and parameters
EXPRESSIONS = {
    # Original SMA crossovers
    'sma_3_gt_sma_5': lambda df: sma_crossover(df, 3, 5, 'gt'),
    'sma_5_gt_sma_10': lambda df: sma_crossover(df, 5, 10, 'gt'),
    'sma_10_gt_sma_20': lambda df: sma_crossover(df, 10, 20, 'gt'),
    'sma_3_lt_sma_5': lambda df: sma_crossover(df, 3, 5, 'lt'),
    'sma_5_lt_sma_10': lambda df: sma_crossover(df, 5, 10, 'lt'),
    'sma_10_lt_sma_20': lambda df: sma_crossover(df, 10, 20, 'lt'),
    
    # Additional SMA crossovers
    'sma_3_gt_sma_10': lambda df: sma_crossover(df, 3, 10, 'gt'),
    'sma_3_lt_sma_10': lambda df: sma_crossover(df, 3, 10, 'lt'),
    'sma_3_gt_sma_20': lambda df: sma_crossover(df, 3, 20, 'gt'),
    'sma_3_lt_sma_20': lambda df: sma_crossover(df, 3, 20, 'lt'),
    'sma_5_gt_sma_20': lambda df: sma_crossover(df, 5, 20, 'gt'),
    'sma_5_lt_sma_20': lambda df: sma_crossover(df, 5, 20, 'lt'),
    
    # EMA crossovers
    'ema_12_gt_ema_26': lambda df: ema_crossover(df, 12, 26, 'gt'),
    'ema_12_lt_ema_26': lambda df: ema_crossover(df, 12, 26, 'lt'),
    
    # Original Price vs SMA
    'close_gt_sma_20_1p05': lambda df: price_vs_sma(df, 20, 1.05, 'gt'),
    'close_lt_sma_20_0p95': lambda df: price_vs_sma(df, 20, 0.95, 'lt'),
    'close_gt_sma_10_1p03': lambda df: price_vs_sma(df, 10, 1.03, 'gt'),
    'close_lt_sma_10_0p97': lambda df: price_vs_sma(df, 10, 0.97, 'lt'),
    
    # Additional Price vs SMA
    'close_gt_sma_5_1p02': lambda df: price_vs_sma(df, 5, 1.02, 'gt'),
    'close_lt_sma_5_0p98': lambda df: price_vs_sma(df, 5, 0.98, 'lt'),
    'close_gt_sma_3_1p01': lambda df: price_vs_sma(df, 3, 1.01, 'gt'),
    'close_lt_sma_3_0p99': lambda df: price_vs_sma(df, 3, 0.99, 'lt'),
    'close_gt_sma_20_1p1': lambda df: price_vs_sma(df, 20, 1.1, 'gt'),
    'close_lt_sma_20_0p9': lambda df: price_vs_sma(df, 20, 0.9, 'lt'),
    
    # Price vs EMA
    'close_gt_ema_12_1p03': lambda df: price_vs_ema(df, 12, 1.03, 'gt'),
    'close_lt_ema_12_0p97': lambda df: price_vs_ema(df, 12, 0.97, 'lt'),
    'close_gt_ema_26_1p05': lambda df: price_vs_ema(df, 26, 1.05, 'gt'),
    'close_lt_ema_26_0p95': lambda df: price_vs_ema(df, 26, 0.95, 'lt'),
    
    # Original Volume analysis
    # 'volume_gt_avg5_1p5': lambda df: volume_vs_average(df, 5, 1.5, 'gt'),  # Low scores < 0.15
    'volume_gt_avg10_1p3': lambda df: volume_vs_average(df, 10, 1.3, 'gt'),
    'volume_gt_avg20_1p2': lambda df: volume_vs_average(df, 20, 1.2, 'gt'),
    'volume_lt_avg5_0p8': lambda df: volume_vs_average(df, 5, 0.8, 'lt'),
    
    # Volume expressions with high volume thresholds (commented out due to low averages ≤0.1)
    # 'volume_gt_avg5_2p0': lambda df: volume_vs_average(df, 5, 2.0, 'gt'),  # Low scores < 0.15
    # 'volume_gt_avg10_2p0': lambda df: volume_vs_average(df, 10, 2.0, 'gt'),  # Low scores < 0.15
    'volume_gt_avg20_1p5': lambda df: volume_vs_average(df, 20, 1.5, 'gt'),
    'volume_lt_avg10_0p7': lambda df: volume_vs_average(df, 10, 0.7, 'lt'),
    'volume_lt_avg20_0p8': lambda df: volume_vs_average(df, 20, 0.8, 'lt'),
    # Volume breakout expressions (commented out due to low averages ≤0.1)
    # 'volume_breakout_10_3p0': lambda df: (df['volume'] > df['volume_sma_10'] * 3.0).astype(int),
    # 'volume_breakout_20_2p5': lambda df: (df['volume'] > df['volume_sma_20'] * 2.5).astype(int),
    
    # Original Close vs previous high
    'close_lt_prev_high_0p9': lambda df: close_vs_previous_high(df, 0.9, 'lt'),
    # 'close_lt_prev_high_0p8': lambda df: close_vs_previous_high(df, 0.8, 'lt'),  # Low scores < 0.15
    # 'close_gt_prev_high_1p02': lambda df: close_vs_previous_high(df, 1.02, 'gt'),  # Low scores < 0.15
    
    # Additional price comparisons
    'close_lt_prev_high_0p95': lambda df: close_vs_previous_high(df, 0.95, 'lt'),
    # 'close_gt_prev_high_1p05': lambda df: close_vs_previous_high(df, 1.05, 'gt'),  # Low scores < 0.15
    'close_vs_open_gt_1p02': lambda df: close_vs_open(df, 1.02, 'gt'),
    'close_vs_open_lt_0p98': lambda df: close_vs_open(df, 0.98, 'lt'),
    # 'close_vs_open_gt_1p05': lambda df: close_vs_open(df, 1.05, 'gt'),  # Low scores < 0.15
    'close_vs_open_lt_0p95': lambda df: close_vs_open(df, 0.95, 'lt'),
    
    # Gap analysis
    'gap_up_gt_1p0': lambda df: open_close_gap(df, 0.01, 'gt'),
    'gap_up_gt_2p0': lambda df: open_close_gap(df, 0.02, 'gt'),
    'gap_down_lt_neg1p0': lambda df: open_close_gap(df, -0.01, 'lt'),
    # 'gap_down_lt_neg2p0': lambda df: gap_analysis(df, -2.0, 'lt'),  # Low scores < 0.15
    
    # High-Low range
    'hl_range_gt_2p0': lambda df: high_low_range(df, 0.02, 'gt'),
    'hl_range_gt_3p0': lambda df: high_low_range(df, 0.03, 'gt'),
    # 'hl_range_lt_1p0': lambda df: high_low_range(df, 0.01, 'lt'),  # Low scores < 0.15
    'hl_range_gt_5p0': lambda df: high_low_range(df, 0.05, 'gt'),
    
    # Original RSI conditions
    'rsi_14_lt_30': lambda df: rsi_threshold(df, 14, 30, 'lt'),
    # 'rsi_14_gt_70': lambda df: rsi_threshold(df, 14, 70, 'gt'),  # Low scores < 0.15
    'rsi_14_lt_50': lambda df: rsi_threshold(df, 14, 50, 'lt'),
    'rsi_14_gt_50': lambda df: rsi_threshold(df, 14, 50, 'gt'),
    
    # Additional RSI conditions
    # 'rsi_14_lt_20': lambda df: rsi_threshold(df, 14, 20, 'lt'),  # Low scores < 0.15
    # 'rsi_14_gt_80': lambda df: rsi_threshold(df, 14, 80, 'gt'),  # Low scores < 0.15
    'rsi_14_lt_40': lambda df: rsi_threshold(df, 14, 40, 'lt'),
    'rsi_14_gt_60': lambda df: rsi_threshold(df, 14, 60, 'gt'),
    'rsi_14_lt_35': lambda df: rsi_threshold(df, 14, 35, 'lt'),
    'rsi_14_gt_65': lambda df: rsi_threshold(df, 14, 65, 'gt'),
    
    # Original MACD signals
    'macd_gt_signal': lambda df: macd_signal_cross(df, 'gt'),
    'macd_lt_signal': lambda df: macd_signal_cross(df, 'lt'),
    
    # Original Bollinger Bands
    'bb_pos_20_lt_0p2': lambda df: bollinger_position(df, 20, 0.2, 'lt'),
    'bb_pos_20_gt_0p8': lambda df: bollinger_position(df, 20, 0.8, 'gt'),
    'bb_pos_15_lt_0p3': lambda df: bollinger_position(df, 15, 0.3, 'lt'),
    
    # Additional Bollinger Bands
    'bb_pos_15_gt_0p7': lambda df: bollinger_position(df, 15, 0.7, 'gt'),
    'bb_pos_25_lt_0p2': lambda df: bollinger_position(df, 25, 0.2, 'lt'),
    'bb_pos_25_gt_0p8': lambda df: bollinger_position(df, 25, 0.8, 'gt'),
    
    # Original Stochastic
    'stoch_14_lt_20': lambda df: stochastic_threshold(df, 14, 3, 20, 'lt'),
    'stoch_14_gt_80': lambda df: stochastic_threshold(df, 14, 3, 80, 'gt'),
    
    # Additional Stochastic
    'stoch_k_gt_stoch_d': lambda df: stochastic_crossover(df, 14, 3, 'gt'),
    'stoch_k_lt_stoch_d': lambda df: stochastic_crossover(df, 14, 3, 'lt'),
    'stoch_14_lt_30': lambda df: stochastic_threshold(df, 14, 3, 30, 'lt'),
    'stoch_14_gt_70': lambda df: stochastic_threshold(df, 14, 3, 70, 'gt'),
    
    # Original Price momentum
    'momentum_5d_gt_1p02': lambda df: price_momentum(df, 5, 1.02, 'gt'),
    'momentum_10d_gt_1p05': lambda df: price_momentum(df, 10, 1.05, 'gt'),
    'momentum_5d_lt_0p98': lambda df: price_momentum(df, 5, 0.98, 'lt'),
    
    # Additional momentum
    'momentum_3d_gt_1p01': lambda df: price_momentum(df, 3, 1.01, 'gt'),
    # 'momentum_3d_lt_0p99': lambda df: price_momentum(df, 3, 0.99, 'lt'),  # Low scores < 0.15
    'momentum_20d_gt_1p1': lambda df: price_momentum(df, 20, 1.1, 'gt'),
    'momentum_20d_lt_0p9': lambda df: price_momentum(df, 20, 0.9, 'lt'),
    
    # Consecutive moves
    # 'consecutive_up_3': lambda df: consecutive_moves(df, 3, 'up'),  # Low scores < 0.15
    # 'consecutive_up_5': lambda df: consecutive_moves(df, 5, 'up'),  # Low scores < 0.15
    # 'consecutive_down_3': lambda df: consecutive_moves(df, 3, 'down'),  # Low scores < 0.15
    # 'consecutive_down_5': lambda df: consecutive_moves(df, 5, 'down'),  # Low scores < 0.15
    
    # Price position in range
    'price_pos_10d_gt_0p8': lambda df: price_position_in_range(df, 10, 0.8, 'gt'),
    'price_pos_10d_lt_0p2': lambda df: price_position_in_range(df, 10, 0.2, 'lt'),
    # 'price_pos_20d_gt_0p9': lambda df: price_position_in_range(df, 20, 0.9, 'gt'),  # Low scores < 0.15
    'price_pos_20d_lt_0p1': lambda df: price_position_in_range(df, 20, 0.1, 'lt'),
    'price_pos_5d_gt_0p7': lambda df: price_position_in_range(df, 5, 0.7, 'gt'),
    'price_pos_5d_lt_0p3': lambda df: price_position_in_range(df, 5, 0.3, 'lt'),
    
    # Original Volatility
    'volatility_10_lt_0p02': lambda df: volatility_threshold(df, 10, 0.02, 'lt'),
    'volatility_20_gt_0p03': lambda df: volatility_threshold(df, 20, 0.03, 'gt'),
    
    # Additional volatility
    'volatility_5_gt_0p02': lambda df: volatility_threshold(df, 5, 0.02, 'gt'),
    # 'volatility_5_lt_0p01': lambda df: volatility_threshold(df, 5, 0.01, 'lt'),  # Low scores < 0.15
    'volatility_10_gt_0p025': lambda df: volatility_threshold(df, 10, 0.025, 'gt'),
    # 'volatility_20_lt_0p015': lambda df: volatility_threshold(df, 20, 0.015, 'lt'),  # Low scores < 0.15
    
    # Original ATR relative
    # 'atr_14_lt_0p03': lambda df: atr_threshold(df, 14, 0.03, 'lt'),  # Low scores < 0.15
    'atr_20_gt_0p05': lambda df: atr_relative(df, 20, 0.05, 'gt'),
    
    # Additional ATR
    'atr_14_gt_0p04': lambda df: atr_relative(df, 14, 0.04, 'gt'),
    # 'atr_20_lt_0p02': lambda df: atr_threshold(df, 20, 0.02, 'lt'),  # Low scores < 0.15
    
    # Candlestick patterns (commented out due to low scores < 0.15)
    # 'hammer_pattern': lambda df: hammer_pattern(df),  # Low scores < 0.15
    # 'doji_pattern': lambda df: doji_pattern(df),  # Low scores < 0.15
    # 'engulfing_bullish': lambda df: engulfing_bullish(df),  # Low scores < 0.15
    # 'engulfing_bearish': lambda df: engulfing_bearish(df),  # Low scores < 0.15
    # 'piercing_line': lambda df: piercing_line(df),  # Low scores < 0.15
    # 'shooting_star': lambda df: shooting_star(df),  # Low scores < 0.15
    # 'three_white_soldiers': lambda df: three_white_soldiers(df),  # Low scores < 0.15
    # 'three_black_crows': lambda df: three_black_crows(df),  # Low scores < 0.15
    # 'morning_star': lambda df: morning_star(df),  # Low scores < 0.15
    # 'evening_star': lambda df: evening_star(df),  # Low scores < 0.15
    
    # Golden/Death crosses (commented out due to low scores < 0.15)
    # 'golden_cross_50_200': lambda df: golden_cross(df, 50, 200),  # Low scores < 0.15
    # 'death_cross_50_200': lambda df: death_cross(df, 50, 200),  # Low scores < 0.15
    # 'golden_cross_20_50': lambda df: golden_cross(df, 20, 50),  # Low scores < 0.15
    # 'death_cross_20_50': lambda df: death_cross(df, 20, 50),  # Low scores < 0.15
    
    # Breakout/Breakdown patterns (commented out due to low scores < 0.15)
    # 'breakout_resistance_20_2pct': lambda df: breakout_resistance(df, 20, 0.02),  # Low scores < 0.15
    # 'breakout_resistance_10_3pct': lambda df: breakout_resistance(df, 10, 0.03),  # Low scores < 0.15
    # 'breakout_resistance_5_5pct': lambda df: breakout_resistance(df, 5, 0.05),  # Low scores < 0.15
    # 'breakdown_support_20_2pct': lambda df: breakdown_support(df, 20, 0.02),  # Low scores < 0.15
    # 'breakdown_support_10_3pct': lambda df: breakdown_support(df, 10, 0.03),  # Low scores < 0.15
    # 'breakdown_support_5_5pct': lambda df: breakdown_support(df, 5, 0.05),  # Low scores < 0.15
    
    # Bar patterns
    'inside_bar': lambda df: inside_bar(df),
    # 'outside_bar': lambda df: outside_bar(df),  # Low scores < 0.15
    
    # === NEW ADVANCED EXPRESSIONS WITH ENHANCED INDICATORS ===
    
    # CCI-based expressions
    'cci_20_gt_100': lambda df: (df['cci_20'] > 100).astype(int),
    'cci_20_lt_neg100': lambda df: (df['cci_20'] < -100).astype(int),
    'cci_20_gt_200': lambda df: (df['cci_20'] > 200).astype(int),
    'cci_20_lt_neg200': lambda df: (df['cci_20'] < -200).astype(int),
    'cci_20_cross_zero_up': lambda df: ((df['cci_20'] > 0) & (df['cci_20'].shift(1) <= 0)).astype(int),
    'cci_20_cross_zero_down': lambda df: ((df['cci_20'] < 0) & (df['cci_20'].shift(1) >= 0)).astype(int),
    
    # ADX-based expressions
    'adx_14_gt_25': lambda df: (df['adx_14'] > 25).astype(int),
    'adx_14_gt_40': lambda df: (df['adx_14'] > 40).astype(int),
    'adx_14_lt_20': lambda df: (df['adx_14'] < 20).astype(int),
    'plus_di_gt_minus_di': lambda df: (df['plus_di_14'] > df['minus_di_14']).astype(int),
    'minus_di_gt_plus_di': lambda df: (df['minus_di_14'] > df['plus_di_14']).astype(int),
    'adx_rising': lambda df: (df['adx_14'] > df['adx_14'].shift(1)).astype(int),
    'adx_falling': lambda df: (df['adx_14'] < df['adx_14'].shift(1)).astype(int),
    
    # ROC-based expressions
    'roc_5_gt_5': lambda df: (df['roc_5'] > 5).astype(int),
    'roc_5_lt_neg5': lambda df: (df['roc_5'] < -5).astype(int),
    'roc_10_gt_10': lambda df: (df['roc_10'] > 10).astype(int),
    'roc_10_lt_neg10': lambda df: (df['roc_10'] < -10).astype(int),
    'roc_20_gt_15': lambda df: (df['roc_20'] > 15).astype(int),
    'roc_20_lt_neg15': lambda df: (df['roc_20'] < -15).astype(int),
    'roc_5_gt_roc_10': lambda df: (df['roc_5'] > df['roc_10']).astype(int),
    'roc_10_gt_roc_20': lambda df: (df['roc_10'] > df['roc_20']).astype(int),
    
    # Williams %R expressions
    'williams_r_14_lt_neg80': lambda df: (df['williams_r_14'] < -80).astype(int),
    'williams_r_14_gt_neg20': lambda df: (df['williams_r_14'] > -20).astype(int),
    'williams_r_14_lt_neg90': lambda df: (df['williams_r_14'] < -90).astype(int),
    'williams_r_14_gt_neg10': lambda df: (df['williams_r_14'] > -10).astype(int),
    'williams_r_rising': lambda df: (df['williams_r_14'] > df['williams_r_14'].shift(1)).astype(int),
    'williams_r_falling': lambda df: (df['williams_r_14'] < df['williams_r_14'].shift(1)).astype(int),
    
    # Aroon-based expressions
    'aroon_up_gt_70': lambda df: (df['aroon_up_25'] > 70).astype(int),
    'aroon_down_gt_70': lambda df: (df['aroon_down_25'] > 70).astype(int),
    'aroon_up_gt_aroon_down': lambda df: (df['aroon_up_25'] > df['aroon_down_25']).astype(int),
    'aroon_oscillator_gt_50': lambda df: (df['aroon_oscillator_25'] > 50).astype(int),
    'aroon_oscillator_lt_neg50': lambda df: (df['aroon_oscillator_25'] < -50).astype(int),
    'aroon_up_gt_90': lambda df: (df['aroon_up_25'] > 90).astype(int),
    'aroon_down_gt_90': lambda df: (df['aroon_down_25'] > 90).astype(int),
    
    # MFI-based expressions
    'mfi_14_gt_80': lambda df: (df['mfi_14'] > 80).astype(int),
    'mfi_14_lt_20': lambda df: (df['mfi_14'] < 20).astype(int),
    'mfi_14_gt_70': lambda df: (df['mfi_14'] > 70).astype(int),
    'mfi_14_lt_30': lambda df: (df['mfi_14'] < 30).astype(int),
    'mfi_rising': lambda df: (df['mfi_14'] > df['mfi_14'].shift(1)).astype(int),
    'mfi_falling': lambda df: (df['mfi_14'] < df['mfi_14'].shift(1)).astype(int),
    
    # TSI-based expressions
    'tsi_gt_0': lambda df: (df['tsi_25_13'] > 0).astype(int),
    'tsi_lt_0': lambda df: (df['tsi_25_13'] < 0).astype(int),
    'tsi_gt_25': lambda df: (df['tsi_25_13'] > 25).astype(int),
    'tsi_lt_neg25': lambda df: (df['tsi_25_13'] < -25).astype(int),
    'tsi_rising': lambda df: (df['tsi_25_13'] > df['tsi_25_13'].shift(1)).astype(int),
    'tsi_falling': lambda df: (df['tsi_25_13'] < df['tsi_25_13'].shift(1)).astype(int),
    
    # Ultimate Oscillator expressions
    'uo_gt_70': lambda df: (df['uo_7_14_28'] > 70).astype(int),
    'uo_lt_30': lambda df: (df['uo_7_14_28'] < 30).astype(int),
    'uo_gt_50': lambda df: (df['uo_7_14_28'] > 50).astype(int),
    'uo_lt_50': lambda df: (df['uo_7_14_28'] < 50).astype(int),
    'uo_rising': lambda df: (df['uo_7_14_28'] > df['uo_7_14_28'].shift(1)).astype(int),
    'uo_falling': lambda df: (df['uo_7_14_28'] < df['uo_7_14_28'].shift(1)).astype(int),
    
    # Keltner Channel expressions
    'close_gt_keltner_upper': lambda df: (df['close'] > df['keltner_upper_20']).astype(int),
    'close_lt_keltner_lower': lambda df: (df['close'] < df['keltner_lower_20']).astype(int),
    'close_in_keltner_middle': lambda df: ((df['close'] > df['keltner_lower_20'] * 1.01) & (df['close'] < df['keltner_upper_20'] * 0.99)).astype(int),
    'keltner_squeeze': lambda df: ((df['keltner_upper_20'] - df['keltner_lower_20']) < (df['keltner_upper_20'] - df['keltner_lower_20']).rolling(20).mean() * 0.8).astype(int),
    
    # Donchian Channel expressions
    'close_gt_donchian_upper': lambda df: (df['close'] > df['donchian_upper_20']).astype(int),
    'close_lt_donchian_lower': lambda df: (df['close'] < df['donchian_lower_20']).astype(int),
    'close_near_donchian_upper': lambda df: (df['close'] > df['donchian_upper_20'] * 0.98).astype(int),
    'close_near_donchian_lower': lambda df: (df['close'] < df['donchian_lower_20'] * 1.02).astype(int),
    'donchian_breakout_up': lambda df: ((df['close'] > df['donchian_upper_20']) & (df['close'].shift(1) <= df['donchian_upper_20'].shift(1))).astype(int),
    'donchian_breakout_down': lambda df: ((df['close'] < df['donchian_lower_20']) & (df['close'].shift(1) >= df['donchian_lower_20'].shift(1))).astype(int),
    
    # Multi-indicator combinations
    'rsi_oversold_and_cci_oversold': lambda df: ((df['rsi_14'] < 30) & (df['cci_20'] < -100)).astype(int),
    'rsi_overbought_and_cci_overbought': lambda df: ((df['rsi_14'] > 70) & (df['cci_20'] > 100)).astype(int),
    'adx_strong_and_plus_di_up': lambda df: ((df['adx_14'] > 25) & (df['plus_di_14'] > df['minus_di_14'])).astype(int),
    'adx_strong_and_minus_di_up': lambda df: ((df['adx_14'] > 25) & (df['minus_di_14'] > df['plus_di_14'])).astype(int),
    'williams_oversold_and_mfi_oversold': lambda df: ((df['williams_r_14'] < -80) & (df['mfi_14'] < 20)).astype(int),
    'williams_overbought_and_mfi_overbought': lambda df: ((df['williams_r_14'] > -20) & (df['mfi_14'] > 80)).astype(int),
    'aroon_bullish_and_roc_positive': lambda df: ((df['aroon_up_25'] > df['aroon_down_25']) & (df['roc_10'] > 0)).astype(int),
    'aroon_bearish_and_roc_negative': lambda df: ((df['aroon_down_25'] > df['aroon_up_25']) & (df['roc_10'] < 0)).astype(int),
    
    # Advanced momentum combinations
    'momentum_acceleration_up': lambda df: ((df['roc_5'] > df['roc_10']) & (df['roc_10'] > df['roc_20'])).astype(int),
    'momentum_acceleration_down': lambda df: ((df['roc_5'] < df['roc_10']) & (df['roc_10'] < df['roc_20'])).astype(int),
    'triple_momentum_bullish': lambda df: ((df['rsi_14'] > 50) & (df['tsi_25_13'] > 0) & (df['roc_10'] > 0)).astype(int),
    'triple_momentum_bearish': lambda df: ((df['rsi_14'] < 50) & (df['tsi_25_13'] < 0) & (df['roc_10'] < 0)).astype(int),
    
    # Volume-price confirmation with new indicators
    'volume_price_momentum_up': lambda df: ((df['volume'] > df['volume_sma_20'] * 1.2) & (df['roc_5'] > 2) & (df['close'] > df['sma_20'])).astype(int),
    'volume_price_momentum_down': lambda df: ((df['volume'] > df['volume_sma_20'] * 1.2) & (df['roc_5'] < -2) & (df['close'] < df['sma_20'])).astype(int),
    'mfi_volume_divergence_bull': lambda df: ((df['mfi_14'] > 50) & (df['volume'] < df['volume_sma_10']) & (df['close'] > df['close'].shift(1))).astype(int),
    'mfi_volume_divergence_bear': lambda df: ((df['mfi_14'] < 50) & (df['volume'] < df['volume_sma_10']) & (df['close'] < df['close'].shift(1))).astype(int),
    
    # Channel and band combinations
    'bb_keltner_squeeze': lambda df: ((df['bb_upper_20'] - df['bb_lower_20']) < (df['keltner_upper_20'] - df['keltner_lower_20'])).astype(int),
    'bb_expansion': lambda df: ((df['bb_upper_20'] - df['bb_lower_20']) > (df['bb_upper_20'] - df['bb_lower_20']).rolling(10).mean() * 1.2).astype(int),
    'multi_channel_breakout_up': lambda df: ((df['close'] > df['bb_upper_20']) & (df['close'] > df['keltner_upper_20']) & (df['close'] > df['donchian_upper_20'])).astype(int),
    'multi_channel_breakout_down': lambda df: ((df['close'] < df['bb_lower_20']) & (df['close'] < df['keltner_lower_20']) & (df['close'] < df['donchian_lower_20'])).astype(int),
    
    # Advanced trend expressions
    'trend_strength_bull': lambda df: ((df['adx_14'] > 25) & (df['sma_5'] > df['sma_20']) & (df['ema_12'] > df['ema_26']) & (df['aroon_up_25'] > 70)).astype(int),
    'trend_strength_bear': lambda df: ((df['adx_14'] > 25) & (df['sma_5'] < df['sma_20']) & (df['ema_12'] < df['ema_26']) & (df['aroon_down_25'] > 70)).astype(int),
    'trend_reversal_bull': lambda df: ((df['rsi_14'] < 30) & (df['williams_r_14'] < -80) & (df['cci_20'] < -100) & (df['close'] > df['close'].shift(1))).astype(int),
    'trend_reversal_bear': lambda df: ((df['rsi_14'] > 70) & (df['williams_r_14'] > -20) & (df['cci_20'] > 100) & (df['close'] < df['close'].shift(1))).astype(int),
    
    # Multi-timeframe momentum
    'short_vs_long_momentum': lambda df: ((df['roc_5'] > 0) & (df['roc_20'] < 0)).astype(int),
    'momentum_divergence_bull': lambda df: ((df['close'] < df['close'].shift(5)) & (df['rsi_14'] > df['rsi_14'].shift(5))).astype(int),
    'momentum_divergence_bear': lambda df: ((df['close'] > df['close'].shift(5)) & (df['rsi_14'] < df['rsi_14'].shift(5))).astype(int),
    
    # Multi-day price comparisons
    'price_vs_2d_ago_gt_1p02': lambda df: price_vs_previous_close(df, 2, 1.02, 'gt'),
    'price_vs_2d_ago_lt_0p98': lambda df: price_vs_previous_close(df, 2, 0.98, 'lt'),
    'price_vs_3d_ago_gt_1p03': lambda df: price_vs_previous_close(df, 3, 1.03, 'gt'),
    'price_vs_3d_ago_lt_0p97': lambda df: price_vs_previous_close(df, 3, 0.97, 'lt'),
    'price_vs_5d_ago_gt_1p05': lambda df: price_vs_previous_close(df, 5, 1.05, 'gt'),
    'price_vs_5d_ago_lt_0p95': lambda df: price_vs_previous_close(df, 5, 0.95, 'lt'),
    'price_vs_10d_ago_gt_1p1': lambda df: price_vs_previous_close(df, 10, 1.1, 'gt'),
    'price_vs_10d_ago_lt_0p9': lambda df: price_vs_previous_close(df, 10, 0.9, 'lt'),
    
    # Volume-price confirmation
    'volume_price_confirm_5d': lambda df: volume_price_confirmation(df, 5),
    'volume_price_confirm_10d': lambda df: volume_price_confirmation(df, 10),
    
    # === EXPERT-RECOMMENDED BUY SIGNAL COMBINATIONS ===
    
    # Strong bullish momentum (RSI + MACD confirmation)
    'rsi_macd_bullish_momentum': lambda df: ((df['rsi_14'] > 50) & (df['rsi_14'] < 70) & (df['macd_line'] > df['macd_signal']) & (df['macd_line'] > df['macd_line'].shift(1))).astype(int),
    'rsi_oversold_macd_bullish': lambda df: ((df['rsi_14'] < 35) & (df['rsi_14'] > df['rsi_14'].shift(1)) & (df['macd_line'] > df['macd_signal'])).astype(int),
    
    # ADX trend strength with directional bias
    'adx_strong_bullish_trend': lambda df: ((df['adx_14'] > 25) & (df['plus_di_14'] > df['minus_di_14']) & (df['plus_di_14'] > df['plus_di_14'].shift(1))).astype(int),
    'adx_trend_reversal_bull': lambda df: ((df['adx_14'] > 20) & (df['plus_di_14'] > df['minus_di_14']) & (df['plus_di_14'].shift(1) <= df['minus_di_14'].shift(1))).astype(int),
    
    # Williams %R oversold bounce with confirmation
    'williams_oversold_bounce': lambda df: ((df['williams_r_14'] < -80) & (df['williams_r_14'] > df['williams_r_14'].shift(1)) & (df['close'] > df['close'].shift(1))).astype(int),
    'williams_momentum_shift': lambda df: ((df['williams_r_14'] < -50) & (df['williams_r_14'] > df['williams_r_14'].shift(2)) & (df['volume'] > df['volume_sma_10'])).astype(int),
    
    # CCI momentum with price confirmation
    'cci_bullish_momentum': lambda df: ((df['cci_20'] > 0) & (df['cci_20'] > df['cci_20'].shift(1)) & (df['close'] > df['sma_20'])).astype(int),
    'cci_oversold_recovery': lambda df: ((df['cci_20'] < -100) & (df['cci_20'] > df['cci_20'].shift(1)) & (df['rsi_14'] > 30)).astype(int),
    
    # MFI money flow with price action
    'mfi_bullish_flow': lambda df: ((df['mfi_14'] > 50) & (df['mfi_14'] > df['mfi_14'].shift(1)) & (df['close'] > df['ema_12'])).astype(int),
    'mfi_oversold_reversal': lambda df: ((df['mfi_14'] < 30) & (df['mfi_14'] > df['mfi_14'].shift(1)) & (df['volume'] > df['volume_sma_20'] * 1.2)).astype(int),
    
    # Multi-indicator bullish convergence (high probability setups)
    'triple_bullish_convergence': lambda df: ((df['rsi_14'] > 50) & (df['macd_line'] > df['macd_signal']) & (df['adx_14'] > 25) & (df['plus_di_14'] > df['minus_di_14'])).astype(int),
    'oversold_multi_bounce': lambda df: ((df['rsi_14'] < 35) & (df['williams_r_14'] < -70) & (df['cci_20'] < -100) & (df['mfi_14'] < 40) & (df['close'] > df['close'].shift(1))).astype(int),
    
    # Volume-confirmed momentum breakouts
    'volume_momentum_breakout': lambda df: ((df['close'] > df['sma_20']) & (df['rsi_14'] > 55) & (df['volume'] > df['volume_sma_20'] * 1.5) & (df['macd_line'] > df['macd_signal'])).astype(int),
    'high_volume_bullish_cross': lambda df: ((df['ema_12'] > df['ema_26']) & (df['ema_12'].shift(1) <= df['ema_26'].shift(1)) & (df['volume'] > df['volume_sma_10'] * 2.0)).astype(int),
    
    # Trend continuation patterns
    'bullish_trend_continuation': lambda df: ((df['sma_5'] > df['sma_20']) & (df['rsi_14'] > 40) & (df['rsi_14'] < 70) & (df['adx_14'] > 20) & (df['close'] > df['sma_5'])).astype(int),
    'momentum_acceleration': lambda df: ((df['roc_5'] > 2) & (df['roc_10'] > 0) & (df['rsi_14'] > df['rsi_14'].shift(2)) & (df['macd_histogram'] > df['macd_histogram'].shift(1))).astype(int),
    
    # Channel and support breakouts
    'bollinger_squeeze_breakout': lambda df: ((df['close'] > df['bb_upper_20']) & (df['volume'] > df['volume_sma_20'] * 1.3) & (df['rsi_14'] > 50)).astype(int),
    'support_bounce_confirmation': lambda df: ((df['close'] > df['bb_lower_20'] * 1.02) & (df['rsi_14'] < 40) & (df['rsi_14'] > df['rsi_14'].shift(1)) & (df['williams_r_14'] > df['williams_r_14'].shift(1))).astype(int),
    
    # Early trend reversal signals
    'early_bullish_reversal': lambda df: ((df['rsi_14'] < 30) & (df['rsi_14'] > df['rsi_14'].shift(1)) & (df['macd_histogram'] > df['macd_histogram'].shift(1)) & (df['cci_20'] > df['cci_20'].shift(1))).astype(int),
    'momentum_divergence_bullish': lambda df: ((df['close'] < df['close'].shift(5)) & (df['rsi_14'] > df['rsi_14'].shift(5)) & (df['macd_line'] > df['macd_line'].shift(5))).astype(int),
}


def calculate_all_expressions(df):
    """Calculate all expressions and return DataFrame with results."""
    results_list = []
    column_names = []
    
    for expr_name, expr_func in EXPRESSIONS.items():
        try:
            result = expr_func(df)
            results_list.append(result)
            column_names.append(expr_name)
        except Exception as e:
            print(f"Error calculating expression {expr_name}: {e}")
            results_list.append(pd.Series([0] * len(df), index=df.index))
            column_names.append(expr_name)
    
    # Use concat for better performance - concatenate all at once
    expression_results = pd.concat(results_list, axis=1, keys=column_names)
    return expression_results


def get_expression_headers():
    """Get list of all expression names."""
    return list(EXPRESSIONS.keys())
