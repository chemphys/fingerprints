#!/usr/bin/env python3
"""
Stock Signal Evaluator - Clean implementation based on original PRD

Check Mode: Evaluate signals for target date and previous N dates, validate against future data
Predict Mode: Issue signals for given date with confidence scores
"""

import sys
import json
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
from indicators import calculate_all_indicators
from expressions import calculate_all_expressions


def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_stock_data(file_path):
    """Load and validate stock data."""
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert numeric columns to float, handling any string values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN values in critical columns
        df = df.dropna(subset=['date', 'close', 'volume'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        return None


def validate_stock_criteria(df, config, target_date):
    """Check if stock meets price and volume criteria for the target date or closest previous date."""
    target_date = pd.to_datetime(target_date)
    
    # Handle timezone awareness
    if df['date'].dt.tz is not None and target_date.tz is None:
        target_date = target_date.tz_localize('UTC')
    elif df['date'].dt.tz is None and target_date.tz is not None:
        target_date = target_date.tz_localize(None)
    
    # Find closest date on or before target date
    valid_dates = df[df['date'] <= target_date]
    if len(valid_dates) == 0:
        return False, "No data on or before target date"
    
    # Get the most recent date
    closest_date_data = valid_dates.iloc[-1]
    
    # Check price criteria for that specific date
    min_price = config['data']['min_price']
    max_price = config['data']['max_price']
    if not (min_price <= closest_date_data['close'] <= max_price):
        return False, f"Price {closest_date_data['close']:.2f} not in range [{min_price}, {max_price}]"
    
    # Check volume criteria for that specific date
    min_volume = config['data']['min_volume']
    if closest_date_data['volume'] < min_volume:
        return False, f"Volume {closest_date_data['volume']} below {min_volume}"
    
    return True, "OK"


def get_evaluation_dates(df, target_date, n_days_back):
    """Get the actual dates to evaluate for signals."""
    target_date = pd.to_datetime(target_date)
    
    # Handle timezone awareness
    if df['date'].dt.tz is not None and target_date.tz is None:
        target_date = target_date.tz_localize('UTC')
    elif df['date'].dt.tz is None and target_date.tz is not None:
        target_date = target_date.tz_localize(None)
    
    # Find all dates on or before target date
    valid_dates = df[df['date'] <= target_date]['date'].tolist()
    
    if len(valid_dates) == 0:
        return []
    
    # Get the last n_days_back dates
    return valid_dates[-n_days_back:]


def classify_future_movement(df, signal_date, n_days_forward, percent_change):
    """Classify future price movement from signal date."""
    signal_date = pd.to_datetime(signal_date)
    
    # Handle timezone awareness
    if df['date'].dt.tz is not None and signal_date.tz is None:
        signal_date = signal_date.tz_localize('UTC')
    elif df['date'].dt.tz is None and signal_date.tz is not None:
        signal_date = signal_date.tz_localize(None)
    
    # Get signal date data
    signal_data = df[df['date'] == signal_date]
    if len(signal_data) == 0:
        return 'UNKNOWN'
    
    signal_close = signal_data['close'].iloc[0]
    
    # Get future data
    future_data = df[df['date'] > signal_date].head(n_days_forward)
    if len(future_data) == 0:
        return 'UNKNOWN'
    
    # Find max high in future period
    max_high = future_data['high'].max()
    max_increase = (max_high - signal_close) / signal_close
    
    # Find min low in future period  
    min_low = future_data['low'].min()
    max_decrease = (signal_close - min_low) / signal_close
    
    # Classify based on movement
    if max_increase >= percent_change * 2:
        return 'BIG_BUY'
    elif max_increase >= percent_change:
        return 'BUY'
    elif max_decrease >= percent_change:
        return 'SELL'
    else:
        return 'NOTHING'


def evaluate_expressions_for_date(df, expression_results, selections, date_idx):
    """Evaluate expressions for a specific date index."""
    matched_expressions = []
    
    for _, row in selections.iterrows():
        expr_name = row['expression']
        
        # Handle compound expressions with AND
        if ' AND ' in expr_name:
            individual_expressions = expr_name.split(' AND ')
            all_match = True
            
            for individual_expr in individual_expressions:
                individual_expr = individual_expr.strip()
                if individual_expr not in expression_results.columns:
                    all_match = False
                    break
                if date_idx >= len(expression_results) or expression_results.iloc[date_idx][individual_expr] != 1:
                    all_match = False
                    break
            
            if all_match:
                matched_expressions.append(expr_name)
        else:
            # Handle single expressions
            if expr_name in expression_results.columns:
                if date_idx < len(expression_results) and expression_results.iloc[date_idx][expr_name] == 1:
                    matched_expressions.append(expr_name)
    
    return matched_expressions


def calculate_confidence(matched_expressions, all_selections, score_column):
    """Calculate confidence based on matched expression scores."""
    if len(matched_expressions) == 0:
        return 0.0
    
    # Get scores for matched expressions
    matched_scores = []
    for expr in matched_expressions:
        expr_row = all_selections[all_selections['expression'] == expr]
        if len(expr_row) > 0:
            score = expr_row.iloc[0].get(score_column, 0.5)
            matched_scores.append(score)
    
    # Get total possible scores
    total_scores = []
    for _, row in all_selections.iterrows():
        score = row.get(score_column, 0.5)
        total_scores.append(score)
    
    if sum(total_scores) == 0:
        return 0.0
    
    return (sum(matched_scores) / sum(total_scores)) * 100


def check_mode(target_date, buy_selections, sell_selections, config):
    """Check mode: evaluate signals and validate against future data."""
    print("STOCK,DATE,SIGNAL,TRUE_SIGNAL,CONFIDENCE,MATCHED_EXPRESSIONS")
    
    data_path = config['data']['stock_data_path_d']
    eval_config = config['evaluation']
    
    min_expressions = eval_config['min_expressions_for_signal']
    n_days_back = eval_config['n_dates_back']
    n_days_forward = eval_config['n_dates_forward']
    max_stocks_printed = eval_config['max_stocks_printed']
    percent_change = config['fingerprints']['percent_change']
    
    # Get all stock files
    stock_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    total_signals = 0
    correct_signals = 0
    n_stocks_printed = 0
    all_results = []  # Store all results for sorting
    
    for file_path in stock_files:
        stock_name = os.path.basename(file_path).replace('.csv', '')
        
        # Load stock data
        df = load_stock_data(file_path)
        if df is None:
            continue
        
        # Validate stock meets criteria for target date
        is_valid, reason = validate_stock_criteria(df, config, target_date)
        if not is_valid:
            continue
        
        # Calculate indicators
        try:
            df_with_indicators = calculate_all_indicators(df, config)
            expression_results = calculate_all_expressions(df_with_indicators)
        except Exception:
            continue
        
        # TODO remove later
        if n_stocks_printed > max_stocks_printed:
            break
        n_stocks_printed += 1
        
        # Get evaluation dates
        eval_dates = get_evaluation_dates(df_with_indicators, target_date, n_days_back)
        
        for eval_date in eval_dates:
            # Find date index
            date_idx = df_with_indicators[df_with_indicators['date'] == eval_date].index
            if len(date_idx) == 0:
                continue
            date_idx = date_idx[0]
            
            # Evaluate buy expressions
            buy_matches = evaluate_expressions_for_date(
                df_with_indicators, expression_results, buy_selections, date_idx
            )
            
            # Evaluate sell expressions  
            sell_matches = evaluate_expressions_for_date(
                df_with_indicators, expression_results, sell_selections, date_idx
            )
            
            # Check for signals
            has_buy_signal = len(buy_matches) >= min_expressions
            has_sell_signal = len(sell_matches) >= min_expressions
            
            if not (has_buy_signal or has_sell_signal):
                continue
            
            # Determine signal type and calculate confidence
            if has_buy_signal:
                signal_type = 'BUY'
                confidence = calculate_confidence(buy_matches, buy_selections, 'big_buy_score')
            else:
                signal_type = 'SELL'
                confidence = calculate_confidence(sell_matches, sell_selections, 'sell_score')
            
            # Classify actual future movement
            actual_movement = classify_future_movement(
                df_with_indicators, eval_date, n_days_forward, percent_change
            )
            
            # Check if signal was correct
            is_correct = False
            if signal_type == 'BUY' and actual_movement in ['BUY', 'BIG_BUY']:
                is_correct = True
            elif signal_type == 'SELL' and actual_movement == 'SELL':
                is_correct = True
            
            # Store result for sorting
            date_str = eval_date.strftime('%Y-%m-%d')
            if signal_type == 'BUY':
                matched_expr_str = '|'.join(buy_matches)
            else:
                matched_expr_str = '|'.join(sell_matches)
            
            result = {
                'stock': stock_name,
                'date': date_str,
                'signal': signal_type,
                'true_signal': actual_movement,
                'confidence': confidence,
                'matched_expressions': matched_expr_str,
                'is_correct': is_correct
            }
            all_results.append(result)
            
            total_signals += 1
            if is_correct:
                correct_signals += 1
    
    # Sort results by confidence (descending) and print
    all_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    for result in all_results:
        print(f"{result['stock']},{result['date']},{result['signal']},{result['true_signal']},{result['confidence']:.1f},{result['matched_expressions']}")
    
    # Print summary
    if total_signals > 0:
        accuracy = (correct_signals / total_signals) * 100
        print(f"\n=== SUMMARY ===", file=sys.stderr)
        print(f"Total signals: {total_signals}", file=sys.stderr)
        print(f"Correct signals: {correct_signals}", file=sys.stderr)
        print(f"Accuracy: {accuracy:.1f}%", file=sys.stderr)
    else:
        print(f"\n=== SUMMARY ===", file=sys.stderr)
        print(f"No signals issued", file=sys.stderr)


def predict_mode(target_date, buy_selections, sell_selections, config):
    """Predict mode: issue signals for given date only."""
    data_path = config['data']['stock_data_path_d']
    eval_config = config['evaluation']
    
    min_expressions = eval_config['min_expressions_for_signal']
    output_prefix = eval_config.get('output_prefix', 'output_')
    
    # Output file
    output_file = f"{output_prefix}{target_date}_signals.csv"
    
    results = []
    
    # Get all stock files
    stock_files = glob.glob(os.path.join(data_path, "*.csv"))
    
    for file_path in stock_files:
        stock_name = os.path.basename(file_path).replace('.csv', '')
        
        # Load stock data
        df = load_stock_data(file_path)
        if df is None:
            continue
        
        # Validate stock meets criteria for target date
        is_valid, reason = validate_stock_criteria(df, config, target_date)
        if not is_valid:
            continue
        
        # Calculate indicators
        try:
            df_with_indicators = calculate_all_indicators(df, config)
            expression_results = calculate_all_expressions(df_with_indicators)
        except Exception:
            continue
        
        # Find target date or closest previous date
        target_date_pd = pd.to_datetime(target_date)
        
        # Handle timezone awareness
        if df_with_indicators['date'].dt.tz is not None and target_date_pd.tz is None:
            target_date_pd = target_date_pd.tz_localize('UTC')
        elif df_with_indicators['date'].dt.tz is None and target_date_pd.tz is not None:
            target_date_pd = target_date_pd.tz_localize(None)
        
        # Find closest date on or before target date
        valid_dates = df_with_indicators[df_with_indicators['date'] <= target_date_pd]
        if len(valid_dates) == 0:
            continue
        
        date_idx = valid_dates.index[-1]  # Get the last (most recent) index
        
        # Evaluate expressions
        buy_matches = evaluate_expressions_for_date(
            df_with_indicators, expression_results, buy_selections, date_idx
        )
        sell_matches = evaluate_expressions_for_date(
            df_with_indicators, expression_results, sell_selections, date_idx
        )
        
        # Check for signals
        has_buy_signal = len(buy_matches) >= min_expressions
        has_sell_signal = len(sell_matches) >= min_expressions
        
        if has_buy_signal:
            confidence = calculate_confidence(buy_matches, buy_selections, 'big_buy_score')
            matched_expr_str = '|'.join(buy_matches)
            results.append({
                'stock': stock_name,
                'signal': 'BUY',
                'confidence': confidence,
                'matched_expressions': matched_expr_str
            })
        elif has_sell_signal:
            confidence = calculate_confidence(sell_matches, sell_selections, 'sell_score')
            matched_expr_str = '|'.join(sell_matches)
            results.append({
                'stock': stock_name,
                'signal': 'SELL', 
                'confidence': confidence,
                'matched_expressions': matched_expr_str
            })
    
    # Sort by confidence and save
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    print(f"Saved {len(results)} signals to {output_file}")
    for result in results[:10]:  # Show top 10
        print(f"{result['stock']}: {result['signal']} ({result['confidence']:.1f}%) - {result['matched_expressions']}")


def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python evaluator_new.py check <date> <buy_selection_file> <sell_selection_file>")
        print("  python evaluator_new.py predict <date> <buy_selection_file> <sell_selection_file>")
        print()
        print("Example:")
        print("  python evaluator_new.py check 2024-12-01 fingerprint_results_file_selection_buy.csv fingerprint_results_file_selection_sell.csv")
        sys.exit(1)
    
    mode = sys.argv[1]
    target_date = sys.argv[2]
    buy_file = sys.argv[3]
    sell_file = sys.argv[4]
    
    # Load config
    config = load_config()
    
    # Load selection files
    buy_selections = pd.read_csv(buy_file)
    sell_selections = pd.read_csv(sell_file)
    
    if mode == 'check':
        check_mode(target_date, buy_selections, sell_selections, config)
    elif mode == 'predict':
        predict_mode(target_date, buy_selections, sell_selections, config)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
