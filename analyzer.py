#!/usr/bin/env python3
"""
Analyzer Script for Stock Fingerprint Results

This script analyzes fingerprint results and selects the best expressions
for buy and sell signals based on difference of averages.

Usage:
    python analyzer.py <fingerprint_results_file>
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime


def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def parse_expression_combination(combination_str):
    """Parse expression combination string into individual expressions."""
    # Remove spaces and split by AND
    expressions = [expr.strip() for expr in combination_str.split(' AND ')]
    return expressions

def get_expression_type(expression):
    """Extract the base type/category of an expression (e.g., 'sma', 'rsi', 'macd')."""
    # Handle common patterns
    if 'sma_' in expression:
        return 'sma'
    elif 'ema_' in expression:
        return 'ema'
    elif 'rsi_' in expression:
        return 'rsi'
    elif 'macd_' in expression:
        return 'macd'
    elif 'bb_' in expression or 'bollinger' in expression:
        return 'bollinger'
    elif 'stoch_' in expression:
        return 'stochastic'
    elif 'atr_' in expression:
        return 'atr'
    elif 'obv_' in expression:
        return 'obv'
    elif 'adx_' in expression:
        return 'adx'
    elif 'williams_' in expression or 'wr_' in expression:
        return 'williams'
    elif 'cci_' in expression:
        return 'cci'
    elif 'mfi_' in expression:
        return 'mfi'
    elif 'roc_' in expression:
        return 'roc'
    elif 'aroon_' in expression:
        return 'aroon'
    elif 'tsi_' in expression:
        return 'tsi'
    elif 'uo_' in expression or 'ultimate_' in expression:
        return 'ultimate_oscillator'
    elif 'keltner_' in expression:
        return 'keltner'
    elif 'donchian_' in expression:
        return 'donchian'
    elif 'volume_' in expression:
        return 'volume'
    elif 'momentum_' in expression:
        return 'momentum'
    elif 'hl_range' in expression:
        return 'hl_range'
    elif 'close_' in expression and ('high' in expression or 'low' in expression):
        return 'price_position'
    else:
        # For unknown patterns, use the first part before underscore or the whole expression
        parts = expression.split('_')
        return parts[0] if len(parts) > 1 else expression

def get_combination_types(combination_str):
    """Get all expression types in a combination."""
    expressions = parse_expression_combination(combination_str)
    return [get_expression_type(expr) for expr in expressions]

def select_diverse_expressions(df, signal_column, max_expressions, min_difference, exclude_patterns=None, max_expression_usage=None):
    """
    Select expressions ordered by signal score, ensuring only one expression of each type.
    """
    exclude_patterns = exclude_patterns or []
    
    # Calculate difference based on signal type
    df_copy = df.copy()
    if signal_column == 'big_buy_score':
        df_copy['difference'] = df_copy['big_buy_score'] - df_copy['sell_score']
    elif signal_column == 'sell_score':
        df_copy['difference'] = df_copy['sell_score'] - df_copy['big_buy_score']
    
    # Filter by minimum difference
    filtered_df = df_copy[df_copy['difference'] >= min_difference].copy()
    
    # Apply exclude patterns
    for pattern in exclude_patterns:
        filtered_df = filtered_df[~filtered_df['combination'].str.contains(pattern, case=False, na=False)]
    
    if len(filtered_df) == 0:
        return pd.DataFrame(columns=['combination', signal_column, 'buy_score', 'sell_score', 'difference'])
    
    # Sort by signal score (descending)
    filtered_df = filtered_df.sort_values(signal_column, ascending=False)
    
    # Track used expression types
    used_types = set()
    selected_expressions = []
    
    for _, row in filtered_df.iterrows():
        combination = row['combination']
        combination_types = get_combination_types(combination)
        
        # Check if any type in this combination is already used
        if any(expr_type in used_types for expr_type in combination_types):
            continue
        
        # Add this combination and mark its types as used
        selected_expressions.append(row)
        used_types.update(combination_types)
        
        # Stop when we have enough expressions
        if len(selected_expressions) >= max_expressions:
            break
    
    if not selected_expressions:
        if signal_column == 'big_buy_score':
            return pd.DataFrame(columns=['combination', 'big_buy_score', 'buy_score', 'sell_score', 'difference'])
        else:  # sell_score
            return pd.DataFrame(columns=['combination', 'sell_score', 'buy_score', 'big_buy_score', 'difference'])
    
    # Convert back to DataFrame
    result_df = pd.DataFrame(selected_expressions)
    
    # Ensure consistent column ordering regardless of signal type
    if signal_column == 'big_buy_score':
        return result_df[['combination', 'big_buy_score', 'buy_score', 'sell_score', 'difference']]
    else:  # sell_score
        return result_df[['combination', 'sell_score', 'buy_score', 'big_buy_score', 'difference']]


def format_expression_combination(expressions):
    """Format list of expressions into combination string."""
    if len(expressions) == 1:
        return expressions[0]
    else:
        return ' AND '.join(expressions)


def select_best_buy_expressions(df, max_expressions, min_difference, exclude_patterns=None, max_expression_usage=None):
    """
    Select best BUY expressions based on big_buy_score - sell_score.
    """
    # Filter out expressions containing exclude patterns
    if exclude_patterns:
        for pattern in exclude_patterns:
            df = df[~df['combination'].str.contains(pattern, case=False, na=False)].copy()
    
    # Calculate difference for buy signals: big_buy_score - sell_score
    df = df.copy()
    df['difference'] = df['big_buy_score'] - df['sell_score']
    
    # Filter by minimum difference
    filtered_df = df[df['difference'] >= min_difference].copy()
    
    # Sort by difference (descending)
    filtered_df = filtered_df.sort_values('difference', ascending=False)
    
    # Apply expression usage limiting if specified
    if max_expression_usage is not None:
        selected_rows = []
        expression_usage = {}
        
        for _, row in filtered_df.iterrows():
            combination = row['combination']
            
            # Parse individual expressions from combination
            if ' AND ' in combination:
                individual_expressions = [expr.strip() for expr in combination.split(' AND ')]
            else:
                individual_expressions = [combination]
            
            # Check if any expression would exceed usage limit
            can_add = True
            for expr in individual_expressions:
                current_usage = expression_usage.get(expr, 0)
                if current_usage >= max_expression_usage:
                    can_add = False
                    break
            
            # If we can add this combination, update usage counts
            if can_add:
                selected_rows.append(row)
                for expr in individual_expressions:
                    expression_usage[expr] = expression_usage.get(expr, 0) + 1
                
                # Stop if we have enough expressions
                if len(selected_rows) >= max_expressions:
                    break
        
        selected_df = pd.DataFrame(selected_rows)
    else:
        # Select top N expressions without usage limiting
        selected_df = filtered_df.head(max_expressions)
    
    if len(selected_df) == 0:
        return pd.DataFrame(columns=['combination', 'big_buy_score', 'buy_score', 'sell_score', 'difference'])
    
    return selected_df[['combination', 'big_buy_score', 'buy_score', 'sell_score', 'difference']]


def select_best_sell_expressions(df, max_expressions, min_difference, exclude_patterns=None, max_expression_usage=None):
    """
    Select best SELL expressions based on sell_score - big_buy_score.
    """
    # Filter out expressions containing exclude patterns
    if exclude_patterns:
        for pattern in exclude_patterns:
            df = df[~df['combination'].str.contains(pattern, case=False, na=False)].copy()
    
    # Calculate difference for sell signals: sell_score - big_buy_score
    df = df.copy()
    df['difference'] = df['sell_score'] - df['big_buy_score']
    
    # Filter by minimum difference
    filtered_df = df[df['difference'] >= min_difference].copy()
    
    # Sort by difference (descending)
    filtered_df = filtered_df.sort_values('difference', ascending=False)
    
    # Apply expression usage limiting if specified
    if max_expression_usage is not None:
        selected_rows = []
        expression_usage = {}
        
        for _, row in filtered_df.iterrows():
            combination = row['combination']
            
            # Parse individual expressions from combination
            if ' AND ' in combination:
                individual_expressions = [expr.strip() for expr in combination.split(' AND ')]
            else:
                individual_expressions = [combination]
            
            # Check if any expression would exceed usage limit
            can_add = True
            for expr in individual_expressions:
                current_usage = expression_usage.get(expr, 0)
                if current_usage >= max_expression_usage:
                    can_add = False
                    break
            
            # If we can add this combination, update usage counts
            if can_add:
                selected_rows.append(row)
                for expr in individual_expressions:
                    expression_usage[expr] = expression_usage.get(expr, 0) + 1
                
                # Stop if we have enough expressions
                if len(selected_rows) >= max_expressions:
                    break
        
        selected_df = pd.DataFrame(selected_rows)
    else:
        # Select top N expressions without usage limiting
        selected_df = filtered_df.head(max_expressions)
    
    if len(selected_df) == 0:
        return pd.DataFrame(columns=['combination', 'big_buy_score', 'buy_score', 'sell_score', 'difference'])
    
    return selected_df[['combination', 'big_buy_score', 'buy_score', 'sell_score', 'difference']]


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyzer.py <fingerprint_results_file>")
        sys.exit(1)
    
    fingerprint_file = sys.argv[1]
    
    # Load configuration
    config = load_config()
    analyzer_config = config.get('analyzer', {})
    
    max_selected_expressions = analyzer_config.get('max_selected_expressions', 10)
    min_difference_buy = analyzer_config.get('min_difference_buy', analyzer_config.get('min_difference', 0.1))
    min_difference_sell = analyzer_config.get('min_difference_sell', analyzer_config.get('min_difference', 0.03))
    max_expression_usage = analyzer_config.get('max_expression_usage', None)
    exclude_patterns = analyzer_config.get('exclude_patterns', [])
    output_buy_file = analyzer_config.get('output_buy_file', 'fingerprint_results_file_selection_buy.csv')
    output_sell_file = analyzer_config.get('output_sell_file', 'fingerprint_results_file_selection_sell.csv')
    output_buy_diverse_file = analyzer_config.get('output_buy_diverse_file', 'fingerprint_results_diverse_selection_buy.csv')
    output_sell_diverse_file = analyzer_config.get('output_sell_diverse_file', 'fingerprint_results_diverse_selection_sell.csv')
    
    print(f"Analyzing fingerprint results from: {fingerprint_file}")
    print(f"Max selected expressions: {max_selected_expressions}")
    print(f"Buy minimum difference threshold: {min_difference_buy}")
    print(f"Sell minimum difference threshold: {min_difference_sell}")
    if max_expression_usage is not None:
        print(f"Max expression usage: {max_expression_usage}")
    if exclude_patterns:
        print(f"Excluding patterns: {exclude_patterns}")
    
    # Load fingerprint results
    try:
        df = pd.read_csv(fingerprint_file)
    except FileNotFoundError:
        print(f"Error: File {fingerprint_file} not found")
        sys.exit(1)
    
    print(f"Loaded {len(df)} expressions from fingerprint results")
    
    # Select best expressions for BIG_BUY signals
    print("\nSelecting best expressions for BIG_BUY signals (big_buy_score - sell_score)...")
    buy_selections = select_best_buy_expressions(
        df, max_selected_expressions, min_difference_buy, exclude_patterns, max_expression_usage
    )
    
    if len(buy_selections) == 0:
        print(f"Warning: No expressions found with difference >= {min_difference_buy} for BIG_BUY signals")
    else:
        print(f"Selected {len(buy_selections)} expressions for BIG_BUY signals")
    
    # Select best expressions for SELL signals
    print("\nSelecting best expressions for SELL signals (sell_score - big_buy_score)...")
    sell_selections = select_best_sell_expressions(
        df, max_selected_expressions, min_difference_sell, exclude_patterns, max_expression_usage
    )
    
    if len(sell_selections) == 0:
        print(f"Warning: No expressions found with difference >= {min_difference_sell} for SELL signals")
    else:
        print(f"Selected {len(sell_selections)} expressions for SELL signals")
    
    # Select diverse expressions for BIG_BUY signals (ordered by signal score, one per type)
    print("\nSelecting diverse expressions for BIG_BUY signals (ordered by big_buy_score, one per type)...")
    buy_diverse_selections = select_diverse_expressions(
        df, 'big_buy_score', max_selected_expressions, min_difference_buy, exclude_patterns, max_expression_usage
    )
    
    if len(buy_diverse_selections) == 0:
        print(f"Warning: No diverse expressions found with difference >= {min_difference_buy} for BIG_BUY signals")
    else:
        print(f"Selected {len(buy_diverse_selections)} diverse expressions for BIG_BUY signals")
    
    # Select diverse expressions for SELL signals (ordered by signal score, one per type)
    print("\nSelecting diverse expressions for SELL signals (ordered by sell_score, one per type)...")
    sell_diverse_selections = select_diverse_expressions(
        df, 'sell_score', max_selected_expressions, min_difference_sell, exclude_patterns, max_expression_usage
    )
    
    if len(sell_diverse_selections) == 0:
        print(f"Warning: No diverse expressions found with difference >= {min_difference_sell} for SELL signals")
    else:
        print(f"Selected {len(sell_diverse_selections)} diverse expressions for SELL signals")
    
    # Add metadata columns
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepare buy selections output
    if len(buy_selections) > 0:
        buy_output = buy_selections.copy()
        buy_output.rename(columns={'combination': 'expression'}, inplace=True)
        buy_output['signal_type'] = 'BIG_BUY'
        buy_output['selection_criteria'] = 'big_buy_score - sell_score'
        buy_output['min_difference_threshold'] = min_difference_buy
        buy_output['analysis_timestamp'] = timestamp
        buy_output['source_file'] = fingerprint_file
        
        # Parse expression combinations for buy signals
        buy_output['expression_components'] = buy_output['expression'].apply(parse_expression_combination)
        buy_output['num_components'] = buy_output['expression_components'].apply(len)
    else:
        # Create empty DataFrame with correct columns
        buy_output = pd.DataFrame(columns=['expression', 'big_buy_score', 'buy_score', 'sell_score', 'difference', 
                                         'signal_type', 'selection_criteria', 'min_difference_threshold', 
                                         'analysis_timestamp', 'source_file', 'expression_components', 'num_components'])
    
    # Prepare sell selections output
    if len(sell_selections) > 0:
        sell_output = sell_selections.copy()
        sell_output.rename(columns={'combination': 'expression'}, inplace=True)
        sell_output['signal_type'] = 'SELL'
        sell_output['selection_criteria'] = 'sell_score - big_buy_score'
        sell_output['min_difference_threshold'] = min_difference_sell
        sell_output['analysis_timestamp'] = timestamp
        sell_output['source_file'] = fingerprint_file
        
        # Parse expression combinations for sell signals
        sell_output['expression_components'] = sell_output['expression'].apply(parse_expression_combination)
        sell_output['num_components'] = sell_output['expression_components'].apply(len)
    else:
        # Create empty DataFrame with correct columns
        sell_output = pd.DataFrame(columns=['expression', 'big_buy_score', 'buy_score', 'sell_score', 'difference', 
                                          'signal_type', 'selection_criteria', 'min_difference_threshold', 
                                          'analysis_timestamp', 'source_file', 'expression_components', 'num_components'])
    
    # Prepare buy diverse selections output
    if len(buy_diverse_selections) > 0:
        buy_diverse_output = buy_diverse_selections.copy()
        buy_diverse_output.rename(columns={'combination': 'expression'}, inplace=True)
        buy_diverse_output['signal_type'] = 'BIG_BUY'
        buy_diverse_output['selection_criteria'] = 'ordered by big_buy_score, diverse types'
        buy_diverse_output['min_difference_threshold'] = min_difference_buy
        buy_diverse_output['analysis_timestamp'] = timestamp
        buy_diverse_output['source_file'] = fingerprint_file
        
        # Parse expression combinations for buy diverse signals
        buy_diverse_output['expression_components'] = buy_diverse_output['expression'].apply(parse_expression_combination)
        buy_diverse_output['num_components'] = buy_diverse_output['expression_components'].apply(len)
        buy_diverse_output['expression_types'] = buy_diverse_output['expression'].apply(get_combination_types)
    else:
        # Create empty DataFrame with correct columns
        buy_diverse_output = pd.DataFrame(columns=['expression', 'big_buy_score', 'buy_score', 'sell_score', 'difference', 
                                                 'signal_type', 'selection_criteria', 'min_difference_threshold', 
                                                 'analysis_timestamp', 'source_file', 'expression_components', 'num_components', 'expression_types'])
    
    # Prepare sell diverse selections output
    if len(sell_diverse_selections) > 0:
        sell_diverse_output = sell_diverse_selections.copy()
        sell_diverse_output.rename(columns={'combination': 'expression'}, inplace=True)
        sell_diverse_output['signal_type'] = 'SELL'
        sell_diverse_output['selection_criteria'] = 'ordered by sell_score, diverse types'
        sell_diverse_output['min_difference_threshold'] = min_difference_sell
        sell_diverse_output['analysis_timestamp'] = timestamp
        sell_diverse_output['source_file'] = fingerprint_file
        
        # Parse expression combinations for sell diverse signals
        sell_diverse_output['expression_components'] = sell_diverse_output['expression'].apply(parse_expression_combination)
        sell_diverse_output['num_components'] = sell_diverse_output['expression_components'].apply(len)
        sell_diverse_output['expression_types'] = sell_diverse_output['expression'].apply(get_combination_types)
    else:
        # Create empty DataFrame with correct columns
        sell_diverse_output = pd.DataFrame(columns=['expression', 'sell_score', 'buy_score', 'big_buy_score', 'difference', 
                                                  'signal_type', 'selection_criteria', 'min_difference_threshold', 
                                                  'analysis_timestamp', 'source_file', 'expression_components', 'num_components', 'expression_types'])
    
    # Save results to CSV files
    buy_output.to_csv(output_buy_file, index=False)
    sell_output.to_csv(output_sell_file, index=False)
    buy_diverse_output.to_csv(output_buy_diverse_file, index=False)
    sell_diverse_output.to_csv(output_sell_diverse_file, index=False)
    
    print(f"\nResults saved:")
    print(f"  BIG_BUY selections: {output_buy_file}")
    print(f"  SELL selections: {output_sell_file}")
    print(f"  BIG_BUY diverse selections: {output_buy_diverse_file}")
    print(f"  SELL diverse selections: {output_sell_diverse_file}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"BIG_BUY expressions selected: {len(buy_selections)}")
    if len(buy_selections) > 0:
        print(f"  Average difference: {buy_selections['difference'].mean():.3f}")
        print(f"  Best difference: {buy_selections['difference'].max():.3f}")
        print(f"  Top 3 expressions:")
        for i, (_, row) in enumerate(buy_selections.head(3).iterrows()):
            print(f"    {i+1}. {row['combination']} (diff: {row['difference']:.3f})")
    
    print(f"\nSELL expressions selected: {len(sell_selections)}")
    if len(sell_selections) > 0:
        print(f"  Average difference: {sell_selections['difference'].mean():.3f}")
        print(f"  Best difference: {sell_selections['difference'].max():.3f}")
        print(f"  Top 3 expressions:")
        for i, (_, row) in enumerate(sell_selections.head(3).iterrows()):
            print(f"    {i+1}. {row['combination']} (diff: {row['difference']:.3f})")
    
    print(f"\nBIG_BUY diverse expressions selected: {len(buy_diverse_selections)}")
    if len(buy_diverse_selections) > 0:
        print(f"  Average big_buy_score: {buy_diverse_selections['big_buy_score'].mean():.3f}")
        print(f"  Best big_buy_score: {buy_diverse_selections['big_buy_score'].max():.3f}")
        print(f"  Top 3 diverse expressions:")
        for i, (_, row) in enumerate(buy_diverse_selections.head(3).iterrows()):
            types = get_combination_types(row['combination'])
            print(f"    {i+1}. {row['combination']} (score: {row['big_buy_score']:.3f}, types: {types})")
    
    print(f"\nSELL diverse expressions selected: {len(sell_diverse_selections)}")
    if len(sell_diverse_selections) > 0:
        try:
            avg_sell_score = sell_diverse_selections['sell_score'].mean()
            best_sell_score = sell_diverse_selections['sell_score'].max()
            print(f"  Average sell_score: {avg_sell_score:.3f}")
            print(f"  Best sell_score: {best_sell_score:.3f}")
        except:
            print(f"  Average sell_score: {sell_diverse_selections['sell_score'].mean()}")
            print(f"  Best sell_score: {sell_diverse_selections['sell_score'].max()}")
        print(f"  Top 3 diverse expressions:")
        for i, (_, row) in enumerate(sell_diverse_selections.head(3).iterrows()):
            types = get_combination_types(row['combination'])
            try:
                score_val = float(row['sell_score'])
                print(f"    {i+1}. {row['combination']} (score: {score_val:.3f}, types: {types})")
            except:
                print(f"    {i+1}. {row['combination']} (score: {row['sell_score']}, types: {types})")


if __name__ == "__main__":
    main()
