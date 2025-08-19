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


def parse_expression_combination(expression_name):
    """Parse expression combination string into individual expressions."""
    if ' AND ' in expression_name:
        return expression_name.split(' AND ')
    else:
        return [expression_name]


def format_expression_combination(expressions):
    """Format list of expressions into combination string."""
    if len(expressions) == 1:
        return expressions[0]
    else:
        return ' AND '.join(expressions)


def select_best_expressions(df, signal_type, opposite_signal, max_expressions, min_difference, exclude_patterns=None, max_expression_usage=None):
    """
    Select best expressions based on difference of averages.
    
    Args:
        df: DataFrame with expression results
        signal_type: Primary signal column (e.g., 'big_buy_score')
        opposite_signal: Opposite signal column (e.g., 'sell_score')
        max_expressions: Maximum number of expressions to select
        min_difference: Minimum difference required
        exclude_patterns: List of patterns to exclude from expression names
        max_expression_usage: Maximum times an individual expression can appear
    
    Returns:
        DataFrame with selected expressions
    """
    # Filter out expressions containing exclude patterns
    if exclude_patterns:
        for pattern in exclude_patterns:
            df = df[~df['combination'].str.contains(pattern, case=False, na=False)].copy()
    
    # Calculate difference between signal and opposite signal
    df = df.copy()  # Ensure we have a proper copy
    df['difference'] = df[signal_type] - df[opposite_signal]
    
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
        return pd.DataFrame(columns=['combination', signal_type, 'buy_score', 'sell_score', 'difference'])
    
    return selected_df[['combination', signal_type, 'buy_score', 'sell_score', 'difference']]


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyzer.py <fingerprint_results_file>")
        sys.exit(1)
    
    fingerprint_file = sys.argv[1]
    
    # Load configuration
    config = load_config()
    analyzer_config = config.get('analyzer', {})
    
    max_selected_expressions = analyzer_config.get('max_selected_expressions', 10)
    min_difference = analyzer_config.get('min_difference', 0.3)
    max_expression_usage = analyzer_config.get('max_expression_usage', None)
    exclude_patterns = analyzer_config.get('exclude_patterns', [])
    output_buy_file = analyzer_config.get('output_buy_file', 'fingerprint_results_file_selection_buy.csv')
    output_sell_file = analyzer_config.get('output_sell_file', 'fingerprint_results_file_selection_sell.csv')
    
    print(f"Analyzing fingerprint results from: {fingerprint_file}")
    print(f"Max selected expressions: {max_selected_expressions}")
    print(f"Minimum difference threshold: {min_difference}")
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
    
    # Select best expressions for BIG_BUY vs SELL
    print("\nSelecting best expressions for BIG_BUY signals (vs SELL)...")
    buy_selections = select_best_expressions(
        df, 'big_buy_score', 'sell_score', 
        max_selected_expressions, min_difference, exclude_patterns, max_expression_usage
    )
    
    if len(buy_selections) == 0:
        print(f"Warning: No expressions found with difference >= {min_difference} for BIG_BUY vs SELL")
    else:
        print(f"Selected {len(buy_selections)} expressions for BIG_BUY signals")
    
    # Select best expressions for SELL vs BIG_BUY
    print("\nSelecting best expressions for SELL signals (vs BIG_BUY)...")
    sell_selections = select_best_expressions(
        df, 'sell_score', 'big_buy_score', 
        max_selected_expressions, min_difference, exclude_patterns, max_expression_usage
    )
    
    if len(sell_selections) == 0:
        print(f"Warning: No expressions found with difference >= {min_difference} for SELL vs BIG_BUY")
    else:
        print(f"Selected {len(sell_selections)} expressions for SELL signals")
    
    # Add metadata columns
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepare buy selections output
    buy_output = buy_selections.copy()
    buy_output.rename(columns={'combination': 'expression'}, inplace=True)
    buy_output['signal_type'] = 'BIG_BUY'
    buy_output['selection_criteria'] = 'big_buy_score - sell_score'
    buy_output['min_difference_threshold'] = min_difference
    buy_output['analysis_timestamp'] = timestamp
    buy_output['source_file'] = fingerprint_file
    
    # Parse expression combinations for buy signals
    buy_output['expression_components'] = buy_output['expression'].apply(parse_expression_combination)
    buy_output['num_components'] = buy_output['expression_components'].apply(len)
    
    # Prepare sell selections output
    sell_output = sell_selections.copy()
    sell_output.rename(columns={'combination': 'expression'}, inplace=True)
    sell_output['signal_type'] = 'SELL'
    sell_output['selection_criteria'] = 'sell_score - big_buy_score'
    sell_output['min_difference_threshold'] = min_difference
    sell_output['analysis_timestamp'] = timestamp
    sell_output['source_file'] = fingerprint_file
    
    # Parse expression combinations for sell signals
    sell_output['expression_components'] = sell_output['expression'].apply(parse_expression_combination)
    sell_output['num_components'] = sell_output['expression_components'].apply(len)
    
    # Save results
    buy_output.to_csv(output_buy_file, index=False)
    sell_output.to_csv(output_sell_file, index=False)
    
    print(f"\nResults saved:")
    print(f"  BIG_BUY selections: {output_buy_file}")
    print(f"  SELL selections: {output_sell_file}")
    
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


if __name__ == "__main__":
    main()
