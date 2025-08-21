"""
Enhanced Stock Fingerprint Analysis System with Expression Combinations
Processes stock data and analyzes individual expressions and their combinations up to max_grouping_expressions.
"""

import json
import os
import glob
import pandas as pd
import numpy as np
import csv
from datetime import datetime
from itertools import combinations
import random

from indicators import calculate_all_indicators
from expressions import EXPRESSIONS, calculate_all_expressions


def load_config(config_path='config.json'):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def filter_low_scoring_expressions(results_file='fingerprint_results.csv', threshold=0.1):
    """Filter out expressions with average scores <= threshold."""
    try:
        results_df = pd.read_csv(results_file)
        # Calculate average score across all signal types
        results_df['avg_score'] = (results_df['big_buy_score'] + results_df['buy_score'] + results_df['sell_score']) / 3
        
        # Filter expressions with average > threshold
        good_expressions = results_df[results_df['avg_score'] > threshold]['expression'].tolist()
        print(f"Filtered {len(results_df) - len(good_expressions)} expressions with avg score <= {threshold}")
        print(f"Remaining expressions: {len(good_expressions)}")
        
        return good_expressions
    except FileNotFoundError:
        print(f"Results file {results_file} not found. Using all expressions.")
        return list(EXPRESSIONS.keys())


def generate_expression_combinations(expression_names, max_combinations=3, max_total_combinations=50000):
    """Generate combinations of expressions up to max_combinations, limited by max_total_combinations.
    Prioritizes single expressions first, then 2-expression, then 3-expression combinations sequentially."""
    all_combinations = []
    
    # Add combinations sequentially by size until we reach the limit
    for combo_size in range(1, max_combinations + 1):
        if combo_size == 1:
            # Add all individual expressions
            for expr in expression_names:
                all_combinations.append([expr])
            print(f"Added {len(expression_names)} individual expressions")
        else:
            # Generate all combinations of current size
            combos_for_size = list(combinations(expression_names, combo_size))
            
            # Check how many we can add without exceeding the limit
            remaining_slots = max_total_combinations - len(all_combinations)
            if remaining_slots <= 0:
                print(f"Reached maximum total combinations limit of {max_total_combinations}")
                break
            
            # Add combinations up to the remaining limit
            if len(combos_for_size) <= remaining_slots:
                # Add all combinations of this size
                for combo in combos_for_size:
                    all_combinations.append(list(combo))
                print(f"Added all {len(combos_for_size)} combinations of size {combo_size}")
            else:
                # Add as many as we can fit, randomly sampled for diversity
                print(f"Adding {remaining_slots} out of {len(combos_for_size)} possible {combo_size}-expression combinations")
                random.shuffle(combos_for_size)  # Shuffle for randomness
                for combo in combos_for_size[:remaining_slots]:
                    all_combinations.append(list(combo))
                print(f"Reached maximum total combinations limit of {max_total_combinations}")
                break
    
    print(f"Generated {len(all_combinations)} expression combinations (1 to {max_combinations} expressions)")
    return all_combinations


def evaluate_expression_combination(df, expression_combination):
    """Evaluate a combination of expressions using AND logic."""
    if len(expression_combination) == 1:
        # Single expression
        expr_name = expression_combination[0]
        if expr_name in df.columns:
            return df[expr_name]
        else:
            return pd.Series([0] * len(df), index=df.index)
    else:
        # Multiple expressions - use AND logic
        result = pd.Series([1] * len(df), index=df.index)
        for expr_name in expression_combination:
            if expr_name in df.columns:
                result = result & df[expr_name]
            else:
                result = result & 0
        return result.astype(int)


def create_combination_name(expression_combination):
    """Create a readable name for an expression combination."""
    if len(expression_combination) == 1:
        return expression_combination[0]
    else:
        return ' AND '.join(expression_combination)


def load_stock_data(file_path):
    """Load stock data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Missing required columns in {file_path}")
            return None
        
        # Convert date column and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Ensure numeric columns are float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def check_stock_criteria(df, config):
    """Check if stock meets minimum criteria (10% of rows must satisfy conditions)."""
    data_config = config['data']
    min_volume = data_config['min_volume']
    min_price = data_config['min_price']
    max_price = data_config['max_price']
    
    # Check conditions for each row
    volume_ok = df['volume'] >= min_volume
    price_ok = (df['close'] >= min_price) & (df['close'] <= max_price)
    all_conditions = volume_ok & price_ok
    
    # Calculate percentage of rows meeting criteria
    valid_rows = all_conditions.sum()
    total_rows = len(df)
    percentage = valid_rows / total_rows if total_rows > 0 else 0
    
    return percentage >= 0.1, percentage, all_conditions


def clean_data(df):
    """Remove rows with NaN or infinity values."""
    # Replace infinity with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Count rows before cleaning
    rows_before = len(df)
    
    # Drop rows with any NaN values
    df_clean = df.dropna()
    
    rows_after = len(df_clean)
    rows_removed = rows_before - rows_after
    
    if rows_removed > 0:
        print(f"  Removed {rows_removed} rows with NaN/infinity values")
    
    return df_clean.reset_index(drop=True)


def classify_signals(df, config):
    """Classify each row as BIG BUY, BUY, SELL, or NONE based on future price movement."""
    fingerprints_config = config['fingerprints']
    n_days = fingerprints_config['n_days']
    percent_change = fingerprints_config['percent_change']
    column_to_check = fingerprints_config.get('column_to_check', 'high')
    
    signals = []
    
    for i in range(len(df)):
        # Skip last N rows as we can't predict future
        if i >= len(df) - n_days:
            signals.append('NONE')
            continue
        
        current_close = df.iloc[i]['close']
        
        # Look at the next N days to find maximum value in specified column
        future_slice = df.iloc[i+1:i+1+n_days]
        if len(future_slice) == 0:
            signals.append('NONE')
            continue
        
        max_future_value = future_slice[column_to_check].max()
        min_future_low = future_slice['low'].min()
        
        # Calculate percentage increase from current close to max future value
        value_increase = (max_future_value - current_close) / current_close
        
        # Calculate percentage decrease from current close to min future low
        low_decrease = (current_close - min_future_low) / current_close
        
        # Classify signal - Fixed to match memory requirements
        # BIG_BUY: high increases >X% within N days, BUY: high increases X/2 to X%
        if value_increase >= percent_change:
            signals.append('BIG_BUY')
        elif value_increase >= percent_change / 2:
            signals.append('BUY')
        elif low_decrease > 0:  # Price decreases
            signals.append('SELL')
        else:
            signals.append('NONE')
    
    return signals


def write_detailed_output(df_clean, combination_results, stock_name, detailed_output_file, combination_names):
    """Write detailed output for each date/stock with all expression values."""
    batch_size = 1000  # Write in batches to avoid memory issues
    batch_data = []
    
    for idx, row in df_clean.iterrows():
        # Create row data: date, stock, signal, then all expression values
        row_data = [
            row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else '',
            stock_name,
            row['signal']
        ] + combination_results[idx].astype(int).tolist()
        
        batch_data.append(row_data)
        
        # Write batch when it reaches batch_size
        if len(batch_data) >= batch_size:
            with open(detailed_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(batch_data)
            batch_data = []
    
    # Write remaining data
    if batch_data:
        with open(detailed_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(batch_data)


def process_single_stock_combinations(file_path, config, valid_conditions, expression_combinations, detailed_output_file=None, combination_names=None):
    """Process a single stock file and return aggregated results for expression combinations."""
    print(f"Processing: {os.path.basename(file_path)}")
    
    # Load and prepare data
    df = load_stock_data(file_path)
    if df is None:
        return None
    
    # Filter to only valid rows
    df = df[valid_conditions].reset_index(drop=True)
    
    if len(df) < 10:  # Need minimum data points
        print(f"  Insufficient data after filtering: {len(df)} rows")
        return None
    
    # Calculate indicators
    df = calculate_all_indicators(df, config)
    
    # Calculate expressions
    expression_results = calculate_all_expressions(df)
    
    # Combine with original data
    df_combined = pd.concat([df, expression_results], axis=1)
    
    # Clean data (remove NaN/infinity)
    df_clean = clean_data(df_combined)
    
    if len(df_clean) < 5:  # Need minimum clean data points
        print(f"  Insufficient clean data: {len(df_clean)} rows")
        return None
    
    # Classify signals
    df_clean['signal'] = classify_signals(df_clean, config)
    
    # Count signals
    signal_counts = df_clean['signal'].value_counts()
    big_buy_count = signal_counts.get('BIG_BUY', 0)
    buy_count = signal_counts.get('BUY', 0)
    sell_count = signal_counts.get('SELL', 0)
    
    # Evaluate expression combinations
    combination_results = []
    for combo in expression_combinations:
        combo_result = evaluate_expression_combination(df_clean, combo)
        combination_results.append(combo_result.values)
    
    combination_results = np.array(combination_results).T  # Transpose to have combinations as columns
    
    # Write detailed output if requested
    if detailed_output_file and combination_names:
        stock_name = os.path.splitext(os.path.basename(file_path))[0]
        write_detailed_output(df_clean, combination_results, stock_name, detailed_output_file, combination_names)
    
    # Aggregate combination results by signal type
    big_buy_rows = df_clean[df_clean['signal'] == 'BIG_BUY']
    buy_rows = df_clean[df_clean['signal'] == 'BUY']
    sell_rows = df_clean[df_clean['signal'] == 'SELL']
    
    # Sum combination values for each signal type
    if len(big_buy_rows) > 0:
        big_buy_indices = big_buy_rows.index.tolist()
        big_buy_sums = combination_results[big_buy_indices].sum(axis=0)
    else:
        big_buy_sums = np.zeros(len(expression_combinations))
    
    if len(buy_rows) > 0:
        buy_indices = buy_rows.index.tolist()
        buy_sums = combination_results[buy_indices].sum(axis=0)
    else:
        buy_sums = np.zeros(len(expression_combinations))
    
    if len(sell_rows) > 0:
        sell_indices = sell_rows.index.tolist()
        sell_sums = combination_results[sell_indices].sum(axis=0)
    else:
        sell_sums = np.zeros(len(expression_combinations))
    
    return {
        'big_buy_array': big_buy_sums,
        'buy_array': buy_sums,
        'sell_array': sell_sums,
        'big_buy_count': big_buy_count,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'total_rows': len(df_clean)
    }


def main():
    """Main execution function for enhanced fingerprint analysis with expression combinations."""
    print("Starting Enhanced Stock Fingerprint Analysis with Expression Combinations...")
    
    # Load configuration
    config = load_config()
    
    # Get stock data path
    data_path = config['data']['stock_data_path_d']
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist")
        return
    
    # Get configuration
    fp_config = config['fingerprints']
    max_grouping_expressions = fp_config.get('max_grouping_expressions', 3)
    max_total_combinations = fp_config.get('max_total_combinations', 50000)
    max_stocks = fp_config.get('max_stocks', None)
    print_detailed_output = fp_config.get('print_detailed_output', False)
    
    # Create dated output filenames
    base_filename = fp_config.get('output_file', 'fingerprint_results.csv')
    detailed_filename = fp_config.get('detailed_output_file', 'fingerprint_results_detailed.csv')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{timestamp}_{base_filename}"
    detailed_output_file = f"{timestamp}_{detailed_filename}"
    
    print(f"Maximum expression combinations: {max_grouping_expressions}")
    print(f"Maximum total combinations: {max_total_combinations:,}")
    if max_stocks:
        print(f"Maximum stocks to process: {max_stocks}")
    print(f"Results will be saved to: {output_file}")
    if print_detailed_output:
        print(f"Detailed results will be saved to: {detailed_output_file}")
    else:
        print("Detailed output disabled (print_detailed_output = false)")
    
    # Filter low-scoring expressions
    good_expressions = filter_low_scoring_expressions()
    
    # Generate expression combinations
    expression_combinations = generate_expression_combinations(good_expressions, max_grouping_expressions, max_total_combinations)
    
    # Create combination names for output
    combination_names = [create_combination_name(combo) for combo in expression_combinations]
    num_combinations = len(expression_combinations)
    
    print(f"Will analyze {num_combinations} expression combinations")
    
    # Initialize detailed output file with headers only if enabled
    if print_detailed_output:
        detailed_headers = ['date', 'stock', 'signal'] + combination_names
        with open(detailed_output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(detailed_headers)
        print(f"Initialized detailed output file with {len(detailed_headers)} columns")
        detailed_output_file_param = detailed_output_file
        combination_names_param = combination_names
    else:
        detailed_output_file_param = None
        combination_names_param = None
    
    # Find all CSV files
    csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {data_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Initialize global aggregation arrays
    global_big_buy_array = np.zeros(num_combinations)
    global_buy_array = np.zeros(num_combinations)
    global_sell_array = np.zeros(num_combinations)
    
    global_big_buy_count = 0
    global_buy_count = 0
    global_sell_count = 0
    
    processed_stocks = 0
    skipped_stocks = 0
    
    # Process each stock
    for file_path in csv_files:
        # Check if we've reached the maximum number of stocks to process
        if max_stocks and processed_stocks >= max_stocks:
            print(f"Reached maximum stocks limit of {max_stocks}. Stopping processing.")
            break
        try:
            # Load stock data for criteria check
            df = load_stock_data(file_path)
            if df is None:
                skipped_stocks += 1
                continue
            
            # Check if stock meets criteria
            meets_criteria, percentage, valid_conditions = check_stock_criteria(df, config)
            
            if not meets_criteria:
                print(f"Skipping {os.path.basename(file_path)}: Only {percentage:.1%} of rows meet criteria")
                skipped_stocks += 1
                continue
            
            # Process the stock with combinations
            result = process_single_stock_combinations(file_path, config, valid_conditions, expression_combinations, detailed_output_file_param, combination_names_param)
            
            if result is None:
                skipped_stocks += 1
                continue
            
            # Aggregate results
            global_big_buy_array += result['big_buy_array']
            global_buy_array += result['buy_array']
            global_sell_array += result['sell_array']
            
            global_big_buy_count += result['big_buy_count']
            global_buy_count += result['buy_count']
            global_sell_count += result['sell_count']
            
            processed_stocks += 1
            
            print(f"  Processed: {result['total_rows']} rows, "
                  f"BIG_BUY: {result['big_buy_count']}, "
                  f"BUY: {result['buy_count']}, "
                  f"SELL: {result['sell_count']}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            skipped_stocks += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"Processed stocks: {processed_stocks}")
    print(f"Skipped stocks: {skipped_stocks}")
    print(f"Total samples - BIG_BUY: {global_big_buy_count}, BUY: {global_buy_count}, SELL: {global_sell_count}")
    
    # Normalize arrays by dividing by counts
    if global_big_buy_count > 0:
        normalized_big_buy = global_big_buy_array / global_big_buy_count
    else:
        normalized_big_buy = np.zeros(num_combinations)
    
    if global_buy_count > 0:
        normalized_buy = global_buy_array / global_buy_count
    else:
        normalized_buy = np.zeros(num_combinations)
    
    if global_sell_count > 0:
        normalized_sell = global_sell_array / global_sell_count
    else:
        normalized_sell = np.zeros(num_combinations)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'combination': combination_names,
        'big_buy_score': normalized_big_buy,
        'buy_score': normalized_buy,
        'sell_score': normalized_sell
    })
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"\nEnhanced fingerprint analysis complete. Results saved to {output_file}")
    
    # Save sample counts
    counts_df = pd.DataFrame({
        'signal_type': ['BIG_BUY', 'BUY', 'SELL'],
        'sample_count': [global_big_buy_count, global_buy_count, global_sell_count],
        'processed_stocks': [processed_stocks] * 3
    })
    
    counts_df.to_csv('fingerprint_enhanced_counts.csv', index=False)
    print("Enhanced sample counts saved to fingerprint_enhanced_counts.csv")
    
    # Print summary statistics
    print("\nTop expression combinations for each signal type:")
    print("\nBIG_BUY (top 5):")
    big_buy_top = results_df.nlargest(5, 'big_buy_score')[['combination', 'big_buy_score']]
    for _, row in big_buy_top.iterrows():
        print(f"  {row['combination']}: {row['big_buy_score']:.3f}")
    
    print("\nBUY (top 5):")
    buy_top = results_df.nlargest(5, 'buy_score')[['combination', 'buy_score']]
    for _, row in buy_top.iterrows():
        print(f"  {row['combination']}: {row['buy_score']:.3f}")
    
    print("\nSELL (top 5):")
    sell_top = results_df.nlargest(5, 'sell_score')[['combination', 'sell_score']]
    for _, row in sell_top.iterrows():
        print(f"  {row['combination']}: {row['sell_score']:.3f}")


if __name__ == "__main__":
    main()
