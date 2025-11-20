#!/usr/bin/env python3
"""
Individual Coin Weight Analysis for Training Package 1
Creates separate charts for each cryptocurrency's weight evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
import re
from datetime import datetime, timedelta
import json

def parse_time(date_str):
    """Convert date string to timestamp"""
    return datetime.strptime(date_str, "%Y/%m/%d").timestamp()

def load_config(config_path):
    """Load configuration file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return None

def extract_weights_from_log(log_path):
    """Extract weight data from backtest log file"""
    weights_data = []
    coin_names = []

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        # Extract coin names
        for line in lines:
            if "Selected coins are:" in line:
                coins_match = re.search(r"\[([^\]]+)\]", line)
                if coins_match:
                    coin_str = coins_match.group(1)
                    coin_names = [coin.strip().strip("'") for coin in coin_str.split(',')]
                break

        # Extract weight data (handle multi-line vectors)
        i = 0
        while i < len(lines):
            if "the raw omega is" in lines[i]:
                # Start of a multi-line vector
                vector_lines = []
                j = i + 1

                # Collect lines that contain vector data
                while j < len(lines) and (re.search(r"^\s*[0-9.e\-+]+", lines[j]) or
                                         re.search(r"\s+[0-9.e\-+]+", lines[j])):
                    vector_lines.append(lines[j].strip())
                    j += 1

                if vector_lines:
                    # Combine all vector parts
                    full_vector = lines[i]
                    for line in vector_lines:
                        # Remove brackets and extract numbers
                        if '[' in line:
                            line = line[line.index('[')+1:]
                        if ']' in line:
                            line = line[:line.index(']')]
                        full_vector += " " + line

                    # Extract complete scientific notation numbers from the vector
                    number_pattern = r"[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?"
                    numbers = re.findall(number_pattern, full_vector)
                    weights = [float(x) for x in numbers]
                    weights_data.append(np.array(weights))

                i = j  # Skip processed lines
            else:
                i += 1

        return weights_data, coin_names

    except Exception as e:
        print(f"Error extracting weights from log file: {e}")
        return [], []

def get_coin_price_data(coin_name, time_dates, db_path="database/Data.db"):
    """Get price data for a specific coin from database"""
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)

        # Convert time dates to timestamps
        start_time = time_dates[0].timestamp()
        end_time = time_dates[-1].timestamp()

        query = """
        SELECT date, close
        FROM History
        WHERE coin = ? AND date >= ? AND date <= ?
        AND close IS NOT NULL
        ORDER BY date
        """

        df = pd.read_sql_query(query, conn, params=(coin_name, start_time, end_time))

        if not df.empty:
            df['datetime'] = pd.to_datetime(df['date'], unit='s')
            df = df.set_index('datetime')
            return df['close']

        conn.close()
        return None

    except Exception as e:
        print(f"Warning: Could not get price data for {coin_name}: {e}")
        return None

def plot_single_coin_weight(time_dates, coin_weights, coin_name, coin_stats, save_path=None):
    """Create a dual-axis chart showing weight evolution and price trend in same coordinate system"""

    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax2 = ax1.twinx()  # Create secondary y-axis

    # Sample data for cleaner visualization
    max_points = 300
    if len(time_dates) > max_points:
        sample_indices = np.linspace(0, len(time_dates)-1, max_points, dtype=int)
        sampled_times = [time_dates[i] for i in sample_indices]
        sampled_weights = coin_weights[sample_indices]
    else:
        sampled_times = time_dates
        sampled_weights = coin_weights

    # Get price data for this coin
    price_data = get_coin_price_data(coin_name, time_dates)

    # Plot weight evolution on primary axis (left)
    line1 = ax1.plot(sampled_times, sampled_weights,
                    linewidth=2.5, color='#2E86AB', alpha=0.9, label=f'{coin_name} Weight')

    # Fill area under the weight curve
    ax1.fill_between(sampled_times, 0, sampled_weights,
                    alpha=0.3, color='#2E86AB')

    # Add average weight line
    avg_weight = coin_stats['avg']
    ax1.axhline(y=avg_weight, color='red', linestyle='--', alpha=0.7,
                linewidth=2, label=f'Weight Avg: {avg_weight:.4f}')

    # Format primary axis (weights)
    ax1.set_ylabel('Portfolio Weight', fontsize=14, labelpad=10, color='#2E86AB')
    ax1.set_ylim(0, max(coin_weights.max() * 1.1, avg_weight * 2))
    ax1.tick_params(axis='y', labelcolor='#2E86AB')

    # Plot price trend on secondary axis (right)
    if price_data is not None and not price_data.empty:
        # Normalize price data for better visualization
        normalized_price = price_data / price_data.iloc[0]

        line2 = ax2.plot(price_data.index, normalized_price.values,
                         linewidth=2.0, color='#E63946', alpha=0.8, label='Normalized Price (Start=1.0)')

        # Add current price info
        current_price = normalized_price.iloc[-1]
        price_change = (current_price - 1.0) * 100
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Format secondary axis (price)
        ax2.set_ylabel('Normalized Price', fontsize=14, labelpad=10, color='#E63946')
        ax2.tick_params(axis='y', labelcolor='#E63946')

        # Calculate correlation for title
        weight_series = pd.Series(coin_weights, index=time_dates)
        aligned_price = price_data.reindex(weight_series.index, method='nearest')
        aligned_weight = weight_series.reindex(aligned_price.index, method='nearest')

        correlation = "N/A"
        if len(aligned_price) == len(aligned_weight) and len(aligned_price) > 10:
            corr_value = aligned_weight.corr(aligned_price)
            correlation = f"{corr_value:.3f}"
    else:
        ax2.text(0.5, 0.5, f'Price data not available for {coin_name}',
                transform=ax2.transAxes, ha='center', va='center',
                fontsize=14, color='red')
        correlation = "N/A"

    # Format both axes
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(5, len(sampled_times)//20)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.set_xlabel('Time', fontsize=14, labelpad=10)
    ax1.grid(True, alpha=0.3)

    # Set title with correlation info
    title = f'{coin_name} - Weight & Price Evolution (Correlation: {correlation})'
    ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    all_lines = lines1 + lines2
    all_labels = labels1 + labels2

    ax1.legend(all_lines, all_labels, loc='upper left', bbox_to_anchor=(1.02, 1),
              frameon=True, fancybox=True, shadow=True)

    # Add statistics text box
    stats_text = f"""Weight Statistics:
Avg: {coin_stats['avg']:.4f} | Std: {coin_stats['std']:.4f} | CV: {coin_stats['cv']:.2f}
Max: {coin_stats['max']:.4f} | Min: {coin_stats['min']:.4f}"""

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            verticalalignment='top', fontsize=11, family='monospace')

    # Add price info box if price data available
    if price_data is not None and not price_data.empty:
        price_change = (normalized_price.iloc[-1] - 1.0) * 100
        price_volatility = normalized_price.std()

        price_text = f"""Price Statistics:
Change: {price_change:+.1f}% | Volatility: {price_volatility:.3f}
Correlation: {correlation}"""

        ax2.text(0.98, 0.98, price_text, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
                verticalalignment='top', horizontalalignment='right', fontsize=11, family='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Dual-axis chart for {coin_name} saved to: {save_path}")
    else:
        plt.show()

    plt.close()  # Close figure to free memory

def calculate_coin_stats(coin_weights):
    """Calculate comprehensive statistics for a coin's weights"""
    return {
        'avg': np.mean(coin_weights),
        'median': np.median(coin_weights),
        'std': np.std(coin_weights),
        'min': np.min(coin_weights),
        'max': np.max(coin_weights),
        'p25': np.percentile(coin_weights, 25),
        'p75': np.percentile(coin_weights, 75),
        'range': np.max(coin_weights) - np.min(coin_weights),
        'cv': np.std(coin_weights) / np.mean(coin_weights) if np.mean(coin_weights) > 0 else 0
    }

def analyze_package(package_name):
    """Analyze a specific training package for individual coin weights"""

    print(f"\n{'='*60}")
    print(f"ANALYZING TRAINING PACKAGE {package_name}")
    print(f"{'='*60}")

    package_path = f"train_package/{package_name}"
    log_path = os.path.join(package_path, "backtestlog")
    config_path = os.path.join(package_path, "net_config.json")

    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return False

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return False

    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    # Load configuration
    config = load_config(config_path)
    if not config:
        return False

    # Extract weights
    weights_data, coin_names = extract_weights_from_log(log_path)
    if not weights_data or not coin_names:
        print("No weight data found")
        return False

    # Filter and prepare data
    expected_length = len(coin_names)
    filtered_weights = []
    for weights in weights_data:
        if len(weights) == expected_length:
            filtered_weights.append(weights)
        elif len(weights) > expected_length:
            filtered_weights.append(weights[:expected_length])
        else:
            padded = np.zeros(expected_length)
            padded[:len(weights)] = weights
            filtered_weights.append(padded)

    weights_matrix = np.array(filtered_weights)

    # Generate time points
    global_start = parse_time(config["input"]["start_date"])
    global_end = parse_time(config["input"]["end_date"])
    span = global_end - global_start
    test_portion = config["input"]["test_portion"]
    test_start = global_end - test_portion * span
    test_end = global_end

    # Extend time range for better visualization
    time_padding = span * 0.05
    extended_start = test_start - time_padding
    extended_end = test_end + time_padding

    n_steps = len(weights_matrix)
    time_points = np.linspace(test_start, test_end, n_steps)
    time_dates = [datetime.fromtimestamp(t) for t in time_points]

    print(f"Time range: {datetime.fromtimestamp(extended_start).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(extended_end).strftime('%Y-%m-%d')}")
    print(f"Processing {len(weights_matrix)} weight vectors over {len(time_dates)} time points")
    print(f"Coins: {coin_names}")

    # Create package-specific output directory
    output_dir = f"individual_coin_charts_{package_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Calculate statistics and print summary
    print(f"\n{'='*60}")
    print(f"PACKAGE {package_name} COIN WEIGHT STATISTICS SUMMARY")
    print(f"{'='*60}")

    all_stats = {}
    for i, coin in enumerate(coin_names):
        coin_weights = weights_matrix[:, i]
        coin_stats = calculate_coin_stats(coin_weights)
        all_stats[coin] = coin_stats

        print(f"{coin:12}: avg={coin_stats['avg']:.4f}, std={coin_stats['std']:.4f}, "
              f"range={coin_stats['range']:.4f}, cv={coin_stats['cv']:.2f}")

    # Sort coins by average weight (descending)
    sorted_coins = sorted(all_stats.items(), key=lambda x: x[1]['avg'], reverse=True)

    print(f"\n{'='*60}")
    print(f"CREATING INDIVIDUAL COIN CHARTS FOR PACKAGE {package_name}")
    print(f"{'='*60}")

    # Create individual chart for each coin
    for i, (coin, stats) in enumerate(sorted_coins):
        coin_idx = coin_names.index(coin)
        coin_weights = weights_matrix[:, coin_idx]

        print(f"\n[{i+1}/{len(coin_names)}] Creating chart for {coin}...")

        save_path = os.path.join(output_dir, f"{coin}_weight_analysis.png")
        plot_single_coin_weight(time_dates, coin_weights, coin, stats, save_path)

    print(f"\n{'='*60}")
    print(f"PACKAGE {package_name} ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Charts saved to: {output_dir}")
    print(f"Generated {len(coin_names)} individual charts")

    # Create summary comparison chart
    create_comparison_chart(time_dates, weights_matrix, coin_names, all_stats, output_dir)

    return True

def main():
    """Main function - analyze training packages 6 and 9 for individual coin weights"""

    print("Analyzing individual coin weights for Training Packages 6 and 9...")

    # Packages to analyze
    packages_to_analyze = ["1","6", "9","13"]

    successful_analyses = 0

    for package in packages_to_analyze:
        if analyze_package(package):
            successful_analyses += 1

    print(f"\n{'='*60}")
    print(f"OVERALL ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully analyzed {successful_analyses}/{len(packages_to_analyze)} packages")
    print("Generated individual coin charts for each package in separate directories:")
    print("  - individual_coin_charts_6/")
    print("  - individual_coin_charts_9/")

def create_comparison_chart(time_dates, weights_matrix, coin_names, all_stats, output_dir):
    """Create a comparison chart showing all coins together"""

    print(f"\nCreating comprehensive comparison chart...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Sample data
    max_points = 200
    if len(time_dates) > max_points:
        sample_indices = np.linspace(0, len(time_dates)-1, max_points, dtype=int)
        sampled_times = [time_dates[i] for i in sample_indices]
        sampled_weights = weights_matrix[sample_indices]
    else:
        sampled_times = time_dates
        sampled_weights = weights_matrix

    # Sort coins by average weight for better visualization
    sorted_coins_by_avg = sorted(coin_names, key=lambda c: all_stats[c]['avg'], reverse=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(coin_names)))

    # Plot all coins on the same chart
    for i, coin in enumerate(sorted_coins_by_avg):
        coin_idx = coin_names.index(coin)
        color = colors[coin_idx]

        ax1.plot(sampled_times, sampled_weights[:, coin_idx],
                label=f'{coin} (avg: {all_stats[coin]["avg"]:.3f})',
                linewidth=1.5, color=color, alpha=0.8)

    ax1.set_title('All Coins Weight Evolution - Comparison View', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Portfolio Weight', fontsize=14, labelpad=10)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Time formatting
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(5, len(sampled_times)//12)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.margins(x=0.02)

    # Bar chart of average weights
    avg_weights = [all_stats[coin]['avg'] for coin in sorted_coins_by_avg]
    std_weights = [all_stats[coin]['std'] for coin in sorted_coins_by_avg]

    bars = ax2.bar(range(len(sorted_coins_by_avg)), avg_weights,
                   yerr=std_weights, capsize=5, alpha=0.7,
                   color=[colors[coin_names.index(coin)] for coin in sorted_coins_by_avg],
                   edgecolor='black', linewidth=1)

    ax2.set_title('Average Weights with Standard Deviation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Cryptocurrency', fontsize=12, labelpad=10)
    ax2.set_ylabel('Average Weight', fontsize=12, labelpad=10)
    ax2.set_xticks(range(len(sorted_coins_by_avg)))
    ax2.set_xticklabels(sorted_coins_by_avg, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, avg, std) in enumerate(zip(bars, avg_weights, std_weights)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + max(avg_weights) * 0.01,
                f'{avg:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    save_path = os.path.join(output_dir, "all_coins_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Comparison chart saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()