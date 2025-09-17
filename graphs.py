import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import calculate_profit_factor

def perform_profit_factor_permutation_test(returns, n_permutations=10000):
    """
    Perform permutation test to evaluate statistical significance of profit factor.
    """
    observed_pf = calculate_profit_factor(returns)
    np.random.seed(21)
    permutation_pfs = np.zeros(n_permutations)

    for i in range(n_permutations):
        random_signs = np.random.choice([-1, 1], size=len(returns))
        permuted_returns = returns * random_signs
        permutation_pfs[i] = calculate_profit_factor(permuted_returns)

    p_value = np.mean(permutation_pfs >= observed_pf)

    plt.figure(figsize=(10, 6))
    plt.hist(permutation_pfs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(observed_pf, color='red', linestyle='dashed', linewidth=2, label=f'Observed PF = {observed_pf:.2f}')
    plt.title(f'Profit Factor Permutation Test (p-value = {p_value:.4f})')
    plt.xlabel('Profit Factor')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n=== Profit Factor Permutation Test ===")
    print(f"Observed Profit Factor: {observed_pf:.4f}")
    print(f"Mean Permutation Profit Factor: {np.mean(permutation_pfs):.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: The observed profit factor is statistically significant (p < 0.05)")
    else:
        print("Result: The observed profit factor is NOT statistically significant (p >= 0.05)")

    return p_value


def plot_cumulative_returns(results, rolling_window=30):
    """
    Plot cumulative log returns with a rolling mean and volatility bands.
    """
    if results.empty:
        print("No data to plot.")
        return

    results = results.copy()
    results['Trade Number'] = range(1, len(results)+1)

    plt.figure(figsize=(12, 6))
    cumulative_log_returns = np.log1p(results['Return']).cumsum()
    plt.plot(results['Trade Number'], cumulative_log_returns, color='red', linewidth=1.5, label='Cumulative Log Returns')
    plt.title('Cumulative Log Returns (Trade Sequence)')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Log Returns')
    plt.grid(True)
    plt.axhline(0, color='black', linestyle='--')

    rolling_mean = cumulative_log_returns.rolling(rolling_window).mean()
    rolling_std = cumulative_log_returns.rolling(rolling_window).std()
    plt.plot(results['Trade Number'], rolling_mean, color='blue', linestyle='--', linewidth=1, label=f'{rolling_window}-Trade Rolling Mean')
    plt.fill_between(results['Trade Number'], rolling_mean - 2*rolling_std, rolling_mean + 2*rolling_std, color='yellow', alpha=0.5, label='2σ Volatility Bands')

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rolling_sharpe_ratio(results, window_hours=6, resample_freq='1h'):
    """
    Plot rolling Sharpe ratio over time.
    Parameters:
    - window_hours: The rolling window for Sharpe calculation (in hours).
    - resample_freq: The frequency to resample returns (e.g., '1h', '4h', '1D').
    """
    if results.empty:
        print("No data to plot.")
        return

    results = results.copy()
    results['Exit Time'] = pd.to_datetime(results['Exit Time'])

    # Resample returns to the desired frequency
    returns = results.set_index('Exit Time')['Return'].resample(resample_freq).sum()
    valid_returns = returns.dropna()

    if len(valid_returns) < window_hours:
        print(f"Not enough data points for a {window_hours}-period rolling window.")
        return

    rolling_sharpe = valid_returns.rolling(window=window_hours).mean() / valid_returns.rolling(window=window_hours).std()

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_sharpe.index, rolling_sharpe, color='purple', linewidth=1.5)
    plt.title(f'Rolling {window_hours}-Period Sharpe Ratio (Resampled: {resample_freq})')
    plt.xlabel('Time')
    plt.ylabel('Sharpe Ratio')
    plt.axhline(0, color='black', linestyle='--')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_drawdown(results):
    """
    Plot the strategy's drawdown over the trade sequence.
    """
    if results.empty:
        print("No data to plot.")
        return

    results = results.copy()
    results['Trade Number'] = range(1, len(results)+1)
    results['Exit Time'] = pd.to_datetime(results['Exit Time'])

    cumulative_log_returns = np.log1p(results['Return']).cumsum()
    cumulative_returns = np.exp(cumulative_log_returns) - 1
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak

    max_dd = drawdown.min()
    max_dd_end_idx = drawdown.idxmin()
    max_dd_start_idx = drawdown[:max_dd_end_idx].idxmax()

    max_dd_start = results.loc[max_dd_start_idx, 'Exit Time']
    max_dd_end = results.loc[max_dd_end_idx, 'Exit Time']
    max_dd_duration = (max_dd_end - max_dd_start).total_seconds() / 86400

    plt.figure(figsize=(12, 6))
    plt.plot(results['Trade Number'], drawdown, color='darkred', linewidth=1.5)
    plt.title(f'Strategy Drawdown (Max: {max_dd:.2%}, Duration: {max_dd_duration:.1f} days)')
    plt.xlabel('Trade Number')
    plt.ylabel('Drawdown')
    plt.axhline(0, color='black', linestyle='--')

    max_dd_trade_num = results.loc[max_dd_end_idx, 'Trade Number']
    prev_trade_num = results.loc[max_dd_start_idx, 'Trade Number']
    plt.annotate(f'Max Drawdown: {max_dd:.2%}\nDuration: {max_dd_duration:.1f} days',
                xy=(max_dd_trade_num, max_dd),
                xytext=(max_dd_trade_num - (max_dd_trade_num - prev_trade_num)/2, max_dd*1.5),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black'))

    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_return_distribution(results, zoom_percentile=0.01):
    """
    Plot the distribution of trade returns.
    """
    if results.empty:
        print("No data to plot.")
        return

    print("\n--- Return Statistics ---")
    stats_df = results['Return'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999])
    print(stats_df.to_string(float_format="{:.4f}".format))

    print(f"\n--- Worst Outcomes (Bottom {zoom_percentile*100}%) ---")
    worst_percentiles = [zoom_percentile, zoom_percentile/10, zoom_percentile/100]
    worst_returns = results['Return'].quantile(worst_percentiles)
    for p, val in zip(worst_percentiles, worst_returns):
        print(f"Worst {p*100:.4f}%: {val:.4f}")

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.hist(results['Return'], bins=100, color='skyblue', edgecolor='black')
    plt.title('Return Distribution (All Trades)')
    plt.xlabel('Return per Trade')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    zoomed_data = results[results['Return'] > results['Return'].quantile(zoom_percentile)]
    plt.hist(zoomed_data['Return'], bins=50, color='lightgreen', edgecolor='black')
    plt.title(f'Return Distribution (Bottom {zoom_percentile*100}% Removed)')
    plt.xlabel('Return per Trade')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_returns_by_hour(results):
    """
    Plot average returns grouped by the hour of day.
    """
    if results.empty:
        print("No data to plot.")
        return

    results = results.copy()
    results['DateTime'] = pd.to_datetime(results['DateTime'])
    hourly_returns = results.groupby(results['DateTime'].dt.hour)['Return'].mean()

    plt.figure(figsize=(12, 6))
    hourly_returns.plot(kind='bar', color='purple', alpha=0.7)
    plt.title('Average Return by Hour of Day')
    plt.xlabel('Hour (24h)')
    plt.ylabel('Mean Return per Trade')
    plt.xticks(rotation=0)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def plot_kelly_criterion_analysis(results, time_windows=None, kelly_fractions=None):
    """
    Perform and plot Kelly Criterion Monte Carlo simulations for different time windows.
    """
    if results.empty:
        print("No data to analyze.")
        return

    if time_windows is None:
        time_windows = [
            ('1 Day', pd.Timedelta(days=1)),
            ('3 Days', pd.Timedelta(days=3)),
            ('1 Week', pd.Timedelta(days=7)),
            ('Full Dataset', None)
        ]

    if kelly_fractions is None:
        kelly_fractions = [1.0, 0.5, 0.3]

    results = results.copy()
    results['DateTime'] = pd.to_datetime(results['DateTime'])
    end_date_dt = results['DateTime'].max()

    def monte_carlo_kelly(returns, kelly_fraction=1.0, n_sims=10000):
        np.random.seed(69)
        all_terminal = []
        for _ in range(n_sims):
            sampled_returns = np.random.choice(returns, size=len(returns), replace=True)
            capital = 1000
            for ret in sampled_returns:
                capital *= (1 + kelly_fraction * ret)
            all_terminal.append(capital)
        return pd.Series(all_terminal)

    def plot_multiple_pnl_paths(returns, kelly_fraction=1.0, num_paths=100, label=''):
        plt.figure(figsize=(10, 4))
        for _ in range(num_paths):
            path = np.cumprod(1 + kelly_fraction * np.random.choice(returns, size=len(returns)))
            plt.plot(path, lw=1, alpha=0.3)
        plt.title(f'Multiple PnL Paths ({label}, {int(kelly_fraction*100)}% Kelly)')
        plt.xlabel('Trade Number')
        plt.ylabel('Normalized Capital')
        plt.grid(True)
        plt.show()

    for label, time_delta in time_windows:
        print(f"\n--- {label} Kelly Simulations ---")
        if time_delta is not None:
            start_date_dt = end_date_dt - time_delta
            subset = results[results['DateTime'] > start_date_dt]
        else:
            subset = results

        if subset.empty:
            print(f"No data for {label} window")
            continue

        returns = subset['Return'].values

        for fraction in kelly_fractions:
            terminal_pnl = monte_carlo_kelly(returns, kelly_fraction=fraction, n_sims=5000)
            stats_table = terminal_pnl.describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

            print(f"\nKelly Fraction: {int(fraction*100)}%")
            print(stats_table.to_string(float_format="{:.2f}".format))

            plt.figure(figsize=(10, 6))
            plt.hist(terminal_pnl, bins=50, alpha=0.7, color='magenta', edgecolor='black')
            plt.title(f'Terminal PnL Distribution ({label}, {int(fraction*100)}% Kelly)')
            plt.xlabel('Terminal Capital')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

            plot_multiple_pnl_paths(returns, kelly_fraction=fraction, num_paths=50, label=label)


def plot_trade_direction_and_exit_analysis(results):
    """
    Plot trade entry direction and exit reason performance.
    """
    if results.empty:
        print("No data to plot.")
        return

    # 1. Trade Entry Distribution
    plt.figure(figsize=(10, 6))
    entry_counts = results['Direction'].value_counts()
    colors = ['green' if dir == 'Long' else 'red' for dir in entry_counts.index]
    entry_counts.plot(kind='bar', color=colors, edgecolor='black')
    plt.title('Trade Entry Distribution')
    plt.xlabel('Trade Direction')
    plt.ylabel('Number of Trades')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()

    # 2. Exit Reason Analysis
    # Handle case where 'Exit Reason' might have custom values
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    reason_counts = results['Exit Reason'].value_counts()
    reason_counts.plot(kind='bar', color='skyblue', ax=ax1, width=0.4, position=1)
    ax1.set_ylabel('Number of Trades', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    ax2 = ax1.twinx()
    avg_returns = results.groupby('Exit Reason')['Return'].mean()
    avg_returns.plot(kind='line', color='crimson', marker='o', ax=ax2, linewidth=2)
    ax2.set_ylabel('Average Return', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')

    plt.title('Exit Reason Performance')
    ax1.set_xlabel('Exit Reason')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print summaries
    print("\n--- Trade Direction Summary ---")
    print(entry_counts.to_string())

    print("\n--- Exit Reason Performance ---")
    summary_df = pd.DataFrame({'Count': reason_counts, 'Avg Return': avg_returns})
    formatters = {
        'Count': lambda x: f"{int(x)}",
        'Avg Return': lambda x: f"{x:.4f}"
    }
    print(summary_df.to_string(formatters=formatters, index=True))


def generate_full_report(results):
    """
    Generate a comprehensive performance report by calling all modular plotting functions.
    """
    if results.empty:
        print("No trades to analyze.")
        return

    print("\n" + "="*50)
    print("PERFORMANCE REPORT")
    print("="*50)

    # 1. Cumulative Returns
    plot_cumulative_returns(results, rolling_window=30)

    # 2. Rolling Sharpe Ratio
    plot_rolling_sharpe_ratio(results, window_hours=6, resample_freq='1h')

    # 3. Drawdown
    plot_drawdown(results)

    # 4. Return Distribution
    plot_return_distribution(results, zoom_percentile=0.01)

    # 5. Returns by Hour
    plot_returns_by_hour(results)

    # 6. Kelly Criterion Analysis
    plot_kelly_criterion_analysis(results)

    # 7. Trade Entry/Exit Analysis
    plot_trade_direction_and_exit_analysis(results)

    print("\n" + "="*50)
    print("END OF REPORT")
    print("="*50)