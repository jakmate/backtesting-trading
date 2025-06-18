import argparse
import os
import sys
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import random
from alpaca_trade_api.rest import REST
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

# 1. Define Objectives & Constraints
TARGET_RETURN = 1.0  # 100% annual return target
INITIAL_CAPITAL = 10000  # paper-trading capital
REBALANCE_FREQUENCY = 'ME'  # Monthly rebalance (changed from weekly)
BASE_UNIVERSE = [
    'SPY', 'QQQ', 'XLK', 'XLF', 'TLT'  # Keep some ETFs as base universe
]
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financial',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLI': 'Industrial',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate'
}
MAX_POSITIONS = 10
VOLATILITY_LOOKBACK = 20  # days for vol scaling
LOOKBACK_WINDOWS = [7, 20, 60, 120]
PERFORMANCE_WINDOW = 60  # days to monitor stock performance
MIN_STOCK_PERFORMANCE = 0.01  # minimum 1% performance over PERFORMANCE_WINDOW

# Screening Parameters
MAX_PE_RATIO = 30  # Maximum PE ratio to consider
MIN_PE_RATIO = 1   # Minimum PE ratio to consider (avoid negative or extremely low PE)
MIN_EPS_GROWTH = 0.05  # Minimum 5% EPS growth year-over-year
MIN_MARKET_CAP = 1e9  # Minimum market cap ($1 billion)
MAX_STOCKS_PER_SECTOR = 3  # Maximum number of stocks per sector
SCREEN_SIZE = 100  # Number of stocks to initially screen from each index

# Alpaca API credentials (use paper trading keys)
load_dotenv()
ALPACA_API_KEY = os.getenv('APCA_API_KEY_ID')
ALPACA_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# 2. Dynamic Universe Selection & Screening
def get_sp500_tickers():
    """
    Scrape the current S&P 500 constituents from Wikipedia.
    Returns a list of ticker symbols.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # pandas.read_html will pull in the first table found on that page
    tables = pd.read_html(requests.get(url).text)
    df = tables[0]
    return df['Symbol'].str.replace(r"\.", "-", regex=True).tolist()

def get_all_us_stocks():
    """Get all active US equities from Alpaca"""
    alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)
    assets = alpaca.list_assets(status='active', asset_class='us_equity')
    # Filter for major exchanges and tradable assets
    major_exchanges = ['NYSE', 'NASDAQ', 'NYSEARCA']
    return [asset.symbol for asset in assets 
            if asset.exchange in major_exchanges 
            and asset.tradable]

fundamentals_cache = {}

def get_fundamental_data(tickers, as_of_date=None):
    """Fetch fundamental data for stock screening"""
    if as_of_date and as_of_date in fundamentals_cache:
        df = fundamentals_cache[as_of_date]
        # reindex to just our tickers (fill missing with NaN)
        return df.reindex(tickers)
    
    fundamental_data = {}
    valid_tickers = []
    
    print(f"Fetching fundamental data for {len(tickers)} stocks...")
    
    # Process in batches to avoid API limits
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            stocks = yf.Tickers(batch)
            for ticker in batch:
                try:
                    stock = stocks.tickers[ticker]
                    info = stock.info
                    
                    # Extract relevant fundamental data
                    fundamentals = {}
                    
                    # Basic info
                    fundamentals['sector'] = info.get('sector', 'Unknown')
                    fundamentals['industry'] = info.get('industry', 'Unknown')
                    fundamentals['market_cap'] = info.get('marketCap', 0)
                    
                    # Valuation metrics
                    fundamentals['pe_ratio'] = info.get('trailingPE', float('nan'))
                    fundamentals['forward_pe'] = info.get('forwardPE', float('nan'))
                    fundamentals['peg_ratio'] = info.get('pegRatio', float('nan'))
                    fundamentals['price_to_book'] = info.get('priceToBook', float('nan'))
                    
                    # Growth & profitability
                    fundamentals['eps'] = info.get('trailingEps', float('nan'))
                    fundamentals['eps_growth'] = info.get('earningsQuarterlyGrowth', float('nan'))
                    fundamentals['profit_margin'] = info.get('profitMargins', float('nan'))
                    fundamentals['roa'] = info.get('returnOnAssets', float('nan'))
                    fundamentals['roe'] = info.get('returnOnEquity', float('nan'))
                    
                    # Get earnings history for EPS growth calculation
                    try:
                        earnings = stock.earnings
                        if not earnings.empty and len(earnings) >= 2:
                            # Calculate year-over-year EPS growth
                            current_eps = earnings.iloc[-1]['Earnings']
                            prev_eps = earnings.iloc[-2]['Earnings']
                            if prev_eps > 0:  # Avoid division by zero or negative EPS
                                fundamentals['annual_eps_growth'] = (current_eps - prev_eps) / abs(prev_eps)
                            else:
                                fundamentals['annual_eps_growth'] = 0.0
                        else:
                            fundamentals['annual_eps_growth'] = float('nan')
                    except:
                        fundamentals['annual_eps_growth'] = float('nan')
                    
                    # Store the data
                    fundamental_data[ticker] = fundamentals
                    valid_tickers.append(ticker)
                    
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {e}")
                    continue
        except Exception as e:
            print(f"Batch error: {e}")
            continue
            
    df = pd.DataFrame.from_dict(fundamental_data, orient="index")
    print(f"Successfully fetched fundamental data for {len(valid_tickers)} stocks")
    
    if as_of_date:
        fundamentals_cache[as_of_date] = df.copy()

    return df

def screen_stocks(fundamentals_df):
    """Apply fundamental screeners to find undervalued growth stocks"""
    if fundamentals_df.empty:
        return []
    
    # Initialize fundamental_score column with zeros
    fundamentals_df['fundamental_score'] = 0.0
    
    # Filter by market cap
    screened = fundamentals_df[fundamentals_df['market_cap'] >= MIN_MARKET_CAP].copy()
    
    # Filter by PE ratio (positive and reasonable)
    pe_filter = (
        (screened['pe_ratio'] >= MIN_PE_RATIO) & 
        (screened['pe_ratio'] <= MAX_PE_RATIO) &
        (~screened['pe_ratio'].isna())
    )
    screened = screened[pe_filter]
    
    # Filter by EPS growth (if available)
    eps_growth_filter = (
        (screened['annual_eps_growth'] >= MIN_EPS_GROWTH) | 
        (screened['eps_growth'] >= MIN_EPS_GROWTH)
    ) & (~screened['eps'].isna())
    
    # Apply if we have enough stocks, otherwise keep all
    if eps_growth_filter.sum() >= MAX_POSITIONS:
        screened = screened[eps_growth_filter]
    
    # Create a composite score for ranking (higher is better)
    # Lower PE is better (inverse relationship)
    if not screened.empty:
        # Normalize metrics to 0-1 range for scoring
        screened['pe_inverse'] = 1 / screened['pe_ratio']
        screened['pe_score'] = (screened['pe_inverse'] - screened['pe_inverse'].min()) / max(0.001, (screened['pe_inverse'].max() - screened['pe_inverse'].min()))
        
        # EPS growth score (higher is better)
        eps_growth_col = 'annual_eps_growth' if 'annual_eps_growth' in screened.columns else 'eps_growth'
        if not all(screened[eps_growth_col].isna()):
            screened['eps_growth_score'] = (screened[eps_growth_col] - screened[eps_growth_col].min()) / max(0.001, (screened[eps_growth_col].max() - screened[eps_growth_col].min()))
        else:
            screened['eps_growth_score'] = 0.5  # Neutral score if no data
            
        # ROE score (higher is better)
        if 'roe' in screened.columns and not all(screened['roe'].isna()):
            screened['roe_score'] = (screened['roe'] - screened['roe'].min()) / max(0.001, (screened['roe'].max() - screened['roe'].min()))
        else:
            screened['roe_score'] = 0.5  # Neutral score if no data
            
        # Composite score: 40% PE ratio, 40% EPS growth, 20% ROE
        screened['fundamental_score'] = (
            0.4 * screened['pe_score'] + 
            0.4 * screened['eps_growth_score'] + 
            0.2 * screened['roe_score']
        )
        
        # Balance sectors: get top stocks from each sector
        sector_balanced = []
        for sector in screened['sector'].unique():
            sector_stocks = screened[screened['sector'] == sector]
            top_sector = sector_stocks.nlargest(MAX_STOCKS_PER_SECTOR, 'fundamental_score')
            sector_balanced.append(top_sector)
        
        if sector_balanced:
            balanced_df = pd.concat(sector_balanced)
            # Ensure we have a score column
            if 'fundamental_score' not in balanced_df.columns:
                balanced_df['fundamental_score'] = balanced_df['market_cap'] / balanced_df['market_cap'].max()
            return balanced_df.nlargest(SCREEN_SIZE, 'fundamental_score').index.tolist()
    
        # Fallback handling
        if len(screened) < MAX_POSITIONS:
            print("Warning: Screening yielded too few stocks. Using fallback universe.")
            fallback = fundamentals_df.nlargest(MAX_POSITIONS, 'market_cap')
            fallback['fundamental_score'] = fallback['market_cap'] / fallback['market_cap'].max()
            return list(fallback.index)
        
        screened['fundamental_score'] = screened['fundamental_score'].fillna(0)
    return list(screened.nlargest(SCREEN_SIZE, 'fundamental_score').index)

def filter_by_liquidity(tickers, lookback_days=60, top_n=500):
    """
    Quick screen: keep only the top_n tickers by average daily dollar volume
    over the past lookback_days.
    """
    # Download volume + close-price for lookback_days
    df = yf.download(
        tickers,
        period=f"{lookback_days}d",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
    )
    # Compute dollar-volume per ticker
    dollar_vol = {}
    for t in tickers:
        try:
            data = df[t]
            dv = (data["Volume"] * data["Close"]).mean()
            dollar_vol[t] = dv
        except Exception:
            continue

    # Sort and take top_n
    ranked = sorted(dollar_vol, key=lambda t: dollar_vol[t], reverse=True)
    return ranked[:min(top_n, len(ranked))]


def update_universe(as_of_date=None):
    """Update the universe of stocks based on fundamental screening"""
    print("Updating stock universe...")
    
    # Get constituents from major indices
    # us_stocks = get_all_us_stocks()
    us_stocks = get_sp500_tickers()
    print("Got stocks...")
    liquid_candidates = filter_by_liquidity(us_stocks, lookback_days=60, top_n=500)
    print("Filtered...")
    # Get fundamental data
    fundamentals = get_fundamental_data(liquid_candidates, as_of_date)
    print("Fundamentals done...")
    # Apply screening criteria
    screened_stocks = screen_stocks(fundamentals)
    
    # Combine with base universe (ETFs)
    updated_universe = BASE_UNIVERSE + screened_stocks
    
    print(f"Updated universe contains {len(updated_universe)} instruments")
    return updated_universe, fundamentals

# 3. Signal & Factor Research
def fetch_price_data(tickers, start, end):
    try:
        data = yf.download(
            tickers, start=start, end=end, interval='1d', auto_adjust=True
        )['Close']
    except Exception as e:
        print(f"Warning: yf.download failed on {tickers}: {e}")
        # Try fetching symbol-by-symbol to isolate failures
        cols = {}
        for sym in tickers:
            try:
                df = yf.download(sym, start=start, end=end, interval='1d', auto_adjust=True)['Close']
                if not df.empty:
                    cols[sym] = df
            except Exception as e2:
                print(f"  • could not fetch {sym}: {e2}")
        if not cols:
            return pd.DataFrame()  # nothing fetched
        data = pd.concat(cols, axis=1)

    # Drop any columns that are entirely NaN (e.g. delisted tickers)
    data = data.dropna(axis=1, how='all')
    if data.empty:
        print("Warning: No price data available after dropping empty tickers.")
    return data

def compute_momentum(prices, lookback_days):
    return prices.pct_change(periods=lookback_days, fill_method=None)

def compute_trend(prices, window=60):
    """Compute linear trend coefficient for each stock"""
    trends = {}
    for col in prices.columns:
        series = prices[col].dropna()
        if len(series) < window:
            continue
        
        # Use only the last 'window' days
        y = series.iloc[-window:].values
        X = np.arange(len(y)).reshape(-1, 1)
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Store slope coefficient
        trends[col] = model.coef_[0] / y.mean()  # Normalized slope
    
    return pd.Series(trends)

def compute_volatility_adjusted_returns(prices, lookback=20):
    """Calculate volatility-adjusted returns (Sharpe-like ratio)"""
    returns = prices.pct_change().dropna()
    # Calculate mean return and volatility over lookback period
    mean_returns = returns.rolling(window=lookback).mean()
    std_returns = returns.rolling(window=lookback).std()
    # Calculate volatility-adjusted returns
    vol_adj_returns = mean_returns / std_returns
    return vol_adj_returns.dropna()

def compute_relative_strength(prices, market_index='SPY', lookback=60):
    """Calculate relative strength compared to market"""
    if market_index not in prices.columns:
        return pd.Series([0] * len(prices.columns), index=prices.columns)
    
    returns = prices.pct_change(periods=lookback)
    market_return = returns[market_index]
    
    # Calculate relative performance
    rel_strength = {}
    for col in prices.columns:
        if col == market_index:
            continue
        rel_strength[col] = returns[col].iloc[-1] - market_return.iloc[-1]
    
    return pd.Series(rel_strength)

def compute_fundamental_scoring(tickers, fundamentals_df):
    """Incorporate fundamental scores into stock selection"""
    if fundamentals_df is None or fundamentals_df.empty or len(tickers) == 0:
        return pd.Series([0] * len(tickers), index=tickers)
    
    # Extract fundamental scores for the tickers in our universe
    valid_tickers = [t for t in tickers if t in fundamentals_df.index]
    if not valid_tickers:
        return pd.Series([0] * len(tickers), index=tickers)
        
    # Get scores
    scores = fundamentals_df.loc[valid_tickers, 'fundamental_score']
    
    # Fill in missing scores with neutral value
    all_scores = pd.Series(0.5, index=tickers)
    all_scores.loc[valid_tickers] = scores
    
    return all_scores

# 4. Risk Management
def check_market_regime(prices, window=20):
    """Determine if market is bullish or bearish based on moving averages"""
    if 'SPY' not in prices.columns:
        return True  # Default to bullish if no market data
    
    spy_price = prices['SPY'].dropna()
    if len(spy_price) < 2*window:
        return True
    
    # Calculate short and long-term moving averages
    short_ma = spy_price.rolling(window=window).mean()
    long_ma = spy_price.rolling(window=2*window).mean()
    
    # Bullish if short MA is above long MA
    is_bullish = short_ma.iloc[-1] > long_ma.iloc[-1]
    return is_bullish

def get_market_exposure(is_bullish):
    """Determine market exposure based on regime"""
    if is_bullish:
        return 1.0  # Full exposure in bullish market
    else:
        return 0.5  # Reduced exposure in bearish market

def filter_negative_performers(prices, window=90, min_return=0.01):
    """Filter out stocks with negative performance over lookback period"""
    performance = prices.pct_change(periods=window, fill_method=None).iloc[-1]
    return performance[performance >= min_return].index.tolist()

# 5. Backtest & Walk-Forward Analysis
def backtest(prices, fundamentals_df=None):
    returns = prices.pct_change(fill_method=None)
    portfolio_returns = []
    dates = []
    
    # Track position history and contributions
    position_history = []
    stock_contributions = {}
    annual_holdings = {}  # Track holdings by year
    blacklist = {}  # Blacklist poor performers

    # Precompute standard momentum
    momentum_series = {w: compute_momentum(prices, w) for w in LOOKBACK_WINDOWS}
    vol_series = returns.rolling(window=VOLATILITY_LOOKBACK).std()
    
    # Rebalance dates (monthly)
    rebalance_dates = prices.resample(REBALANCE_FREQUENCY).last().index

    for dt in rebalance_dates:
        # Update blacklist for poor performers
        current_year = dt.year
        if current_year not in annual_holdings:
            annual_holdings[current_year] = {}
            blacklist[current_year] = set()  # Reset blacklist for new year
            
        # Skip if not enough data
        if dt <= prices.index[LOOKBACK_WINDOWS[-1]]:
            continue
            
        # Check market regime
        is_bullish = check_market_regime(prices.loc[:dt])
        market_exposure = get_market_exposure(is_bullish)
        
        # Filter universe based on performance
        valid_universe = prices.columns.tolist()
        valid_universe = [
            ticker for ticker in valid_universe 
            if ticker not in blacklist.get(current_year, set())
        ]
        
        # Filter out negative performers
        if len(prices.loc[:dt]) > PERFORMANCE_WINDOW:
            positive_performers = filter_negative_performers(
                prices.loc[:dt], 
                window=PERFORMANCE_WINDOW, 
                min_return=MIN_STOCK_PERFORMANCE
            )
            valid_universe = [t for t in valid_universe if t in positive_performers]
        
        # Ensure we have enough stocks
        if len(valid_universe) < MAX_POSITIONS:
            valid_universe = prices.columns.tolist()[:MAX_POSITIONS]  # Fall back to default universe
        
        # Build momentum snapshot
        snapshot = {}
        for w, mom in momentum_series.items():
            valid = mom.loc[:dt].dropna()
            if not valid.empty:
                snapshot[f"mom_{w}"] = valid.iloc[-1]
        if not snapshot:
            continue
            
        df_mom = pd.DataFrame(snapshot)
        
        # Calculate additional factors
        trend = compute_trend(prices.loc[:dt])

        # Volatility-adjusted returns: guard against empty DataFrame
        vol_adj_df = compute_volatility_adjusted_returns(prices.loc[:dt])
        if vol_adj_df.empty:
            vol_adj_returns = pd.Series(dtype=float)
        else:
            vol_adj_returns = vol_adj_df.iloc[-1]

        rel_strength = compute_relative_strength(prices.loc[:dt])
        
        # Get fundamental scores (if available)
        fundamental_scores = compute_fundamental_scoring(valid_universe, fundamentals_df)
        
        # Combine factors into a composite score
        df_mom['avg_mom'] = df_mom.mean(axis=1)
        
        # Create composite score with heavier weight on fundamentals
        composite_score = pd.Series(0.0, index=prices.columns)
        for ticker in valid_universe:
            score = 0
            # Technical factors (60% weight)
            if ticker in df_mom.index:
                score += 0.2 * df_mom.loc[ticker, 'avg_mom']
            if ticker in trend.index:
                score += 0.2 * trend[ticker]
            if ticker in vol_adj_returns.index:
                score += 0.1 * vol_adj_returns[ticker]
            if ticker in rel_strength.index:
                score += 0.1 * rel_strength[ticker]
                
            # Fundamental factors (40% weight)
            if ticker in fundamental_scores.index:
                score += 0.4 * fundamental_scores[ticker]
                
            composite_score[ticker] = score
        
        # Filter out everything except valid universe
        composite_score = composite_score[valid_universe]
        
        # Select top tickers
        top = composite_score.nlargest(MAX_POSITIONS).index.tolist()
        
        # Update annual holdings
        for ticker in top:
            if ticker not in annual_holdings[current_year]:
                annual_holdings[current_year][ticker] = 0
            annual_holdings[current_year][ticker] += 1
        
        # Compute equal-risk weights
        vs = vol_series.loc[:dt].dropna()
        if vs.empty:
            # No vol history yet?  just assign uniform vol=1 so weights equal
            vol = pd.Series(1.0, index=top)
        else:
            # take the last available vol row, reindex to our top tickers,
            # fill any missing with the average of that row
            row = vs.iloc[-1]
            vol = row.reindex(top)
            if vol.isna().any():
                vol.fillna(row.mean(), inplace=True)

        inv_vol = 1.0 / vol
        weights = inv_vol / inv_vol.sum()
        
        # Adjust weights based on market exposure
        weights = weights * market_exposure
        
        # If bearish market, allocate remaining to "TLT" (treasury bonds) if available
        if market_exposure < 1.0 and 'TLT' in prices.columns:
            tlt_weight = 1.0 - market_exposure
            # Adjust weights to include TLT
            weights = weights * (1 - tlt_weight)
            if 'TLT' not in top:
                top.append('TLT')
                weights_dict = dict(zip(top[:-1], weights))
                weights_dict['TLT'] = tlt_weight
                weights = pd.Series(weights_dict)
        
        # Store position info
        position_history.append({
            'date': dt,
            'positions': dict(zip(top, weights))
        })

        # Next period return
        next_dt = dt + pd.offsets.MonthEnd()  # Proper month-end alignment
        valid_dates = returns.index[(returns.index > dt) & (returns.index <= next_dt)]
        if valid_dates.empty:
            continue
            
        # Get returns for this period and selected stocks
        period_returns = returns.loc[valid_dates, top].fillna(0)
        
        # Convert weights to numpy array aligned with columns
        weight_vector = period_returns[top].columns.map(weights).values
        
        # Calculate daily portfolio returns (vectorized operation)
        daily_portfolio_returns = period_returns.dot(weight_vector)
        
        # Calculate total period return using compounding
        total_period_return = (1 + daily_portfolio_returns).prod() - 1
        
        # Store contributions (individual stock returns over period)
        for stock in top:
            stock_cum_return = (1 + period_returns[stock]).prod() - 1
            stock_contribution = weights[stock] * stock_cum_return
            
            if stock not in stock_contributions:
                stock_contributions[stock] = []
            stock_contributions[stock].append({
                'date': valid_dates[-1],
                'contribution': stock_contribution
            })
            
            # Update blacklist if stock is performing poorly
            if stock_cum_return < -0.1:  # Blacklist stocks with >10% loss
                blacklist[current_year].add(stock)
        
        # Store portfolio return
        portfolio_returns.append(total_period_return)
        dates.append(valid_dates[-1])

    perf = pd.Series(portfolio_returns, index=dates).dropna()
    if perf.empty:
        print("Warning: No valid returns generated during backtest. Returning zeroed metrics.")
        # Return empty structures matching the expected signature
        empty_cum = pd.Series(dtype=float)
        empty_contrib = pd.DataFrame()
        return perf, empty_cum, empty_contrib, {}, {}, {}, []

    cum_return = (1 + perf).cumprod() - 1
    # Guard against an empty cum_return (shouldn’t happen if perf isn’t empty, but just in case)
    if cum_return.empty:
        annualized = 0.0
    else:
        annualized = (1 + cum_return.iloc[-1]) ** (252 / len(perf)) - 1
    
    # Convert stock contributions to DataFrame
    contributions_df = {}
    for stock in stock_contributions:
        dates = [item['date'] for item in stock_contributions[stock]]
        values = [item['contribution'] for item in stock_contributions[stock]]
        contributions_df[stock] = pd.Series(values, index=dates)
    
    contributions_df = pd.DataFrame(contributions_df)
    
    # Calculate annual returns
    annual_returns = calculate_annual_returns(perf)
    
    # Calculate cumulative contribution by stock
    stock_total_contribution = {
        stock: sum(item['contribution'] for item in contribs)
        for stock, contribs in stock_contributions.items()
    }
    
    print(f"Backtest period: {perf.index[0].date()} to {perf.index[-1].date()}")
    print(f"Annualized return: {annualized:.2%}")
    
    return perf, cum_return, contributions_df, annual_returns, stock_total_contribution, annual_holdings, position_history

def calculate_annual_returns(returns_series):
    """Calculate returns by calendar year"""
    annual_returns = {}
    
    # Group returns by year and calculate cumulative return for each year
    for year in set(returns_series.index.year):
        year_returns = returns_series[returns_series.index.year == year]
        year_cum_return = (1 + year_returns).cumprod().iloc[-1] - 1
        annual_returns[year] = year_cum_return
        
    return annual_returns

def drawdown_analysis(returns):
    """Calculate maximum drawdown and related metrics."""
    # 1) cumulative returns & drop duplicate timestamps
    cum_returns = (1 + returns).cumprod()
    cum_returns = cum_returns.loc[~cum_returns.index.duplicated(keep='first')]

    # 2) running max & drawdowns
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max) - 1

    # 3) find max drawdown and its timestamp
    max_dd = drawdowns.min()
    max_dd_date = drawdowns.idxmin()

    # 4) recovery: find first time cum_returns returns to or above prior peak
    recovery_date = None
    recovery_time_days = None
    if max_dd < 0:
        # integer‐position slice to avoid label duplication
        pos = cum_returns.index.get_indexer([max_dd_date], method='bfill')[0]
        after_max = cum_returns.iloc[pos:]
        # check where it equals or exceeds the running max at max_dd_date
        prior_peak = running_max.loc[max_dd_date]
        recovered = after_max >= prior_peak
        if recovered.any():
            recovery_date = recovered.idxmax()
            recovery_time_days = (recovery_date - max_dd_date).days

    return {
        'max_drawdown': max_dd,
        'max_drawdown_date': max_dd_date,
        'recovery_date': recovery_date,
        'recovery_time_days': recovery_time_days
    }


# 6. Implementation for Monthly Screening & Live Trading
alpaca = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

def place_orders(targets, weights):
    # Get current positions
    positions = {p.symbol: p.qty for p in alpaca.list_positions()}
    
    # Cancel open orders
    for order in alpaca.list_orders(status='open'):
        alpaca.cancel_order(order.id)

    account = alpaca.get_account()
    buying_power = float(account.cash)
    portfolio_value = float(account.portfolio_value)

    # First sell positions that are not in our target
    for symbol, qty in positions.items():
        if symbol not in targets:
            alpaca.submit_order(symbol, qty, 'sell', 'market', 'day')
            print(f"Selling all {qty} shares of {symbol}")

    # Calculate target position sizes
    target_positions = {}
    for symbol in targets:
        price = alpaca.get_last_trade(symbol).price
        target_value = portfolio_value * weights[symbol]
        target_qty = int(target_value / price)
        target_positions[symbol] = target_qty

    # Adjust existing positions
    for symbol in targets:
        current_qty = int(positions.get(symbol, 0))
        target_qty = target_positions[symbol]
        delta = target_qty - current_qty
        
        if delta > 0:  # Buy
            alpaca.submit_order(symbol, delta, 'buy', 'market', 'day')
            print(f"Buying {delta} shares of {symbol}")
        elif delta < 0:  # Sell
            alpaca.submit_order(symbol, abs(delta), 'sell', 'market', 'day')
            print(f"Selling {abs(delta)} shares of {symbol}")

def monthly_rebalance():
    print("Starting monthly rebalance...")
    
    # 1. Update universe with fundamental screening
    UNIVERSE, fundamentals = update_universe()
    
    # 2. Fetch price data
    today = datetime.today()
    start_date = today - timedelta(days=365 * 2)  # 2 years of data for signals
    prices = fetch_price_data(
        UNIVERSE, start_date.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')
    )
    
    # 3. Check market regime
    is_bullish = check_market_regime(prices)
    market_exposure = get_market_exposure(is_bullish)
    
    # 4. Filter universe based on performance
    positive_performers = filter_negative_performers(
        prices, window=PERFORMANCE_WINDOW, min_return=MIN_STOCK_PERFORMANCE
    )
    valid_universe = [t for t in UNIVERSE if t in positive_performers]
    
    # Ensure we have enough stocks
    if len(valid_universe) < MAX_POSITIONS:
        valid_universe = UNIVERSE[:MAX_POSITIONS]  # Fall back to default universe
    
    # 5. Calculate technical factors
    momentums = {w: compute_momentum(prices, w).dropna().iloc[-1] for w in LOOKBACK_WINDOWS}
    df_mom = pd.DataFrame(momentums)
    df_mom['avg_mom'] = df_mom.mean(axis=1)
    
    trend = compute_trend(prices)
    # Continuing from the previous code

    vol_adj_returns = compute_volatility_adjusted_returns(prices).iloc[-1]
    rel_strength = compute_relative_strength(prices)
    
    # 6. Get fundamental scores
    fundamental_scores = compute_fundamental_scoring(valid_universe, fundamentals)
    
    # 7. Combine factors into a composite score (with heavier weight on fundamentals)
    composite_score = pd.Series(0.0, index=prices.columns)
    for ticker in valid_universe:
        score = 0
        # Technical factors (60% weight)
        if ticker in df_mom.index:
            score += 0.2 * df_mom.loc[ticker, 'avg_mom']
        if ticker in trend.index:
            score += 0.2 * trend[ticker]
        if ticker in vol_adj_returns.index:
            score += 0.1 * vol_adj_returns[ticker]
        if ticker in rel_strength.index:
            score += 0.1 * rel_strength[ticker]
            
        # Fundamental factors (40% weight)
        if ticker in fundamental_scores.index:
            score += 0.4 * fundamental_scores[ticker]
            
        composite_score[ticker] = score
    
    # Filter and select top tickers
    composite_score = composite_score[valid_universe]
    top = composite_score.nlargest(MAX_POSITIONS).index.tolist()
    
    # Calculate volatility for risk-based weighting
    vols = prices.pct_change().rolling(VOLATILITY_LOOKBACK).std().iloc[-1]
    vols = vols[top]
    inv_vol = 1 / vols
    weights = inv_vol / inv_vol.sum()
    
    # Adjust weights based on market exposure
    weights = weights * market_exposure
    
    # If bearish market, allocate remaining to "TLT" (treasury bonds) if available
    if market_exposure < 1.0 and 'TLT' in prices.columns:
        tlt_weight = 1.0 - market_exposure
        # Adjust weights to include TLT
        weights = weights * (1 - tlt_weight)
        if 'TLT' not in top:
            top.append('TLT')
            weights_dict = dict(zip(top[:-1], weights))
            weights_dict['TLT'] = tlt_weight
            weights = pd.Series(weights_dict)
    
    # Print current selection
    print(f"\nSelected {len(top)} stocks for this month:")
    for stock, weight in zip(top, weights):
        print(f"{stock}: {weight:.2%}")
        
    # Place orders
    place_orders(top, weights)
    
    return top, weights, fundamentals

# 7. Run main script
# Check if in backtest mode or live trading mode
parser = argparse.ArgumentParser(description='Run dynamic stock screener strategy')
parser.add_argument('--mode', choices=['backtest', 'live'], default='backtest', 
                    help='Run in backtest or live trading mode')
parser.add_argument('--years', type=int, default=10, 
                    help='Number of years for backtest')
args = parser.parse_args()

if args.mode == 'backtest':
    print(f"Running backtest for {args.years} years...")
    
    # Initialize with base universe
    UNIVERSE = BASE_UNIVERSE.copy()
    
    # Set up backtest period
    today = datetime.today()
    start_date = today - timedelta(days=365 * args.years)
    end_date = today
    
    # Run backtest with periodic universe updates
    backtest_results = []
    all_contributions = pd.DataFrame()
    fundamentals_history = {}
    
    # Split the backtest period into annual chunks for universe updates
    current_date = start_date
    
    while current_date < end_date:
        period_end = min(current_date + timedelta(days=365), end_date)

        rebalance_date = current_date.date()

        print(f"\nUpdating universe for period: {current_date.date()} to {period_end.date()}")
        if current_date > start_date:
            UNIVERSE, fundamentals = update_universe(as_of_date=rebalance_date)
        else:
            UNIVERSE, fundamentals = BASE_UNIVERSE.copy(), None

        # Now fetch price data and backtest exactly as before
        fetch_start = current_date - timedelta(days=365)
        price_data = fetch_price_data(
            UNIVERSE,
            fetch_start.strftime('%Y-%m-%d'),
            period_end.strftime('%Y-%m-%d')
        )

        perf, cum, contrib, annual, stock_contrib, holdings, positions = backtest(price_data, fundamentals)
        backtest_results.append({
            'period': (current_date.date(), period_end.date()),
            'performance': perf,
            'cumulative': cum,
            'annual_returns': annual,
            'holdings': holdings,
            'positions': positions
        })

        current_date = period_end
    
    # Combine results
    perf_list = [r['performance'] for r in backtest_results if not r['performance'].empty]
    if not perf_list:
        print("Error: No backtest periods produced any returns.")
        sys.exit(1)
    all_perf = pd.concat(perf_list)
    all_cum = (1 + all_perf).cumprod() - 1
    
    # Calculate overall metrics
    total_days = (all_perf.index[-1] - all_perf.index[0]).days
    annualized_return = (1 + all_cum.iloc[-1]) ** (252 / len(all_perf)) - 1
    volatility = all_perf.std() * (252 ** 0.5)
    sharpe = annualized_return / volatility if volatility > 0 else 0
    
    # Display results
    print("\n===== BACKTEST RESULTS =====")
    print(f"Period: {all_perf.index[0].date()} to {all_perf.index[-1].date()} ({total_days} days)")
    print(f"Final Return: {all_cum.iloc[-1]:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Display annual returns
    print("\nAnnual Returns:")
    annual_returns = calculate_annual_returns(all_perf)
    for year, ret in sorted(annual_returns.items()):
        print(f"{year}: {ret:.2%}")
    
    # Display drawdown analysis
    dd_analysis = drawdown_analysis(all_perf)
    print("\nDrawdown Analysis:")
    print(f"Maximum Drawdown: {dd_analysis['max_drawdown']:.2%}")
    print(f"Maximum Drawdown Date: {dd_analysis['max_drawdown_date'].date()}")
    if dd_analysis['recovery_date']:
        print(f"Recovery Date: {dd_analysis['recovery_date'].date()}")
        print(f"Recovery Time: {dd_analysis['recovery_time_days']} days")
    else:
        print("Recovery: Not yet recovered")
    
    # Create visualizations
    try:        
        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        all_cum.plot()
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.grid(True)
        plt.savefig('cumulative_returns.png')
        
        # Plot annual returns
        years = list(annual_returns.keys())
        returns = list(annual_returns.values())
        plt.figure(figsize=(10, 6))
        plt.bar(years, [r*100 for r in returns])
        plt.title('Annual Returns (%)')
        plt.xlabel('Year')
        plt.ylabel('Return (%)')
        plt.grid(axis='y')
        plt.savefig('annual_returns.png')
        
        # Plot drawdowns
        plt.figure(figsize=(12, 6))
        cum_returns = (1 + all_perf).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = (cum_returns / running_max) - 1
        drawdowns.plot()
        plt.title('Portfolio Drawdowns')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.savefig('drawdowns.png')
        
        print("\nSaved visualization charts to:")
        print("- cumulative_returns.png")
        print("- annual_returns.png")
        print("- drawdowns.png")
    except ImportError:
        print("Matplotlib not installed. Install it to visualize results.")

elif args.mode == 'live':
    print("Starting live trading mode...")
    
    # Run initial screening and rebalance
    selected_stocks, weights, fundamentals = monthly_rebalance()
    
    # Setup scheduler for monthly rebalance
    # In a real implementation, you'd use a proper scheduler like APScheduler
    print("\nSetup complete. In a real implementation, the strategy would now:")
    print("1. Run the monthly_rebalance() function on the last trading day of each month")
    print("2. Log performance metrics and positions")
    print("3. Monitor for any emergency rebalance conditions")
    
    print("\nCurrent portfolio allocation:")
    for stock, weight in zip(selected_stocks, weights):
        print(f"{stock}: {weight:.2%}")