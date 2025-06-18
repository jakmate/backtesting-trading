import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from pytz import timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import os
import math

load_dotenv()

# Alpaca API configuration
ALPACA_API_KEY = os.getenv('APCA_API_KEY_ID')
ALPACA_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')

# Define market timezone
MARKET_TZ = timezone('US/Eastern')

def calculate_fees(price, shares, is_sell):
    """Calculate SEC and FINRA TAF fees for a sell order."""
    if not is_sell:
        return 0.0
    # Calculate SEC fee
    principal = price * shares
    sec_fee = (principal * 22.90) / 1e6
    sec_fee = math.ceil(sec_fee * 100) / 100  # Round up to nearest penny
    # Calculate FINRA TAF
    taf_fee = shares * 0.000119
    taf_fee = math.ceil(taf_fee * 100) / 100
    taf_fee = min(taf_fee, 5.95)  # Cap at $5.95
    total_fee = sec_fee + taf_fee
    return total_fee

def calculate_vwap(data):
    """Calculate intraday VWAP for each day."""
    data = data.copy()
    data['Date'] = data.index.date
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    data['TPV'] = typical_price * data['Volume']
    data['Cumul_TPV'] = data.groupby('Date')['TPV'].cumsum()
    data['Cumul_Vol'] = data.groupby('Date')['Volume'].cumsum()
    data['VWAP'] = data['Cumul_TPV'] / data['Cumul_Vol']
    return data

def calculate_rsi(data, period=14):
    """Calculate RSI with protection against division by zero."""
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    avg_loss = avg_loss.replace(0, np.finfo(float).eps)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def apply_strategy(data):
    """Apply the HedgeScalp strategy."""
    if len(data) < 20:
        return pd.DataFrame()
        
    data = calculate_vwap(data)
    data['RSI'] = calculate_rsi(data, period=14)
    data['Vol_MA'] = data['Volume'].rolling(window=20, min_periods=20).mean()
    
    threshold = 0.003
    
    data['ShortSignal'] = (data['Close'] > data['VWAP'] * (1 + threshold)) & \
                          (data['RSI'] > 70) & \
                          (data['Volume'] < data['Vol_MA'] * 0.75)
                          
    data['LongSignal'] = (data['Close'] < data['VWAP'] * (1 - threshold)) & \
                         (data['RSI'] < 30) & \
                         (data['Volume'] < data['Vol_MA'] * 0.75)
    
    data['Signal'] = 0
    data.loc[data['LongSignal'], 'Signal'] = 1
    data.loc[data['ShortSignal'], 'Signal'] = -1
    
    return data

def run_backtest(tickers, start_date, end_date, use_trailing_stop=True, exit_bars=60, capital_per_trade=1000):
    """Run backtest using Alpaca historical data with regulatory fees."""
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    all_trades = []
    trade_logs = {}

    # Convert dates to market timezone
    start_date = start_date if start_date.tzinfo else MARKET_TZ.localize(start_date)
    end_date = end_date if end_date.tzinfo else MARKET_TZ.localize(end_date)
    start_utc = start_date.astimezone(timezone('UTC'))
    end_utc = end_date.astimezone(timezone('UTC'))

    pbar = tqdm(tickers, desc="Processing Stocks")
    for ticker in pbar:
        pbar.set_description(f"Processing {ticker}")
        try:
            # Fetch historical data from Alpaca
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=start_utc,
                end=end_utc,
                adjustment='all',
                feed='iex'
            )
            bars = client.get_stock_bars(request)
            if not bars or bars.df.empty:
                print(f"No data available for {ticker}")
                continue

            data = bars.df
            if isinstance(data.index, pd.MultiIndex):
                data = data.droplevel('symbol')
            data.index = data.index.tz_convert(MARKET_TZ)
            
            # Rename columns and select required fields
            data = data.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })[['Open', 'High', 'Low', 'Close', 'Volume']]

            # Apply strategy
            data = apply_strategy(data)
            if 'Signal' not in data.columns or data['Signal'].abs().sum() == 0:
                continue

            # Initialize tracking for this ticker
            active_trades = {}
            ticker_trades = []
            
            # Process day by day
            for day, day_data in data.groupby(data.index.date):
                # Skip days with insufficient data
                if len(day_data) < 2:
                    continue
                    
                # Track VWAP components for this day
                cumul_tpv = 0
                cumul_vol = 0
                
                # Process each bar (minute) during the day
                for t, row in day_data.iterrows():
                    # Update VWAP for this bar
                    tp = (row['High'] + row['Low'] + row['Close']) / 3
                    cumul_tpv += tp * row['Volume']
                    cumul_vol += row['Volume']
                    
                    vwap = cumul_tpv / cumul_vol if cumul_vol > 0 else row['Close']
                    
                    # Check trailing stops for active trades
                    trades_to_close = set()
                    for entry_time, trade in active_trades.items():
                        if use_trailing_stop:
                            try:
                                # For long positions, trailing stop is previous candle's close
                                if trade['Direction'] == 'Long':
                                    # Get index safely
                                    t_idx = day_data.index.get_loc(t)
                                    if t_idx > 0:  # Make sure we're not at the first bar
                                        prev_bar = day_data.iloc[t_idx - 1]
                                        new_stop = prev_bar['Close']
                                        if trade['Stop'] is None or new_stop > trade['Stop']:
                                            active_trades[entry_time]['Stop'] = new_stop
                                        
                                        # Check if current price hits trailing stop
                                        if row['Low'] <= trade['Stop']:
                                            trade['Exit Price'] = trade['Stop']
                                            trade['Exit Time'] = t
                                            trade['Exit Reason'] = 'Trailing Stop'
                                            
                                            # Calculate return with fees
                                            gross_ret = (trade['Exit Price'] - trade['Entry Price']) / trade['Entry Price']
                                            fees = calculate_fees(trade['Exit Price'], trade['Shares'], True)
                                            investment = trade['Entry Price'] * trade['Shares']
                                            fee_percent = fees / investment
                                            net_ret = gross_ret - fee_percent
                                            trade['Return'] = net_ret
                                            trade['Return %'] = net_ret * 100
                                            
                                            ticker_trades.append(trade)
                                            trades_to_close.add(entry_time)
                            
                                # For short positions, trailing stop is previous candle's open
                                elif trade['Direction'] == 'Short':
                                    t_idx = day_data.index.get_loc(t)
                                    if t_idx > 0:  # Make sure we're not at the first bar
                                        prev_bar = day_data.iloc[t_idx - 1]
                                        new_stop = prev_bar['Open']
                                        if trade['Stop'] is None or new_stop < trade['Stop']:
                                            active_trades[entry_time]['Stop'] = new_stop
                                        
                                        # Check if current price hits trailing stop
                                        if row['High'] >= trade['Stop']:
                                            trade['Exit Price'] = trade['Stop']
                                            trade['Exit Time'] = t
                                            trade['Exit Reason'] = 'Trailing Stop'
                                            
                                            # Calculate return with fees
                                            gross_ret = (trade['Entry Price'] - trade['Exit Price']) / trade['Entry Price']
                                            fees = calculate_fees(trade['Entry Price'], trade['Shares'], True)
                                            investment = trade['Entry Price'] * trade['Shares']
                                            fee_percent = fees / investment
                                            net_ret = gross_ret - fee_percent
                                            trade['Return'] = net_ret
                                            trade['Return %'] = net_ret * 100
                                            
                                            ticker_trades.append(trade)
                                            trades_to_close.add(entry_time)
                            except Exception as e:
                                pass
                        
                        # Time-based exit (after exit_bars minutes)
                        time_passed = (t - entry_time).total_seconds() / 60
                        if time_passed >= exit_bars:
                            trade['Exit Price'] = row['Close']
                            trade['Exit Time'] = t
                            trade['Exit Reason'] = 'Time-based'
                            
                            # Calculate return with fees
                            if trade['Direction'] == 'Long':
                                gross_ret = (trade['Exit Price'] - trade['Entry Price']) / trade['Entry Price']
                            else:
                                gross_ret = (trade['Entry Price'] - trade['Exit Price']) / trade['Entry Price']
                                
                            fees = calculate_fees(trade['Exit Price'], trade['Shares'], True)
                            investment = trade['Entry Price'] * trade['Shares']
                            fee_percent = fees / investment
                            net_ret = gross_ret - fee_percent
                            trade['Return'] = net_ret
                            trade['Return %'] = net_ret * 100
                            
                            ticker_trades.append(trade)
                            trades_to_close.add(entry_time)
                    
                    # Remove closed trades
                    for entry_time in trades_to_close:
                        del active_trades[entry_time]
                    
                    # Process new entry signals when not in a position
                    if t not in active_trades and row['Signal'] != 0:
                        direction = 'Long' if row['Signal'] == 1 else 'Short'
                        
                        # Calculate shares based on capital
                        price = row['Close']
                        shares = int(capital_per_trade / price)
                        
                        # Make sure we have at least 1 share
                        if shares < 1:
                            continue
                        
                        # Set initial trailing stop
                        initial_stop = None
                        try:
                            # Get index safely
                            t_loc = day_data.index.get_indexer([t])[0]
                            prev_idx = t_loc - 1
                            if prev_idx >= 0 and prev_idx < len(day_data):
                                if direction == 'Long':
                                    initial_stop = day_data['Close'].iloc[prev_idx]
                                else:
                                    initial_stop = day_data['Open'].iloc[prev_idx]
                        except:
                            # If index lookup fails, proceed without initial stop
                            pass
                        
                        active_trades[t] = {
                            'Symbol': ticker,
                            'DateTime': t,
                            'Direction': direction,
                            'Entry Price': price,
                            'Stop': initial_stop,
                            'RSI': row['RSI'],
                            'VWAP': vwap,
                            'Volume': row['Volume'],
                            'Vol_MA': row['Vol_MA'],
                            'Shares': shares,
                            'Capital': capital_per_trade
                        }
            
            # Close any remaining open trades at the end of data
            for entry_time, trade in active_trades.items():
                last_row = data.iloc[-1]
                trade['Exit Price'] = last_row['Close'] 
                trade['Exit Time'] = data.index[-1]
                trade['Exit Reason'] = 'End of Data'
                
                # Calculate return with fees
                if trade['Direction'] == 'Long':
                    gross_ret = (trade['Exit Price'] - trade['Entry Price']) / trade['Entry Price']
                else:
                    gross_ret = (trade['Entry Price'] - trade['Exit Price']) / trade['Entry Price']
                    
                fees = calculate_fees(trade['Exit Price'], trade['Shares'], True)
                investment = trade['Entry Price'] * trade['Shares']
                fee_percent = fees / investment
                net_ret = gross_ret - fee_percent
                trade['Return'] = net_ret
                trade['Return %'] = net_ret * 100
                
                ticker_trades.append(trade)
            
            # Add ticker trades to overall results
            all_trades.extend(ticker_trades)
            
            # Store in trade logs for performance summary
            if ticker_trades:
                trade_logs[ticker] = [t['Return'] for t in ticker_trades]
                
                # Print performance for this ticker
                total_pnl = sum(trade_logs[ticker])
                num_trades = len(trade_logs[ticker])
                print(f"{ticker}: Total PnL: {total_pnl:.4f} from {num_trades} trades")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    # Return results
    if all_trades:
        results = pd.DataFrame(all_trades)
        results.sort_values('DateTime', inplace=True)
        
        # Print overall performance summary
        print("\n=== Overall Performance Summary ===")
        print(f"Total Return: {results['Return'].sum():.4f} ({results['Return %'].sum():.2f}%)")
        print(f"Average Return per Trade: {results['Return'].mean():.4f} ({results['Return %'].mean():.2f}%)")
        print(f"Win Rate: {(results['Return'] > 0).mean():.2%}")
        print(f"Total Number of Trades: {len(results)}")
        
        # Print per-stock summary
        print("\n=== Per-stock Trade Performance Summary ===")
        for symbol, pnl_list in trade_logs.items():
            total_pnl = sum(pnl_list)
            num_trades = len(pnl_list)
            win_rate = sum(1 for pnl in pnl_list if pnl > 0) / num_trades if num_trades > 0 else 0
            print(f"{symbol}: Total PnL: {total_pnl:.4f} from {num_trades} trades (Win rate: {win_rate:.2%})")
        
        return results
    else:
        print("No trades were generated.")
        return pd.DataFrame()

if __name__ == "__main__":
    tickers = [
        "MSTR", "COIN", "RIVN",
        "CVNA", "PLTR", "DJT", "AMD", 
        "MRNA", "DKNG", "SNAP", "MARA", "RIOT",
        "KMX", "CLSK", "CORZ", "HUT", "CZR",
        "NVDA", "AVGO",
    ]
    
    end_date = datetime.now(MARKET_TZ)
    start_date = end_date - timedelta(days=7)
    
    results = run_backtest(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        use_trailing_stop=True,
        exit_bars=5,
        capital_per_trade=1000
    )