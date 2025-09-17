import csv
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from pytz import timezone
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
import os
import logging
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()

# Configuration
ALPACA_API_KEY = os.getenv('APCA_API_KEY_ID')
ALPACA_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
MARKET_TZ = timezone('US/Eastern')

class LiveTrader:
    def __init__(self):
        self.trading_client = TradingClient(
            ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True
        )
        self.data_stream = StockDataStream(
            ALPACA_API_KEY, ALPACA_SECRET_KEY, feed=DataFeed.IEX
        )
        self.historical_client = StockHistoricalDataClient(
            ALPACA_API_KEY, ALPACA_SECRET_KEY
        )
        
        self.active_positions = {}
        self.historical_data = {}
        self.exit_bars = 30  # 30-minute time-based exit
        self.position_size = 0.1  # 10% of equity per position
        self.max_positions = 5
        
        # Strategy parameters
        self.atr_period = 14
        self.vol_ma_window = 30
        self.ema_period = 7
        self.threshold_multiplier = 1.5

        self.trade_log_file = 'trades.csv'
        self.initialize_trade_log()

        self.data_stream._on_connect = self.on_connect
        self.data_stream._on_disconnect = self.on_disconnect

    async def on_connect(self):
        logging.info("Connected to Alpaca WebSocket")

    async def on_disconnect(self):
        logging.warning("Disconnected from Alpaca WebSocket")

    async def fetch_historical_data(self, symbols, lookback_days=3):
        end = datetime.now(MARKET_TZ)
        start = end - timedelta(days=lookback_days)
        
        for symbol in symbols:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    start=start,
                    end=end,
                    adjustment='all',
                    feed='iex'
                )
                bars = await asyncio.to_thread(
                    self.historical_client.get_stock_bars, request
                )
                
                if bars.df.empty:
                    continue
                    
                df = bars.df.droplevel('symbol') if isinstance(bars.df.index, pd.MultiIndex) else bars.df
                df.index = df.index.tz_convert(MARKET_TZ)
                self.historical_data[symbol] = df[['open','high','low','close','volume']]
                
            except Exception as e:
                logging.error(f"Error loading {symbol} history: {str(e)}")
        
        self.calculate_historical_indicators()

    def initialize_trade_log(self):
        """Create CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.trade_log_file):
            with open(self.trade_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'direction', 'entry_price',
                    'exit_price', 'quantity', 'pnl', 'fees', 'reason'
                ])

    def calculate_historical_indicators(self):
        for symbol, df in self.historical_data.items():
            try:
                df = df.copy()
                # Calculate VWAP
                tp = (df['high'] + df['low'] + df['close']) / 3
                cumul_tpv = tp * df['volume']
                cumul_vol = df['volume']
                df['vwap'] = cumul_tpv.cumsum() / cumul_vol.cumsum()
                
                # Calculate ATR
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr'] = tr.rolling(self.atr_period).mean().bfill()
                
                # Calculate indicators
                df['vol_ma'] = df['volume'].rolling(self.vol_ma_window).mean()
                df['ema_7'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
                
                # Threshold calculation
                df['threshold'] = self.threshold_multiplier * (df['atr'] / df['close']).rolling(14).mean()
                
                # Generate signals
                df['long_signal'] = (
                    (df['close'] < df['vwap'] * (1 - df['threshold'])) & 
                    (df['volume'] < df['vol_ma'] * 0.9) &
                    (df['close'] > df['ema_7']) &
                    (df['volume'] > 100))
                
                df['short_signal'] = (
                    (df['close'] > df['vwap'] * (1 + df['threshold'])) & 
                    (df['volume'] < df['vol_ma'] * 0.9) &
                    (df['close'] < df['ema_7']) &
                    (df['volume'] > 100))
                
                self.historical_data[symbol] = df
                
            except Exception as e:
                logging.error(f"Indicator error for {symbol}: {str(e)}")

    async def run(self, symbols):
        await self.fetch_historical_data(symbols)
        self.data_stream.subscribe_bars(self.on_bar, *symbols)
        await self.data_stream._run_forever()

    async def on_bar(self, bar):
        symbol = bar.symbol
        try:
            # Update historical data with new bar
            new_bar = pd.DataFrame({
                'open': [bar.open],
                'high': [bar.high],
                'low': [bar.low],
                'close': [bar.close],
                'volume': [bar.volume]
            }, index=pd.to_datetime([bar.timestamp]).tz_convert(MARKET_TZ))
            
            if symbol not in self.historical_data:
                self.historical_data[symbol] = new_bar
            else:
                self.historical_data[symbol] = pd.concat([
                    self.historical_data[symbol], new_bar
                ]).iloc[-1000:]  # Keep last 1000 bars
                
            # Recalculate indicators
            df = self.historical_data[symbol]
            self.calculate_indicators(df)
            
            # Get latest data
            latest = df.iloc[-1]
            
            # Manage existing position
            if symbol in self.active_positions:
                await self.manage_position(symbol, latest, bar.timestamp)
            else:
                await self.check_new_entry(symbol, latest, bar.timestamp)
                
        except Exception as e:
            logging.error(f"Error processing {symbol}: {str(e)}")

    def calculate_indicators(self, df):
        try:
            # VWAP
            tp = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
            
            # ATR
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift()).abs(),
                (df['low'] - df['close'].shift()).abs()
            ], axis=1).max(axis=1)
            df['atr'] = tr.rolling(self.atr_period).mean().bfill()
            
            # Volume MA
            df['vol_ma'] = df['volume'].rolling(self.vol_ma_window).mean()
            
            # EMA
            df['ema_7'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
            
            # Dynamic threshold
            df['threshold'] = self.threshold_multiplier * (df['atr'] / df['close']).rolling(14).mean()
            
            # Signals
            df['long_signal'] = (
                (df['close'] < df['vwap'] * (1 - df['threshold'])) & 
                (df['volume'] < df['vol_ma'] * 0.9) &
                (df['close'] > df['ema_7']) &
                (df['volume'] > 100))
            
            df['short_signal'] = (
                (df['close'] > df['vwap'] * (1 + df['threshold'])) & 
                (df['volume'] < df['vol_ma'] * 0.9) &
                (df['close'] < df['ema_7']) &
                (df['volume'] > 100))
                
        except Exception as e:
            logging.error(f"Indicator calculation error: {str(e)}")

    async def check_new_entry(self, symbol, latest, timestamp):
        if len(self.active_positions) >= self.max_positions:
            return
            
        try:
            equity = float((await asyncio.to_thread(
                self.trading_client.get_account
            )).equity)
            position_size = equity * self.position_size
            
            if latest['long_signal']:
                qty = int(position_size / latest['close'])
                await self.enter_position(symbol, qty, OrderSide.BUY, latest, timestamp)
                
            elif latest['short_signal']:
                qty = int(position_size / latest['close'])
                await self.enter_position(symbol, qty, OrderSide.SELL, latest, timestamp)
                
        except Exception as e:
            logging.error(f"Entry error for {symbol}: {str(e)}")

    async def enter_position(self, symbol, qty, side, latest, timestamp):
        if qty < 1:
            return
            
        try:
            await asyncio.to_thread(
                self.trading_client.submit_order,
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.GTC
                )
            )
            
            # Get previous bar for stop calculation
            prev_bar = self.historical_data[symbol].iloc[-2] if len(self.historical_data[symbol]) > 1 else latest
            
            self.active_positions[symbol] = {
                'entry_time': timestamp,
                'entry_price': latest['close'],
                'direction': 'long' if side == OrderSide.BUY else 'short',
                'stop_price': self.calculate_initial_stop(side, prev_bar),
                'quantity': qty
            }
            
            logging.info(f"Entered {side} {qty} {symbol} @ {latest['close']:.2f}")
            
        except Exception as e:
            logging.error(f"Order failed for {symbol}: {str(e)}")

    def calculate_initial_stop(self, side, prev_bar):
        if side == OrderSide.BUY:
            return prev_bar['close']  # Previous close for long stop
        return prev_bar['open']  # Previous open for short stop

    async def manage_position(self, symbol, latest, timestamp):
        if symbol not in self.active_positions:
            return
        
        position = self.active_positions[symbol]
        
        # Time-based exit
        if (timestamp - position['entry_time']).total_seconds() / 60 >= self.exit_bars:
            await self.exit_position(symbol, "Time-based exit")
            return
            
        # Trailing stop logic
        current_price = latest['close']
        new_stop = position['stop_price']
        
        if position['direction'] == 'long':
            new_stop = max(new_stop, latest['low'] * 0.999)
            if current_price <= new_stop:
                await self.exit_position(symbol, "Trailing stop hit")
        else:
            new_stop = min(new_stop, latest['high'] * 1.001)
            if current_price >= new_stop:
                await self.exit_position(symbol, "Trailing stop hit")
        
        self.active_positions[symbol]['stop_price'] = new_stop

    async def exit_position(self, symbol, reason):
        try:
            # Validate position exists before proceeding
            if symbol not in self.active_positions:
                logging.warning(f"No active position for {symbol} during exit")
                return

            # Get position details before any async operations
            position = self.active_positions[symbol]
            entry_price = position['entry_price']
            qty = position['quantity']
            direction = position['direction']
            
            # Execute closing order
            await asyncio.to_thread(
                self.trading_client.close_position, symbol
            )
            
            # Calculate fees and PnL
            exit_price = await self.get_current_price(symbol)
            fees = calculate_fees(entry_price, qty)
            pnl = (exit_price - entry_price) * qty if direction == 'long' \
                else (entry_price - exit_price) * qty
            
            # Log trade details
            self.log_trade(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=qty,
                pnl=pnl,
                fees=fees,
                reason=reason
            )
            
            # Remove position only after successful close
            del self.active_positions[symbol]
            
            logging.info(
                f"Closed {symbol} {direction} | "
                f"Entry: {entry_price:.2f} | Exit: {exit_price:.2f} | "
                f"PnL: ${pnl:.2f} | Fees: ${fees:.2f}"
            )

        except Exception as e:
            logging.error(f"Error closing {symbol}: {str(e)}")

    async def get_current_price(self, symbol):
        """Get most recent price from historical data"""
        try:
            return self.historical_data[symbol]['close'].iloc[-1]
        except (KeyError, IndexError):
            # Fallback to account value if historical data missing
            position = await asyncio.to_thread(
                self.trading_client.get_open_position, symbol
            )
            return float(position.current_price)
        
    def log_trade(self, **trade_data):
        """Record trade details to CSV"""
        with open(self.trade_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                trade_data['symbol'],
                trade_data['direction'],
                trade_data['entry_price'],
                trade_data['exit_price'],
                trade_data['quantity'],
                trade_data['pnl'],
                trade_data['fees'],
                trade_data['reason']
            ])

def calculate_fees(price, shares):
    """Calculate regulatory fees (SEC + FINRA)"""
    principal = price * shares
    sec_fee = (principal / 1e6) * 27.80
    sec_fee = math.ceil(sec_fee * 100) / 100
    taf_fee = shares * 0.000166
    taf_fee = math.ceil(taf_fee * 100) / 100
    taf_fee = min(taf_fee, 8.30)
    return sec_fee + taf_fee

if __name__ == "__main__":
    symbols = [
        "MSTR", "CVNA", "COIN", "AFRM", "AR", "PR", "DJT", "MARA",
        "ACHR", "BE", "W", "SOUN", "SM", "RIOT", "CLSK", "BHVN",
        "JANX", "SEZL", "IREN"
    ]
    
    trader = LiveTrader()
    loop = asyncio.new_event_loop()
    
    try:
        loop.run_until_complete(trader.run(symbols))
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()