import pandas as pd
import numpy as np
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

# Configure logging at the top
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()

# Configuration
ALPACA_API_KEY = os.getenv('APCA_API_KEY_ID')
ALPACA_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
PAPER_BASE_URL = 'https://paper-api.alpaca.markets'
MARKET_TZ = timezone('US/Eastern')

class LiveTrader:
    def __init__(self):
        self.connected = False
        self.running = False

        # Initialize trading client with explicit parameters
        logging.info("Initializing trading client...")
        self.trading_client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=True
        )
        
        # Initialize data stream with enum feed
        logging.info("Initializing data stream...")
        self.data_stream = StockDataStream(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            feed=DataFeed.IEX
        )

        # Initialize historical data client
        self.historical_client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY
        )

        self.data_stream._on_connect = self.on_connect
        self.data_stream._on_disconnect = self.on_disconnect
        self.ws_task = None
        self._cleaned_up = False
        
        self.active_positions = {}
        self.historical_data = {}
        self.threshold = 0.003
        self.exit_bars = 5
        self.position_percentage = 1.0
        self.max_positions = 1

        self.last_equity_log = datetime.now()
        self.equity_log_interval = timedelta(minutes=15)

    async def on_connect(self):
        """Handle successful WebSocket connection"""
        self.connected = True
        logging.info("Connected to Alpaca WebSocket")

    async def on_disconnect(self):
        """Handle WebSocket disconnection"""
        self.connected = False
        logging.warning("Disconnected from Alpaca WebSocket")

    async def fetch_historical_data(self, symbols, lookback_days=1):
        """Fetch historical data for warm starting"""
        logging.info(f"Fetching historical data for {len(symbols)} symbols (last {lookback_days} days)...")
        
        end_date = datetime.now(MARKET_TZ)
        start_date = end_date - timedelta(days=lookback_days)
        
        # Convert to UTC for API request
        start_utc = start_date.astimezone(timezone('UTC'))
        end_utc = end_date.astimezone(timezone('UTC'))
        
        for symbol in symbols:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    start=start_utc,
                    end=end_utc,
                    adjustment='all',
                    feed='iex'
                )
                
                bars = await asyncio.to_thread(self.historical_client.get_stock_bars, request)
                
                if not bars or bars.df.empty:
                    logging.warning(f"No historical data available for {symbol}")
                    continue
                    
                # Extract and process data for this symbol
                df = bars.df
                if isinstance(df.index, pd.MultiIndex):
                    df = df.droplevel('symbol')
                
                # Convert to market timezone
                df.index = df.index.tz_convert(MARKET_TZ)
                
                # Store in appropriate format for our strategy
                historical_bars = []
                for timestamp, row in df.iterrows():
                    historical_bars.append({
                        'timestamp': timestamp,
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    })
                
                self.historical_data[symbol] = historical_bars
                logging.info(f"Loaded {len(historical_bars)} historical bars for {symbol}")
                
            except Exception as e:
                logging.error(f"Error fetching historical data for {symbol}: {str(e)}")
        
        # Calculate indicators on historical data
        self.prepare_indicators()
        
    def prepare_indicators(self):
        """Calculate indicators on historical data"""
        for symbol, bars in self.historical_data.items():
            if len(bars) < 20:  # Need minimum data for indicators
                continue
                
            df = pd.DataFrame(bars)
            df.set_index('timestamp', inplace=True)
            df = self.calculate_indicators(df)
            
            # Update our historical data with processed indicators
            updated_bars = []
            for timestamp, row in df.iterrows():
                bar_dict = {
                    'timestamp': timestamp,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
                
                # Add calculated indicators
                if 'vwap' in row:
                    bar_dict['vwap'] = row['vwap']
                if 'rsi' in row:
                    bar_dict['rsi'] = row['rsi']
                if 'vol_ma' in row:
                    bar_dict['vol_ma'] = row['vol_ma']
                    
                updated_bars.append(bar_dict)
                
            self.historical_data[symbol] = updated_bars
            
        logging.info("Calculated indicators on historical data")

    async def run(self, symbols):
        """Main trading loop"""
        # First, fetch historical data for warm start
        await self.fetch_historical_data(symbols, lookback_days=3)
        
        # Now subscribe to real-time data
        self.data_stream.subscribe_bars(self.on_bar, *symbols)
        logging.info(f"Starting WebSocket connection... Subscribed to {len(symbols)} symbols")
        
        account = self.trading_client.get_account()
        logging.info(f"Account connected: {account.status}")

        await self.get_account_equity()

        self.running = True
        logging.info(f"Trader running state: {self.running}")

        equity_task = asyncio.create_task(self.periodic_equity_logging())
        self.ws_task = asyncio.create_task(self.data_stream._run_forever())

        try:
            await asyncio.gather(self.ws_task, equity_task)
        except asyncio.CancelledError:
            logging.warning("WebSocket connection cancelled")
        except Exception as e:
            logging.error(f"WebSocket error: {e}", exc_info=True)
        finally:
            self.running = False
            await self.cleanup()
    
    async def periodic_equity_logging(self):
        """Periodically log account equity"""
        while self.running:
            await self.get_account_equity()
            await asyncio.sleep(900)

    async def cleanup(self):
        """Cleanup resources"""
        if self._cleaned_up:
            return
                
        logging.info("Starting cleanup...")
        self._cleaned_up = True

        # Close positions first using asyncio.to_thread
        try:
            await asyncio.to_thread(self.trading_client.close_all_positions, cancel_orders=True)
            logging.info("Closed all positions")
        except Exception as e:
            logging.error(f"Error closing positions: {str(e)}")

        # Close WebSocket connection
        if self.connected:
            try:
                await self.data_stream.stop()
                logging.info("WebSocket connection closed")
            except Exception as e:
                logging.error(f"Error closing WebSocket: {str(e)}")

        # Cancel remaining tasks
        try:
            if self.ws_task and not self.ws_task.done():
                self.ws_task.cancel()
                await self.ws_task
        except asyncio.CancelledError:
            logging.info("WebSocket task cancelled")
        except Exception as e:
            logging.error(f"Error cancelling WebSocket task: {str(e)}")

        logging.info("Cleanup complete")

    async def get_account_equity(self):
        """Get current account equity"""
        try:
            account = self.trading_client.get_account()
            equity = float(account.equity)
            logging.info(f"Current account equity: ${equity:.2f}")
            return equity
        except Exception as e:
            logging.error(f"Error getting account equity: {str(e)}")
            return 0

    async def on_bar(self, bar):
        """Handle incoming real-time bars"""
        try:
            logging.info(f"BAR RECEIVED: {bar.symbol} @ {bar.timestamp} - Close: {bar.close}")
            
            #if not self.connected:
            #   logging.warning("Received bar but not connected - discarding")
            #    return
            
            symbol = bar.symbol
            current_time = bar.timestamp.astimezone(MARKET_TZ)
            
            # Update historical data
            if symbol not in self.historical_data:
                self.historical_data[symbol] = []
                
            self.historical_data[symbol].append({
                'timestamp': current_time,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })
            
            # Log receipt of bar data
            # logging.info(f"Processed bar: {symbol} @ {current_time} - Close: {bar.close}")
            
            # Process signals - make sure this is called for every bar
            await self.process_symbol(symbol, current_time)
            
        except Exception as e:
            logging.error(f"Error processing bar: {e}", exc_info=True)

    async def process_symbol(self, symbol, current_time):
        """Process trading logic for a symbol"""
        try:
            if len(self.historical_data.get(symbol, [])) < 20:
                logging.debug(f"Not enough data for {symbol}: {len(self.historical_data.get(symbol, []))} bars")
                return
                
            df = pd.DataFrame(self.historical_data[symbol])
            df.set_index('timestamp', inplace=True)
            df = self.calculate_indicators(df)
            
            latest = df.iloc[-1]
            
            # More detailed signal logging
            vwap_threshold_short = latest['vwap'] * (1 + self.threshold)
            rsi_threshold = 70
            vol_threshold = latest['vol_ma'] * 0.75
            
            #logging.info(f"{symbol} analysis: Close: {latest['close']:.2f}, VWAP: {latest['vwap']:.2f} (threshold: {vwap_threshold_short:.2f}), " 
            #            f"RSI: {latest['rsi']:.2f} (threshold: {rsi_threshold}), "
            #            f"Volume: {latest['volume']} (threshold: {vol_threshold:.2f})")
            
            # Explicitly log signal components
            price_above_vwap = latest['close'] > vwap_threshold_short
            rsi_above_threshold = latest['rsi'] > rsi_threshold
            volume_below_threshold = latest['volume'] < vol_threshold
            
            logging.info(f"{symbol} signal components - Price>VWAP: {price_above_vwap}, RSI>70: {rsi_above_threshold}, Vol<Threshold: {volume_below_threshold}")
            
            # Process trading logic
            position = self.active_positions.get(symbol)
            if position:
                await self.manage_position(symbol, latest, current_time)
            else:
                await self.check_entry(symbol, latest, current_time)
                
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")

    def calculate_indicators(self, data):
        """Calculate technical indicators - KEEP IDENTICAL TO BACKTEST VERSION"""
        # VWAP calculation
        data['typical'] = (data['high'] + data['low'] + data['close']) / 3
        data['cumul_tpv'] = data['typical'] * data['volume']
        data['cumul_vol'] = data['volume']
        
        # Calculate VWAP for each day separately (reset at market open)
        data['date'] = data.index.date
        grouped = data.groupby('date')
        
        data['vwap'] = grouped['cumul_tpv'].cumsum() / grouped['cumul_vol'].cumsum()
        
        # RSI calculation
        delta = data['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume MA
        data['vol_ma'] = data['volume'].rolling(20).mean()
        
        # Generate signals
        data['long_signal'] = (
            (data['close'] < data['vwap'] * (1 - self.threshold)) &
            (data['rsi'] < 30) &
            (data['volume'] < data['vol_ma'] * 0.75)
        )
        
        data['short_signal'] = (
            (data['close'] > data['vwap'] * (1 + self.threshold)) &
            (data['rsi'] > 70) &
            (data['volume'] < data['vol_ma'] * 0.75)
        )
        
        data['signal'] = 0
        data.loc[data['long_signal'], 'signal'] = 1
        data.loc[data['short_signal'], 'signal'] = -1
        
        return data

    async def check_entry(self, symbol, latest, current_time):
        """Check for new entry signals"""
        if len(self.active_positions) >= self.max_positions:
            logging.debug(f"Max positions reached, skipping {symbol}")
            return
            
        signal = latest['signal']
        if signal == 0:
            return
            
        logging.info(f"Signal detected for {symbol}: {signal}")
        side = OrderSide.BUY if signal == 1 else OrderSide.SELL
        
        try:
            # Get account equity and calculate position size
            equity = await self.get_account_equity()
            position_value = equity * self.position_percentage
            
            # Calculate quantity based on percentage of account
            qty = round(position_value / latest['close'])
            
            if qty <= 0:
                logging.warning(f"Calculated quantity too small for {symbol}, skipping")
                return
                
            logging.info(f"Allocating ${position_value:.2f} ({self.position_percentage*100}% of ${equity:.2f}) to {symbol}")

            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            
            trade = await asyncio.to_thread(self.trading_client.submit_order, order)
            logging.info(f"Entered {side} {qty} {symbol} @ {latest['close']:.2f} (${position_value:.2f})")
            
            self.active_positions[symbol] = {
                'entry_time': current_time,
                'entry_price': latest['close'],
                'direction': 'long' if side == OrderSide.BUY else 'short',
                'stop_price': self.calculate_stop_price(side, latest),
                'quantity': qty
            }
            
        except Exception as e:
            logging.error(f"Order failed for {symbol}: {e}")

    def calculate_stop_price(self, side, latest):
        """Calculate initial stop price"""
        return latest['low'] * 0.98 if side == OrderSide.BUY else latest['high'] * 1.02

    async def manage_position(self, symbol, latest, current_time):
        """Manage existing position"""
        position = self.active_positions[symbol]
        
        # Time-based exit
        time_in_trade = (current_time - position['entry_time']).total_seconds() / 60
        if time_in_trade >= self.exit_bars:
            await self.exit_position(symbol, 'Time-based exit')
            return
            
        # Trailing stop logic
        current_price = latest['close']
        new_stop = None
        
        if position['direction'] == 'long':
            new_stop = max(position['stop_price'], latest['low'] * 0.98)
            if current_price <= new_stop:
                await self.exit_position(symbol, 'Trailing stop hit')
        else:
            new_stop = min(position['stop_price'], latest['high'] * 1.02)
            if current_price >= new_stop:
                await self.exit_position(symbol, 'Trailing stop hit')
                
        if new_stop:
            self.active_positions[symbol]['stop_price'] = new_stop

    async def exit_position(self, symbol, reason):
        """Close an existing position"""
        try:
            await asyncio.to_thread(self.trading_client.close_position, symbol)
            logging.info(f"Closed {symbol} position: {reason}")
            del self.active_positions[symbol]
        except Exception as e:
            logging.error(f"Error closing {symbol}: {e}")

    async def close_all_positions(self):
        """Close all active positions"""
        try:
            await asyncio.to_thread(self.trading_client.close_all_positions, cancel_orders=True)
            self.active_positions.clear()
        except Exception as e:
            logging.error(f"Error closing positions: {str(e)}")

if __name__ == "__main__":
    symbols = [
        "MSTR", "COIN", "RIVN",
        "CVNA", "PLTR", "DJT", "AMD", 
        "MRNA", "DKNG", "SNAP", "MARA", "RIOT",
        "KMX", "CLSK", "CORZ", "HUT", "CZR",
        "NVDA", "AVGO",
    ]
    
    trader = LiveTrader()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        logging.info("Starting trader...")
        main_task = loop.create_task(trader.run(symbols))
        loop.run_forever()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    finally:
        logging.info("Final shutdown sequence...")
        try:
            if not main_task.done():
                main_task.cancel()
                loop.run_until_complete(main_task)
            loop.run_until_complete(trader.cleanup())
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            logging.info("Shutdown complete")