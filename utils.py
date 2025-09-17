import math
from datetime import timedelta
import pandas_market_calendars as mcal
from pytz import timezone

# Define market timezone (US/Eastern) as a constant
MARKET_TZ = timezone('US/Eastern')

def is_market_open(date):
    """Check if the given date is a trading day in the US market."""
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=date, end_date=date)
    return not schedule.empty

def get_previous_trading_day(date):
    """Get the previous trading day for a given date."""
    current_date = date
    while True:
        current_date -= timedelta(days=1)
        if is_market_open(current_date):
            return current_date

def get_next_trading_day(date):
    """Get the next trading day for a given date."""
    current_date = date + timedelta(days=1)
    while True:
        if is_market_open(current_date):
            return current_date
        current_date += timedelta(days=1)

def calculate_regulatory_fees(sell_price, shares):
    """Calculate SEC and FINRA fees for a sell order."""
    # SEC fee: $27.80 per $1,000,000 of principal
    principal = sell_price * shares
    sec_fee = (principal / 1e6) * 27.80
    sec_fee = math.ceil(sec_fee * 100) / 100  # Round up to nearest penny
    # FINRA TAF: $0.000166 per share
    taf_fee = shares * 0.000166
    taf_fee = math.ceil(taf_fee * 100) / 100
    taf_fee = min(taf_fee, 8.30)  # Maximum $8.30
    return sec_fee + taf_fee

def calculate_profit_factor(returns):
    """Calculate profit factor: sum of profits / sum of losses."""
    if len(returns) == 0:
        return 0
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float('inf') if profits > 0 else 0
    return profits / losses