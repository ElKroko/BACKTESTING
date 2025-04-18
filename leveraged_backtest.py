# leveraged_backtest.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta
from strategies import (
    ma_crossover,
    bollinger_breakout,
    rsi_reversion,
    macd_momentum,
    sr_breakout
)
from config import PALETTES

# --- Utility functions for data fetching ---
def fetch_ohlc_data(symbol, interval, start_date=None, end_date=None, limit=1000):
    """
    Fetches OHLC data from Binance API
    
    Args:
        symbol: Trading pair (e.g. 'BTCUSDT')
        interval: Timeframe (e.g. '1h', '4h', '1d')
        start_date: Start date as datetime or date
        end_date: End date as datetime or date
        limit: Maximum number of candles to fetch
        
    Returns:
        pandas DataFrame with OHLC data
    """
    # Convert date to datetime if needed and then to milliseconds timestamp
    if start_date:
        # Check if it's a date but not a datetime
        if hasattr(start_date, 'date') and callable(getattr(start_date, 'date')):
            # It's a datetime object
            start_dt = start_date
        else:
            # It's a date object, convert to datetime
            start_dt = datetime.combine(start_date, datetime.min.time())
        start_ts = int(start_dt.timestamp() * 1000)
    else:
        start_ts = None
        
    if end_date:
        # Check if it's a date but not a datetime
        if hasattr(end_date, 'date') and callable(getattr(end_date, 'date')):
            # It's a datetime object
            end_dt = end_date
        else:
            # It's a date object, convert to datetime
            end_dt = datetime.combine(end_date, datetime.max.time())
        end_ts = int(end_dt.timestamp() * 1000)
    else:
        end_ts = None
    
    # Build URL with appropriate parameters
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    if start_ts:
        url += f"&startTime={start_ts}"
    if end_ts:
        url += f"&endTime={end_ts}"
    
    # Fetch and process data
    cols = ['Open time','Open','High','Low','Close','Volume','Close time',
            'Quote asset volume','Num trades','Taker buy base','Taker buy quote','Ignore']
    df = pd.DataFrame(requests.get(url).json(), columns=cols)
    
    # Convert types and set index
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = df[c].astype(float)
    
    return df

def fetch_funding_rates(symbol, start_date=None, end_date=None, limit=1000):
    """
    Fetches historical funding rates from Binance API
    
    Args:
        symbol: Trading pair (e.g. 'BTCUSDT')
        start_date: Start date as datetime or date
        end_date: End date as datetime or date
        limit: Maximum number of funding rates to fetch
        
    Returns:
        pandas DataFrame with funding rates data
    """
    # Convert date to datetime if needed and then to milliseconds timestamp
    if start_date:
        # Check if it's a date but not a datetime
        if hasattr(start_date, 'date') and callable(getattr(start_date, 'date')):
            # It's a datetime object
            start_dt = start_date
        else:
            # It's a date object, convert to datetime
            start_dt = datetime.combine(start_date, datetime.min.time())
        start_ts = int(start_dt.timestamp() * 1000)
    else:
        start_ts = None
        
    if end_date:
        # Check if it's a date but not a datetime
        if hasattr(end_date, 'date') and callable(getattr(end_date, 'date')):
            # It's a datetime object
            end_dt = end_date
        else:
            # It's a date object, convert to datetime
            end_dt = datetime.combine(end_date, datetime.max.time())
        end_ts = int(end_dt.timestamp() * 1000)
    else:
        end_ts = None
    
    # Build URL with appropriate parameters
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit={limit}"
    if start_ts:
        url += f"&startTime={start_ts}"
    if end_ts:
        url += f"&endTime={end_ts}"
    
    try:
        # Fetch funding rates
        response = requests.get(url)
        data = response.json()
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Convert types and set index
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['fundingRate'] = df['fundingRate'].astype(float)
        df.set_index('fundingTime', inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error fetching funding rates: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['fundingTime', 'fundingRate']).set_index('fundingTime')

# --- Liquidation price calculation ---
def calculate_liquidation_price(entry_price, leverage, maintenance_margin, is_long=True):
    """
    Calculate the liquidation price for a leveraged position
    
    Args:
        entry_price: Entry price of the position
        leverage: Leverage used (e.g. 10 for 10x)
        maintenance_margin: Maintenance margin ratio (e.g. 0.005 for 0.5%)
        is_long: True if long position, False if short
        
    Returns:
        Liquidation price
    """
    if is_long:
        # Long position liquidation price
        liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin)
    else:
        # Short position liquidation price
        liquidation_price = entry_price * (1 + (1 / leverage) - maintenance_margin)
    
    return liquidation_price

# --- Leveraged backtesting engine ---
def run_leveraged_backtest(
    symbol, interval, strategy_fn, initial_cash,
    commission=0.001, slippage=0.0005,
    leverage=1, maintenance_margin=0.005,
    start_date=None, end_date=None):
    """
    Execute a backtest with leverage, funding rates and liquidation simulation
    
    Args:
        symbol: Trading pair (e.g. 'BTCUSDT')
        interval: Timeframe (e.g. '1h', '4h', '1d')
        strategy_fn: Trading strategy function
        initial_cash: Initial capital
        commission: Trading commission as a fraction (e.g. 0.001 for 0.1%)
        slippage: Slippage as a fraction (e.g. 0.0005 for 0.05%)
        leverage: Leverage multiplier (e.g. 10 for 10x)
        maintenance_margin: Maintenance margin ratio (e.g. 0.005 for 0.5%)
        start_date: Start date for backtesting
        end_date: End date for backtesting
        
    Returns:
        Dictionary of metrics, DataFrame of trades, equity curve DataFrame, price DataFrame
    """
    # Fetch OHLC data
    price_df = fetch_ohlc_data(symbol, interval, start_date, end_date)
    
    # Fetch funding rates data
    funding_df = fetch_funding_rates(symbol, start_date, end_date)
    
    # If data is empty, return empty results
    if price_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Generate trading signals based on the strategy
    raw_trades = strategy_fn(price_df)
    
    # Initialize tracking variables
    cash = initial_cash
    equity = initial_cash
    margin_used = 0
    position = 0
    entry_price = None
    leverage_used = leverage
    is_long = None
    liquidation_price = None
    
    # Results containers
    equity_curve = []
    trade_results = []
    funding_payments = []
    liquidations = []
    
    # Reindex funding rates to match OHLC data frequency
    if not funding_df.empty:
        # Create continuous funding rate series
        # For each price timestamp, get the most recent funding rate
        continuous_funding = pd.Series(index=price_df.index)
        for ts in price_df.index:
            # Find last funding rate before this timestamp
            prev_fundings = funding_df[funding_df.index <= ts]
            if not prev_fundings.empty:
                continuous_funding[ts] = prev_fundings['fundingRate'].iloc[-1]
            else:
                continuous_funding[ts] = 0  # Default to zero if no previous funding
    else:
        # If no funding data, set all to zero
        continuous_funding = pd.Series(0, index=price_df.index)
    
    # Add continuous funding to price_df
    price_df['FundingRate'] = continuous_funding
    
    # Process timestamps in chronological order
    for idx, ts in enumerate(price_df.index):
        current_price = price_df.at[ts, 'Close']
        current_funding = price_df.at[ts, 'FundingRate']
        
        # Check for signals at this timestamp
        current_signals = [t for t in raw_trades if t['timestamp'] == ts]
        
        # Check for liquidation if we have an open position
        if position != 0 and liquidation_price is not None:
            # Check if the current bar crossed the liquidation price
            if (is_long and price_df.at[ts, 'Low'] <= liquidation_price) or \
               (not is_long and price_df.at[ts, 'High'] >= liquidation_price):
                # Position is liquidated
                liquidations.append({
                    'timestamp': ts,
                    'price': liquidation_price,
                    'position': position,
                    'pnl': -margin_used  # Total loss equals margin used
                })
                
                # Close position, update cash and equity
                position = 0
                cash -= margin_used  # Lose the margin
                margin_used = 0
                liquidation_price = None
                is_long = None
                entry_price = None
        
        # Apply funding rate payments/charges if we have an open position
        if position != 0:
            # Calculate notional value
            notional_value = abs(position * current_price)
            
            # Calculate funding payment (positive = payment to user, negative = charge to user)
            # For shorts, multiply by -1 (shorts receive negative funding and pay positive funding)
            funding_adjustment = notional_value * current_funding * (-1 if not is_long else 1)
            
            # Apply funding to cash balance
            cash += funding_adjustment
            
            # Track funding payments
            if funding_adjustment != 0:
                funding_payments.append({
                    'timestamp': ts,
                    'rate': current_funding,
                    'payment': funding_adjustment,
                    'position': 'Short' if not is_long else 'Long'
                })
        
        # Process trading signals
        for signal in current_signals:
            action = signal['action']
            signal_price = signal['price'] * (1 + slippage if action == 'buy' else 1 - slippage)
            
            if action == 'buy' and position <= 0:  # Open long or close short
                if position < 0:  # Close existing short position
                    # Calculate profit or loss
                    pnl = margin_used + (entry_price - signal_price) * -position
                    
                    # Record trade result
                    trade_results.append({
                        'Entry Time': entry_time,
                        'Exit Time': ts,
                        'Entry Price': entry_price,
                        'Exit Price': signal_price,
                        'Position': 'Short',
                        'Leverage': leverage_used,
                        'PnL': round(pnl, 2),
                        'Liquidated': False
                    })
                    
                    # Reset position tracking
                    cash += margin_used + pnl - (abs(position) * signal_price * commission)
                    position = 0
                    margin_used = 0
                    entry_price = None
                    liquidation_price = None
                
                # Now open new long position
                if cash > 0:
                    # Calculate position size based on leverage
                    notional = cash * leverage
                    position_size = notional / signal_price
                    
                    # Account for trading commission
                    position_after_fee = position_size * (1 - commission)
                    position = position_after_fee
                    
                    # Calculate required margin
                    margin_used = cash
                    cash = 0
                    
                    # Record entry details
                    entry_time = ts
                    entry_price = signal_price
                    is_long = True
                    leverage_used = leverage
                    
                    # Calculate liquidation price
                    liquidation_price = calculate_liquidation_price(
                        entry_price, leverage, maintenance_margin, is_long=True
                    )
            
            elif action == 'sell' and position >= 0:  # Open short or close long
                if position > 0:  # Close existing long position
                    # Calculate profit or loss
                    pnl = margin_used + (signal_price - entry_price) * position
                    
                    # Record trade result
                    trade_results.append({
                        'Entry Time': entry_time,
                        'Exit Time': ts,
                        'Entry Price': entry_price,
                        'Exit Price': signal_price,
                        'Position': 'Long',
                        'Leverage': leverage_used,
                        'PnL': round(pnl, 2),
                        'Liquidated': False
                    })
                    
                    # Reset position tracking
                    cash += margin_used + pnl - (position * signal_price * commission)
                    position = 0
                    margin_used = 0
                    entry_price = None
                    liquidation_price = None
                
                # Now open new short position
                if cash > 0:
                    # Calculate position size based on leverage
                    notional = cash * leverage
                    position_size = notional / signal_price
                    
                    # Account for trading commission
                    position_after_fee = position_size * (1 - commission)
                    position = -position_after_fee  # Negative for shorts
                    
                    # Calculate required margin
                    margin_used = cash
                    cash = 0
                    
                    # Record entry details
                    entry_time = ts
                    entry_price = signal_price
                    is_long = False
                    leverage_used = leverage
                    
                    # Calculate liquidation price
                    liquidation_price = calculate_liquidation_price(
                        entry_price, leverage, maintenance_margin, is_long=False
                    )
        
        # Calculate unrealized PnL and update equity
        unrealized_pnl = 0
        if position != 0 and entry_price is not None:
            if is_long:
                unrealized_pnl = (current_price - entry_price) * position
            else:
                unrealized_pnl = (entry_price - current_price) * -position
        
        equity = cash + margin_used + unrealized_pnl
        
        # Track equity curve
        equity_curve.append({
            'timestamp': ts,
            'equity': equity,
            'cash': cash,
            'position': position,
            'unrealized_pnl': unrealized_pnl,
            'liquidation_price': liquidation_price if position != 0 else None
        })
    
    # Convert results to DataFrames
    equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
    trades_df = pd.DataFrame(trade_results) if trade_results else pd.DataFrame()
    funding_df = pd.DataFrame(funding_payments) if funding_payments else pd.DataFrame()
    liquidations_df = pd.DataFrame(liquidations) if liquidations else pd.DataFrame()
    
    # Add liquidated trades to trades_df
    for liq in liquidations:
        # Find the corresponding entry in equity_curve to get the entry details
        entry_idx = next((i for i, e in enumerate(equity_curve) 
                         if e['timestamp'] < liq['timestamp'] and e['position'] != 0), None)
        
        if entry_idx is not None:
            entry_record = equity_curve[entry_idx]
            entry_time = entry_record['timestamp']
            # Add to trades dataframe
            liq_trade = {
                'Entry Time': entry_time,
                'Exit Time': liq['timestamp'],
                'Entry Price': entry_price,  # Using the last known entry_price
                'Exit Price': liq['price'],
                'Position': 'Long' if is_long else 'Short',
                'Leverage': leverage_used,
                'PnL': round(liq['pnl'], 2),
                'Liquidated': True
            }
            if not trades_df.empty:
                trades_df = trades_df.append(liq_trade, ignore_index=True)
            else:
                trades_df = pd.DataFrame([liq_trade])
    
    # Calculate metrics
    metrics = calculate_leveraged_metrics(equity_df['equity'], trades_df, funding_df, liquidations_df)
    
    return metrics, trades_df, equity_df, price_df, funding_df

# --- Metrics calculation ---
def calculate_leveraged_metrics(equity_series, trade_results, funding_df, liquidations_df):
    """
    Calculate performance metrics for a leveraged backtest
    
    Args:
        equity_series: Series of equity values over time
        trade_results: DataFrame of completed trades
        funding_df: DataFrame of funding payments
        liquidations_df: DataFrame of liquidation events
        
    Returns:
        Dictionary of metrics
    """
    if equity_series.empty or len(equity_series) < 2:
        return {}
    
    # Calculate basic metrics
    net_profit = equity_series.iloc[-1] - equity_series.iloc[0]
    drawdown = equity_series.cummax() - equity_series
    max_drawdown = drawdown.max()
    returns = equity_series.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * (252**0.5) if returns.std() != 0 else 0
    
    # Trading metrics
    if not trade_results.empty:
        total_trades = len(trade_results)
        wins = trade_results[trade_results['PnL'] > 0]
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = wins['PnL'].mean() if len(wins) > 0 else 0
        
        losses = trade_results[trade_results['PnL'] <= 0]
        avg_loss = losses['PnL'].mean() if len(losses) > 0 else 0
        
        # Calculate expectancy
        expectancy = (win_rate/100) * avg_win + ((100-win_rate)/100) * avg_loss
        
        # Liquidations
        liquidated_trades = trade_results[trade_results['Liquidated'] == True]
        num_liquidations = len(liquidated_trades)
        
        # Funding metrics
        total_funding = funding_df['payment'].sum() if not funding_df.empty else 0
    else:
        total_trades = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        expectancy = 0
        num_liquidations = 0
        total_funding = 0
    
    # Return all metrics
    return {
        'Net Profit': round(net_profit, 2),
        'Max Drawdown': round(max_drawdown, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Total Trades': total_trades,
        'Win Rate (%)': round(win_rate, 2),
        'Avg Win': round(avg_win, 2),
        'Avg Loss': round(avg_loss, 2),
        'Expectancy': round(expectancy, 2),
        'Liquidations': num_liquidations,
        'Funding Impact': round(total_funding, 2)
    }

# --- Streamlit UI component ---
def render_leveraged_backtest(tooltip_func=None):
    """
    Render the leveraged backtest UI component
    
    Args:
        tooltip_func: Function to create tooltips (passed from container.py to avoid circular imports)
    """
    # Use tooltip_func if provided, otherwise create a simple version
    tooltip = tooltip_func if tooltip_func else lambda title, content: f"<span title='{content}'>{title}</span>"
    
    st.markdown(tooltip("🔄 Leveraged Backtesting", 
                      "El backtesting apalancado simula operaciones de trading con apalancamiento, donde utilizas más capital del que tienes disponible. Incluye simulación de liquidaciones y pagos de funding rate para reflejar condiciones reales de mercado."), 
                unsafe_allow_html=True)
    
    # Initialize session state variables if they don't exist
    if 'lev_metrics' not in st.session_state:
        st.session_state.lev_metrics = {}
    if 'lev_trades_df' not in st.session_state:
        st.session_state.lev_trades_df = pd.DataFrame()
    if 'lev_eq_df' not in st.session_state:
        st.session_state.lev_eq_df = pd.DataFrame()
    if 'lev_price_df' not in st.session_state:
        st.session_state.lev_price_df = pd.DataFrame()
    if 'lev_funding_df' not in st.session_state:
        st.session_state.lev_funding_df = pd.DataFrame()
    if 'lev_last_run' not in st.session_state:
        st.session_state.lev_last_run = False
    if 'lev_summary' not in st.session_state:
        st.session_state.lev_summary = {
            'symbol': 'BTCUSDT',
            'interval': '1h',
            'strategy': 'MA Crossover',
            'start_date': (datetime.utcnow().date() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'end_date': datetime.utcnow().date().strftime('%Y-%m-%d'),
            'initial_cash': 10000.0,
            'commission': 0.1,
            'slippage': 0.05,
            'leverage': 3,
            'maintenance_margin': 0.5
        }
    
    # Determine default date range: last 30 days
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=30)
    
    # Settings form in a collapsible section
    with st.expander("Leveraged Backtest Settings", expanded=not st.session_state.lev_last_run):
        # Use a form for more compact UI
        form = st.form('leveraged_backtest_form', clear_on_submit=False)
        col1, col2, col3 = form.columns(3)
        
        with col1:
            symbol = st.text_input('Symbol', value='BTCUSDT', key='lev_symbol').strip().upper()
            interval = st.selectbox('Timeframe', options=['1m','5m','15m','1h','4h','1d'], key='lev_interval')
            strategy_name = st.selectbox('Strategy', 
                                       options=['MA Crossover','Bollinger Breakout','RSI Reversion',
                                               'MACD Momentum','SR Breakout'],
                                       key='lev_strategy',
                                       help="MA Crossover: Cruces de medias móviles. Bollinger: Ruptura de bandas. RSI: Sobrecompra/sobreventa. MACD: Momentum. SR: Soporte/Resistencia.")
        
        with col2:
            start_date = st.date_input('Start Date', value=default_start, key='lev_start')
            end_date = st.date_input('End Date', value=today, key='lev_end')
            initial_cash = st.number_input('Initial Capital (USD)', min_value=100.0, value=10000.0, step=100.0, key='lev_cash')
        
        with col3:
            leverage = st.slider('Leverage', min_value=1, max_value=100, value=3, step=1, key='lev_leverage')
            maintenance_margin = st.slider('Maintenance Margin (%)', min_value=0.1, max_value=5.0, value=0.5, step=0.1, key='lev_maintenance')
            commission = st.number_input('Commission (%)', min_value=0.0, max_value=1.0, value=0.1, step=0.01, key='lev_commission')
            slippage = st.number_input('Slippage (%)', min_value=0.0, max_value=1.0, value=0.05, step=0.01, key='lev_slippage')
        
        run = form.form_submit_button('Run Leveraged Backtest')
    
    # Execute backtest when button is clicked
    if run:
        # Map strategy names to functions
        strategy_map = {
            'MA Crossover': ma_crossover,
            'Bollinger Breakout': bollinger_breakout,
            'RSI Reversion': rsi_reversion,
            'MACD Momentum': macd_momentum,
            'SR Breakout': sr_breakout
        }
        strategy_fn = strategy_map[strategy_name]
        
        # Show progress bar during backtest
        with st.spinner('Running leveraged backtest...'):
            # Run the backtest
            metrics, trades_df, eq_df, price_df, funding_df = run_leveraged_backtest(
                symbol, interval, strategy_fn,
                initial_cash, 
                commission/100, slippage/100,
                leverage, maintenance_margin/100,
                start_date, end_date
            )
        
        # Store results in session state
        st.session_state.lev_metrics = metrics
        st.session_state.lev_trades_df = trades_df
        st.session_state.lev_eq_df = eq_df
        st.session_state.lev_price_df = price_df
        st.session_state.lev_funding_df = funding_df
        st.session_state.lev_last_run = True
        
        # Update summary info
        st.session_state.lev_summary = {
            'symbol': symbol,
            'interval': interval,
            'strategy': strategy_name,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'initial_cash': initial_cash,
            'commission': commission,
            'slippage': slippage,
            'leverage': leverage,
            'maintenance_margin': maintenance_margin
        }
    
    # If no backtest has been run yet, run a default one
    if not st.session_state.lev_last_run:
        # Map strategy names to functions
        strategy_fn = ma_crossover  # Default strategy
        
        # Run a default backtest
        with st.spinner('Running initial leveraged backtest...'):
            metrics, trades_df, eq_df, price_df, funding_df = run_leveraged_backtest(
                'BTCUSDT', '1h', strategy_fn,
                10000.0, 0.001, 0.0005,
                3, 0.005,  # Default leverage: 3x, margin: 0.5%
                default_start, today
            )
        
        # Store results in session state
        st.session_state.lev_metrics = metrics
        st.session_state.lev_trades_df = trades_df
        st.session_state.lev_eq_df = eq_df
        st.session_state.lev_price_df = price_df
        st.session_state.lev_funding_df = funding_df
        st.session_state.lev_last_run = True
    
    # Get current color palette
    from config import PALETTE
    current_palette = PALETTE
    
    # Display summary of backtest parameters
    st.markdown("""
    <style>
    .backtest-summary {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        padding: 10px 15px;
        margin-bottom: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .summary-item {
        display: flex;
        align-items: center;
        margin: 5px 10px;
    }
    .summary-icon {
        font-size: 1.2rem;
        margin-right: 8px;
        width: 24px;
        text-align: center;
    }
    .summary-label {
        font-size: 0.8rem;
        opacity: 0.8;
        margin-right: 4px;
    }
    .summary-value {
        font-weight: bold;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create the summary row with all parameters
    summary = st.session_state.lev_summary
    summary_html = f"""
    <div class="backtest-summary" style="background-color: {current_palette['secondary']}; color: {current_palette['text']}; border: 1px solid {current_palette['border']};">
        <div class="summary-item">
            <div class="summary-icon">💱</div>
            <div class="summary-label">Symbol:</div>
            <div class="summary-value">{summary['symbol']}</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">⏱️</div>
            <div class="summary-label">Timeframe:</div>
            <div class="summary-value">{summary['interval']}</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">🧠</div>
            <div class="summary-label">Strategy:</div>
            <div class="summary-value">{summary['strategy']}</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">📅</div>
            <div class="summary-label">Period:</div>
            <div class="summary-value">{summary['start_date']} to {summary['end_date']}</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">💰</div>
            <div class="summary-label">Capital:</div>
            <div class="summary-value">${summary['initial_cash']:,.2f}</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">📈</div>
            <div class="summary-label">Leverage:</div>
            <div class="summary-value">{summary['leverage']}x</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">🛡️</div>
            <div class="summary-label">Margin:</div>
            <div class="summary-value">{summary['maintenance_margin']}%</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">💸</div>
            <div class="summary-label">Commission:</div>
            <div class="summary-value">{summary['commission']}%</div>
        </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)
    
    # Results section
    metrics_section, chart_section = st.columns(2)
    
    with metrics_section:
        st.markdown(tooltip('📊 Performance Metrics', 
                           'Métricas clave que resumen el rendimiento de tu estrategia. Incluyen rentabilidad, drawdown (caída máxima), ratio de Sharpe (rentabilidad ajustada al riesgo) y estadísticas de operaciones.'), 
                    unsafe_allow_html=True)
        
        # Display metrics in a grid
        mcols = st.columns(3)
        for i, (name, val) in enumerate(st.session_state.lev_metrics.items()):
            mcols[i % 3].metric(name, val)
        
        # Display funding impacts table if available
        if not st.session_state.lev_funding_df.empty:
            st.markdown(tooltip('💰 Funding Impacts', 
                               'Resumen de los pagos de funding rate. Los mercados de futuros perpetuos utilizan pagos periódicos (funding) para mantener el precio del contrato cerca del mercado spot.'), 
                       unsafe_allow_html=True)
            
            # Group funding by day and position type for summary
            funding_summary = st.session_state.lev_funding_df.copy()
            if 'timestamp' in funding_summary.columns:
                funding_summary['day'] = funding_summary['timestamp'].dt.date
                daily_funding = funding_summary.groupby(['day', 'position'])['payment'].sum().reset_index()
                
                # Format for display
                daily_funding['payment'] = daily_funding['payment'].map('${:,.2f}'.format)
                st.dataframe(daily_funding, use_container_width=True)
            else:
                st.info("No funding payments in this period.")
        
        # Equity curve
        st.markdown(tooltip('💹 Equity Curve', 
                           'Gráfico que muestra la evolución de tu capital a lo largo del tiempo. Una curva ascendente indica rentabilidad, mientras que las caídas representan pérdidas.'), 
                   unsafe_allow_html=True)
        
        if not st.session_state.lev_eq_df.empty:
            eq_df = st.session_state.lev_eq_df.reset_index()
            
            # Create equity curve chart with funding payment markers
            fig_eq = px.line(
                eq_df, x='timestamp', y='equity', 
                title=None,
                line_shape='linear',
                color_discrete_sequence=[current_palette['equity']]
            )
            
            # Add funding payments as markers if available
            if not st.session_state.lev_funding_df.empty:
                funding_df = st.session_state.lev_funding_df
                if 'timestamp' in funding_df.columns:
                    # Merge with equity curve to get corresponding equity values
                    funding_with_equity = pd.merge_asof(
                        funding_df.sort_values('timestamp'),
                        eq_df[['timestamp', 'equity']].sort_values('timestamp'),
                        on='timestamp',
                        direction='nearest'
                    )
                    
                    # Add positive funding (receiving) markers
                    positive_funding = funding_with_equity[funding_with_equity['payment'] > 0]
                    if not positive_funding.empty:
                        fig_eq.add_trace(
                            go.Scatter(
                                x=positive_funding['timestamp'],
                                y=positive_funding['equity'],
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-up',
                                    size=10,
                                    color='green'
                                ),
                                name='Funding Received'
                            )
                        )
                    
                    # Add negative funding (paying) markers
                    negative_funding = funding_with_equity[funding_with_equity['payment'] < 0]
                    if not negative_funding.empty:
                        fig_eq.add_trace(
                            go.Scatter(
                                x=negative_funding['timestamp'],
                                y=negative_funding['equity'],
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-down',
                                    size=10,
                                    color='red'
                                ),
                                name='Funding Paid'
                            )
                        )
            
            # Update chart layout with palette colors
            fig_eq.update_layout(
                template=current_palette['template'],
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor=current_palette['chart_bg'],
                plot_bgcolor=current_palette['chart_bg'],
                font=dict(color=current_palette['chart_text']),
                xaxis=dict(
                    gridcolor=current_palette['chart_grid'],
                    zerolinecolor=current_palette['chart_grid']
                ),
                yaxis=dict(
                    gridcolor=current_palette['chart_grid'],
                    zerolinecolor=current_palette['chart_grid']
                )
            )
            
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.warning('No equity data available to plot.')
    
    with chart_section:
        st.markdown(tooltip('📝 Price Chart & Trades', 
                           'Gráfico de precios con marcadores de entradas, salidas y niveles de liquidación. Te permite visualizar cuándo tu estrategia entró y salió del mercado.'), 
                   unsafe_allow_html=True)
        
        # Create price chart with trades and liquidation levels
        if not st.session_state.lev_price_df.empty:
            fig_price = go.Figure(
                data=[
                    go.Candlestick(
                        x=st.session_state.lev_price_df.index,
                        open=st.session_state.lev_price_df['Open'],
                        high=st.session_state.lev_price_df['High'],
                        low=st.session_state.lev_price_df['Low'],
                        close=st.session_state.lev_price_df['Close'],
                        name='Price'
                    )
                ]
            )
            
            # Add trade entries and exits
            if not st.session_state.lev_trades_df.empty:
                # Long entries (buys)
                long_entries = st.session_state.lev_trades_df[
                    (st.session_state.lev_trades_df['Position'] == 'Long')
                ]
                if not long_entries.empty:
                    fig_price.add_trace(
                        go.Scatter(
                            x=long_entries['Entry Time'],
                            y=long_entries['Entry Price'],
                            mode='markers',
                            marker=dict(
                                color=current_palette['entries'],
                                symbol='triangle-up',
                                size=10
                            ),
                            name='Long Entries'
                        )
                    )
                
                # Long exits
                if not long_entries.empty:
                    fig_price.add_trace(
                        go.Scatter(
                            x=long_entries['Exit Time'],
                            y=long_entries['Exit Price'],
                            mode='markers',
                            marker=dict(
                                color=current_palette['exits'],
                                symbol='triangle-down',
                                size=10
                            ),
                            name='Long Exits'
                        )
                    )
                
                # Short entries (sells)
                short_entries = st.session_state.lev_trades_df[
                    (st.session_state.lev_trades_df['Position'] == 'Short')
                ]
                if not short_entries.empty:
                    fig_price.add_trace(
                        go.Scatter(
                            x=short_entries['Entry Time'],
                            y=short_entries['Entry Price'],
                            mode='markers',
                            marker=dict(
                                color='red',
                                symbol='triangle-down',
                                size=10
                            ),
                            name='Short Entries'
                        )
                    )
                
                # Short exits
                if not short_entries.empty:
                    fig_price.add_trace(
                        go.Scatter(
                            x=short_entries['Exit Time'],
                            y=short_entries['Exit Price'],
                            mode='markers',
                            marker=dict(
                                color='green',
                                symbol='triangle-up',
                                size=10
                            ),
                            name='Short Exits'
                        )
                    )
                
                # Liquidations
                liquidations = st.session_state.lev_trades_df[
                    st.session_state.lev_trades_df['Liquidated'] == True
                ]
                if not liquidations.empty:
                    fig_price.add_trace(
                        go.Scatter(
                            x=liquidations['Exit Time'],
                            y=liquidations['Exit Price'],
                            mode='markers',
                            marker=dict(
                                color='yellow',
                                symbol='x',
                                size=12,
                                line=dict(width=2, color='black')
                            ),
                            name='Liquidations'
                        )
                    )
            
            # Add liquidation price lines from equity curve
            if not st.session_state.lev_eq_df.empty and 'liquidation_price' in st.session_state.lev_eq_df.columns:
                liq_df = st.session_state.lev_eq_df.reset_index()
                
                # Filter to only rows with liquidation prices
                liq_df = liq_df[liq_df['liquidation_price'].notnull()]
                
                if not liq_df.empty:
                    # Group by continuous segments of liquidation prices
                    liq_df['group'] = (liq_df['liquidation_price'].isna() != 
                                      liq_df['liquidation_price'].shift().isna()).cumsum()
                    
                    for group, data in liq_df.groupby('group'):
                        if data['liquidation_price'].notnull().any():
                            fig_price.add_trace(
                                go.Scatter(
                                    x=data['timestamp'],
                                    y=data['liquidation_price'],
                                    mode='lines',
                                    line=dict(
                                        color='red',
                                        width=1,
                                        dash='dash'
                                    ),
                                    name='Liquidation Level'
                                )
                            )
            
            # Update chart layout with palette colors
            fig_price.update_layout(
                template=current_palette['template'],
                height=400,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor=current_palette['chart_bg'],
                plot_bgcolor=current_palette['chart_bg'],
                font=dict(color=current_palette['chart_text']),
                xaxis=dict(
                    gridcolor=current_palette['chart_grid'],
                    zerolinecolor=current_palette['chart_grid']
                ),
                yaxis=dict(
                    gridcolor=current_palette['chart_grid'],
                    zerolinecolor=current_palette['chart_grid']
                )
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.warning('No price data available to plot.')
        
        # Trade results table
        st.markdown(tooltip('📝 Trade Results', 
                           'Tabla detallada de todas las operaciones realizadas. Incluye precios de entrada y salida, tipo de posición, apalancamiento usado, resultado (PnL) y si hubo liquidación.'), 
                   unsafe_allow_html=True)
        
        if not st.session_state.lev_trades_df.empty:
            # Add highlight for liquidated trades
            st.dataframe(st.session_state.lev_trades_df.style.apply(
                lambda row: ['background-color: rgba(255,0,0,0.2)' if row['Liquidated'] else '' 
                           for _ in row], axis=1
            ), use_container_width=True)
        else:
            st.info('No trades executed in this period.')