# backtest_tab.py
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from strategies import (
    ma_crossover,
    bollinger_breakout,
    rsi_reversion,
    macd_momentum,
    sr_breakout
)

# --- Utility functions for metrics ---
def calculate_metrics(equity_series, trade_results):
    net_profit = equity_series.iloc[-1] - equity_series.iloc[0]
    drawdown = equity_series.cummax() - equity_series
    max_drawdown = drawdown.max()
    returns = equity_series.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * (252**0.5) if returns.std() != 0 else 0

    wins = [r['PnL'] for r in trade_results if r['PnL'] > 0]
    losses = [r['PnL'] for r in trade_results if r['PnL'] <= 0]
    win_rate = len(wins) / len(trade_results) * 100 if trade_results else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    expectancy = (win_rate/100) * avg_win + ((100-win_rate)/100) * avg_loss

    return {
        'Net Profit':    round(net_profit, 2),
        'Max Drawdown':  round(max_drawdown, 2),
        'Sharpe Ratio':  round(sharpe, 2),
        'Total Trades':  len(trade_results),
        'Win Rate (%)':  round(win_rate, 2),
        'Avg Win':       round(avg_win, 2),
        'Avg Loss':      round(avg_loss, 2),
        'Expectancy':    round(expectancy, 2)
    }

# --- Backtest engine with commission & slippage ---
def run_backtest(symbol, interval, strategy_fn, initial_cash, commission=0.001, slippage=0.0005):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1000"
    cols = ['Open time','Open','High','Low','Close','Volume','Close time',
            'Quote asset volume','Num trades','Taker buy base','Taker buy quote','Ignore']
    df = pd.DataFrame(requests.get(url).json(), columns=cols)
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = df[c].astype(float)

    raw_trades = strategy_fn(df)
    cash = initial_cash
    position = 0.0
    equity_curve = []
    trade_results = []
    entry_time = entry_cash = entry_price = None

    for trade in raw_trades:
        ts = trade['timestamp']
        price = trade['price'] * (1 + slippage if trade['action']=='buy' else 1 - slippage)
        if trade['action'] == 'buy' and cash > 0:
            cost = price * (1 + commission)
            position = cash / cost
            entry_time, entry_cash, entry_price = ts, cash, price
            cash = 0
        elif trade['action'] == 'sell' and position > 0:
            proceeds = position * price * (1 - commission)
            pnl = proceeds - entry_cash
            trade_results.append({
                'Entry Time': entry_time,
                'Exit Time': ts,
                'Entry Price': entry_price,
                'Exit Price': price,
                'PnL': round(pnl, 2)
            })
            cash, position = proceeds, 0
        equity_curve.append({'timestamp': ts, 'equity': cash + position * price})

    eq_df = pd.DataFrame(equity_curve).set_index('timestamp')
    metrics = calculate_metrics(eq_df['equity'], trade_results)
    trades_df = pd.DataFrame(trade_results)
    return metrics, trades_df, eq_df

# --- Streamlit UI ---

def render_backtests():
    st.header('🔄 Backtesting Engine')
    left, right = st.columns([1, 1])

    # Left column: input form
    with left:
        with st.form('backtest_form'):
            symbol = st.text_input('Symbol (e.g. BTCUSDT)', 'BTCUSDT', key='bt_symbol').strip().upper()
            interval = st.selectbox('Timeframe', ['1m','5m','15m','1h','4h','1d'], key='bt_interval')
            strategy_name = st.selectbox(
                'Strategy',
                ['MA Crossover','Bollinger Breakout','RSI Reversion','MACD Momentum','SR Breakout'],
                key='bt_strategy'
            )
            initial_cash = st.number_input('Initial Capital (USD)', min_value=100.0, value=10000.0, step=100.0, key='bt_cash')
            commission = st.number_input('Commission (%)', min_value=0.0, max_value=1.0, value=0.1, step=0.01, key='bt_commission')
            slippage = st.number_input('Slippage (%)', min_value=0.0, max_value=1.0, value=0.05, step=0.01, key='bt_slippage')
            run = st.form_submit_button('Run Backtest')

    # Execute backtest and display results
    if run:
        strategy_map = {
            'MA Crossover': ma_crossover,
            'Bollinger Breakout': bollinger_breakout,
            'RSI Reversion': rsi_reversion,
            'MACD Momentum': macd_momentum,
            'SR Breakout': sr_breakout
        }
        strategy_fn = strategy_map[strategy_name]
        metrics, trades_df, eq_df = run_backtest(
            symbol, interval, strategy_fn,
            initial_cash, commission/100, slippage/100
        )

        # Right column: metrics and interactive chart
        with right:
            st.subheader('📊 Performance Metrics')
            cols = st.columns(4)
            for i, (name, val) in enumerate(metrics.items()):
                cols[i % 4].metric(name, val)

            st.subheader('📈 Equity Curve')
            df_plot = eq_df.reset_index()
            fig = px.line(df_plot, x='timestamp', y='equity', title='Equity Curve')
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader('📝 Trade Results')
            st.dataframe(trades_df)
