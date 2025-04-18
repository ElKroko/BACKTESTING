# backtest_tab.py
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from strategies import (
    ma_crossover,
    bollinger_breakout,
    rsi_reversion,
    macd_momentum,
    sr_breakout
)
from config import PALETTES  # Importamos PALETTES para acceder a las paletas

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

# --- Backtest engine with commission, slippage & date filtering ---
def run_backtest(symbol, interval, strategy_fn, initial_cash,
                 commission=0.001, slippage=0.0005,
                 start_date=None, end_date=None):
    # Fetch OHLC data
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1000"
    cols = ['Open time','Open','High','Low','Close','Volume','Close time',
            'Quote asset volume','Num trades','Taker buy base','Taker buy quote','Ignore']
    df = pd.DataFrame(requests.get(url).json(), columns=cols)
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = df[c].astype(float)

    # Filter by date range
    if start_date:
        df = df.loc[df.index >= pd.to_datetime(start_date)]
    if end_date:
        # include end_date entire day
        df = df.loc[df.index <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)]
    if df.empty:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
    # Return OHLC df as price_df for charting
    return metrics, trades_df, eq_df, df

# --- Streamlit UI ---
def render_backtests():
    st.header('🔄 Backtesting Engine')
    
    # Inicializar variables de estado en session_state si no existen
    if 'bt_metrics' not in st.session_state:
        st.session_state.bt_metrics = {}
    if 'bt_trades_df' not in st.session_state:
        st.session_state.bt_trades_df = pd.DataFrame()
    if 'bt_eq_df' not in st.session_state:
        st.session_state.bt_eq_df = pd.DataFrame()
    if 'bt_price_df' not in st.session_state:
        st.session_state.bt_price_df = pd.DataFrame()
    if 'bt_last_run' not in st.session_state:
        st.session_state.bt_last_run = False
    if 'bt_summary' not in st.session_state:
        st.session_state.bt_summary = {
            'symbol': 'BTCUSDT',
            'interval': '1m',
            'strategy': 'MA Crossover',
            'start_date': (datetime.utcnow().date() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'end_date': datetime.utcnow().date().strftime('%Y-%m-%d'),
            'initial_cash': 10000.0,
            'commission': 0.1,
            'slippage': 0.05
        }
    
    # Determine default date range: last 30 days
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=30)

    # Form más compacto usando expander
    with st.expander("Backtest Settings", expanded=not st.session_state.bt_last_run):
        # Formulario compacto en 3 columnas para ocupar menos espacio
        form = st.form('backtest_form', clear_on_submit=False)
        col1, col2, col3 = form.columns(3)
        
        with col1:
            symbol = st.text_input('Symbol', 'BTCUSDT', key='bt_symbol').strip().upper()
            interval = st.selectbox('Timeframe', ['1m','5m','15m','1h','4h','1d'], key='bt_interval')
            strategy_name = st.selectbox('Strategy',
                ['MA Crossover','Bollinger Breakout','RSI Reversion','MACD Momentum','SR Breakout'],
                key='bt_strategy')
        
        with col2:
            start_date = st.date_input('Start Date', value=default_start, key='bt_start')
            end_date = st.date_input('End Date', value=today, key='bt_end')
        
        with col3:
            initial_cash = st.number_input('Initial Capital (USD)', min_value=100.0, value=10000.0, step=100.0, key='bt_cash')
            commission = st.number_input('Commission (%)', min_value=0.0, max_value=1.0, value=0.1, step=0.01, key='bt_commission')
            slippage = st.number_input('Slippage (%)', min_value=0.0, max_value=1.0, value=0.05, step=0.01, key='bt_slippage')
        
        run = form.form_submit_button('Run Backtest')

    # Ejecutar backtest si se presiona el botón o si es la primera vez
    if run:
        strategy_map = {
            'MA Crossover': ma_crossover,
            'Bollinger Breakout': bollinger_breakout,
            'RSI Reversion': rsi_reversion,
            'MACD Momentum': macd_momentum,
            'SR Breakout': sr_breakout
        }
        strategy_fn = strategy_map[strategy_name]
        
        metrics, trades_df, eq_df, price_df = run_backtest(
            symbol, interval, strategy_fn,
            initial_cash, commission/100, slippage/100,
            start_date, end_date
        )
        
        # Guardar resultados en session_state
        st.session_state.bt_metrics = metrics
        st.session_state.bt_trades_df = trades_df
        st.session_state.bt_eq_df = eq_df
        st.session_state.bt_price_df = price_df
        st.session_state.bt_last_run = True
        
        # Actualizar el resumen
        st.session_state.bt_summary = {
            'symbol': symbol,
            'interval': interval,
            'strategy': strategy_name,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'initial_cash': initial_cash,
            'commission': commission,
            'slippage': slippage
        }
    
    # Si aún no se ha ejecutado ningún backtest, ejecutar uno con valores predeterminados
    if not st.session_state.bt_last_run:
        strategy_fn = ma_crossover  # Estrategia predeterminada
        metrics, trades_df, eq_df, price_df = run_backtest(
            'BTCUSDT', '1m', strategy_fn,
            10000.0, 0.001, 0.0005,
            default_start, today
        )
        
        # Guardar resultados en session_state
        st.session_state.bt_metrics = metrics
        st.session_state.bt_trades_df = trades_df
        st.session_state.bt_eq_df = eq_df
        st.session_state.bt_price_df = price_df
        st.session_state.bt_last_run = True
    
    # Obtener la paleta actual desde config mediante un container key
    # Esto asegura que los gráficos se actualicen cuando cambie la paleta sin necesidad de reejecutar el backtest
    from config import PALETTE
    current_palette = PALETTE
    
    # Nueva fila de resumen con 100% de ancho
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
    
    # Crear la fila de resumen con los parámetros del último backtest ejecutado
    summary = st.session_state.bt_summary
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
            <div class="summary-icon">💸</div>
            <div class="summary-label">Commission:</div>
            <div class="summary-value">{summary['commission']}%</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">🧮</div>
            <div class="summary-label">Slippage:</div>
            <div class="summary-value">{summary['slippage']}%</div>
        </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)
    
    # Mostrar resultados del backtest
    res_left, res_right = st.columns(2)
    with res_left:
        
        # Interactive equity curve
        st.subheader('Equity Curve')
        if not st.session_state.bt_eq_df.empty:
            df_plot = st.session_state.bt_eq_df.reset_index()
            if 'timestamp' not in df_plot.columns and 'index' in df_plot.columns:
                df_plot = df_plot.rename(columns={'index': 'timestamp'})
            
            fig_eq = px.line(
                df_plot, x='timestamp', y='equity', title=None,
                line_shape='linear',
                color_discrete_sequence=[current_palette['equity']]
            )
            fig_eq.update_layout(
                template=current_palette['template'],
                height=250, 
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
            
        st.subheader('📊 Performance Metrics & Equity Curve')
        # Metrics
        mcols = st.columns(4)
        for i, (name, val) in enumerate(st.session_state.bt_metrics.items()):
            mcols[i % 4].metric(name, val)
        

    with res_right:
        st.subheader('📝 Price Action & Trade Entries/Exits')
        if not st.session_state.bt_price_df.empty:
            fig_price = go.Figure(
                data=[
                    go.Candlestick(
                        x=st.session_state.bt_price_df.index,
                        open=st.session_state.bt_price_df['Open'], 
                        high=st.session_state.bt_price_df['High'],
                        low=st.session_state.bt_price_df['Low'], 
                        close=st.session_state.bt_price_df['Close'],
                        name='Price'
                    )
                ]
            )
            if not st.session_state.bt_trades_df.empty:
                fig_price.add_trace(
                    go.Scatter(
                        x=st.session_state.bt_trades_df['Entry Time'], 
                        y=st.session_state.bt_trades_df['Entry Price'],
                        mode='markers', 
                        marker=dict(color=current_palette['entries'], symbol='triangle-up', size=10),
                        name='Entries'
                    )
                )
                fig_price.add_trace(
                    go.Scatter(
                        x=st.session_state.bt_trades_df['Exit Time'], 
                        y=st.session_state.bt_trades_df['Exit Price'],
                        mode='markers', 
                        marker=dict(color=current_palette['exits'], symbol='triangle-down', size=10),
                        name='Exits'
                    )
                )
            fig_price.update_layout(
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
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.warning('No price data available.')
        
        st.subheader('📝 Trade Results')
        if not st.session_state.bt_trades_df.empty:
            st.dataframe(st.session_state.bt_trades_df)
        else:
            st.info('No trades executed in this period.')
