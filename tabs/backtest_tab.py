"""
Módulo de backtesting para el sistema de análisis de estrategias.

Este módulo implementa un motor de backtesting para evaluar estrategias
de trading en diferentes activos y timeframes con soporte para:
- Comisiones y slippage
- Filtrado por fechas
- Múltiples estrategias predefinidas
- Visualización de resultados
"""
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from models.strategies import (
    ma_crossover,
    bollinger_breakout,
    rsi_reversion,
    macd_momentum,
    sr_breakout
)
import config
from utils.html_utils import tooltip
from utils.data_utils import format_backtest_summary

# --- Utility functions for metrics ---
def calculate_metrics(equity_series, trade_results):
    """
    Calcula métricas de rendimiento para una estrategia de trading
    
    Args:
        equity_series: Serie temporal con la evolución del capital
        trade_results: Lista de diccionarios con los resultados de las operaciones
        
    Returns:
        Diccionario con las métricas calculadas
    """
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
    """
    Ejecuta un backtest para una estrategia de trading
    
    Args:
        symbol: Par de trading (ej. 'BTCUSDT')
        interval: Timeframe (ej. '1h', '4h', '1d')
        strategy_fn: Función de estrategia de trading
        initial_cash: Capital inicial
        commission: Comisión como fracción (ej. 0.001 para 0.1%)
        slippage: Slippage como fracción (ej. 0.0005 para 0.05%)
        start_date: Fecha de inicio para el backtest
        end_date: Fecha de fin para el backtest
        
    Returns:
        Métricas, DataFrame de operaciones, curva de capital, y DataFrame de precios
    """
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
    """
    Renderiza la interfaz de usuario para el backtesting
    """
    st.markdown(tooltip('🔄 Backtesting Engine', 
                      'El backtesting te permite probar estrategias de trading utilizando datos históricos para simular operaciones y evaluar su rendimiento. Ayuda a perfeccionar estrategias antes de usar dinero real.'),
               unsafe_allow_html=True)
    
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
            symbol = st.text_input('Symbol', value='BTCUSDT', key='bt_symbol').strip().upper()
            interval = st.selectbox('Timeframe', options=['1m','5m','15m','1h','4h','1d'], key='bt_interval')
            strategy_name = st.selectbox('Strategy', 
                                       options=['MA Crossover','Bollinger Breakout','RSI Reversion',
                                               'MACD Momentum','SR Breakout'],
                                       key='bt_strategy',
                                       help="MA Crossover: Cruces de medias móviles. Bollinger: Ruptura de bandas. RSI: Sobrecompra/sobreventa. MACD: Momentum. SR: Soporte/Resistencia.")
        
        with col2:
            start_date = st.date_input('Start Date', value=default_start, key='bt_start')
            end_date = st.date_input('End Date', value=today, key='bt_end')
        
        with col3:
            initial_cash = st.number_input('Initial Capital (USD)', 
                                         min_value=100.0, value=10000.0, step=100.0, key='bt_cash')
            commission = st.number_input('Commission (%)', 
                                       min_value=0.0, max_value=1.0, value=0.1, step=0.01, key='bt_commission')
            slippage = st.number_input('Slippage (%)', 
                                     min_value=0.0, max_value=1.0, value=0.05, step=0.01, key='bt_slippage')
        
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
    
    # Obtener la paleta actual desde config
    current_palette = config.PALETTE
    
    # Mostrar resumen del backtest usando la función de utilidad
    summary = st.session_state.bt_summary
    summary_html = format_backtest_summary(summary, current_palette)
    st.markdown(summary_html, unsafe_allow_html=True)
    
    # Mostrar resultados del backtest
    res_left, res_right = st.columns(2)
    with res_left:
        
        # Interactive equity curve
        st.markdown(tooltip('Equity Curve', 
                           'La curva de capital muestra la evolución de tu inversión a lo largo del tiempo. Una curva ascendente indica rentabilidad, mientras que las caídas representan pérdidas.'),
                  unsafe_allow_html=True)
        
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
            
        st.markdown(tooltip('📊 Performance Metrics', 
                           'Métricas clave que resumen el rendimiento de tu estrategia. Incluyen rentabilidad, drawdown (caída máxima), ratio de Sharpe (rentabilidad ajustada al riesgo) y estadísticas de operaciones.'),
                   unsafe_allow_html=True)
        
        # Metrics
        mcols = st.columns(4)
        for i, (name, val) in enumerate(st.session_state.bt_metrics.items()):
            mcols[i % 4].metric(name, val)
        

    with res_right:
        st.markdown(tooltip('📝 Price Action & Trade Entries/Exits', 
                           'Gráfico de precios con marcadores que indican los puntos de entrada y salida de tus operaciones. Te permite visualizar cuándo tu estrategia entró y salió del mercado.'),
                   unsafe_allow_html=True)
        
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
        
        st.markdown(tooltip('📝 Trade Results', 
                           'Tabla detallada de todas las operaciones realizadas. Incluye precios de entrada y salida, resultado (PnL - Profit and Loss) y fechas de cada operación.'),
                   unsafe_allow_html=True)
        
        if not st.session_state.bt_trades_df.empty:
            st.dataframe(st.session_state.bt_trades_df)
        else:
            st.info('No trades executed in this period.')