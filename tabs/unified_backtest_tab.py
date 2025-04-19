"""
Módulo de backtesting unificado para spot y futuros apalancados.

Este módulo implementa una interfaz unificada para backtesting que soporta:
- Trading spot
- Trading de futuros con apalancamiento
- Simulación de liquidaciones
- Funding rates para futuros perpetuos
- Exportación de resultados a PDF
- Backtesting por lotes (batch)
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import base64
import ta

# Importamos las estrategias existentes
from models.strategies import (
    ma_crossover,
    bollinger_breakout,
    rsi_reversion,
    macd_momentum,
    sr_breakout
)

# Importamos utilidades
import config
from utils.html_utils import tooltip
from utils.data_utils import format_backtest_summary

# Importamos funciones de los módulos de backtesting existentes
from tabs.backtest_tab import run_backtest as run_spot_backtest
from tabs.leveraged_backtest import (
    run_leveraged_backtest,
    fetch_ohlc_data,
    fetch_funding_rates,
    calculate_liquidation_price
)

# --- Importaciones para PDF ---
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import io

# --- Constantes para los modos de operación ---
MODES = {
    "spot": "Spot Trading",
    "futures": "Futures Trading w/ Leverage"
}

# --- Backtest unificado ---
def run_unified_backtest(
    symbol, interval, strategy_fn, initial_cash,
    commission=0.001, slippage=0.0005,
    mode="spot", leverage=1, maintenance_margin=0.005,
    start_date=None, end_date=None):
    """
    Ejecuta un backtest unificado que puede ser spot o futures con apalancamiento
    
    Args:
        symbol: Par de trading (ej. 'BTCUSDT')
        interval: Timeframe (ej. '1h', '4h', '1d')
        strategy_fn: Función de estrategia de trading
        initial_cash: Capital inicial
        commission: Comisión como fracción (ej. 0.001 para 0.1%)
        slippage: Slippage como fracción (ej. 0.0005 para 0.05%)
        mode: Modo de trading ('spot' o 'futures')
        leverage: Apalancamiento (solo para futures)
        maintenance_margin: Margen de mantenimiento (solo para futures)
        start_date: Fecha de inicio del backtest
        end_date: Fecha de fin del backtest
        
    Returns:
        Tuple con métricas, trades_df, equity_df, price_df y funding_df (si aplica)
    """
    if mode == "spot":
        # Ejecutar backtest spot
        metrics, trades_df, equity_df, price_df = run_spot_backtest(
            symbol, interval, strategy_fn, initial_cash,
            commission, slippage, start_date, end_date
        )
        # Para mantener la consistencia en el formato de retorno
        funding_df = pd.DataFrame()
        
    else:  # mode == "futures"
        # Ejecutar backtest con apalancamiento
        metrics, trades_df, equity_df, price_df, funding_df = run_leveraged_backtest(
            symbol, interval, strategy_fn, initial_cash,
            commission, slippage, leverage, maintenance_margin,
            start_date, end_date
        )
    
    return metrics, trades_df, equity_df, price_df, funding_df

# --- Función para backtesting por lotes ---
def run_batch_backtest(
    symbols, interval, strategy_fn, initial_cash,
    commission=0.001, slippage=0.0005,
    mode="spot", leverage=1, maintenance_margin=0.005,
    start_date=None, end_date=None):
    """
    Ejecuta un backtest por lotes para múltiples símbolos
    
    Args:
        symbols: Lista de pares de trading (ej. ['BTCUSDT', 'ETHUSDT'])
        interval: Timeframe (ej. '1h', '4h', '1d')
        strategy_fn: Función de estrategia de trading
        initial_cash: Capital inicial
        commission: Comisión como fracción (ej. 0.001 para 0.1%)
        slippage: Slippage como fracción (ej. 0.0005 para 0.05%)
        mode: Modo de trading ('spot' o 'futures')
        leverage: Apalancamiento (solo para futures)
        maintenance_margin: Margen de mantenimiento (solo para futures)
        start_date: Fecha de inicio del backtest
        end_date: Fecha de fin del backtest
        
    Returns:
        DataFrame con resultados comparativos
    """
    results = []
    
    for symbol in symbols:
        try:
            # Ejecutar backtest para este símbolo
            metrics, trades_df, equity_df, _, _ = run_unified_backtest(
                symbol, interval, strategy_fn, initial_cash,
                commission, slippage, mode, leverage, maintenance_margin,
                start_date, end_date
            )
            
            if metrics:  # Si hay resultados válidos
                # Calcular métricas adicionales
                if not equity_df.empty:
                    # Calcular retorno porcentual
                    if 'equity' in equity_df:
                        initial_equity = equity_df['equity'].iloc[0]
                        final_equity = equity_df['equity'].iloc[-1]
                        returns_pct = ((final_equity / initial_equity) - 1) * 100
                    else:
                        returns_pct = 0
                    
                    # Crear fila de resultados
                    result = {
                        'Symbol': symbol,
                        'Net Profit': metrics.get('Net Profit', 0),
                        'Return (%)': round(returns_pct, 2),
                        'Max Drawdown': metrics.get('Max Drawdown', 0),
                        'Sharpe': metrics.get('Sharpe Ratio', 0),
                        'Trades': metrics.get('Total Trades', 0),
                        'Win Rate (%)': metrics.get('Win Rate (%)', 0),
                    }
                    
                    # Añadir liquidaciones si es modo futures
                    if mode == 'futures':
                        result['Liquidations'] = metrics.get('Liquidations', 0)
                        result['Funding Impact'] = metrics.get('Funding Impact', 0)
                    
                    results.append(result)
        except Exception as e:
            st.error(f"Error en backtest para {symbol}: {str(e)}")
    
    # Convertir a DataFrame
    if results:
        results_df = pd.DataFrame(results)
        # Ordenar por retorno descendente
        results_df = results_df.sort_values('Return (%)', ascending=False)
        return results_df
    else:
        return pd.DataFrame()

# --- Función para generar PDF ---
def generate_backtest_pdf(
    symbol, interval, strategy, mode, leverage, maintenance_margin,
    start_date, end_date, initial_cash, commission, slippage,
    metrics, trades_df, equity_df, price_df, funding_df=None):
    """
    Genera un informe PDF con los resultados del backtest
    
    Args:
        symbol, interval, strategy: Información básica del backtest
        mode: 'spot' o 'futures'
        leverage, maintenance_margin: Parámetros de futuros
        start_date, end_date: Rango de fechas
        initial_cash, commission, slippage: Parámetros de trading
        metrics: Diccionario con métricas calculadas
        trades_df: DataFrame con operaciones
        equity_df: DataFrame con equity curve
        price_df: DataFrame con precios
        funding_df: DataFrame con funding rates (opcional)
        
    Returns:
        PDF como bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    
    # Estilo de título
    title_style = styles['Heading1']
    title_style.alignment = 1  # Centrado
    
    # Estilo de subtítulo
    subtitle_style = styles['Heading2']
    
    # Estilo de texto normal
    normal_style = styles['Normal']
    
    # --- Título del reporte ---
    elements.append(Paragraph(f"Backtest Report: {symbol}", title_style))
    elements.append(Spacer(1, 0.25*inch))
    
    # --- Parámetros del backtest ---
    elements.append(Paragraph("Backtest Parameters", subtitle_style))
    
    # Tabla de parámetros
    params_data = [
        ["Parameter", "Value"],
        ["Symbol", symbol],
        ["Timeframe", interval],
        ["Strategy", strategy],
        ["Mode", MODES[mode]],
        ["Start Date", start_date],
        ["End Date", end_date],
        ["Initial Capital", f"${initial_cash:,.2f}"],
        ["Commission", f"{commission*100:.2f}%"],
        ["Slippage", f"{slippage*100:.2f}%"],
    ]
    
    # Añadir parámetros específicos de futuros si aplica
    if mode == "futures":
        params_data.append(["Leverage", f"{leverage}x"])
        params_data.append(["Maintenance Margin", f"{maintenance_margin*100:.2f}%"])
    
    params_table = Table(params_data, colWidths=[2.5*inch, 3*inch])
    params_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.white),
        ('GRID', (0, 0), (1, -1), 1, colors.black),
    ]))
    
    elements.append(params_table)
    elements.append(Spacer(1, 0.5*inch))
    
    # --- Métricas de rendimiento ---
    elements.append(Paragraph("Performance Metrics", subtitle_style))
    
    # Crear tabla de métricas
    metrics_data = [["Metric", "Value"]]
    
    for key, value in metrics.items():
        # Formatear según tipo de métrica
        if "Profit" in key or "Drawdown" in key or "Impact" in key:
            formatted_value = f"${value}"
        elif "%" in key:
            formatted_value = f"{value}%"
        elif "Ratio" in key:
            formatted_value = f"{value}"
        else:
            formatted_value = f"{value}"
        
        metrics_data.append([key, formatted_value])
    
    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.white),
        ('GRID', (0, 0), (1, -1), 1, colors.black),
    ]))
    
    elements.append(metrics_table)
    elements.append(Spacer(1, 0.5*inch))
    
    # --- Gráficos ---
    elements.append(Paragraph("Equity Curve", subtitle_style))
    
    # Crear gráfico de equity curve
    if not equity_df.empty:
        plt.figure(figsize=(7, 3))
        plt.plot(equity_df.index, equity_df['equity'])
        plt.title('Equity Curve')
        plt.tight_layout()
        
        # Guardar gráfico en buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close()
        img_buffer.seek(0)
        
        # Añadir imagen al PDF
        equity_img = Image(img_buffer, width=6*inch, height=3*inch)
        elements.append(equity_img)
    else:
        elements.append(Paragraph("No equity data available", normal_style))
    
    elements.append(Spacer(1, 0.25*inch))
    
    # --- Tabla de operaciones ---
    elements.append(Paragraph("Trade Results", subtitle_style))
    
    if not trades_df.empty:
        # Limitar a primeras 20 operaciones para que no sea muy largo
        table_df = trades_df.head(20).copy()
        
        # Format the DataFrame for display
        for col in table_df.columns:
            if 'Time' in col:
                table_df[col] = table_df[col].dt.strftime('%Y-%m-%d %H:%M')
            elif 'Price' in col or 'PnL' in col:
                table_df[col] = table_df[col].apply(lambda x: f"${x:.2f}")
        
        # Create table data from DataFrame
        trade_data = [table_df.columns.tolist()]
        trade_data += table_df.values.tolist()
        
        # Calculate column widths
        col_widths = [1.2*inch] * len(trade_data[0])
        
        # Create the table
        trade_table = Table(trade_data, colWidths=col_widths)
        trade_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(trade_table)
        
        # Añadir nota si hay más operaciones
        if len(trades_df) > 20:
            elements.append(Paragraph(f"Note: Showing first 20 of {len(trades_df)} trades", normal_style))
    else:
        elements.append(Paragraph("No trades were executed during this period", normal_style))
    
    # Construir el PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- Función para exportar resultados del backtest ---
def export_backtest_results(
    symbol, interval, strategy, mode, leverage, maintenance_margin,
    start_date, end_date, initial_cash, commission, slippage,
    metrics, trades_df, equity_df, price_df, funding_df=None):
    """
    Exporta los resultados del backtest a PDF
    
    Args:
        Mismos parámetros que generate_backtest_pdf
        
    Returns:
        href para descarga del PDF
    """
    pdf_buffer = generate_backtest_pdf(
        symbol, interval, strategy, mode, leverage, maintenance_margin,
        start_date, end_date, initial_cash, commission, slippage,
        metrics, trades_df, equity_df, price_df, funding_df
    )
    
    # Convertir buffer a base64
    b64_pdf = base64.b64encode(pdf_buffer.read()).decode()
    
    # Crear enlace de descarga
    pdf_name = f"backtest_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.pdf"
    href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="{pdf_name}">Download PDF Report</a>'
    
    return href

# --- Interfaz de usuario para el backtesting unificado ---
def render_unified_backtest():
    """
    Renderiza la interfaz de usuario para el backtesting unificado
    """
    st.markdown(tooltip("🔄 Unified Backtesting",
                      "Prueba estrategias de trading en modo spot o con futuros apalancados. " + 
                      "Simula operaciones con datos históricos para evaluar rendimiento, " +
                      "incluyendo comisiones, slippage, funding rates y liquidaciones."),
               unsafe_allow_html=True)
    
    # Inicializar variables de estado en session_state si no existen
    if 'bt_mode' not in st.session_state:
        st.session_state.bt_mode = "spot"
    if 'bt_metrics' not in st.session_state:
        st.session_state.bt_metrics = {}
    if 'bt_trades_df' not in st.session_state:
        st.session_state.bt_trades_df = pd.DataFrame()
    if 'bt_eq_df' not in st.session_state:
        st.session_state.bt_eq_df = pd.DataFrame()
    if 'bt_price_df' not in st.session_state:
        st.session_state.bt_price_df = pd.DataFrame()
    if 'bt_funding_df' not in st.session_state:
        st.session_state.bt_funding_df = pd.DataFrame()
    if 'bt_last_run' not in st.session_state:
        st.session_state.bt_last_run = False
    if 'bt_batch_results' not in st.session_state:
        st.session_state.bt_batch_results = pd.DataFrame()
    if 'bt_symbols_list' not in st.session_state:
        st.session_state.bt_symbols_list = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    if 'bt_summary' not in st.session_state:
        st.session_state.bt_summary = {
            'symbol': 'BTCUSDT',
            'interval': '1h',
            'strategy': 'MA Crossover',
            'start_date': (datetime.utcnow().date() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'end_date': datetime.utcnow().date().strftime('%Y-%m-%d'),
            'initial_cash': 10000.0,
            'commission': 0.1,
            'slippage': 0.05,
            'mode': 'spot',
            'leverage': 3,
            'maintenance_margin': 0.5
        }
    
    # Función para cambiar el modo
    def change_mode():
        st.session_state.bt_mode = st.session_state.selected_mode
    
    # --- Header con selector de modo y botón de exportación ---
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        # Radio buttons para seleccionar modo
        st.radio(
            "Trading Mode",
            options=list(MODES.keys()),
            format_func=lambda x: MODES[x],
            index=0 if st.session_state.bt_mode == "spot" else 1,
            key="selected_mode",
            on_change=change_mode,
            horizontal=True
        )
    
    with header_col2:
        # Botón para exportar a PDF (visible si hay resultados)
        if st.session_state.bt_last_run:
            summary = st.session_state.bt_summary
            
            # Crear botón de exportación
            export_href = export_backtest_results(
                summary['symbol'], summary['interval'], summary['strategy'],
                summary['mode'], summary.get('leverage', 1), summary.get('maintenance_margin', 0.005),
                summary['start_date'], summary['end_date'], summary['initial_cash'],
                summary['commission']/100, summary['slippage']/100,
                st.session_state.bt_metrics, st.session_state.bt_trades_df,
                st.session_state.bt_eq_df, st.session_state.bt_price_df,
                st.session_state.bt_funding_df
            )
            
            st.markdown(export_href, unsafe_allow_html=True)
    
    # Determine default date range: last 30 days
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=30)
    
    # --- Formulario de configuración en dos columnas ---
    with st.expander("Backtest Settings", expanded=not st.session_state.bt_last_run):
        col1, col2 = st.columns(2)
        
        with col1:
            # Columna A: Símbolo, Timeframe, Fecha inicio/fin, Modo, Leverage, Maintenance Margin
            symbol = st.text_input('Symbol', value='BTCUSDT', key='bt_symbol').strip().upper()
            
            # Opción de backtesting por lotes
            batch_mode = st.checkbox("Batch Backtest", value=False, key="bt_batch_mode",
                                    help="Run backtest on multiple symbols at once")
            
            if batch_mode:
                # Campo para lista de símbolos
                symbols_str = st.text_area(
                    "Symbols List (comma separated)",
                    value=", ".join(st.session_state.bt_symbols_list),
                    key="bt_symbols_str"
                )
                symbols_list = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
                st.session_state.bt_symbols_list = symbols_list
            
            interval = st.selectbox('Timeframe', 
                                  options=['1m','5m','15m','1h','4h','1d'], 
                                  key='bt_interval')
            
            strategy_name = st.selectbox('Strategy', 
                                       options=['MA Crossover','Bollinger Breakout','RSI Reversion',
                                               'MACD Momentum','SR Breakout'],
                                       key='bt_strategy',
                                       help="MA Crossover: Cruces de medias móviles. Bollinger: Ruptura de bandas. RSI: Sobrecompra/sobreventa. MACD: Momentum. SR: Soporte/Resistencia.")
            
            # Fechas en la misma línea
            date_cols = st.columns(2)
            with date_cols[0]:
                start_date = st.date_input('Start Date', value=default_start, key='bt_start')
            with date_cols[1]:
                end_date = st.date_input('End Date', value=today, key='bt_end')
            
            # Parámetros específicos del modo futures
            if st.session_state.bt_mode == "futures":
                leverage = st.slider('Leverage', min_value=1, max_value=100, value=3, step=1, 
                                   key='bt_leverage')
                maintenance_margin = st.slider('Maintenance Margin (%)', 
                                             min_value=0.1, max_value=5.0, value=0.5, step=0.1, 
                                             key='bt_maintenance')
            else:
                # Valores default si estamos en spot
                leverage = 1
                maintenance_margin = 0.005
                
        with col2:
            # Columna B: Capital inicial, Comisión, Slippage, Botón para ejecutar
            initial_cash = st.number_input('Initial Capital (USD)', min_value=100.0, 
                                         value=10000.0, step=100.0, key='bt_cash')
            
            # Comisión y slippage en la misma línea
            fee_cols = st.columns(2)
            with fee_cols[0]:
                commission = st.number_input('Commission (%)', min_value=0.0, max_value=1.0, 
                                           value=0.1, step=0.01, key='bt_commission')
            with fee_cols[1]:
                slippage = st.number_input('Slippage (%)', min_value=0.0, max_value=1.0, 
                                         value=0.05, step=0.01, key='bt_slippage')
            
            # Espacio para alinear el botón en la parte inferior
            st.write("")
            st.write("")
            st.write("")
            
            # Botón según el modo
            if batch_mode:
                run = st.button('Run Batch Backtest', key='bt_run_batch',
                              help="Run backtest on all symbols in the list")
            else:
                run = st.button('Run Backtest', key='bt_run_single')
    
    # --- Ejecutar backtest cuando se presiona el botón ---
    if run:
        # Mapear nombres de estrategia a funciones
        strategy_map = {
            'MA Crossover': ma_crossover,
            'Bollinger Breakout': bollinger_breakout,
            'RSI Reversion': rsi_reversion,
            'MACD Momentum': macd_momentum,
            'SR Breakout': sr_breakout
        }
        strategy_fn = strategy_map[strategy_name]
        mode = st.session_state.bt_mode
        
        # Ejecutar backtesting (single o batch)
        with st.spinner(f"Running {'batch' if batch_mode else ''} backtest..."):
            try:
                if batch_mode:
                    # Backtesting por lotes
                    results_df = run_batch_backtest(
                        symbols_list, interval, strategy_fn, initial_cash,
                        commission/100, slippage/100, mode, leverage, maintenance_margin/100,
                        start_date, end_date
                    )
                    st.session_state.bt_batch_results = results_df
                else:
                    # Backtesting individual
                    metrics, trades_df, equity_df, price_df, funding_df = run_unified_backtest(
                        symbol, interval, strategy_fn, initial_cash,
                        commission/100, slippage/100, mode, leverage, maintenance_margin/100,
                        start_date, end_date
                    )
                    
                    # Guardar resultados en session_state
                    st.session_state.bt_metrics = metrics
                    st.session_state.bt_trades_df = trades_df
                    st.session_state.bt_eq_df = equity_df
                    st.session_state.bt_price_df = price_df
                    st.session_state.bt_funding_df = funding_df
                
                # Marcar que se ha ejecutado un backtest
                st.session_state.bt_last_run = True
                
                # Actualizar el resumen
                st.session_state.bt_summary = {
                    'symbol': symbol if not batch_mode else ", ".join(symbols_list[:3]) + ("..." if len(symbols_list) > 3 else ""),
                    'interval': interval,
                    'strategy': strategy_name,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'initial_cash': initial_cash,
                    'commission': commission,
                    'slippage': slippage,
                    'mode': mode,
                    'leverage': leverage,
                    'maintenance_margin': maintenance_margin
                }
            except Exception as e:
                st.error(f"Error executing backtest: {str(e)}")
    
    # --- Mostrar resultados del batch si es necesario ---
    if batch_mode and not st.session_state.bt_batch_results.empty:
        st.subheader("Batch Results")
        
        # Mostrar tabla de resultados
        st.dataframe(st.session_state.bt_batch_results, use_container_width=True)
        
        # Gráfico de comparación (top 5 por rentabilidad)
        if len(st.session_state.bt_batch_results) > 1:
            top_symbols = st.session_state.bt_batch_results.head(min(5, len(st.session_state.bt_batch_results)))
            
            fig = px.bar(
                top_symbols, 
                x='Symbol', 
                y='Return (%)',
                title="Top Performing Symbols",
                color='Return (%)',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # --- Mostrar resultados del backtesting individual ---
    elif not batch_mode and st.session_state.bt_last_run:
        # Actualizar paleta de colores
        current_palette = config.PALETTE
        
        # Mostrar resumen del backtest
        summary = st.session_state.bt_summary
        summary_html = format_backtest_summary(summary, current_palette)
        st.markdown(summary_html, unsafe_allow_html=True)
        
        # Resultados en dos paneles
        metrics_col, trades_col = st.columns(2)
        
        # Panel izquierdo: Métricas y Equity Curve
        with metrics_col:
            st.subheader("Performance Metrics")
            
            # Métricas en cards
            metric_cols = st.columns(3)
            for i, (name, val) in enumerate(st.session_state.bt_metrics.items()):
                if name in ['Net Profit', 'Max Drawdown', 'Expectancy', 'Funding Impact']:
                    # Format as currency with thousands separator
                    if isinstance(val, (int, float)):
                        # Verificar si el valor es realista (máximo 100 veces el capital inicial)
                        max_realistic_value = st.session_state.bt_summary.get('initial_cash', 10000) * 100
                        if abs(val) > max_realistic_value:
                            st.warning(f"¡Valor sospechosamente alto detectado en {name}: ${val:,.2f}!")
                            # Usar un valor más realista (limitado)
                            val = max_realistic_value if val > 0 else -max_realistic_value
                        metric_cols[i % 3].metric(name, f"${val:,.2f}")
                    else:
                        metric_cols[i % 3].metric(name, f"${val}")
                elif "%" in name:
                    # Format as percentage with 2 decimal places
                    if isinstance(val, (int, float)):
                        metric_cols[i % 3].metric(name, f"{val:.2f}%")
                    else:
                        metric_cols[i % 3].metric(name, f"{val}%")
                elif "Ratio" in name or isinstance(val, float):
                    # Format numbers with 2 decimal places
                    if isinstance(val, (int, float)):
                        metric_cols[i % 3].metric(name, f"{val:,.2f}")
                    else:
                        metric_cols[i % 3].metric(name, val)
                else:
                    # Leave integers as is
                    metric_cols[i % 3].metric(name, f"{val:,}" if isinstance(val, int) else val)
            
            # Equity Curve interactiva
            st.subheader("Equity Curve")
            
            if not st.session_state.bt_eq_df.empty:
                df_plot = st.session_state.bt_eq_df.reset_index()
                
                # Asegurar que tenemos la columna timestamp
                if 'timestamp' not in df_plot.columns and 'index' in df_plot.columns:
                    df_plot = df_plot.rename(columns={'index': 'timestamp'})
                
                # Crear gráfico base
                fig_eq = go.Figure()
                fig_eq.add_trace(
                    go.Scatter(
                        x=df_plot['timestamp'], 
                        y=df_plot['equity'],
                        mode='lines', 
                        line=dict(color=current_palette['equity'], width=2),
                        name='Equity'
                    )
                )
                
                # Añadir funding payments como markers si es futures
                if st.session_state.bt_mode == "futures" and not st.session_state.bt_funding_df.empty:
                    funding_df = st.session_state.bt_funding_df
                    if 'timestamp' in funding_df.columns:
                        # Merge with equity curve to get corresponding equity values
                        funding_with_equity = pd.merge_asof(
                            funding_df.sort_values('timestamp'),
                            df_plot[['timestamp', 'equity']].sort_values('timestamp'),
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
                
                # Configuración de layout
                fig_eq.update_layout(
                    template=current_palette['template'],
                    height=350, 
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
        
        # Panel derecho: Gráfico de precios y tabla de trades
        with trades_col:
            st.subheader("Price Action & Trades")
            
            # Crear figura base con velas
            fig_price = go.Figure()

            # Separar datos para velas alcistas (close >= open) y bajistas (close < open)
            df_bull = st.session_state.bt_price_df[st.session_state.bt_price_df['Close'] >= st.session_state.bt_price_df['Open']].copy()
            df_bear = st.session_state.bt_price_df[st.session_state.bt_price_df['Close'] < st.session_state.bt_price_df['Open']].copy()

            # Color principal para todas las velas
            candle_color = current_palette['equity']

            # Añadir velas alcistas (rellenas)
            if not df_bull.empty:
                fig_price.add_trace(
                    go.Candlestick(
                        x=df_bull.index,
                        open=df_bull['Open'], 
                        high=df_bull['High'],
                        low=df_bull['Low'], 
                        close=df_bull['Close'],
                        name='Bullish',
                        increasing=dict(line=dict(color=candle_color), fillcolor=candle_color),
                        decreasing=dict(line=dict(color=candle_color), fillcolor=candle_color)
                    )
                )

            # Añadir velas bajistas (vacías)
            if not df_bear.empty:
                fig_price.add_trace(
                    go.Candlestick(
                        x=df_bear.index,
                        open=df_bear['Open'], 
                        high=df_bear['High'],
                        low=df_bear['Low'], 
                        close=df_bear['Close'],
                        name='Bearish',
                        increasing=dict(line=dict(color=candle_color), fillcolor=candle_color),
                        decreasing=dict(line=dict(color=candle_color), fillcolor='rgba(0,0,0,0)')
                    )
                )

            # Añadir trades si hay
            if not st.session_state.bt_trades_df.empty:
                # En modo futures, separamos long y short
                if st.session_state.bt_mode == "futures" and 'Position' in st.session_state.bt_trades_df.columns:
                    # Long trades
                    long_trades = st.session_state.bt_trades_df[
                        st.session_state.bt_trades_df['Position'] == 'Long'
                    ]
                    if not long_trades.empty:
                        # Long entries
                        fig_price.add_trace(
                            go.Scatter(
                                x=long_trades['Entry Time'], 
                                y=long_trades['Entry Price'],
                                mode='markers', 
                                marker=dict(color=current_palette['entries'], symbol='triangle-up', size=10),
                                name='Long Entries'
                            )
                        )
                        # Long exits
                        fig_price.add_trace(
                            go.Scatter(
                                x=long_trades['Exit Time'], 
                                y=long_trades['Exit Price'],
                                mode='markers', 
                                marker=dict(color=current_palette['exits'], symbol='triangle-down', size=10),
                                name='Long Exits'
                            )
                        )
                    
                    # Short trades
                    short_trades = st.session_state.bt_trades_df[
                        st.session_state.bt_trades_df['Position'] == 'Short'
                    ]
                    if not short_trades.empty:
                        # Short entries
                        fig_price.add_trace(
                            go.Scatter(
                                x=short_trades['Entry Time'], 
                                y=short_trades['Entry Price'],
                                mode='markers', 
                                marker=dict(color='red', symbol='triangle-down', size=10),
                                name='Short Entries'
                            )
                        )
                        # Short exits
                        fig_price.add_trace(
                            go.Scatter(
                                x=short_trades['Exit Time'], 
                                y=short_trades['Exit Price'],
                                mode='markers', 
                                marker=dict(color='green', symbol='triangle-up', size=10),
                                name='Short Exits'
                            )
                        )
                    
                    # Liquidaciones
                    if 'Liquidated' in st.session_state.bt_trades_df.columns:
                        liquidations = st.session_state.bt_trades_df[
                            st.session_state.bt_trades_df['Liquidated'] == True
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
                else:
                    # Trades normales (no futures o sin posición especificada)
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
            
            # Crear un segundo eje y para las líneas de liquidación
            liquidation_fig = None
            
            # Añadir niveles de liquidación si estamos en futures y tenemos los datos
            if st.session_state.bt_mode == "futures" and not st.session_state.bt_eq_df.empty:
                if 'liquidation_price' in st.session_state.bt_eq_df.columns:
                    liq_df = st.session_state.bt_eq_df.reset_index()
                    liq_df = liq_df[liq_df['liquidation_price'].notnull()]
                    
                    if not liq_df.empty:
                        # Mostrar opciones para ver liquidación
                        show_liquidation = st.checkbox("Mostrar precios de liquidación", value=False, key="show_liquidation")
                        
                        if show_liquidation:
                            # Group by continuous segments
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
            
            # Añadir líneas de funding rate si es futures y tenemos los datos
            if st.session_state.bt_mode == "futures" and 'FundingRate' in st.session_state.bt_price_df.columns:
                # Mostrar opciones para ver funding rate
                show_funding = st.checkbox("Mostrar funding rates", value=False, key="show_funding")
                
                if show_funding:
                    # Crear un eje secundario para el funding rate
                    funding_df = st.session_state.bt_price_df.reset_index()[['Open time', 'FundingRate']]
                    funding_df = funding_df.rename(columns={'Open time': 'timestamp'})
                    
                    # Añadir la traza de funding rate
                    fig_price.add_trace(
                        go.Scatter(
                            x=funding_df['timestamp'],
                            y=funding_df['FundingRate'] * 100,  # Convertir a porcentaje para mejor visualización
                            mode='lines',
                            name='Funding Rate (%)',
                            line=dict(color='orange', width=1),
                            yaxis='y2'  # Usar el eje secundario
                        )
                    )
                    
                    # Configurar el eje secundario
                    fig_price.update_layout(
                        yaxis2=dict(
                            title='Funding Rate (%)',
                            overlaying='y',
                            side='right',
                            showgrid=False
                        )
                    )
            
            # Actualizar layout
            fig_price.update_layout(
                template=current_palette['template'],
                height=350, 
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
            
            # Quitar rangeslider para ahorrar espacio
            fig_price.update_layout(xaxis_rangeslider_visible=False)
            
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Tabla de trades
            st.subheader("Trade Results")
            
            if not st.session_state.bt_trades_df.empty:
                # Si estamos en modo futures, darle formato a liquidaciones
                if st.session_state.bt_mode == "futures" and 'Liquidated' in st.session_state.bt_trades_df.columns:
                    st.dataframe(st.session_state.bt_trades_df.style.apply(
                        lambda row: ['background-color: rgba(255,0,0,0.2)' if row['Liquidated'] else '' 
                                   for _ in row], axis=1
                    ), use_container_width=True)
                else:
                    st.dataframe(st.session_state.bt_trades_df, use_container_width=True)
            else:
                st.info('No trades executed in this period.')
                
        # Sección opcional para analytics avanzados
        with st.expander("Advanced Analytics", expanded=False):
            if not st.session_state.bt_trades_df.empty:
                # Crear pestañas para diferentes análisis
                tabs = st.tabs(["Trade Statistics", "Monthly Returns", "Daily Returns"])
                
                with tabs[0]:
                    st.subheader("Trade Statistics")
                    
                    # Profit factor
                    if len(st.session_state.bt_trades_df) > 0:
                        winning_trades = st.session_state.bt_trades_df[st.session_state.bt_trades_df['PnL'] > 0]
                        losing_trades = st.session_state.bt_trades_df[st.session_state.bt_trades_df['PnL'] <= 0]
                        
                        total_wins = winning_trades['PnL'].sum() if not winning_trades.empty else 0
                        total_losses = abs(losing_trades['PnL'].sum()) if not losing_trades.empty else 0
                        
                        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Profit Factor", f"{profit_factor:.2f}")
                        col2.metric("Avg. Trade", f"${st.session_state.bt_trades_df['PnL'].mean():.2f}")
                        col3.metric("Median Trade", f"${st.session_state.bt_trades_df['PnL'].median():.2f}")
                        
                        # Histograma de PnL
                        fig = px.histogram(
                            st.session_state.bt_trades_df, 
                            x='PnL',
                            nbins=20,
                            color_discrete_sequence=[current_palette['equity']]
                        )
                        fig.update_layout(
                            title="Distribution of Trade Results",
                            xaxis_title="Profit/Loss ($)",
                            yaxis_title="Number of Trades"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with tabs[1]:
                    st.subheader("Monthly Returns")
                    
                    # Convertir las columnas de tiempo a datetime
                    if 'Exit Time' in st.session_state.bt_trades_df.columns:
                        # Asegurar que Exit Time es datetime
                        if not pd.api.types.is_datetime64_any_dtype(st.session_state.bt_trades_df['Exit Time']):
                            st.session_state.bt_trades_df['Exit Time'] = pd.to_datetime(st.session_state.bt_trades_df['Exit Time'])
                        
                        # Extraer mes y año
                        st.session_state.bt_trades_df['Month'] = st.session_state.bt_trades_df['Exit Time'].dt.to_period('M')
                        
                        # Agrupar por mes
                        monthly_returns = st.session_state.bt_trades_df.groupby('Month')['PnL'].sum().reset_index()
                        monthly_returns['Month'] = monthly_returns['Month'].astype(str)
                        
                        # Graficar retornos mensuales
                        fig = px.bar(
                            monthly_returns, 
                            x='Month', 
                            y='PnL',
                            color='PnL',
                            color_continuous_scale=px.colors.diverging.RdYlGn,
                            color_continuous_midpoint=0
                        )
                        fig.update_layout(
                            title="Monthly Returns",
                            xaxis_title="Month",
                            yaxis_title="Profit/Loss ($)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with tabs[2]:
                    st.subheader("Daily Equity")
                    
                    if not st.session_state.bt_eq_df.empty:
                        # Resample a diario
                        if isinstance(st.session_state.bt_eq_df.index, pd.DatetimeIndex):
                            daily_equity = st.session_state.bt_eq_df['equity'].resample('D').last().fillna(method='ffill')
                            daily_returns = daily_equity.pct_change().dropna()
                            
                            # Crear DataFrame para plotly
                            daily_df = pd.DataFrame({
                                'date': daily_returns.index,
                                'return': daily_returns * 100,  # Convertir a porcentaje
                                'positive': daily_returns > 0
                            })
                            
                            # Gráfico de retornos diarios (arreglado para evitar error de listas en x e y)
                            fig = px.bar(
                                daily_df,
                                x='date', 
                                y='return',
                                color='positive',
                                color_discrete_map={True: 'green', False: 'red'},
                            )
                            fig.update_layout(
                                title="Daily Returns (%)",
                                xaxis_title="Date",
                                yaxis_title="Return (%)",
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("Daily analysis requires datetime index data")
                    else:
                        st.warning("No equity data available for daily analysis")

# Ejecutar en modo standalone para testing
if __name__ == "__main__":
    st.set_page_config(page_title="Unified Backtesting", layout="wide")
    render_unified_backtest()