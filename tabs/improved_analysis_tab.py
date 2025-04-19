"""
Centro de Mando Diario para Análisis de Criptomonedas.

Este módulo implementa un dashboard avanzado que incluye:
- Resumen diario automático con KPIs macro
- Panel de watchlist con múltiples monedas
- Análisis técnico con semáforos de indicadores 
- Análisis de sentimiento y eventos
- Plan de trading personalizable
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import datetime
import pytz
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit.components.v1 import html as components_html
import os
from pathlib import Path

# Importamos funciones del módulo de análisis existente
from tabs.analysis_tab import (
    detect_support_resistance, detect_pivots, 
    indicator_recommendation, get_funding_rate, 
    get_open_interest, plot_open_interest,
    get_orderflow_delta, plot_delta, get_data_perp,
    PALETTE
)

# Importamos utilidades
from utils.html_utils import tooltip
from utils.data_utils import get_data, calculate_indicators

# Constantes
DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT']
TIMEFRAMES = {
    '1h': {'label': '1 Hora', 'weight': 1},
    '4h': {'label': '4 Horas', 'weight': 2},
    '1d': {'label': '1 Día', 'weight': 3}
}
INDICATORS = {
    'SMA50': {'name': 'SMA 50/100', 'weight': 1},
    'MACD': {'name': 'MACD', 'weight': 1},
    'RSI': {'name': 'RSI', 'weight': 1},
    'MFI': {'name': 'MFI', 'weight': 0.5}
}

# Carpeta para guardar planes de trading
PLANS_FOLDER = Path('data/trading_plans')
PLANS_FOLDER.mkdir(parents=True, exist_ok=True)

def get_current_session():
    """
    Determina la sesión actual de mercado basada en la hora UTC.
    
    Returns:
        str: 'Asia', 'Europa' o 'América'
    """
    utc_now = datetime.datetime.now(pytz.utc)
    hour = utc_now.hour
    
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 16:
        return "Europa"
    else:
        return "América"

def get_market_summary(symbol):
    """
    Obtiene un resumen de las principales métricas del mercado para un símbolo
    
    Args:
        symbol: Símbolo de la criptomoneda
        
    Returns:
        dict: Diccionario con métricas clave
    """
    try:
        # Obtenemos datos de funding rate
        funding_df = get_funding_rate(symbol)
        current_funding = funding_df['fundingRate'].iloc[-1] if not funding_df.empty else 0
        
        # Obtenemos datos de Open Interest
        oi_df = get_open_interest(symbol, '1h')
        current_oi = oi_df['openInterest'].iloc[-1] if not oi_df.empty else 0
        oi_change = ((current_oi / oi_df['openInterest'].iloc[-2]) - 1) * 100 if not oi_df.empty and len(oi_df) > 1 else 0
        
        # Obtenemos datos de Delta (presión compradora vs vendedora)
        delta_df = get_orderflow_delta(symbol, '1h')
        current_delta = delta_df['delta'].iloc[-1] if not delta_df.empty else 0
        delta_percent = delta_df['delta_percent'].iloc[-1] if not delta_df.empty else 0
        
        # Precio actual
        price_df = get_data_perp(symbol, '1h')
        current_price = price_df['Close'].iloc[-1] if not price_df.empty else 0
        price_change_24h = ((current_price / price_df['Close'].iloc[-24]) - 1) * 100 if not price_df.empty and len(price_df) >= 24 else 0
        
        # Retornar resumen
        return {
            'funding_rate': current_funding,
            'open_interest': current_oi,
            'oi_change': oi_change,
            'delta': current_delta,
            'delta_percent': delta_percent,
            'price': current_price,
            'price_change_24h': price_change_24h
        }
    except Exception as e:
        st.error(f"Error al obtener resumen del mercado: {str(e)}")
        return {
            'funding_rate': 0,
            'open_interest': 0,
            'oi_change': 0,
            'delta': 0, 
            'delta_percent': 0,
            'price': 0,
            'price_change_24h': 0
        }

def render_kpi_row(symbol):
    """
    Renderiza la fila de KPIs con métricas clave de trading
    
    Args:
        symbol: Símbolo de la criptomoneda
    """
    # Obtenemos el resumen del mercado
    summary = get_market_summary(symbol)
    
    # Creamos una fila con 4 columnas para los KPIs
    cols = st.columns(4)
    
    # 1. Funding Rate
    with cols[0]:
        funding_color = PALETTE['green'] if summary['funding_rate'] >= 0 else PALETTE['red']
        funding_text = "Largos pagan" if summary['funding_rate'] >= 0 else "Cortos pagan"
        st.markdown(f"""
        <div style="border:1px solid {funding_color}; border-radius:5px; padding:10px; text-align:center;">
            <h4>Funding Rate</h4>
            <p style="font-size:24px; color:{funding_color};">{summary['funding_rate']*100:.4f}%</p>
            <p>{funding_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 2. Open Interest
    with cols[1]:
        oi_color = PALETTE['green'] if summary['oi_change'] >= 0 else PALETTE['red']
        oi_formatted = f"{summary['open_interest']:,.0f}".replace(',', ' ')
        st.markdown(f"""
        <div style="border:1px solid {oi_color}; border-radius:5px; padding:10px; text-align:center;">
            <h4>Open Interest</h4>
            <p style="font-size:20px; color:{oi_color}">{oi_formatted}</p>
            <p style="color:{oi_color}">Cambio: {summary['oi_change']:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 3. Order Flow Delta
    with cols[2]:
        delta_color = PALETTE['green'] if summary['delta'] >= 0 else PALETTE['red']
        delta_text = "Compras dominan" if summary['delta'] >= 0 else "Ventas dominan"
        st.markdown(f"""
        <div style="border:1px solid {delta_color}; border-radius:5px; padding:10px; text-align:center;">
            <h4>Order Flow</h4>
            <p style="font-size:20px; color:{delta_color}">{summary['delta_percent']:.2f}%</p>
            <p>{delta_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 4. Precio y cambio 24h
    with cols[3]:
        price_color = PALETTE['green'] if summary['price_change_24h'] >= 0 else PALETTE['red']
        st.markdown(f"""
        <div style="border:1px solid {price_color}; border-radius:5px; padding:10px; text-align:center;">
            <h4>Precio</h4>
            <p style="font-size:20px;">${summary['price']:,.2f}</p>
            <p style="color:{price_color}">{summary['price_change_24h']:.2f}% (24h)</p>
        </div>
        """, unsafe_allow_html=True)

def calculate_technical_score(df, timeframe_weight=1):
    """
    Calcula un score técnico basado en diferentes indicadores
    
    Args:
        df: DataFrame con indicadores calculados
        timeframe_weight: Peso del timeframe para el score global
        
    Returns:
        float: Score técnico normalizado (-100 a +100)
    """
    if df.empty or len(df) < 2:
        return 0
    
    last = df.iloc[-1]
    score = 0
    
    # SMA 50/100
    if last['SMA50'] > last['SMA100']:
        score += 25 * INDICATORS['SMA50']['weight']  # Bullish
    else:
        score -= 25 * INDICATORS['SMA50']['weight']  # Bearish
    
    # MACD
    if last['MACD'] > last['Signal']:
        score += 25 * INDICATORS['MACD']['weight']  # Bullish
    else:
        score -= 25 * INDICATORS['MACD']['weight']  # Bearish
    
    # RSI
    if last['RSI'] <= 30:
        score += 25 * INDICATORS['RSI']['weight']  # Oversold (potencial alcista)
    elif last['RSI'] >= 70:
        score -= 25 * INDICATORS['RSI']['weight']  # Overbought (potencial bajista)
    else:
        # Neutral pero con sesgo
        rsi_score = ((last['RSI'] - 50) / 20) * 25 * INDICATORS['RSI']['weight']
        score += rsi_score
    
    # MFI
    if last['MFI'] <= 20:
        score += 25 * INDICATORS['MFI']['weight']  # Oversold (potencial alcista)
    elif last['MFI'] >= 80:
        score -= 25 * INDICATORS['MFI']['weight']  # Overbought (potencial bajista)
    else:
        # Neutral pero con sesgo
        mfi_score = ((last['MFI'] - 50) / 30) * 25 * INDICATORS['MFI']['weight']
        score += mfi_score
    
    # Aplicar peso del timeframe
    return score * timeframe_weight

def render_technical_indicators(data_dict):
    """
    Renderiza los indicadores técnicos en formato de semáforo
    
    Args:
        data_dict: Diccionario con DataFrames por timeframe
    """
    st.markdown("### Semáforo de Indicadores")
    
    # Preparamos la tabla de semáforos
    indicators_df = pd.DataFrame(columns=['Indicador'] + list(TIMEFRAMES.keys()) + ['Score'])
    
    # Para cada indicador, calculamos su estado en cada timeframe
    for ind_key, ind_info in INDICATORS.items():
        row = {'Indicador': ind_info['name']}
        ind_score = 0
        
        for tf_key, tf_info in TIMEFRAMES.items():
            df = data_dict.get(tf_key, pd.DataFrame())
            if not df.empty:
                last = df.iloc[-1]
                
                # Determinamos el estado del indicador
                if ind_key == 'SMA50':
                    if last['SMA50'] > last['SMA100']:
                        status = "🟢"  # Bullish
                        ind_score += 1 * tf_info['weight']
                    else:
                        status = "🔴"  # Bearish
                        ind_score -= 1 * tf_info['weight']
                
                elif ind_key == 'MACD':
                    if last['MACD'] > last['Signal']:
                        status = "🟢"  # Bullish
                        ind_score += 1 * tf_info['weight']
                    else:
                        status = "🔴"  # Bearish
                        ind_score -= 1 * tf_info['weight']
                
                elif ind_key == 'RSI':
                    if last['RSI'] <= 30:
                        status = "🟢"  # Oversold (potencial alcista)
                        ind_score += 1 * tf_info['weight']
                    elif last['RSI'] >= 70:
                        status = "🔴"  # Overbought (potencial bajista)
                        ind_score -= 1 * tf_info['weight']
                    else:
                        status = "🟡"  # Neutral
                
                elif ind_key == 'MFI':
                    if last['MFI'] <= 20:
                        status = "🟢"  # Oversold (potencial alcista)
                        ind_score += 1 * tf_info['weight']
                    elif last['MFI'] >= 80:
                        status = "🔴"  # Overbought (potencial bajista)
                        ind_score -= 1 * tf_info['weight']
                    else:
                        status = "🟡"  # Neutral
                
                row[tf_key] = status
            else:
                row[tf_key] = "⚪"  # No hay datos
        
        # Calculamos el score normalizado para este indicador
        max_possible_score = sum([tf_info['weight'] for tf_key, tf_info in TIMEFRAMES.items()])
        normalized_score = (ind_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        row['Score'] = f"{normalized_score:.1f}"
        
        # Añadimos la fila a la tabla
        indicators_df = pd.concat([indicators_df, pd.DataFrame([row])], ignore_index=True)
    
    # Calculamos el score global
    global_score = indicators_df['Score'].astype(float).mean()
    
    # Mostramos el score global con un color indicativo
    score_color = PALETTE['green'] if global_score > 0 else PALETTE['red'] if global_score < 0 else PALETTE['neutral']
    
    st.markdown(f"""
    <div style="border:1px solid {score_color}; border-radius:5px; padding:10px; text-align:center; margin-bottom:15px;">
        <h3>Score Técnico Global</h3>
        <p style="font-size:36px; color:{score_color};">{global_score:.1f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostramos la tabla de semáforos
    st.dataframe(indicators_df, use_container_width=True)

def get_watchlist():
    """
    Obtiene la watchlist guardada o crea una por defecto
    
    Returns:
        list: Lista de símbolos en la watchlist
    """
    if 'watchlist' not in st.session_state:
        # Cargar watchlist desde un archivo si existe
        watchlist_path = Path('data/watchlist.json')
        if watchlist_path.exists():
            try:
                with open(watchlist_path, 'r') as f:
                    st.session_state.watchlist = json.load(f)
            except:
                st.session_state.watchlist = DEFAULT_SYMBOLS
        else:
            st.session_state.watchlist = DEFAULT_SYMBOLS
    
    return st.session_state.watchlist

def save_watchlist(symbols):
    """
    Guarda la watchlist actualizada
    
    Args:
        symbols: Lista de símbolos a guardar
    """
    # Crear directorio si no existe
    Path('data').mkdir(exist_ok=True)
    
    # Guardamos la watchlist
    with open('data/watchlist.json', 'w') as f:
        json.dump(symbols, f)
    
    # Actualizamos la session state
    st.session_state.watchlist = symbols

def render_mini_chart(symbol):
    """
    Renderiza un mini-chart para un símbolo en la watchlist
    
    Args:
        symbol: Símbolo de la criptomoneda
    """
    # Obtenemos datos diarios
    df = get_data(symbol, '1d')
    if df.empty:
        st.warning(f"No hay datos disponibles para {symbol}")
        return
    
    # Calculamos indicadores
    df = calculate_indicators(df)
    
    # Detectamos niveles de soporte/resistencia
    res, sup = detect_support_resistance(df)
    
    # Creamos gráfico
    fig = go.Figure()
    
    # Añadimos gráfico de velas
    fig.add_trace(go.Candlestick(
        x=df.index[-30:],  # Últimos 30 días
        open=df['Open'][-30:],
        high=df['High'][-30:],
        low=df['Low'][-30:],
        close=df['Close'][-30:],
        name='Price'
    ))
    
    # Añadimos SMA50
    fig.add_trace(go.Scatter(
        x=df.index[-30:],
        y=df['SMA50'][-30:],
        line=dict(color='rgba(255, 213, 79, 0.7)', width=1.5),
        name='SMA 50'
    ))
    
    # Añadimos niveles de soporte/resistencia
    for level in res:
        fig.add_shape(
            type="line",
            x0=df.index[-30],
            y0=level,
            x1=df.index[-1],
            y1=level,
            line=dict(color="rgba(239, 83, 80, 0.7)", width=1, dash="dash")
        )
    
    for level in sup:
        fig.add_shape(
            type="line",
            x0=df.index[-30],
            y0=level,
            x1=df.index[-1],
            y1=level,
            line=dict(color="rgba(38, 166, 154, 0.7)", width=1, dash="dash")
        )
    
    # Configuración del layout
    fig.update_layout(
        title=symbol,
        height=200,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        xaxis_rangeslider_visible=False,
        template=PALETTE['template']
    )
    
    # Mostramos el gráfico
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostramos algunos datos clave
    last = df.iloc[-1]
    price_change = ((last['Close'] / df.iloc[-2]['Close']) - 1) * 100 if len(df) > 1 else 0
    price_color = PALETTE['green'] if price_change >= 0 else PALETTE['red']
    
    cols = st.columns(3)
    with cols[0]:
        st.metric("Precio", f"${last['Close']:.2f}", f"{price_change:.2f}%")
    with cols[1]:
        st.metric("RSI", f"{last['RSI']:.1f}", "")
    with cols[2]:
        signal = "Compra" if last['MACD'] > last['Signal'] else "Venta"
        signal_color = PALETTE['green'] if signal == "Compra" else PALETTE['red']
        st.markdown(f"<p>Señal MACD: <span style='color:{signal_color}'>{signal}</span></p>", unsafe_allow_html=True)

def get_crypto_news(symbol):
    """
    Obtiene noticias recientes sobre una criptomoneda
    
    Args:
        symbol: Símbolo de la criptomoneda
        
    Returns:
        list: Lista de noticias
    """
    # Eliminar el 'USDT' del símbolo para buscar la moneda
    coin = symbol.replace('USDT', '')
    
    # Esta es una función simulada. En un entorno real, conectarías con una API de noticias
    simulated_news = [
        {
            "title": f"Análisis técnico de {coin}: ¿Rumbo a nuevos máximos?",
            "description": f"Los analistas predicen que {coin} podría estar preparando un movimiento alcista significativo en los próximos días.",
            "source": "CryptoAnalytics",
            "date": datetime.datetime.now().strftime("%Y-%m-%d")
        },
        {
            "title": f"Instituciones aumentan exposición a {coin}",
            "description": f"Grandes fondos de inversión han aumentado su posición en {coin} durante la última semana.",
            "source": "CoinDesk",
            "date": (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        },
        {
            "title": f"Nuevo desarrollo en el ecosistema de {coin}",
            "description": f"El equipo de desarrollo de {coin} ha anunciado mejoras en su protocolo que podrían impulsar la adopción.",
            "source": "CryptoBriefing",
            "date": (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d")
        }
    ]
    
    return simulated_news

def get_economic_calendar():
    """
    Obtiene eventos económicos relevantes para el mercado crypto
    
    Returns:
        list: Lista de eventos
    """
    # Esta es una función simulada. En un entorno real, conectarías con una API de calendario económico
    today = datetime.datetime.now()
    
    simulated_events = [
        {
            "date": today.strftime("%Y-%m-%d"),
            "time": "14:30 UTC",
            "event": "Datos de inflación (CPI) de EE.UU.",
            "impact": "Alto",
            "forecast": "3.2%"
        },
        {
            "date": (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
            "time": "18:00 UTC",
            "event": "Reunión de la FED - Decisión de tipos",
            "impact": "Alto",
            "forecast": "Sin cambios"
        },
        {
            "date": (today + datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
            "time": "12:00 UTC",
            "event": "Datos de empleo (Non-Farm Payrolls)",
            "impact": "Medio",
            "forecast": "+180K"
        }
    ]
    
    return simulated_events

def load_trading_plan(symbol):
    """
    Carga el plan de trading guardado para un símbolo
    
    Args:
        symbol: Símbolo de la criptomoneda
        
    Returns:
        str: Plan de trading guardado o string vacío
    """
    plan_path = PLANS_FOLDER / f"{symbol}_plan.txt"
    if plan_path.exists():
        with open(plan_path, 'r') as f:
            return f.read()
    return ""

def save_trading_plan(symbol, plan_text):
    """
    Guarda el plan de trading para un símbolo
    
    Args:
        symbol: Símbolo de la criptomoneda
        plan_text: Texto del plan de trading
    """
    plan_path = PLANS_FOLDER / f"{symbol}_plan.txt"
    with open(plan_path, 'w') as f:
        f.write(plan_text)

def render_improved_analysis():
    """
    Renderiza la interfaz mejorada del centro de mando diario
    """
    # Header con información del mercado
    utc_now = datetime.datetime.now(pytz.utc)
    current_session = get_current_session()
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>Centro de Mando Diario - Crypto</h1>
        <p>Fecha UTC: {utc_now.strftime('%Y-%m-%d %H:%M')} | Sesión: {current_session}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Selección de símbolo principal
    watchlist = get_watchlist()
    main_symbol = st.selectbox("Símbolo Principal", watchlist, key="main_symbol")
    
    # 1. Dashboard con KPIs del día
    st.markdown("## Today's Dashboard")
    render_kpi_row(main_symbol)
    
    # Obtenemos datos para todos los timeframes
    data_dict = {}
    for tf_key in TIMEFRAMES.keys():
        df_raw = get_data(main_symbol, tf_key)
        data_dict[tf_key] = calculate_indicators(df_raw).dropna() if not df_raw.empty else pd.DataFrame()
    
    # 2. Layout principal: TradingView (60%) + Semáforos (40%)
    col1, col2 = st.columns([0.6, 0.4])
    
    # Columna 1: TradingView Chart
    with col1:
        st.subheader('Chart & Analysis')
        
        # TradingView Chart
        tv_height = 500
        tv_chart = f"""
        <div style="height:{tv_height}px; margin-bottom:20px;">
            <iframe 
                src="https://s.tradingview.com/widgetembed/?symbol=BINANCE:{main_symbol}&interval=D&theme=dark&hide_side_toolbar=true&allow_symbol_change=false" 
                style="width:100%; height:100%; border: none;"
                allowtransparency="true"
                scrolling="no">
            </iframe>
        </div>
        """
        components_html(tv_chart, height=tv_height+10)
        
        # Plan de Trading
        st.subheader("Plan de Trading")
        plan_text = load_trading_plan(main_symbol)
        new_plan = st.text_area("Escribe tu plan de trading para hoy:", plan_text, height=200)
        
        if st.button("Guardar Plan"):
            save_trading_plan(main_symbol, new_plan)
            st.success(f"Plan de trading para {main_symbol} guardado correctamente.")
    
    # Columna 2: Semáforos y Watchlist
    with col2:
        # Semáforo de indicadores
        render_technical_indicators(data_dict)
        
        # Watchlist mínima
        st.subheader("Watchlist")
        # Excluimos el símbolo principal para no duplicar
        other_symbols = [s for s in watchlist if s != main_symbol]
        
        for symbol in other_symbols:
            with st.expander(symbol, expanded=False):
                render_mini_chart(symbol)
        
        # Editor de Watchlist
        with st.expander("Editar Watchlist"):
            watchlist_str = st.text_area("Ingresa símbolos separados por coma:", 
                                         ", ".join(watchlist))
            if st.button("Actualizar Watchlist"):
                new_watchlist = [s.strip().upper() for s in watchlist_str.split(",")]
                save_watchlist(new_watchlist)
                st.experimental_rerun()
    
    # 3. News & Calendar (colapsable)
    with st.expander("Noticias & Calendario Económico", expanded=False):
        news_cal_cols = st.columns(2)
        
        # Noticias
        with news_cal_cols[0]:
            st.subheader("Últimas Noticias")
            coin_news = get_crypto_news(main_symbol)
            
            for news in coin_news:
                st.markdown(f"""
                <div style="border:1px solid gray; border-radius:5px; padding:10px; margin-bottom:10px;">
                    <h4>{news['title']}</h4>
                    <p>{news['description']}</p>
                    <p><small>Fuente: {news['source']} - {news['date']}</small></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Calendario económico
        with news_cal_cols[1]:
            st.subheader("Calendario Económico")
            events = get_economic_calendar()
            
            for event in events:
                impact_color = "red" if event['impact'] == "Alto" else "orange" if event['impact'] == "Medio" else "green"
                st.markdown(f"""
                <div style="border:1px solid gray; border-radius:5px; padding:10px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between;">
                        <h4>{event['event']}</h4>
                        <span style="color:{impact_color}">■ {event['impact']}</span>
                    </div>
                    <p>Fecha: {event['date']} {event['time']}</p>
                    <p>Previsión: {event['forecast']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # 4. Métricas de derivados (similar al análisis original)
    st.header('Métricas de Mercados de Derivados')
    
    # Crear una sección con tabs para cada métrica de derivados
    derivative_tabs = st.tabs(["Funding Rate", "Open Interest", "Order Flow Delta"])
    
    # Tab 1: Funding Rate
    with derivative_tabs[0]:
        funding_df = get_funding_rate(main_symbol)
        from tabs.analysis_tab import plot_funding
        funding_fig = plot_funding(funding_df)
        st.plotly_chart(funding_fig, use_container_width=True)
    
    # Tab 2: Open Interest
    with derivative_tabs[1]:
        oi_interval = st.selectbox('Select Open Interest Interval', ['5m', '15m', '1h', '4h', '1d'], index=2)
        oi_df = get_open_interest(main_symbol, oi_interval)
        price_df = get_data_perp(main_symbol, oi_interval)
        oi_fig = plot_open_interest(oi_df, price_df)
        st.plotly_chart(oi_fig, use_container_width=True)
    
    # Tab 3: Order Flow Delta
    with derivative_tabs[2]:
        delta_interval = st.selectbox('Select Order Flow Delta Interval', ['1m', '5m', '15m', '30m', '1h', '4h', '1d'], index=4)
        delta_df = get_orderflow_delta(main_symbol, delta_interval)
        delta_fig = plot_delta(delta_df, price_df)
        st.plotly_chart(delta_fig, use_container_width=True)