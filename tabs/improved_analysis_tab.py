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
    
    # Preparamos la tabla de semáforos y almacenamos los scores para cálculos posteriores
    indicators_df = pd.DataFrame(columns=['Indicador'] + list(TIMEFRAMES.keys()) + ['Score'])
    scores = []
    
    # Para cada indicador, calculamos su estado en cada timeframe
    for ind_key, ind_info in INDICATORS.items():
        row = {'Indicador': ind_info['name']}
        ind_score = 0
        
        for tf_key, tf_info in TIMEFRAMES.items():
            df = data_dict.get(tf_key, pd.DataFrame())
            if not df.empty and len(df) > 1:  # Aseguramos que hay suficientes datos
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
                        # Añadir un pequeño sesgo basado en la dirección del RSI desde el nivel neutral (50)
                        rsi_bias = (last['RSI'] - 50) / 20  # Normalizado entre -1 y 1
                        ind_score += rsi_bias * tf_info['weight']
                
                elif ind_key == 'MFI':
                    if last['MFI'] <= 20:
                        status = "🟢"  # Oversold (potencial alcista)
                        ind_score += 1 * tf_info['weight']
                    elif last['MFI'] >= 80:
                        status = "🔴"  # Overbought (potencial bajista)
                        ind_score -= 1 * tf_info['weight']
                    else:
                        status = "🟡"  # Neutral
                        # Añadir un pequeño sesgo basado en la dirección del MFI desde el nivel neutral (50)
                        mfi_bias = ((last['MFI'] - 50) / 30) * tf_info['weight']
                        ind_score += mfi_bias
                
                row[tf_key] = status
            else:
                row[tf_key] = "⚪"  # No hay datos
                # No afectamos el score si no hay datos
        
        # Calculamos el score normalizado para este indicador
        max_possible_score = sum([tf_info['weight'] for tf_key, tf_info in TIMEFRAMES.items()])
        normalized_score = (ind_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        
        # Guardar el score normalizado redondeado a 1 decimal
        score_value = round(normalized_score, 1)
        row['Score'] = f"{score_value}"
        scores.append(score_value)  # Guardamos el valor numérico para cálculos posteriores
        
        # Añadimos la fila a la tabla
        indicators_df = pd.concat([indicators_df, pd.DataFrame([row])], ignore_index=True)
    
    # Calculamos el score global como promedio de los scores individuales
    if scores:
        # Calculamos el score global
        weights = [INDICATORS[ind]['weight'] for ind in INDICATORS]
        total_weight = sum(weights)
        
        # Aplicamos los pesos para dar más importancia a ciertos indicadores
        weighted_scores = [score * weight for score, weight in zip(scores, weights)]
        global_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0
    else:
        global_score = 0
    
    # Mostramos el score global con un color indicativo
    score_color = PALETTE['green'] if global_score > 0 else PALETTE['red'] if global_score < 0 else PALETTE['neutral']
    
    st.markdown(f"""
    <div style="border:1px solid {score_color}; border-radius:5px; padding:10px; text-align:center; margin-bottom:15px;">
        <h3>Score Técnico Global</h3>
        <p style="font-size:36px; color:{score_color};">{global_score:.1f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mostramos la tabla de semáforos con formato mejorado
    # Convertimos a dataframe para mejor visualización y evitamos índice
    st.dataframe(indicators_df, use_container_width=True, hide_index=True)

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

def analyze_funding_rate(funding_df):
    """
    Analiza el funding rate y genera recomendaciones
    
    Args:
        funding_df: DataFrame con el historial de funding rate
        
    Returns:
        dict: Diccionario con análisis y recomendaciones
    """
    if funding_df.empty:
        return {
            'status': 'neutral',
            'trend': 'neutral',
            'analysis': 'No hay suficientes datos para analizar el funding rate.',
            'recommendation': 'Esperar a tener más datos para tomar decisiones basadas en funding rate.'
        }
    
    # Calculamos estadísticas del funding rate
    current = funding_df['fundingRate'].iloc[-1]
    avg_7d = funding_df['fundingRate'].iloc[-7:].mean() if len(funding_df) >= 7 else current
    avg_30d = funding_df['fundingRate'].iloc[-30:].mean() if len(funding_df) >= 30 else avg_7d
    
    # Determinar si el funding está en niveles extremos
    is_extreme = abs(current) > 0.001  # Más del 0.1% se considera extremo
    
    # Determinar tendencia
    if len(funding_df) >= 3:
        last_3 = funding_df['fundingRate'].iloc[-3:].values
        if last_3[2] > last_3[0]:
            trend = 'up'
        elif last_3[2] < last_3[0]:
            trend = 'down'
        else:
            trend = 'neutral'
    else:
        trend = 'neutral'
    
    # Determinar estatus y análisis
    if current > 0:
        status = 'bullish'
        if is_extreme:
            analysis = f"Funding rate positivo extremo ({current*100:.4f}%), indicando un fuerte sesgo alcista en el mercado de perpetuos. Los largos están pagando primas significativas a los cortos."
            recommendation = "Considerar estrategias de hedge o reducción de exposición larga en el corto plazo, posible corrección próxima."
        else:
            analysis = f"Funding rate positivo moderado ({current*100:.4f}%), señalando optimismo en el mercado. Los largos están pagando a los cortos, pero en niveles normales."
            recommendation = "Sesgo alcista prevaleciendo, mantén una postura neutra o ligeramente larga según tu estrategia."
    elif current < 0:
        status = 'bearish'
        if is_extreme:
            analysis = f"Funding rate negativo extremo ({current*100:.4f}%), indicando un fuerte sesgo bajista en el mercado de perpetuos. Los cortos están pagando primas significativas a los largos."
            recommendation = "Posible rebote técnico cercano. Considera reducir exposición corta o buscar oportunidades de entrada en largo en soportes."
        else:
            analysis = f"Funding rate negativo moderado ({current*100:.4f}%), señalando pesimismo en el mercado. Los cortos están pagando a los largos, pero en niveles normales."
            recommendation = "Sesgo bajista prevaleciendo, mantén una postura neutra o ligeramente corta según tu estrategia."
    else:
        status = 'neutral'
        analysis = "Funding rate neutral, indicando equilibrio entre posiciones largas y cortas en el mercado de perpetuos."
        recommendation = "No hay señal clara desde funding rate. Considera otros indicadores para tus decisiones."
    
    # Añadir contexto de tendencia
    if trend == 'up' and status != 'bullish':
        analysis += " Sin embargo, el funding rate está aumentando recientemente, lo que puede indicar un cambio hacia un sentimiento más alcista."
    elif trend == 'down' and status != 'bearish':
        analysis += " Sin embargo, el funding rate está disminuyendo recientemente, lo que puede indicar un cambio hacia un sentimiento más bajista."
    
    # Comparar con promedios históricos
    if abs(current - avg_30d) > abs(avg_30d) * 0.5:  # 50% de desviación del promedio
        analysis += f" El funding actual está significativamente {'por encima' if current > avg_30d else 'por debajo'} de su promedio de 30 días ({avg_30d*100:.4f}%)."
        if current > 0 and current > avg_30d:
            recommendation += " Esta desviación extrema sugiere posible sobrecompra y podría preceder a una corrección."
        elif current < 0 and current < avg_30d:
            recommendation += " Esta desviación extrema sugiere posible sobreventa y podría preceder a un rebote."
    
    return {
        'status': status,
        'trend': trend,
        'analysis': analysis,
        'recommendation': recommendation
    }

def analyze_open_interest(oi_df, price_df):
    """
    Analiza el open interest y genera recomendaciones
    
    Args:
        oi_df: DataFrame con el historial de open interest
        price_df: DataFrame con el historial de precios
        
    Returns:
        dict: Diccionario con análisis y recomendaciones
    """
    if oi_df.empty or price_df.empty:
        return {
            'status': 'neutral',
            'trend': 'neutral',
            'analysis': 'No hay suficientes datos para analizar el open interest.',
            'recommendation': 'Esperar a tener más datos para tomar decisiones basadas en open interest.'
        }
    
    # Asegurar que tenemos suficientes datos
    if len(oi_df) < 3 or len(price_df) < 3:
        return {
            'status': 'neutral',
            'trend': 'neutral',
            'analysis': 'Insuficientes datos históricos para un análisis completo de open interest.',
            'recommendation': 'Recolectar más datos para mejorar la calidad del análisis.'
        }
    
    # Calculamos estadísticas de OI
    current_oi = oi_df['openInterest'].iloc[-1]
    prev_oi = oi_df['openInterest'].iloc[-2]
    oi_change = ((current_oi / prev_oi) - 1) * 100
    
    # Calculamos estadísticas de precio
    current_price = price_df['Close'].iloc[-1]
    prev_price = price_df['Close'].iloc[-2]
    price_change = ((current_price / prev_price) - 1) * 100
    
    # Determinar tendencia del OI (últimos 3 intervalos)
    oi_values = oi_df['openInterest'].iloc[-3:].values
    if oi_values[2] > oi_values[0]:
        oi_trend = 'up'
    elif oi_values[2] < oi_values[0]:
        oi_trend = 'down'
    else:
        oi_trend = 'neutral'
    
    # Analizar la relación entre OI y precio
    if oi_change > 1:  # Aumento significativo del OI
        if price_change > 0:
            status = 'bullish'
            analysis = f"Open Interest incrementando ({oi_change:.2f}%) junto con el precio ({price_change:.2f}%). Esto sugiere nueva liquidez entrando en posiciones largas, fortaleciendo la tendencia alcista."
            recommendation = "Considerar mantener o aumentar exposición alcista mientras este patrón continúe."
        else:
            status = 'bearish'
            analysis = f"Open Interest incrementando ({oi_change:.2f}%) mientras el precio cae ({price_change:.2f}%). Esto sugiere nueva liquidez entrando en posiciones cortas, fortaleciendo la tendencia bajista."
            recommendation = "Cautela con posiciones largas. La presión vendedora está aumentando."
    elif oi_change < -1:  # Disminución significativa del OI
        if price_change > 0:
            status = 'mixed'
            analysis = f"Open Interest disminuyendo ({oi_change:.2f}%) mientras el precio sube ({price_change:.2f}%). Esto sugiere cierre de posiciones cortas (short squeeze) o toma de ganancias en posiciones largas."
            recommendation = "Posible rebote técnico o alivio de sobreventa. La tendencia alcista podría ser temporal."
        else:
            status = 'mixed'
            analysis = f"Open Interest disminuyendo ({oi_change:.2f}%) junto con el precio ({price_change:.2f}%). Esto sugiere cierre de posiciones largas y posible agotamiento vendedor."
            recommendation = "Considerar reducir exposición corta en soportes clave. Posible formación de suelo cercana."
    else:  # OI estable
        status = 'neutral'
        analysis = f"Open Interest relativamente estable ({oi_change:.2f}%) con precio {('subiendo' if price_change > 0 else 'bajando')} ({price_change:.2f}%). No hay señales fuertes de nuevas posiciones."
        recommendation = "Mantener estrategia actual y monitorear cambios en volumen y precio para señales más claras."
    
    return {
        'status': status,
        'trend': oi_trend,
        'analysis': analysis,
        'recommendation': recommendation
    }

def analyze_order_flow(delta_df, price_df):
    """
    Analiza el order flow delta y genera recomendaciones
    
    Args:
        delta_df: DataFrame con delta de órdenes
        price_df: DataFrame con precios
        
    Returns:
        dict: Diccionario con análisis y recomendaciones
    """
    if delta_df.empty or price_df.empty or len(delta_df) < 3:
        return {
            'status': 'neutral',
            'trend': 'neutral',
            'analysis': 'Datos insuficientes para analizar order flow delta.',
            'recommendation': 'Recopilar más datos de mercado para un análisis efectivo.'
        }
    
    # Estadísticas clave
    current_delta = delta_df['delta'].iloc[-1]
    current_delta_pct = delta_df['delta_percent'].iloc[-1] if 'delta_percent' in delta_df else 0
    cum_delta = delta_df['cum_delta'].iloc[-1] if 'cum_delta' in delta_df else delta_df['delta'].sum()
    
    # Tendencia reciente del delta
    recent_delta = delta_df['delta'].iloc[-5:].mean() if len(delta_df) >= 5 else delta_df['delta'].mean()
    
    # Determinar tendencia
    delta_values = delta_df['delta'].iloc[-3:].values
    if delta_values[2] > delta_values[0]:
        delta_trend = 'improving'  # Mejorando (más compras o menos ventas)
    elif delta_values[2] < delta_values[0]:
        delta_trend = 'deteriorating'  # Empeorando (menos compras o más ventas)
    else:
        delta_trend = 'neutral'
    
    # Análisis básico
    if current_delta > 0:
        if current_delta_pct > 10:  # Dominancia compradora fuerte
            status = 'strongly_bullish'
            analysis = f"Fuerte dominancia compradora (Delta: {current_delta_pct:.2f}%). Los compradores están siendo significativamente más agresivos que los vendedores."
            recommendation = "Sesgo alcista fuerte. Considerar entradas largas en retrocesos menores."
        else:
            status = 'bullish'
            analysis = f"Presión compradora moderada (Delta: {current_delta_pct:.2f}%). Hay más volumen de compra que de venta, pero no es dominante."
            recommendation = "Sesgo alcista presente. Mantener posiciones largas con stops adecuados."
    elif current_delta < 0:
        if current_delta_pct < -10:  # Dominancia vendedora fuerte
            status = 'strongly_bearish'
            analysis = f"Fuerte dominancia vendedora (Delta: {current_delta_pct:.2f}%). Los vendedores están siendo significativamente más agresivos que los compradores."
            recommendation = "Sesgo bajista fuerte. Considerar proteger posiciones largas o buscar oportunidades cortas."
        else:
            status = 'bearish'
            analysis = f"Presión vendedora moderada (Delta: {current_delta_pct:.2f}%). Hay más volumen de venta que de compra, pero no es dominante."
            recommendation = "Sesgo bajista presente. Mantener cautela con posiciones largas."
    else:
        status = 'neutral'
        analysis = "Equilibrio entre compradores y vendedores. No hay presión dominante en el order flow."
        recommendation = "Sin sesgo claro desde order flow. Buscar confirmación en otros indicadores."
    
    # Añadir contexto de delta acumulativo
    if cum_delta > 0:
        if status.endswith('bearish'):
            analysis += f" Sin embargo, el delta acumulativo sigue siendo positivo ({cum_delta:.2f}), lo que sugiere que la presión compradora ha dominado históricamente."
        else:
            analysis += f" El delta acumulativo positivo ({cum_delta:.2f}) confirma la presión compradora sostenida."
    else:
        if status.endswith('bullish'):
            analysis += f" Sin embargo, el delta acumulativo sigue siendo negativo ({cum_delta:.2f}), lo que sugiere que la presión vendedora ha dominado históricamente."
        else:
            analysis += f" El delta acumulativo negativo ({cum_delta:.2f}) confirma la presión vendedora sostenida."
    
    # Analizar divergencia con precio
    current_price = price_df['Close'].iloc[-1]
    prev_price = price_df['Close'].iloc[-5] if len(price_df) >= 5 else price_df['Close'].iloc[0]
    price_change = ((current_price / prev_price) - 1) * 100
    
    if (price_change > 2 and recent_delta < 0) or (price_change < -2 and recent_delta > 0):
        analysis += f" DIVERGENCIA DETECTADA: El precio se mueve {('al alza' if price_change > 0 else 'a la baja')} ({price_change:.2f}%) mientras el order flow sugiere lo contrario."
        recommendation += " La divergencia entre precio y order flow sugiere posible reversión. Considera reducir exposición y esperar confirmación."
    
    return {
        'status': status,
        'trend': delta_trend,
        'analysis': analysis,
        'recommendation': recommendation
    }

def render_derivatives_analysis(symbol):
    """
    Renderiza un análisis completo de los datos de derivados con las tres métricas juntas
    
    Args:
        symbol: Símbolo de la criptomoneda
    """
    st.header('Análisis de Mercados de Derivados')
    
    # Obtenemos todos los datos que necesitamos
    funding_df = get_funding_rate(symbol)
    oi_interval = '1h'  # Predefino un intervalo para simplificar
    oi_df = get_open_interest(symbol, oi_interval)
    price_df = get_data_perp(symbol, oi_interval)
    delta_df = get_orderflow_delta(symbol, oi_interval)
    
    # Ejecutamos los análisis
    funding_analysis = analyze_funding_rate(funding_df)
    oi_analysis = analyze_open_interest(oi_df, price_df)
    flow_analysis = analyze_order_flow(delta_df, price_df)
    
    # 1. Tablero de resumen con las tres métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        funding_color = PALETTE['green'] if funding_analysis['status'] == 'bullish' else PALETTE['red'] if funding_analysis['status'] == 'bearish' else PALETTE['neutral']
        st.markdown(f"""
        <div style="border:1px solid {funding_color}; border-radius:5px; padding:10px;">
            <h3 style="text-align:center;">Funding Rate</h3>
            <p style="color:{funding_color}; font-weight:bold; text-align:center;">
                {funding_df['fundingRate'].iloc[-1]*100:.4f}% 
                {' ▲' if funding_analysis['trend'] == 'up' else ' ▼' if funding_analysis['trend'] == 'down' else ''}
            </p>
            <h4>Análisis:</h4>
            <p>{funding_analysis['analysis']}</p>
            <h4>Recomendación:</h4>
            <p>{funding_analysis['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        oi_color = PALETTE['green'] if oi_analysis['status'] == 'bullish' else PALETTE['red'] if oi_analysis['status'] == 'bearish' else PALETTE['neutral']
        oi_change = ((oi_df['openInterest'].iloc[-1] / oi_df['openInterest'].iloc[-2]) - 1) * 100 if not oi_df.empty and len(oi_df) > 1 else 0
        st.markdown(f"""
        <div style="border:1px solid {oi_color}; border-radius:5px; padding:10px;">
            <h3 style="text-align:center;">Open Interest</h3>
            <p style="color:{oi_color}; font-weight:bold; text-align:center;">
                {oi_change:.2f}% 
                {' ▲' if oi_analysis['trend'] == 'up' else ' ▼' if oi_analysis['trend'] == 'down' else ''}
            </p>
            <h4>Análisis:</h4>
            <p>{oi_analysis['analysis']}</p>
            <h4>Recomendación:</h4>
            <p>{oi_analysis['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        flow_color = PALETTE['green'] if flow_analysis['status'] in ['bullish', 'strongly_bullish'] else PALETTE['red'] if flow_analysis['status'] in ['bearish', 'strongly_bearish'] else PALETTE['neutral']
        current_delta_pct = delta_df['delta_percent'].iloc[-1] if not delta_df.empty and 'delta_percent' in delta_df else 0
        st.markdown(f"""
        <div style="border:1px solid {flow_color}; border-radius:5px; padding:10px;">
            <h3 style="text-align:center;">Order Flow</h3>
            <p style="color:{flow_color}; font-weight:bold; text-align:center;">
                {current_delta_pct:.2f}% 
                {' ▲' if flow_analysis['trend'] == 'improving' else ' ▼' if flow_analysis['trend'] == 'deteriorating' else ''}
            </p>
            <h4>Análisis:</h4>
            <p>{flow_analysis['analysis']}</p>
            <h4>Recomendación:</h4>
            <p>{flow_analysis['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 2. Análisis integrado y conclusión general
    overall_bullish = sum([1 if a['status'] in ['bullish', 'strongly_bullish'] else 0 for a in [funding_analysis, oi_analysis, flow_analysis]])
    overall_bearish = sum([1 if a['status'] in ['bearish', 'strongly_bearish'] else 0 for a in [funding_analysis, oi_analysis, flow_analysis]])
    
    if overall_bullish > overall_bearish:
        overall_status = "bullish"
        overall_color = PALETTE['green']
        if overall_bullish == 3:
            overall_strength = "fuerte"
            overall_message = "Los tres indicadores de derivados muestran un sesgo alcista claro. Alta probabilidad de continuación de tendencia alcista."
        else:
            overall_strength = "moderado"
            overall_message = f"Mayoría de indicadores ({overall_bullish}/3) muestran sesgo alcista. Considerar posiciones largas con gestión adecuada de riesgo."
    elif overall_bearish > overall_bullish:
        overall_status = "bearish"
        overall_color = PALETTE['red']
        if overall_bearish == 3:
            overall_strength = "fuerte"
            overall_message = "Los tres indicadores de derivados muestran un sesgo bajista claro. Alta probabilidad de continuación de tendencia bajista."
        else:
            overall_strength = "moderado"
            overall_message = f"Mayoría de indicadores ({overall_bearish}/3) muestran sesgo bajista. Cautela con posiciones largas; considerar estrategias de protección."
    else:
        overall_status = "neutral"
        overall_color = PALETTE['neutral']
        overall_strength = "mixto"
        overall_message = "Señales mixtas de los indicadores de derivados. No hay un sesgo claro; enfócate en niveles técnicos y catalistas fundamentales."
    
    st.markdown(f"""
    <div style="border:2px solid {overall_color}; border-radius:5px; padding:15px; margin-top:20px;">
        <h3 style="text-align:center;">Conclusión del Análisis de Derivados</h3>
        <p style="font-size:18px; text-align:center; color:{overall_color};">
            Sesgo {overall_status.upper()} {overall_strength}
        </p>
        <p style="font-size:16px;">{overall_message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Gráficos integrados (opcional)
    with st.expander("Ver gráficos detallados de derivados"):
        # Aquí podríamos mostrar los gráficos de cada métrica
        st.subheader("Funding Rate")
        from tabs.analysis_tab import plot_funding
        funding_fig = plot_funding(funding_df)
        st.plotly_chart(funding_fig, use_container_width=True)
        
        st.subheader("Open Interest vs Precio")
        oi_fig = plot_open_interest(oi_df, price_df)
        st.plotly_chart(oi_fig, use_container_width=True)
        
        st.subheader("Order Flow Delta")
        delta_fig = plot_delta(delta_df, price_df)
        st.plotly_chart(delta_fig, use_container_width=True)

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
    
    # 4. Métricas de derivados (versión mejorada con análisis)
    render_derivatives_analysis(main_symbol)