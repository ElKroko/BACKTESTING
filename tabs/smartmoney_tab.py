"""
Módulo de análisis de Smart Money Concepts.

Este módulo implementa visualizaciones y análisis basados en la teoría
de Smart Money Concepts (SMC) incluyendo:
- Order Blocks (bloques de órdenes)
- Fair Value Gaps (FVG)
- Liquidity Pools (piscinas de liquidez)
"""
import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

from utils.data_utils import get_data
from utils.html_utils import tooltip

def detect_order_blocks(df: pd.DataFrame, pct: float = 0.01) -> list:
    """
    Detecta Order Blocks basados en velas con movimientos significativos
    
    Args:
        df: DataFrame con datos OHLC
        pct: Porcentaje mínimo de movimiento para considerar un Order Block
        
    Returns:
        Lista de timestamps de Order Blocks
    """
    moves = df['Close'].pct_change().shift(-1)
    return df.index[moves.abs() > pct].tolist()

def detect_fvg(df: pd.DataFrame) -> list:
    """
    Detecta Fair Value Gaps (FVG) donde hay gaps de precio
    
    Args:
        df: DataFrame con datos OHLC
        
    Returns:
        Lista de gaps con timestamp y niveles de precio
    """
    gaps = []
    for i in range(len(df) - 2):
        h1, l3 = df['High'].iat[i], df['Low'].iat[i+2]
        if h1 < l3:
            gaps.append((df.index[i+1], h1, l3))
    return gaps

def detect_liquidity_pools(df: pd.DataFrame) -> list:
    """
    Detecta áreas de liquidez basadas en niveles de precio tocados múltiples veces
    
    Args:
        df: DataFrame con datos OHLC
        
    Returns:
        Lista de niveles de precio con alta liquidez
    """
    counts = df['High'].round(2).value_counts() + df['Low'].round(2).value_counts()
    return counts[counts >= 2].index.tolist()

def render_smart_money():
    """
    Renderiza la interfaz de usuario para el análisis de Smart Money Concepts
    """
    st.markdown(tooltip('💡 Smart Money Concepts', 
                       'Análisis basado en la teoría de Smart Money Concepts (SMC), que estudia patrones de acumulación y distribución utilizados por operadores institucionales.'),
              unsafe_allow_html=True)
    
    symbol = st.text_input('Symbol', value='BTCUSDT', key='smc_symbol').strip().upper()
    st.header(f'Smart Money Concepts for {symbol}')
    
    intervals = [('1 Hour','1h'), ('4 Hours','4h'), ('Daily','1d')]

    style = mpf.make_mpf_style(base_mpf_style='nightclouds')
    mc = mpf.make_marketcolors(up='lime', down='red', inherit=True)
    style['marketcolors'] = mc

    for title, interval in intervals:
        df = get_data(symbol, interval)
        st.subheader(title)

        if df.empty:
            st.warning(f"No data available for {title} ({interval}). Skipping visualization.")
            continue

        obs   = detect_order_blocks(df)
        gaps  = detect_fvg(df)
        pools = detect_liquidity_pools(df)

        cols = st.columns(3)

        with cols[0]:
            st.markdown(tooltip('**Order Blocks (>1% next candle)**', 
                              'Los Order Blocks son áreas de precio donde los operadores institucionales acumulan o distribuyen posiciones, típicamente antes de movimientos significativos.'),
                       unsafe_allow_html=True)
            
            fig, axlist = mpf.plot(df, type='candle', style=style, mav=(50,100),
                                   volume=False, returnfig=True, figsize=(4,3))
            ax = axlist[0]
            y_vals = df.loc[obs, 'Close']
            ax.scatter(obs, y_vals, color='cyan', s=20, label='Order Block')
            ax.set_xlim(df.index[0], df.index[-1])
            ax.legend(loc='upper left', fontsize=6)
            st.pyplot(fig)

        with cols[1]:
            st.markdown(tooltip('**Fair Value Gaps**',
                              'Los Fair Value Gaps (FVGs) son áreas de precio que se saltaron durante movimientos rápidos. El precio suele volver a "rellenar" estos gaps.'),
                       unsafe_allow_html=True)
            
            fig, axlist = mpf.plot(df, type='candle', style=style,
                                   volume=False, returnfig=True, figsize=(4,3))
            ax = axlist[0]
            for _, h1, l3 in gaps:
                ax.axhspan(l3, h1, color='orange', alpha=0.3)
            st.pyplot(fig)

        with cols[2]:
            st.markdown(tooltip('**Liquidity Pools (tocados ≥2)**',
                              'Las piscinas de liquidez son áreas donde se concentran órdenes de compra/venta. Los operadores institucionales suelen buscar esta liquidez.'),
                       unsafe_allow_html=True)
            
            fig, axlist = mpf.plot(df, type='candle', style=style,
                                   volume=False, returnfig=True, figsize=(4,3))
            ax = axlist[0]
            start, end = df.index[0], df.index[-1]
            for lvl in pools:
                ax.hlines(lvl, start, end, linestyles='--', linewidth=1)
            st.pyplot(fig)

        st.markdown('---')

    st.write(
        'Estos gráficos muestran conceptos avanzados de análisis de mercado siguiendo la teoría de Smart Money Concepts: '
        '**Order Blocks** (bloques de órdenes que generan impulsos), **Fair Value Gaps** (gaps que tienden a rellenarse), '
        'y **Liquidity Pools** (áreas donde se concentran órdenes que atraen a los operadores institucionales).'
    )