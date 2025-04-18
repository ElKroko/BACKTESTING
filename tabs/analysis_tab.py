"""
Módulo de análisis técnico para el sistema de backtesting.

Este módulo implementa el análisis técnico de activos con soporte para:
- Múltiples timeframes (semanal, diario, por hora)
- Múltiples indicadores técnicos (SMA, MACD, RSI, MFI)
- Detección de niveles de soporte/resistencia
- Generación de reportes PDF
- Métricas de mercados de derivados (Funding Rate, Open Interest, Order Flow Delta)
"""
import streamlit as st
import pandas as pd
import requests
import ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from streamlit.components.v1 import html as components_html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.html_utils import tooltip
from utils.data_utils import get_data, calculate_indicators

# Paleta de colores para mantener consistencia
PALETTE = {
    'template': 'plotly_dark',
    'green': '#26a69a',
    'red': '#ef5350',
    'neutral': '#ffd54f'
}

def detect_support_resistance(df: pd.DataFrame):
    """
    Detecta niveles de soporte y resistencia basados en puntos de giro
    
    Args:
        df: DataFrame con datos OHLC
        
    Returns:
        Tupla con listas de niveles de resistencia y soporte
    """
    highs = df['High']; lows = df['Low']
    piv_high = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    piv_low  = lows[(lows.shift(1) > lows)   & (lows.shift(-1) > lows)]
    res = piv_high.iloc[-2:].sort_values(ascending=False).tolist() if len(piv_high) >= 2 else []
    sup = piv_low.iloc[-2:].sort_values().tolist()             if len(piv_low)  >= 2 else []
    return res, sup

def detect_pivots(df: pd.DataFrame):
    """
    Detecta máximos y mínimos pivotes para el análisis técnico
    
    Args:
        df: DataFrame con datos OHLC
        
    Returns:
        Tupla con listas de Higher Highs y Lower Lows
    """
    hh = df['High'].rolling(20).max().dropna().iloc[-2:].tolist()[::-1]
    ll = df['Low'].rolling(20).min().dropna().iloc[-2:].tolist()[::-1]
    return hh, ll

def indicator_recommendation(ind: str, value: float, extra=None) -> str:
    """
    Genera recomendaciones basadas en valores de indicadores
    
    Args:
        ind: Nombre del indicador
        value: Valor actual del indicador
        extra: Valor adicional para comparaciones (ej. señal MACD)
        
    Returns:
        Recomendación como string
    """
    if ind == 'SMA50' and extra is not None:
        return 'Bullish' if value > extra else 'Bearish'
    if ind == 'MACD'  and extra is not None:
        return 'Bullish' if value > extra else 'Bearish'
    if ind == 'RSI':
        if value > 70: return 'Overbought'
        if value < 30: return 'Oversold'
        return 'Neutral'
    if ind == 'MFI':
        if value > 80: return 'Overbought'
        if value < 20: return 'Oversold'
        return 'Neutral'
    return ''

def generate_pdf(symbol: str, data_dict: dict, pdf_filename: str = 'analysis_report.pdf') -> str:
    """
    Genera un reporte PDF con el análisis técnico
    
    Args:
        symbol: Símbolo del activo analizado
        data_dict: Diccionario con DataFrames por timeframe
        pdf_filename: Nombre del archivo PDF a generar
        
    Returns:
        Ruta del archivo PDF generado
    """
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height-50, f"Analysis Report: {symbol}")
    y = height - 80
    for label, df in data_dict.items():
        if df.empty: continue
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, label)
        y -= 20
        last = df.iloc[-1]
        for name in ['SMA50','SMA100','MACD','Signal','RSI','MFI']:
            c.setFont("Helvetica", 12)
            c.drawString(60, y, f"{name}: {round(last[name],2)}")
            y -= 15
            if y < 60:
                c.showPage()
                y = height - 50
        y -= 10
    c.save()
    return pdf_filename

def get_funding_rate(symbol: str) -> pd.DataFrame:
    """
    Obtiene el histórico de funding rate de Binance para un símbolo de perpetual futures
    
    Args:
        symbol: Símbolo del contrato perpetuo (ej. BTCUSDT)
        
    Returns:
        DataFrame con el historial de funding rate
    """
    try:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=500"
        response = requests.get(url)
        if response.status_code != 200:
            st.warning(f"Error al obtener datos de funding rate: {response.text}")
            return pd.DataFrame()
            
        fr_df = pd.DataFrame(response.json())
        fr_df['fundingTime'] = pd.to_datetime(fr_df['fundingTime'], unit='ms')
        fr_df.set_index('fundingTime', inplace=True)
        fr_df['fundingRate'] = fr_df['fundingRate'].astype(float)
        fr_df.sort_index(inplace=True)
        return fr_df
    except Exception as e:
        st.error(f"Error al procesar funding rate: {str(e)}")
        return pd.DataFrame()
        
def plot_funding(funding_df: pd.DataFrame) -> go.Figure:
    """
    Crea un gráfico de línea para visualizar el histórico de funding rate
    
    Args:
        funding_df: DataFrame con datos de funding rate
        
    Returns:
        Figura de Plotly con el gráfico de funding rate
    """
    if funding_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No hay datos disponibles", showarrow=False)
        fig.update_layout(template=PALETTE['template'])
        return fig
        
    # Crear línea base en 0
    zero_line = go.Scatter(
        x=funding_df.index, 
        y=[0] * len(funding_df),
        mode='lines',
        line=dict(color='white', width=1, dash='dash'),
        name='Neutral'
    )
    
    # Crear áreas coloreadas para values positivos y negativos
    positive_mask = funding_df['fundingRate'] >= 0
    negative_mask = funding_df['fundingRate'] < 0
    
    positive_scatter = go.Scatter(
        x=funding_df.index[positive_mask],
        y=funding_df['fundingRate'][positive_mask],
        fill='tozeroy',
        mode='none',
        fillcolor=f"rgba({int(PALETTE['green'][1:3], 16)}, {int(PALETTE['green'][3:5], 16)}, {int(PALETTE['green'][5:7], 16)}, 0.5)",
        name='Bullish (largos pagan)'
    )
    
    negative_scatter = go.Scatter(
        x=funding_df.index[negative_mask],
        y=funding_df['fundingRate'][negative_mask],
        fill='tozeroy',
        mode='none',
        fillcolor=f"rgba({int(PALETTE['red'][1:3], 16)}, {int(PALETTE['red'][3:5], 16)}, {int(PALETTE['red'][5:7], 16)}, 0.5)",
        name='Bearish (cortos pagan)'
    )
    
    # Crear el gráfico
    fig = go.Figure(data=[zero_line, positive_scatter, negative_scatter])
    
    # Actualizar layout
    fig.update_layout(
        title="Historial de Funding Rate (Mercado de Perpetuos)",
        xaxis_title="Fecha",
        yaxis_title="Funding Rate (%)",
        template=PALETTE['template'],
        hovermode="x unified",
        height=400
    )
    
    # Agregar anotación explicativa
    fig.add_annotation(
        text="Funding positivo: Largos pagan a cortos (mercado alcista)<br>Funding negativo: Cortos pagan a largos (mercado bajista)",
        align="left",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="gray",
        borderwidth=1
    )
    
    return fig

def get_data_perp(symbol: str, interval: str) -> pd.DataFrame:
    """
    Obtiene datos OHLC de contratos perpetuos
    
    Args:
        symbol: Símbolo del contrato perpetuo (ej. BTCUSDT)
        interval: Intervalo de tiempo (1h, 4h, 1d)
        
    Returns:
        DataFrame con datos OHLC
    """
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit=500"
        response = requests.get(url)
        if response.status_code != 200:
            st.warning(f"Error al obtener datos OHLC de perpetuos: {response.text}")
            return pd.DataFrame()
            
        # Formato de columnas para klines de Binance
        cols = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                'Close time', 'Quote asset volume', 'Number of trades',
                'Taker buy base volume', 'Taker buy quote volume', 'Ignore']
        
        df = pd.DataFrame(response.json(), columns=cols)
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df.set_index('Open time', inplace=True)
        
        # Convertir columnas numéricas
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
            
        return df
    except Exception as e:
        st.error(f"Error al procesar datos OHLC de perpetuos: {str(e)}")
        return pd.DataFrame()

def get_open_interest(symbol: str, interval: str) -> pd.DataFrame:
    """
    Obtiene el histórico de Open Interest de Binance para un símbolo de perpetual futures
    
    Args:
        symbol: Símbolo del contrato perpetuo (ej. BTCUSDT)
        interval: Intervalo de tiempo (5m, 15m, 1h, 4h, 1d)
        
    Returns:
        DataFrame con el historial de Open Interest
    """
    try:
        url = f"https://fapi.binance.com/fapi/v1/openInterestHist?symbol={symbol}&period={interval}&limit=500"
        response = requests.get(url)
        if response.status_code != 200:
            st.warning(f"Error al obtener datos de open interest: {response.text}")
            return pd.DataFrame()
            
        oi_df = pd.DataFrame(response.json())
        oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
        oi_df.set_index('timestamp', inplace=True)
        oi_df['openInterest'] = oi_df['sumOpenInterest'].astype(float)
        oi_df['sumOpenInterestValue'] = oi_df['sumOpenInterestValue'].astype(float)
        oi_df.sort_index(inplace=True)
        return oi_df
    except Exception as e:
        st.error(f"Error al procesar open interest: {str(e)}")
        return pd.DataFrame()

def plot_open_interest(oi_df: pd.DataFrame, price_df: pd.DataFrame) -> go.Figure:
    """
    Crea un gráfico combinado de Open Interest y precio
    
    Args:
        oi_df: DataFrame con datos de Open Interest
        price_df: DataFrame con datos OHLC
        
    Returns:
        Figura de Plotly con el gráfico combinado
    """
    if oi_df.empty or price_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No hay datos disponibles", showarrow=False)
        fig.update_layout(template=PALETTE['template'])
        return fig
    
    # Crear un gráfico con subplots
    fig = make_subplots(rows=2, cols=1, 
                         shared_xaxes=True, 
                         row_heights=[0.7, 0.3],
                         vertical_spacing=0.05,
                         subplot_titles=("Precio", "Open Interest"))
    
    # Añadir el gráfico de velas para el precio
    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df['Open'],
            high=price_df['High'],
            low=price_df['Low'],
            close=price_df['Close'],
            name="Precio"
        ),
        row=1, col=1
    )
    
    # Añadir el gráfico de barras para el Open Interest
    colors = [PALETTE['green'] if oi_df['openInterest'].iloc[i] >= oi_df['openInterest'].iloc[i-1] 
              else PALETTE['red'] for i in range(1, len(oi_df))]
    colors.insert(0, PALETTE['neutral'])  # Para el primer elemento
    
    fig.add_trace(
        go.Bar(
            x=oi_df.index,
            y=oi_df['openInterest'],
            name="Open Interest",
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Actualizar layout
    fig.update_layout(
        title="Open Interest vs. Precio",
        xaxis_title="Fecha",
        template=PALETTE['template'],
        height=600,
        hovermode="x unified",
        xaxis_rangeslider_visible=False
    )
    
    # Personalizar los ejes Y
    fig.update_yaxes(title_text="Precio", row=1, col=1)
    fig.update_yaxes(title_text="Contratos", row=2, col=1)
    
    # Agregar anotación explicativa
    fig.add_annotation(
        text="Aumento en Open Interest con precio subiendo: Mayor convicción alcista<br>Aumento en Open Interest con precio bajando: Mayor convicción bajista",
        align="left",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="gray",
        borderwidth=1
    )
    
    return fig

def get_orderflow_delta(symbol: str, interval: str) -> pd.DataFrame:
    """
    Obtiene el Order Flow Delta (diferencia entre volumen comprador y vendedor)
    
    Args:
        symbol: Símbolo del contrato perpetuo (ej. BTCUSDT)
        interval: Intervalo de tiempo para la agregación
        
    Returns:
        DataFrame con el delta de orden calculado
    """
    try:
        # Obtenemos datos de trades agregados
        url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={symbol}&limit=1000"
        response = requests.get(url)
        if response.status_code != 200:
            st.warning(f"Error al obtener trades: {response.text}")
            return pd.DataFrame()
            
        # Procesar los datos
        trades = response.json()
        df_trades = pd.DataFrame(trades)
        
        # Convertir datos y calcular delta
        df_trades['timestamp'] = pd.to_datetime(df_trades['T'], unit='ms')
        df_trades['price'] = df_trades['p'].astype(float)
        df_trades['qty'] = df_trades['q'].astype(float)
        
        # Si m=True, el maker es un comprador (el taker es vendedor)
        # Si m=False, el maker es un vendedor (el taker es comprador)
        df_trades['delta'] = df_trades.apply(
            lambda r: r['qty'] if r['m'] == False else -r['qty'], 
            axis=1
        )
        
        # Calcular volumen total y delta acumulado
        df_trades['total_volume'] = df_trades['qty']
        
        # Establecer el timestamp como índice
        df_trades.set_index('timestamp', inplace=True)
        
        # Determinar el formato de resampleo basado en el intervalo
        resample_map = {
            '1m': '1min', '5m': '5min', '15m': '15min',
            '30m': '30min', '1h': '1H', '4h': '4H',
            '1d': '1D', '1w': '1W'
        }
        resample_interval = resample_map.get(interval, '1H')
        
        # Resamplear los datos por el intervalo deseado
        agg_dict = {
            'price': 'mean',
            'qty': 'sum',
            'delta': 'sum',
            'total_volume': 'sum'
        }
        
        df_resampled = df_trades.resample(resample_interval).agg(agg_dict)
        
        # Calcular delta acumulativo
        df_resampled['cum_delta'] = df_resampled['delta'].cumsum()
        
        # Calcular porcentaje de delta respecto al volumen
        df_resampled['delta_percent'] = (df_resampled['delta'] / df_resampled['total_volume']) * 100
        
        return df_resampled
    except Exception as e:
        st.error(f"Error al procesar delta: {str(e)}")
        return pd.DataFrame()

def plot_delta(delta_df: pd.DataFrame, price_df: pd.DataFrame) -> go.Figure:
    """
    Crea un gráfico combinado de Order Flow Delta y precio
    
    Args:
        delta_df: DataFrame con datos de delta calculados
        price_df: DataFrame con datos OHLC
        
    Returns:
        Figura de Plotly con el gráfico combinado
    """
    if delta_df.empty or price_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No hay datos disponibles", showarrow=False)
        fig.update_layout(template=PALETTE['template'])
        return fig
    
    # Crear gráfico con eje Y secundario
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Añadir gráfico de velas para el precio
    fig.add_trace(
        go.Candlestick(
            x=price_df.index,
            open=price_df['Open'],
            high=price_df['High'],
            low=price_df['Low'],
            close=price_df['Close'],
            name="Precio"
        ),
        secondary_y=False
    )
    
    # Colorear barras de delta según su valor
    colors = [PALETTE['green'] if d >= 0 else PALETTE['red'] for d in delta_df['delta']]
    
    # Añadir barras de delta
    fig.add_trace(
        go.Bar(
            x=delta_df.index,
            y=delta_df['delta'],
            name="Delta (Compras - Ventas)",
            marker_color=colors
        ),
        secondary_y=True
    )
    
    # Añadir línea de delta acumulativo
    fig.add_trace(
        go.Scatter(
            x=delta_df.index,
            y=delta_df['cum_delta'],
            name="Delta Acumulativo",
            line=dict(color=PALETTE['neutral'], width=2)
        ),
        secondary_y=True
    )
    
    # Actualizar layout
    fig.update_layout(
        title="Order Flow Delta vs. Precio",
        xaxis_title="Fecha",
        template=PALETTE['template'],
        height=500,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        barmode='relative'
    )
    
    # Personalizar los ejes Y
    fig.update_yaxes(title_text="Precio", secondary_y=False)
    fig.update_yaxes(title_text="Delta de Volumen", secondary_y=True)
    
    # Agregar anotación explicativa
    fig.add_annotation(
        text="Delta positivo: Predominan compras (presión alcista)<br>Delta negativo: Predominan ventas (presión bajista)",
        align="left",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="gray",
        borderwidth=1
    )
    
    return fig

def render_analysis():
    """
    Renderiza la interfaz de usuario para el análisis técnico
    """
    st.markdown(tooltip('📊 Technical Analysis', 
                      'Analiza activos en múltiples timeframes con indicadores técnicos, niveles de soporte/resistencia y patrones de precio.'),
               unsafe_allow_html=True)
    
    symbol = st.text_input('Symbol (e.g. BTCUSDT)', 'BTCUSDT', key='analysis_symbol').strip().upper()
    st.title(f'Analysis: {symbol}')

    timeframes = [
        ('Weekly', '1w', 'W'),
        ('Daily',  '1d', 'D'),
        ('Hourly','1h', '60')
    ]

    data_dict, levels_dict, pivots_dict = {}, {}, {}
    for label, api_int, tv_int in timeframes:
        df_raw = get_data(symbol, api_int)
        df = calculate_indicators(df_raw).dropna()
        data_dict[label] = df
        if not df.empty:
            res, sup = detect_support_resistance(df)
            hh, ll = detect_pivots(df)
            levels_dict[label] = {'res': res, 'sup': sup}
            pivots_dict[label] = {'hh': hh, 'll': ll}
        else:
            levels_dict[label] = {'res': [], 'sup': []}
            pivots_dict[label] = {'hh': [], 'll': []}

    st.subheader('Combined View by Timeframe')
    cols = st.columns(3)
    hide = '&hide_top_toolbar=true&hide_side_toolbar=true&hide_legend=true&allow_symbol_change=false'

    for col, (label, _, tv_int) in zip(cols, timeframes):
        df     = data_dict[label]
        levels = levels_dict[label]
        pivots = pivots_dict[label]
        with col:
            st.markdown(f'### {label}')
            src = f'https://s.tradingview.com/widgetembed/?symbol=BINANCE:{symbol}&interval={tv_int}&theme=Dark{hide}'
            components_html(f'<iframe src="{src}" width="100%" height="200" frameborder="0"></iframe>', height=210)

            if df.empty:
                st.warning(f'Not enough data for {label}')
            else:
                last = df.iloc[-1]
                rows = []
                for ind in ['SMA50','SMA100','MACD','Signal','RSI','MFI']:
                    extra = last['SMA100'] if ind=='SMA50' else (last['Signal'] if ind=='MACD' else None)
                    rec   = indicator_recommendation(ind, last[ind], extra)
                    rows.append({'Indicator': ind, 'Value': round(last[ind],2), 'Recommendation': rec})
                for i, r in enumerate(levels['res']): rows.append({'Indicator': f'R{i+1}','Value': round(r,2),'Recommendation':''})
                for i, s in enumerate(levels['sup']): rows.append({'Indicator': f'S{i+1}','Value': round(s,2),'Recommendation':''})
                for i, h in enumerate(pivots['hh']): rows.append({'Indicator': f'HH{i+1}','Value': round(h,2),'Recommendation':''})
                for i, l in enumerate(pivots['ll']): rows.append({'Indicator': f'LL{i+1}','Value': round(l,2),'Recommendation':''})
                st.table(pd.DataFrame(rows))

                fig, ax = plt.subplots(figsize=(4,3))
                ax.plot(df.index, df['Close'], label='Close')
                for price in levels['res']: ax.hlines(price, df.index[0], df.index[-1], linestyles='--', alpha=0.7)
                for price in levels['sup']: ax.hlines(price, df.index[0], df.index[-1], linestyles='--', alpha=0.7)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.legend(fontsize='small')
                st.pyplot(fig)

    funding_df = get_funding_rate(symbol)
    st.subheader('Funding Rate Analysis')
    funding_fig = plot_funding(funding_df)
    st.plotly_chart(funding_fig, use_container_width=True)

    oi_interval = st.selectbox('Select Open Interest Interval', ['5m', '15m', '1h', '4h', '1d'], index=2)
    oi_df = get_open_interest(symbol, oi_interval)
    price_df = get_data_perp(symbol, oi_interval)
    st.subheader('Open Interest Analysis')
    oi_fig = plot_open_interest(oi_df, price_df)
    st.plotly_chart(oi_fig, use_container_width=True)

    delta_interval = st.selectbox('Select Order Flow Delta Interval', ['1m', '5m', '15m', '30m', '1h', '4h', '1d'], index=4)
    delta_df = get_orderflow_delta(symbol, delta_interval)
    st.subheader('Order Flow Delta Analysis')
    delta_fig = plot_delta(delta_df, price_df)
    st.plotly_chart(delta_fig, use_container_width=True)

    if st.button('Generate and Download PDF', key='analysis_download'):
        pdf_file = generate_pdf(symbol, data_dict, pdf_filename=f'analysis_{symbol}.pdf')
        with open(pdf_file, 'rb') as f:
            st.download_button('Download PDF', f, file_name=pdf_file, mime='application/pdf', key='download_pdf')