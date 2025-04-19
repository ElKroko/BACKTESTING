# Módulo de Análisis Técnico

## Descripción

El módulo de análisis técnico (`analysis_tab.py`) proporciona herramientas avanzadas para el análisis de criptomonedas, incluyendo indicadores técnicos tradicionales, detección de niveles de soporte/resistencia y métricas especializadas de mercados de derivados.

## Características Principales

- **Visualización con TradingView**: Gráficos interactivos en múltiples timeframes
- **Análisis Plotly**: Gráficos personalizados para análisis detallado
- **Métricas de Derivados**: Funding Rate, Open Interest y Order Flow Delta
- **Indicadores Técnicos**: SMA, MACD, RSI, MFI
- **Niveles Clave**: Soporte, resistencia y niveles pivote
- **Generación de PDF**: Reportes descargables con análisis completo

## Estructura de Interfaz

La interfaz del módulo de análisis está diseñada con un layout 60/40:
- **Panel Izquierdo (60%)**: Gráfico principal de TradingView e indicadores clave
- **Panel Derecho (40%)**: Gráficos por timeframe (semanal, diario, horario) en pestañas
- **Sección Inferior**: Métricas de derivados organizadas en pestañas

## Métricas de Derivados

### Funding Rate

El funding rate es un mecanismo importante en contratos perpetuos que refleja el sentimiento del mercado:

- **Funding Positivo**: Los posiciones largas pagan a las cortas (sentimiento alcista)
- **Funding Negativo**: Las posiciones cortas pagan a las largas (sentimiento bajista)

```python
# Ejemplo de cómo obtener el funding rate
def get_funding_rate(symbol: str) -> pd.DataFrame:
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=500"
    response = requests.get(url)
    if response.status_code != 200:
        return pd.DataFrame()
        
    fr_df = pd.DataFrame(response.json())
    fr_df['fundingTime'] = pd.to_datetime(fr_df['fundingTime'], unit='ms')
    fr_df.set_index('fundingTime', inplace=True)
    fr_df['fundingRate'] = fr_df['fundingRate'].astype(float)
    return fr_df
```

### Open Interest

El Open Interest muestra el volumen total de contratos abiertos, proporcionando información sobre la liquidez del mercado y la convicción de los traders:

```python
# Ejemplo de visualización de Open Interest con Plotly
def plot_open_interest(oi_df: pd.DataFrame, price_df: pd.DataFrame) -> go.Figure:
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
    fig.add_trace(
        go.Bar(
            x=oi_df.index,
            y=oi_df['openInterest'],
            name="Open Interest",
        ),
        row=2, col=1
    )
    
    return fig
```

### Order Flow Delta

El Order Flow Delta mide la diferencia entre volumen comprador y vendedor, revelando la presión dominante en el mercado:

```python
# Ejemplo de cálculo de Delta
df_trades['delta'] = df_trades.apply(
    lambda r: r['qty'] if r['m'] == False else -r['qty'], 
    axis=1
)

# Acumulando por intervalo
df_resampled = df_trades.resample(resample_interval).agg({
    'price': 'mean',
    'qty': 'sum',
    'delta': 'sum',
    'total_volume': 'sum'
})
```

## Uso Avanzado

### Personalización de Gráficos

Los gráficos de Plotly utilizan una paleta de colores consistente definida en el módulo:

```python
PALETTE = {
    'template': 'plotly_dark',
    'green': '#26a69a',
    'red': '#ef5350',
    'neutral': '#ffd54f'
}
```

### Extendiendo los Indicadores

Para añadir un nuevo indicador técnico:

1. Ampliar la función `calculate_indicators()` en `data_utils.py`
2. Actualizar la visualización en `render_analysis()`
3. Ajustar la función `indicator_recommendation()` para proporcionar interpretación

## Referencias Adicionales

- [Documentación de Plotly](https://plotly.com/python/)
- [API de Binance](https://binance-docs.github.io/apidocs/)