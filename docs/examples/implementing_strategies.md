# Implementación de Nuevas Estrategias

Este documento muestra paso a paso cómo implementar una nueva estrategia de trading en la plataforma.

## Estructura Básica

Todas las estrategias deben seguir la misma estructura básica y estar definidas en `models/strategies.py`. Una estrategia debe:

1. Aceptar un DataFrame con datos OHLC
2. Aplicar alguna lógica para generar señales de entrada/salida
3. Devolver un DataFrame con las operaciones resultantes

## Ejemplo: Estrategia de Cruce de Medias Móviles Triple

A continuación, se muestra un ejemplo completo de cómo implementar una estrategia que utiliza tres medias móviles para generar señales:

```python
def triple_sma_strategy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Estrategia que utiliza tres medias móviles: corta, media y larga.
    Genera señales cuando la media corta cruza la media y ambas están por encima de la larga (alcista)
    o cuando la media corta cruza hacia abajo la media y ambas están por debajo de la larga (bajista).
    
    Args:
        df: DataFrame con datos OHLC
        params: Diccionario de parámetros con claves 'short_window', 'medium_window', 'long_window'
        
    Returns:
        DataFrame con señales generadas
    """
    # Extraer parámetros o usar valores por defecto
    short_window = params.get('short_window', 5)
    medium_window = params.get('medium_window', 20)
    long_window = params.get('long_window', 50)
    
    # Calcular medias móviles
    df = df.copy()
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_medium'] = df['Close'].rolling(window=medium_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    
    # Inicializar columnas para las señales
    df['signal'] = 0
    df['position'] = 0
    
    # Generar señales
    # Señal de compra: SMA corta cruza hacia arriba SMA media y ambas están por encima de SMA larga
    buy_condition = ((df['SMA_short'] > df['SMA_medium']) & 
                    (df['SMA_short'].shift(1) <= df['SMA_medium'].shift(1)) &
                    (df['SMA_medium'] > df['SMA_long']))
    
    # Señal de venta: SMA corta cruza hacia abajo SMA media y ambas están por debajo de SMA larga
    sell_condition = ((df['SMA_short'] < df['SMA_medium']) & 
                     (df['SMA_short'].shift(1) >= df['SMA_medium'].shift(1)) &
                     (df['SMA_medium'] < df['SMA_long']))
    
    # Asignar valores a las señales
    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1
    
    # Calcular posiciones acumulativas (1 = long, -1 = short, 0 = sin posición)
    df['position'] = df['signal'].cumsum()
    
    # Eliminar posiciones duplicadas (mantener solo la primera entrada y la salida)
    df['position_changed'] = df['position'] != df['position'].shift(1)
    df.loc[~df['position_changed'], 'signal'] = 0
    
    # Crear el DataFrame de trades
    trades = df[df['signal'] != 0].copy()
    trades['type'] = trades['signal'].apply(lambda x: 'buy' if x > 0 else 'sell')
    trades['price'] = df['Close']
    trades['timestamp'] = trades.index
    
    return trades[['timestamp', 'type', 'price', 'signal']]
```

## Integración con el Sistema

Para integrar esta nueva estrategia en la plataforma:

1. Añade la función a `models/strategies.py`
2. Registra la estrategia en la lista de estrategias disponibles en `backtest_tab.py`:

```python
# En backtest_tab.py
AVAILABLE_STRATEGIES = {
    'Triple SMA': strategies.triple_sma_strategy,
    # ... otras estrategias existentes
}
```

3. Define los parámetros configurables en la interfaz:

```python
# En backtest_tab.py, dentro de render_backtest()
if selected_strategy == 'Triple SMA':
    st.sidebar.subheader('Strategy Parameters')
    short_window = st.sidebar.slider('Short SMA Window', 2, 20, 5)
    medium_window = st.sidebar.slider('Medium SMA Window', 10, 50, 20)
    long_window = st.sidebar.slider('Long SMA Window', 30, 200, 50)
    
    strategy_params = {
        'short_window': short_window,
        'medium_window': medium_window,
        'long_window': long_window
    }
```

## Visualización de la Estrategia

Para ayudar a visualizar cómo funciona la estrategia, puedes añadir un gráfico específico:

```python
def plot_triple_sma_strategy(df, trades, params):
    """
    Visualiza la estrategia de triple SMA con las señales generadas
    """
    fig = make_subplots(rows=1, cols=1)
    
    # Añadir gráfico de velas
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Añadir medias móviles
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['SMA_short'],
        line=dict(color='rgba(255, 213, 79, 0.7)', width=1.5),
        name=f'SMA {params["short_window"]}'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['SMA_medium'],
        line=dict(color='rgba(38, 166, 154, 0.7)', width=1.5),
        name=f'SMA {params["medium_window"]}'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df['SMA_long'],
        line=dict(color='rgba(239, 83, 80, 0.7)', width=1.5),
        name=f'SMA {params["long_window"]}'
    ))
    
    # Añadir señales de compra/venta
    buys = trades[trades['type'] == 'buy']
    sells = trades[trades['type'] == 'sell']
    
    fig.add_trace(go.Scatter(
        x=buys['timestamp'],
        y=buys['price'],
        mode='markers',
        marker=dict(
            symbol='triangle-up',
            size=12,
            color='green',
            line=dict(width=1, color='darkgreen')
        ),
        name='Buy Signal'
    ))
    
    fig.add_trace(go.Scatter(
        x=sells['timestamp'],
        y=sells['price'],
        mode='markers',
        marker=dict(
            symbol='triangle-down',
            size=12,
            color='red',
            line=dict(width=1, color='darkred')
        ),
        name='Sell Signal'
    ))
    
    fig.update_layout(
        title=f'Triple SMA Strategy ({params["short_window"]}, {params["medium_window"]}, {params["long_window"]})',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        xaxis_rangeslider_visible=False
    )
    
    return fig
```

## Evaluación del Rendimiento

Puedes evaluar el rendimiento de tu nueva estrategia utilizando el módulo de backtesting:

1. Selecciona tu estrategia en la interfaz
2. Ajusta los parámetros según sea necesario
3. Ejecuta el backtest para obtener métricas de rendimiento
4. Compara los resultados con otras estrategias

Recuerda que una buena estrategia debe tener:
- Un Profit Factor > 1.5
- Un porcentaje de operaciones ganadoras > 50%
- Un drawdown máximo controlado
- Un ratio de Sharpe > 1.0 para un buen rendimiento ajustado al riesgo