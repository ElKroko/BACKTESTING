"""
Estrategias de trading para el backtesting.

Este módulo contiene varias estrategias de trading que pueden ser utilizadas
en el sistema de backtesting. Cada estrategia recibe un DataFrame con datos OHLC
y devuelve una lista de señales de trading con timestamp, acción y precio.
"""
import pandas as pd
import ta

def ma_crossover(data: pd.DataFrame, short_window: int = 50, long_window: int = 100) -> list:
    """
    Estrategia de cruce de medias móviles:
    - Compra cuando la SMA corta cruza por encima de la SMA larga
    - Vende cuando la SMA corta cruza por debajo de la SMA larga
    
    Args:
        data: DataFrame con datos OHLC
        short_window: Ventana de la media móvil corta
        long_window: Ventana de la media móvil larga
    
    Returns:
        Lista de señales con timestamp, acción y precio
    """
    df = data.copy()
    df['SMA_short'] = df['Close'].rolling(short_window).mean()
    df['SMA_long']  = df['Close'].rolling(long_window).mean()

    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    signals.loc[df.index[short_window:], 'signal'] = (
        df['SMA_short'][short_window:] > df['SMA_long'][short_window:]
    ).astype(int)
    signals['positions'] = signals['signal'].diff()

    trades = []
    for ts, pos in signals['positions'].items():
        if pos == 1:
            trades.append({'timestamp': ts, 'action': 'buy',  'price': df.at[ts, 'Close']})
        elif pos == -1:
            trades.append({'timestamp': ts, 'action': 'sell', 'price': df.at[ts, 'Close']})
    return trades


def bollinger_breakout(data: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> list:
    """
    Estrategia de ruptura de bandas de Bollinger:
    - Compra cuando el precio cierra por encima de la banda superior
    - Vende cuando el precio cierra por debajo de la banda inferior
    
    Args:
        data: DataFrame con datos OHLC
        window: Ventana para el cálculo de las bandas
        n_std: Número de desviaciones estándar para las bandas
    
    Returns:
        Lista de señales con timestamp, acción y precio
    """
    df = data.copy()
    mean = df['Close'].rolling(window).mean()
    std  = df['Close'].rolling(window).std()
    df['upper_band'] = mean + (n_std * std)
    df['lower_band'] = mean - (n_std * std)

    trades = []
    for ts, row in df.iterrows():
        price = row['Close']
        if price > row['upper_band']:
            trades.append({'timestamp': ts, 'action': 'buy',  'price': price})
        elif price < row['lower_band']:
            trades.append({'timestamp': ts, 'action': 'sell', 'price': price})
    return trades


def rsi_reversion(data: pd.DataFrame, period: int = 14, overbought: int = 70, oversold: int = 30) -> list:
    """
    Estrategia de reversión basada en RSI:
    - Compra cuando RSI < umbral de sobreventa
    - Vende cuando RSI > umbral de sobrecompra
    
    Args:
        data: DataFrame con datos OHLC
        period: Período para el cálculo del RSI
        overbought: Umbral de sobrecompra
        oversold: Umbral de sobreventa
    
    Returns:
        Lista de señales con timestamp, acción y precio
    """
    df = data.copy()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], period).rsi()

    trades = []
    for ts, rsi in df['RSI'].items():
        price = df.at[ts, 'Close']
        if rsi < oversold:
            trades.append({'timestamp': ts, 'action': 'buy',  'price': price})
        elif rsi > overbought:
            trades.append({'timestamp': ts, 'action': 'sell', 'price': price})
    return trades


def macd_momentum(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> list:
    """
    Estrategia de momento basada en MACD:
    - Compra cuando la línea MACD cruza por encima de la línea de señal
    - Vende cuando la línea MACD cruza por debajo de la línea de señal
    
    Args:
        data: DataFrame con datos OHLC
        fast: Período rápido para el cálculo del MACD
        slow: Período lento para el cálculo del MACD
        signal: Período para la línea de señal
    
    Returns:
        Lista de señales con timestamp, acción y precio
    """
    df = data.copy()
    macd = ta.trend.MACD(df['Close'], fast, slow, signal)
    df['macd']        = macd.macd()
    df['signal_line'] = macd.macd_signal()

    trades = []
    prev_state = None
    for ts, (m_val, s_val) in df[['macd', 'signal_line']].iterrows():
        curr_state = 'buy' if m_val > s_val else 'sell'
        if prev_state and curr_state != prev_state:
            trades.append({'timestamp': ts, 'action': curr_state, 'price': df.at[ts, 'Close']})
        prev_state = curr_state
    return trades


def sr_breakout(data: pd.DataFrame, window: int = 20) -> list:
    """
    Estrategia de ruptura de soporte/resistencia:
    - Compra en ruptura por encima del máximo móvil
    - Vende en ruptura por debajo del mínimo móvil
    
    Args:
        data: DataFrame con datos OHLC
        window: Ventana para los niveles de soporte/resistencia
    
    Returns:
        Lista de señales con timestamp, acción y precio
    """
    df = data.copy()
    highs = df['High'].rolling(window).max()
    lows  = df['Low'].rolling(window).min()

    trades = []
    for ts in df.index[window:]:
        price     = df.at[ts, 'Close']
        prev_high = highs.loc[ts - pd.Timedelta(window, unit='m')]
        prev_low  = lows.loc[ts - pd.Timedelta(window, unit='m')]
        if price > prev_high:
            trades.append({'timestamp': ts, 'action': 'buy',  'price': price})
        elif price < prev_low:
            trades.append({'timestamp': ts, 'action': 'sell', 'price': price})
    return trades


# Funciones adicionales que podrían ser útiles en el futuro

def volatility_breakout(data: pd.DataFrame, window: int = 10, multiplier: float = 1.5) -> list:
    """
    Estrategia de ruptura de volatilidad:
    - Compra cuando el precio rompe hacia arriba con mayor volatilidad
    - Vende cuando el precio rompe hacia abajo con mayor volatilidad
    
    Args:
        data: DataFrame con datos OHLC
        window: Ventana para el cálculo de la volatilidad
        multiplier: Multiplicador para determinar la ruptura
    
    Returns:
        Lista de señales con timestamp, acción y precio
    """
    df = data.copy()
    df['ATR'] = ta.volatility.AverageTrueRange(
        df['High'], df['Low'], df['Close'], window=window
    ).average_true_range()
    
    df['range_high'] = df['High'].rolling(window).max()
    df['range_low'] = df['Low'].rolling(window).min()
    
    trades = []
    for i in range(window, len(df)):
        curr_candle = df.iloc[i]
        prev_candle = df.iloc[i-1]
        
        # Comprar si el precio rompe hacia arriba con alta volatilidad
        if (curr_candle['High'] > prev_candle['range_high'] and 
            curr_candle['High'] - curr_candle['Low'] > multiplier * curr_candle['ATR']):
            trades.append({
                'timestamp': df.index[i], 
                'action': 'buy', 
                'price': curr_candle['Close']
            })
        
        # Vender si el precio rompe hacia abajo con alta volatilidad
        elif (curr_candle['Low'] < prev_candle['range_low'] and
              curr_candle['High'] - curr_candle['Low'] > multiplier * curr_candle['ATR']):
            trades.append({
                'timestamp': df.index[i], 
                'action': 'sell', 
                'price': curr_candle['Close']
            })
    
    return trades