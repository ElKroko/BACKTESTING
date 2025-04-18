# strategies.py
import pandas as pd
import ta

# --- Strategy Definitions for Backtesting ---

def ma_crossover(data: pd.DataFrame, short_window: int = 50, long_window: int = 100) -> list:
    """
    Moving Average Crossover Strategy:
    - Buy when short SMA crosses above long SMA
    - Sell when short SMA crosses below long SMA
    Returns list of trades with timestamp, action, price
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
    Bollinger Bands Breakout:
    - Buy when price closes above upper band
    - Sell when price closes below lower band
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
    RSI Reversion Strategy:
    - Buy when RSI < oversold threshold
    - Sell when RSI > overbought threshold
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
    MACD Momentum Strategy:
    - Buy when MACD line crosses above signal line
    - Sell when MACD line crosses below signal line
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
    Support/Resistance Breakout Strategy:
    - Buy on breakout above the rolling high
    - Sell on breakdown below the rolling low
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
