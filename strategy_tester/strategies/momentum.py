"""
Estrategias basadas en momentum
"""

import pandas as pd
import pandas_ta as ta
import numpy as np


def momentum_strategy(df: pd.DataFrame, momentum_period: int = 10, threshold: float = 0.02,
                     smoothing_period: int = 3) -> pd.DataFrame:
    """
    Estrategia basada en momentum de precios
    
    Args:
        df: DataFrame con datos OHLCV
        momentum_period: Período para calcular momentum
        threshold: Umbral mínimo de momentum para generar señal
        smoothing_period: Período para suavizar señales
        
    Returns:
        DataFrame con señales
    """
    # Calcular momentum
    df['momentum'] = ta.mom(df['close'], length=momentum_period)
    df['momentum_pct'] = df['momentum'] / df['close'].shift(momentum_period) * 100
    
    # Suavizar momentum
    df['momentum_smooth'] = ta.sma(df['momentum_pct'], length=smoothing_period)
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: momentum positivo y fuerte
    buy_condition = (df['momentum_smooth'] > threshold) & (df['momentum_smooth'].shift(1) <= threshold)
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: momentum se debilita o se vuelve negativo
    sell_condition = (df['momentum_smooth'] < -threshold) & (df['momentum_smooth'].shift(1) >= -threshold)
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def price_channel_breakout(df: pd.DataFrame, period: int = 20, breakout_threshold: float = 0.01) -> pd.DataFrame:
    """
    Estrategia de ruptura de canal de precios
    
    Args:
        df: DataFrame con datos OHLCV
        period: Período para calcular el canal
        breakout_threshold: Umbral mínimo de ruptura (como porcentaje)
        
    Returns:
        DataFrame con señales
    """
    # Calcular canal de precios
    df['channel_high'] = df['high'].rolling(window=period).max()
    df['channel_low'] = df['low'].rolling(window=period).min()
    df['channel_middle'] = (df['channel_high'] + df['channel_low']) / 2
    
    # Calcular ancho del canal
    df['channel_width'] = (df['channel_high'] - df['channel_low']) / df['channel_middle']
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: ruptura alcista del canal superior
    breakout_up = (df['close'] > df['channel_high'].shift(1) * (1 + breakout_threshold)) & \
                  (df['close'].shift(1) <= df['channel_high'].shift(1))
    df.loc[breakout_up, 'signal'] = 1
    
    # Señales de venta: ruptura bajista del canal inferior
    breakout_down = (df['close'] < df['channel_low'].shift(1) * (1 - breakout_threshold)) & \
                    (df['close'].shift(1) >= df['channel_low'].shift(1))
    df.loc[breakout_down, 'signal'] = -1
    
    return df


def volatility_breakout(df: pd.DataFrame, atr_period: int = 14, atr_multiplier: float = 2.0,
                       volume_confirmation: bool = True) -> pd.DataFrame:
    """
    Estrategia de ruptura basada en volatilidad (ATR)
    
    Args:
        df: DataFrame con datos OHLCV
        atr_period: Período para calcular ATR
        atr_multiplier: Multiplicador del ATR para las bandas
        volume_confirmation: Requerir confirmación de volumen
        
    Returns:
        DataFrame con señales
    """
    # Calcular ATR
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    
    # Calcular bandas de volatilidad
    df['upper_band'] = df['close'].shift(1) + (df['atr'] * atr_multiplier)
    df['lower_band'] = df['close'].shift(1) - (df['atr'] * atr_multiplier)
    
    # Filtro de volumen
    if volume_confirmation:
        volume_ma = ta.sma(df['vol'], length=20)
        high_volume = df['vol'] > volume_ma * 1.5
    else:
        high_volume = True
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: precio rompe banda superior
    buy_condition = (df['close'] > df['upper_band']) & high_volume
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: precio rompe banda inferior
    sell_condition = (df['close'] < df['lower_band']) & high_volume
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def rate_of_change_strategy(df: pd.DataFrame, roc_period: int = 12, threshold: float = 5,
                           smoothing: int = 3) -> pd.DataFrame:
    """
    Estrategia basada en Rate of Change (ROC)
    
    Args:
        df: DataFrame con datos OHLCV
        roc_period: Período para calcular ROC
        threshold: Umbral para generar señales
        smoothing: Período de suavizado
        
    Returns:
        DataFrame con señales
    """
    # Calcular Rate of Change
    df['roc'] = ta.roc(df['close'], length=roc_period)
    df['roc_smooth'] = ta.sma(df['roc'], length=smoothing)
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: ROC fuertemente positivo
    buy_condition = (df['roc_smooth'] > threshold) & (df['roc_smooth'].shift(1) <= threshold)
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: ROC fuertemente negativo
    sell_condition = (df['roc_smooth'] < -threshold) & (df['roc_smooth'].shift(1) >= -threshold)
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def aroon_strategy(df: pd.DataFrame, period: int = 25, threshold: float = 70) -> pd.DataFrame:
    """
    Estrategia basada en indicador Aroon
    
    Args:
        df: DataFrame con datos OHLCV
        period: Período para Aroon
        threshold: Umbral para considerar tendencia fuerte
        
    Returns:
        DataFrame con señales
    """
    # Calcular Aroon
    aroon = ta.aroon(df['high'], df['low'], length=period)
    df['aroon_up'] = aroon[f'AROONU_{period}']
    df['aroon_down'] = aroon[f'AROOND_{period}']
    
    # Calcular oscilador Aroon
    df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: Aroon Up fuerte y dominante
    buy_condition = (df['aroon_up'] > threshold) & (df['aroon_up'] > df['aroon_down']) & \
                   (df['aroon_osc'] > 0) & (df['aroon_osc'].shift(1) <= 0)
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: Aroon Down fuerte y dominante
    sell_condition = (df['aroon_down'] > threshold) & (df['aroon_down'] > df['aroon_up']) & \
                    (df['aroon_osc'] < 0) & (df['aroon_osc'].shift(1) >= 0)
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def momentum_divergence_strategy(df: pd.DataFrame, momentum_period: int = 14, 
                                lookback_period: int = 5) -> pd.DataFrame:
    """
    Estrategia basada en divergencias de momentum
    
    Args:
        df: DataFrame con datos OHLCV
        momentum_period: Período para calcular momentum
        lookback_period: Período para detectar divergencias
        
    Returns:
        DataFrame con señales
    """
    # Calcular momentum e RSI
    df['momentum'] = ta.mom(df['close'], length=momentum_period)
    df['rsi'] = ta.rsi(df['close'], length=momentum_period)
    
    # Encontrar picos y valles en precio
    df['price_high'] = df['high'].rolling(window=lookback_period, center=True).max() == df['high']
    df['price_low'] = df['low'].rolling(window=lookback_period, center=True).min() == df['low']
    
    # Encontrar picos y valles en momentum
    df['momentum_high'] = df['momentum'].rolling(window=lookback_period, center=True).max() == df['momentum']
    df['momentum_low'] = df['momentum'].rolling(window=lookback_period, center=True).min() == df['momentum']
    
    # Detectar divergencias (simplificado)
    # Divergencia alcista: precio hace mínimos más bajos, momentum hace mínimos más altos
    bullish_div = df['price_low'] & (df['momentum'] > df['momentum'].shift(lookback_period))
    
    # Divergencia bajista: precio hace máximos más altos, momentum hace máximos más bajos
    bearish_div = df['price_high'] & (df['momentum'] < df['momentum'].shift(lookback_period))
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: divergencia alcista
    df.loc[bullish_div, 'signal'] = 1
    
    # Señales de venta: divergencia bajista
    df.loc[bearish_div, 'signal'] = -1
    
    return df


def squeeze_momentum_strategy(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2,
                             kc_period: int = 20, kc_multiplier: float = 1.5,
                             momentum_period: int = 12) -> pd.DataFrame:
    """
    Estrategia TTM Squeeze (Bollinger Bands vs Keltner Channels)
    
    Args:
        df: DataFrame con datos OHLCV
        bb_period: Período de Bollinger Bands
        bb_std: Desviación estándar de BB
        kc_period: Período de Keltner Channels
        kc_multiplier: Multiplicador de KC
        momentum_period: Período para momentum
        
    Returns:
        DataFrame con señales
    """
    # Calcular Bollinger Bands
    bb = ta.bbands(df['close'], length=bb_period, std=bb_std)
    df['bb_upper'] = bb[f'BBU_{bb_period}_{bb_std}']
    df['bb_lower'] = bb[f'BBL_{bb_period}_{bb_std}']
    
    # Calcular Keltner Channels
    kc = ta.kc(df['high'], df['low'], df['close'], length=kc_period, scalar=kc_multiplier)
    df['kc_upper'] = kc[f'KCUe_{kc_period}_{kc_multiplier}']
    df['kc_lower'] = kc[f'KCLe_{kc_period}_{kc_multiplier}']
    
    # Detectar squeeze: BB dentro de KC
    df['squeeze_on'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
    
    # Calcular momentum
    df['momentum'] = ta.mom(df['close'], length=momentum_period)
    df['momentum_smooth'] = ta.sma(df['momentum'], length=3)
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: squeeze termina con momentum positivo
    buy_condition = (~df['squeeze_on']) & df['squeeze_on'].shift(1) & (df['momentum_smooth'] > 0)
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: squeeze termina con momentum negativo
    sell_condition = (~df['squeeze_on']) & df['squeeze_on'].shift(1) & (df['momentum_smooth'] < 0)
    df.loc[sell_condition, 'signal'] = -1
    
    return df
