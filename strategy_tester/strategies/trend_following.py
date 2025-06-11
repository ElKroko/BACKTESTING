"""
Estrategias de seguimiento de tendencia
"""

import pandas as pd
import pandas_ta as ta
import numpy as np


def ema_crossover_strategy(df: pd.DataFrame, fast_ema: int = 12, slow_ema: int = 26, volume_filter: bool = False) -> pd.DataFrame:
    """
    Estrategia de cruce de EMAs
    
    Args:
        df: DataFrame con datos OHLCV
        fast_ema: Período de EMA rápida
        slow_ema: Período de EMA lenta
        volume_filter: Aplicar filtro de volumen
        
    Returns:
        DataFrame con señales
    """
    # Calcular EMAs
    df['ema_fast'] = ta.ema(df['close'], length=fast_ema)
    df['ema_slow'] = ta.ema(df['close'], length=slow_ema)
    
    # Condición básica de tendencia
    bullish_trend = df['ema_fast'] > df['ema_slow']
    
    # Filtro de volumen (opcional)
    if volume_filter:
        volume_ma = ta.sma(df['vol'], length=20)
        high_volume = df['vol'] > volume_ma * 1.2
    else:
        high_volume = True
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: EMA rápida cruza por encima de EMA lenta
    buy_condition = bullish_trend & (~bullish_trend.shift(1).fillna(False)) & high_volume
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: EMA rápida cruza por debajo de EMA lenta
    sell_condition = (~bullish_trend) & bullish_trend.shift(1).fillna(False)
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def sma_crossover_strategy(df: pd.DataFrame, fast_sma: int = 10, slow_sma: int = 30) -> pd.DataFrame:
    """
    Estrategia de cruce de SMAs
    
    Args:
        df: DataFrame con datos OHLCV
        fast_sma: Período de SMA rápida
        slow_sma: Período de SMA lenta
        
    Returns:
        DataFrame con señales
    """
    # Calcular SMAs
    df['sma_fast'] = ta.sma(df['close'], length=fast_sma)
    df['sma_slow'] = ta.sma(df['close'], length=slow_sma)
    
    # Condición de tendencia
    bullish_trend = df['sma_fast'] > df['sma_slow']
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra
    buy_condition = bullish_trend & (~bullish_trend.shift(1).fillna(False))
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta
    sell_condition = (~bullish_trend) & bullish_trend.shift(1).fillna(False)
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def macd_strategy(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9, 
                 histogram_threshold: float = 0) -> pd.DataFrame:
    """
    Estrategia MACD
    
    Args:
        df: DataFrame con datos OHLCV
        fast: Período EMA rápida
        slow: Período EMA lenta
        signal_period: Período de línea de señal
        histogram_threshold: Umbral del histograma
        
    Returns:
        DataFrame con señales
    """
    # Calcular MACD
    macd_data = ta.macd(df['close'], fast=fast, slow=slow, signal=signal_period)
    df['macd'] = macd_data[f'MACD_{fast}_{slow}_{signal_period}']
    df['macd_signal'] = macd_data[f'MACDs_{fast}_{slow}_{signal_period}']
    df['macd_histogram'] = macd_data[f'MACDh_{fast}_{slow}_{signal_period}']
    
    # Condiciones de señal
    macd_bullish = (df['macd'] > df['macd_signal']) & (df['macd_histogram'] > histogram_threshold)
    macd_bearish = (df['macd'] < df['macd_signal']) & (df['macd_histogram'] < histogram_threshold)
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: MACD cruza por encima de línea de señal
    buy_condition = macd_bullish & (~macd_bullish.shift(1).fillna(False))
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: MACD cruza por debajo de línea de señal
    sell_condition = macd_bearish & (~macd_bearish.shift(1).fillna(False))
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def adx_trend_strategy(df: pd.DataFrame, adx_period: int = 14, adx_threshold: float = 25,
                      di_period: int = 14) -> pd.DataFrame:
    """
    Estrategia basada en ADX para detectar tendencias fuertes
    
    Args:
        df: DataFrame con datos OHLCV
        adx_period: Período del ADX
        adx_threshold: Umbral mínimo de ADX para considerar tendencia fuerte
        di_period: Período de los indicadores direccionales
        
    Returns:
        DataFrame con señales
    """
    # Calcular ADX y DI
    adx_data = ta.adx(df['high'], df['low'], df['close'], length=adx_period)
    df['adx'] = adx_data[f'ADX_{adx_period}']
    df['di_plus'] = adx_data[f'DMP_{adx_period}']
    df['di_minus'] = adx_data[f'DMN_{adx_period}']
    
    # Condiciones
    strong_trend = df['adx'] > adx_threshold
    bullish_direction = df['di_plus'] > df['di_minus']
    bearish_direction = df['di_plus'] < df['di_minus']
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: ADX fuerte + DI+ > DI-
    buy_condition = strong_trend & bullish_direction & (
        ~(strong_trend & bullish_direction).shift(1).fillna(False)
    )
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: ADX fuerte + DI- > DI+
    sell_condition = strong_trend & bearish_direction & (
        ~(strong_trend & bearish_direction).shift(1).fillna(False)
    )
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def triple_ema_strategy(df: pd.DataFrame, fast: int = 8, medium: int = 21, slow: int = 55) -> pd.DataFrame:
    """
    Estrategia de triple EMA
    
    Args:
        df: DataFrame con datos OHLCV
        fast: EMA rápida
        medium: EMA media
        slow: EMA lenta
        
    Returns:
        DataFrame con señales
    """
    # Calcular EMAs
    df['ema_fast'] = ta.ema(df['close'], length=fast)
    df['ema_medium'] = ta.ema(df['close'], length=medium)
    df['ema_slow'] = ta.ema(df['close'], length=slow)
    
    # Condiciones de alineación
    bullish_alignment = (df['ema_fast'] > df['ema_medium']) & (df['ema_medium'] > df['ema_slow'])
    bearish_alignment = (df['ema_fast'] < df['ema_medium']) & (df['ema_medium'] < df['ema_slow'])
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: alineación alcista
    buy_condition = bullish_alignment & (~bullish_alignment.shift(1).fillna(False))
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: alineación bajista
    sell_condition = bearish_alignment & (~bearish_alignment.shift(1).fillna(False))
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def parabolic_sar_strategy(df: pd.DataFrame, acceleration: float = 0.02, 
                          maximum: float = 0.2) -> pd.DataFrame:
    """
    Estrategia basada en Parabolic SAR
    
    Args:
        df: DataFrame con datos OHLCV
        acceleration: Factor de aceleración
        maximum: Valor máximo de aceleración
        
    Returns:
        DataFrame con señales
    """
    # Calcular Parabolic SAR
    df['psar'] = ta.psar(df['high'], df['low'], acceleration=acceleration, maximum=maximum)
    
    # Determinar dirección de la tendencia
    df['psar_trend'] = np.where(df['close'] > df['psar'], 1, -1)
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: precio cruza por encima del SAR
    buy_condition = (df['psar_trend'] == 1) & (df['psar_trend'].shift(1) == -1)
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: precio cruza por debajo del SAR
    sell_condition = (df['psar_trend'] == -1) & (df['psar_trend'].shift(1) == 1)
    df.loc[sell_condition, 'signal'] = -1
    
    return df
