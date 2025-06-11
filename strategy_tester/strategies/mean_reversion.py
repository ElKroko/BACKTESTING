"""
Estrategias de reversión a la media
"""

import pandas as pd
import pandas_ta as ta
import numpy as np


def rsi_strategy(df: pd.DataFrame, rsi_period: int = 14, oversold: float = 30, 
                overbought: float = 70, exit_middle: bool = False) -> pd.DataFrame:
    """
    Estrategia basada en RSI
    
    Args:
        df: DataFrame con datos OHLCV
        rsi_period: Período del RSI
        oversold: Nivel de sobreventa
        overbought: Nivel de sobrecompra
        exit_middle: Salir en el nivel medio (50) o en extremos opuestos
        
    Returns:
        DataFrame con señales
    """
    # Calcular RSI
    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
    
    # Generar señales
    df['signal'] = 0
    
    if exit_middle:
        # Entrar en extremos, salir en nivel medio
        # Comprar cuando sale de sobreventa
        buy_condition = (df['rsi'] > oversold) & (df['rsi'].shift(1) <= oversold)
        df.loc[buy_condition, 'signal'] = 1
        
        # Vender cuando llega al nivel medio desde abajo
        sell_condition = (df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)
        df.loc[sell_condition, 'signal'] = -1
        
    else:
        # Estrategia tradicional: entrar en un extremo, salir en el otro
        # Comprar cuando sale de sobreventa
        buy_condition = (df['rsi'] > oversold) & (df['rsi'].shift(1) <= oversold)
        df.loc[buy_condition, 'signal'] = 1
        
        # Vender cuando entra en sobrecompra
        sell_condition = (df['rsi'] > overbought) & (df['rsi'].shift(1) <= overbought)
        df.loc[sell_condition, 'signal'] = -1
    
    return df


def bollinger_bands_strategy(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2,
                           mean_reversion: bool = True) -> pd.DataFrame:
    """
    Estrategia de Bandas de Bollinger
    
    Args:
        df: DataFrame con datos OHLCV
        bb_period: Período de las bandas
        bb_std: Desviación estándar
        mean_reversion: True para reversión, False para breakout
        
    Returns:
        DataFrame con señales
    """
    # Calcular Bollinger Bands
    bb = ta.bbands(df['close'], length=bb_period, std=bb_std)
    df['bb_upper'] = bb[f'BBU_{bb_period}_{bb_std}']
    df['bb_middle'] = bb[f'BBM_{bb_period}_{bb_std}']
    df['bb_lower'] = bb[f'BBL_{bb_period}_{bb_std}']
    
    # Calcular posición relativa en las bandas
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Generar señales
    df['signal'] = 0
    
    if mean_reversion:
        # Estrategia de reversión a la media
        # Comprar cuando toca la banda inferior
        buy_condition = (df['close'] <= df['bb_lower']) & (df['close'].shift(1) > df['bb_lower'].shift(1))
        df.loc[buy_condition, 'signal'] = 1
        
        # Vender cuando toca la banda superior o regresa a la media
        sell_condition = ((df['close'] >= df['bb_upper']) & (df['close'].shift(1) < df['bb_upper'].shift(1))) | \
                        ((df['close'] >= df['bb_middle']) & (df['close'].shift(1) < df['bb_middle'].shift(1)))
        df.loc[sell_condition, 'signal'] = -1
        
    else:
        # Estrategia de breakout
        # Comprar cuando rompe la banda superior
        buy_condition = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
        df.loc[buy_condition, 'signal'] = 1
        
        # Vender cuando rompe la banda inferior
        sell_condition = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
        df.loc[sell_condition, 'signal'] = -1
    
    return df


def stochastic_strategy(df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
                       oversold: float = 20, overbought: float = 80) -> pd.DataFrame:
    """
    Estrategia basada en Stochastic Oscillator
    
    Args:
        df: DataFrame con datos OHLCV
        k_period: Período del %K
        d_period: Período del %D (media móvil del %K)
        oversold: Nivel de sobreventa
        overbought: Nivel de sobrecompra
        
    Returns:
        DataFrame con señales
    """
    # Calcular Stochastic
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=k_period, d=d_period)
    df['stoch_k'] = stoch[f'STOCHk_{k_period}_{d_period}_{d_period}']
    df['stoch_d'] = stoch[f'STOCHd_{k_period}_{d_period}_{d_period}']
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: ambas líneas salen de sobreventa y %K cruza por encima de %D
    buy_condition = (df['stoch_k'] > oversold) & (df['stoch_d'] > oversold) & \
                   (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: ambas líneas entran en sobrecompra o %K cruza por debajo de %D
    sell_condition = ((df['stoch_k'] < overbought) & (df['stoch_d'] < overbought)) | \
                    ((df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)))
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def williams_r_strategy(df: pd.DataFrame, period: int = 14, oversold: float = -20,
                       overbought: float = -80) -> pd.DataFrame:
    """
    Estrategia basada en Williams %R
    
    Args:
        df: DataFrame con datos OHLCV
        period: Período del indicador
        oversold: Nivel de sobreventa (ej: -20)
        overbought: Nivel de sobrecompra (ej: -80)
        
    Returns:
        DataFrame con señales
    """
    # Calcular Williams %R
    df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=period)
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: sale de sobrecompra (valores más negativos)
    buy_condition = (df['williams_r'] > overbought) & (df['williams_r'].shift(1) <= overbought)
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: sale de sobreventa (hacia valores más negativos)
    sell_condition = (df['williams_r'] < oversold) & (df['williams_r'].shift(1) >= oversold)
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def mean_reversion_bands_strategy(df: pd.DataFrame, period: int = 20, multiplier: float = 1.5,
                                 base_indicator: str = 'sma') -> pd.DataFrame:
    """
    Estrategia de reversión a la media usando bandas personalizadas
    
    Args:
        df: DataFrame con datos OHLCV
        period: Período para el cálculo
        multiplier: Multiplicador para las bandas
        base_indicator: 'sma', 'ema', o 'vwap'
        
    Returns:
        DataFrame con señales
    """
    # Calcular indicador base
    if base_indicator == 'sma':
        df['base'] = ta.sma(df['close'], length=period)
    elif base_indicator == 'ema':
        df['base'] = ta.ema(df['close'], length=period)
    elif base_indicator == 'vwap':
        df['base'] = ta.vwap(df['high'], df['low'], df['close'], df['vol'])
    else:
        raise ValueError("base_indicator debe ser 'sma', 'ema' o 'vwap'")
    
    # Calcular desviación
    df['deviation'] = df['close'].rolling(window=period).std()
    
    # Crear bandas
    df['upper_band'] = df['base'] + (df['deviation'] * multiplier)
    df['lower_band'] = df['base'] - (df['deviation'] * multiplier)
    
    # Generar señales
    df['signal'] = 0
    
    # Comprar cuando el precio toca la banda inferior
    buy_condition = (df['close'] <= df['lower_band']) & (df['close'].shift(1) > df['lower_band'].shift(1))
    df.loc[buy_condition, 'signal'] = 1
    
    # Vender cuando el precio regresa a la media o toca la banda superior
    sell_condition = ((df['close'] >= df['base']) & (df['close'].shift(1) < df['base'].shift(1))) | \
                    ((df['close'] >= df['upper_band']) & (df['close'].shift(1) < df['upper_band'].shift(1)))
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def cci_strategy(df: pd.DataFrame, period: int = 20, overbought: float = 100,
                oversold: float = -100) -> pd.DataFrame:
    """
    Estrategia basada en Commodity Channel Index (CCI)
    
    Args:
        df: DataFrame con datos OHLCV
        period: Período del CCI
        overbought: Nivel de sobrecompra
        oversold: Nivel de sobreventa
        
    Returns:
        DataFrame con señales
    """
    # Calcular CCI
    df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=period)
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: CCI sale de sobreventa
    buy_condition = (df['cci'] > oversold) & (df['cci'].shift(1) <= oversold)
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: CCI entra en sobrecompra
    sell_condition = (df['cci'] > overbought) & (df['cci'].shift(1) <= overbought)
    df.loc[sell_condition, 'signal'] = -1
    
    return df
