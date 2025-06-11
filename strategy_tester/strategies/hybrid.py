"""
Estrategias híbridas que combinan múltiples indicadores
"""

import pandas as pd
import pandas_ta as ta
import numpy as np


def multi_indicator_strategy(df: pd.DataFrame, ema_fast: int = 12, ema_slow: int = 26,
                           rsi_period: int = 14, rsi_oversold: float = 30, rsi_overbought: float = 70,
                           volume_factor: float = 1.5) -> pd.DataFrame:
    """
    Estrategia que combina EMA, RSI y filtro de volumen
    
    Args:
        df: DataFrame con datos OHLCV
        ema_fast: Período EMA rápida
        ema_slow: Período EMA lenta
        rsi_period: Período RSI
        rsi_oversold: Nivel RSI sobreventa
        rsi_overbought: Nivel RSI sobrecompra
        volume_factor: Factor de volumen vs media
        
    Returns:
        DataFrame con señales
    """
    # Calcular indicadores
    df['ema_fast'] = ta.ema(df['close'], length=ema_fast)
    df['ema_slow'] = ta.ema(df['close'], length=ema_slow)
    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
    
    # Filtro de volumen
    volume_ma = ta.sma(df['vol'], length=20)
    df['high_volume'] = df['vol'] > volume_ma * volume_factor
    
    # Condiciones combinadas
    trend_bullish = df['ema_fast'] > df['ema_slow']
    trend_bearish = df['ema_fast'] < df['ema_slow']
    rsi_neutral = (df['rsi'] > rsi_oversold) & (df['rsi'] < rsi_overbought)
    rsi_oversold_condition = df['rsi'] < rsi_oversold
    rsi_overbought_condition = df['rsi'] > rsi_overbought
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: tendencia alcista + RSI no sobrecomprado + alto volumen
    buy_condition = trend_bullish & (~trend_bullish.shift(1).fillna(False)) & \
                   rsi_neutral & df['high_volume']
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: tendencia bajista o RSI sobrecomprado
    sell_condition = (trend_bearish & (~trend_bearish.shift(1).fillna(False))) | \
                    (rsi_overbought_condition & (~rsi_overbought_condition.shift(1).fillna(False)))
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def trend_momentum_hybrid(df: pd.DataFrame, sma_period: int = 50, atr_period: int = 14,
                         atr_multiplier: float = 2, rsi_period: int = 14,
                         momentum_threshold: float = 0.02) -> pd.DataFrame:
    """
    Estrategia híbrida que combina tendencia, volatilidad y momentum
    
    Args:
        df: DataFrame con datos OHLCV
        sma_period: Período SMA para tendencia
        atr_period: Período ATR
        atr_multiplier: Multiplicador ATR
        rsi_period: Período RSI
        momentum_threshold: Umbral de momentum
        
    Returns:
        DataFrame con señales
    """
    # Indicadores de tendencia
    df['sma'] = ta.sma(df['close'], length=sma_period)
    df['price_above_sma'] = df['close'] > df['sma']
    
    # Indicadores de volatilidad
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    df['atr_high'] = df['atr'] > df['atr'].rolling(20).mean() * 1.2
    
    # Indicadores de momentum
    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
    df['momentum'] = df['close'].pct_change(10) * 100
    
    # Conditions
    strong_uptrend = df['price_above_sma'] & (df['rsi'] > 50) & (df['momentum'] > momentum_threshold)
    strong_downtrend = (~df['price_above_sma']) & (df['rsi'] < 50) & (df['momentum'] < -momentum_threshold)
    high_volatility = df['atr_high']
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: tendencia alcista fuerte + alta volatilidad
    buy_condition = strong_uptrend & (~strong_uptrend.shift(1).fillna(False)) & high_volatility
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta: tendencia bajista fuerte
    sell_condition = strong_downtrend & (~strong_downtrend.shift(1).fillna(False))
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def scalping_strategy(df: pd.DataFrame, ema_fast: int = 5, ema_slow: int = 15,
                     stoch_period: int = 14, volume_spike: float = 2.0,
                     profit_target: float = 0.005) -> pd.DataFrame:
    """
    Estrategia de scalping para timeframes cortos
    
    Args:
        df: DataFrame con datos OHLCV
        ema_fast: EMA rápida
        ema_slow: EMA lenta
        stoch_period: Período Stochastic
        volume_spike: Factor de spike de volumen
        profit_target: Target de ganancia
        
    Returns:
        DataFrame con señales
    """
    # Indicadores rápidos
    df['ema_fast'] = ta.ema(df['close'], length=ema_fast)
    df['ema_slow'] = ta.ema(df['close'], length=ema_slow)
    
    # Stochastic para timing
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=stoch_period)
    df['stoch_k'] = stoch[f'STOCHk_{stoch_period}_3_3']
    
    # Filtro de volumen para confirmación
    volume_ma = ta.sma(df['vol'], length=10)
    df['volume_spike'] = df['vol'] > volume_ma * volume_spike
    
    # Volatilidad reciente
    df['recent_volatility'] = df['close'].rolling(5).std() / df['close'].rolling(5).mean()
    
    # Condiciones de entrada
    quick_bullish = (df['ema_fast'] > df['ema_slow']) & (df['stoch_k'] < 80)
    quick_bearish = (df['ema_fast'] < df['ema_slow']) & (df['stoch_k'] > 20)
    
    # Generar señales
    df['signal'] = 0
    
    # Señales de compra: cruce alcista rápido + volumen + momentum
    buy_condition = quick_bullish & (~quick_bullish.shift(1).fillna(False)) & df['volume_spike']
    df.loc[buy_condition, 'signal'] = 1
    
    # Señales de venta rápida: cruce bajista o take profit
    sell_condition = quick_bearish & (~quick_bearish.shift(1).fillna(False))
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def adaptive_strategy(df: pd.DataFrame, base_period: int = 20, volatility_factor: float = 0.1,
                     trend_threshold: float = 0.02) -> pd.DataFrame:
    """
    Estrategia adaptativa que ajusta parámetros según condiciones de mercado
    
    Args:
        df: DataFrame con datos OHLCV
        base_period: Período base para cálculos
        volatility_factor: Factor de ajuste por volatilidad
        trend_threshold: Umbral para detectar tendencia
        
    Returns:
        DataFrame con señales
    """
    # Medir volatilidad del mercado
    df['volatility'] = df['close'].rolling(base_period).std() / df['close'].rolling(base_period).mean()
    df['volatility_regime'] = np.where(df['volatility'] > df['volatility'].rolling(100).mean() * 1.2, 'high', 'low')
    
    # Medir fuerza de tendencia
    df['sma_short'] = ta.sma(df['close'], length=int(base_period/2))
    df['sma_long'] = ta.sma(df['close'], length=base_period)
    df['trend_strength'] = abs(df['sma_short'] - df['sma_long']) / df['sma_long']
    df['trending'] = df['trend_strength'] > trend_threshold
    
    # Ajustar períodos según volatilidad
    df['adaptive_period'] = np.where(df['volatility_regime'] == 'high', 
                                    int(base_period * 0.7),  # Períodos más cortos en alta volatilidad
                                    int(base_period * 1.3))  # Períodos más largos en baja volatilidad
    
    # Calcular indicadores adaptativos (simplificado con períodos fijos por limitaciones)
    df['ema_adaptive'] = ta.ema(df['close'], length=base_period)
    df['rsi_adaptive'] = ta.rsi(df['close'], length=base_period)
    
    # Estrategia adaptativa
    df['signal'] = 0
    
    # En mercados trending: seguir tendencia
    trending_buy = df['trending'] & (df['close'] > df['ema_adaptive']) & \
                   (df['close'].shift(1) <= df['ema_adaptive'].shift(1))
    trending_sell = df['trending'] & (df['close'] < df['ema_adaptive']) & \
                    (df['close'].shift(1) >= df['ema_adaptive'].shift(1))
    
    # En mercados laterales: reversión a la media
    ranging_buy = (~df['trending']) & (df['rsi_adaptive'] < 30) & (df['rsi_adaptive'].shift(1) >= 30)
    ranging_sell = (~df['trending']) & (df['rsi_adaptive'] > 70) & (df['rsi_adaptive'].shift(1) <= 70)
    
    # Combinar señales
    df.loc[trending_buy | ranging_buy, 'signal'] = 1
    df.loc[trending_sell | ranging_sell, 'signal'] = -1
    
    return df


def machine_learning_features_strategy(df: pd.DataFrame, feature_window: int = 20) -> pd.DataFrame:
    """
    Estrategia que genera features para machine learning
    
    Args:
        df: DataFrame con datos OHLCV
        feature_window: Ventana para calcular features
        
    Returns:
        DataFrame con señales y features
    """
    # Features de precio
    df['price_change'] = df['close'].pct_change()
    df['price_volatility'] = df['price_change'].rolling(feature_window).std()
    df['price_momentum'] = df['close'] / df['close'].shift(feature_window) - 1
    
    # Features técnicos
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
    df['bb_position'] = (df['close'] - ta.bbands(df['close'])['BBL_20_2.0']) / \
                       (ta.bbands(df['close'])['BBU_20_2.0'] - ta.bbands(df['close'])['BBL_20_2.0'])
    
    # Features de volumen
    df['volume_sma'] = ta.sma(df['vol'], length=feature_window)
    df['volume_ratio'] = df['vol'] / df['volume_sma']
    df['volume_trend'] = ta.slope(df['vol'], length=feature_window)
    
    # Features de volatilidad
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['atr_ratio'] = df['atr'] / df['close']
    
    # Señales simplificadas basadas en combinación de features
    # (En un caso real, aquí iría un modelo ML entrenado)
    strong_bullish = (df['rsi'] > 50) & (df['macd'] > 0) & (df['bb_position'] > 0.5) & \
                    (df['volume_ratio'] > 1.2) & (df['price_momentum'] > 0.02)
    
    strong_bearish = (df['rsi'] < 50) & (df['macd'] < 0) & (df['bb_position'] < 0.5) & \
                    (df['volume_ratio'] > 1.2) & (df['price_momentum'] < -0.02)
    
    # Generar señales
    df['signal'] = 0
    df.loc[strong_bullish & (~strong_bullish.shift(1).fillna(False)), 'signal'] = 1
    df.loc[strong_bearish & (~strong_bearish.shift(1).fillna(False)), 'signal'] = -1
    
    return df


def risk_managed_strategy(df: pd.DataFrame, base_strategy_func, stop_loss_pct: float = 0.02,
                         take_profit_pct: float = 0.04, **strategy_params) -> pd.DataFrame:
    """
    Wrapper que añade gestión de riesgo a cualquier estrategia
    
    Args:
        df: DataFrame con datos OHLCV
        base_strategy_func: Función de estrategia base
        stop_loss_pct: Porcentaje de stop loss
        take_profit_pct: Porcentaje de take profit
        **strategy_params: Parámetros para la estrategia base
        
    Returns:
        DataFrame con señales y gestión de riesgo
    """
    # Aplicar estrategia base
    df = base_strategy_func(df, **strategy_params)
    
    # Añadir gestión de riesgo
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['take_profit'] = np.nan
    df['risk_managed_signal'] = 0
    
    position = 0
    entry_price = 0
    
    for i, row in df.iterrows():
        if row['signal'] == 1 and position == 0:  # Nueva entrada
            position = 1
            entry_price = row['close']
            df.loc[i, 'entry_price'] = entry_price
            df.loc[i, 'stop_loss'] = entry_price * (1 - stop_loss_pct)
            df.loc[i, 'take_profit'] = entry_price * (1 + take_profit_pct)
            df.loc[i, 'risk_managed_signal'] = 1
            
        elif position == 1:  # En posición
            # Verificar stop loss o take profit
            if row['low'] <= entry_price * (1 - stop_loss_pct):  # Stop loss hit
                df.loc[i, 'risk_managed_signal'] = -1
                position = 0
            elif row['high'] >= entry_price * (1 + take_profit_pct):  # Take profit hit
                df.loc[i, 'risk_managed_signal'] = -1
                position = 0
            elif row['signal'] == -1:  # Señal de salida original
                df.loc[i, 'risk_managed_signal'] = -1
                position = 0
    
    return df
