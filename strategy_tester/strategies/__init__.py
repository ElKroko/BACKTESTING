"""
Biblioteca de Estrategias de Trading
"""

from .trend_following import (
    ema_crossover_strategy,
    sma_crossover_strategy,
    macd_strategy,
    adx_trend_strategy
)

from .mean_reversion import (
    rsi_strategy,
    bollinger_bands_strategy,
    stochastic_strategy,
    williams_r_strategy
)

from .momentum import (
    momentum_strategy,
    price_channel_breakout,
    volatility_breakout
)

from .hybrid import (
    multi_indicator_strategy,
    trend_momentum_hybrid,
    scalping_strategy
)

# Diccionario con todas las estrategias disponibles
AVAILABLE_STRATEGIES = {
    # Trend Following
    'EMA_Crossover': ema_crossover_strategy,
    'SMA_Crossover': sma_crossover_strategy,
    'MACD': macd_strategy,
    'ADX_Trend': adx_trend_strategy,
    
    # Mean Reversion
    'RSI': rsi_strategy,
    'Bollinger_Bands': bollinger_bands_strategy,
    'Stochastic': stochastic_strategy,
    'Williams_R': williams_r_strategy,
    
    # Momentum
    'Momentum': momentum_strategy,
    'Price_Channel': price_channel_breakout,
    'Volatility_Breakout': volatility_breakout,
    
    # Hybrid
    'Multi_Indicator': multi_indicator_strategy,
    'Trend_Momentum': trend_momentum_hybrid,
    'Scalping': scalping_strategy
}

def get_strategy(name: str):
    """Obtener estrategia por nombre"""
    if name not in AVAILABLE_STRATEGIES:
        available = list(AVAILABLE_STRATEGIES.keys())
        raise ValueError(f"Estrategia '{name}' no encontrada. Disponibles: {available}")
    
    return AVAILABLE_STRATEGIES[name]

def list_strategies() -> list:
    """Listar todas las estrategias disponibles"""
    return list(AVAILABLE_STRATEGIES.keys())

__all__ = [
    'ema_crossover_strategy', 'sma_crossover_strategy', 'macd_strategy', 'adx_trend_strategy',
    'rsi_strategy', 'bollinger_bands_strategy', 'stochastic_strategy', 'williams_r_strategy',
    'momentum_strategy', 'price_channel_breakout', 'volatility_breakout',
    'multi_indicator_strategy', 'trend_momentum_hybrid', 'scalping_strategy',
    'AVAILABLE_STRATEGIES', 'get_strategy', 'list_strategies'
]
