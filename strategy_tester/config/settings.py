"""
Configuración del Strategy Tester
"""

# Símbolos por defecto para probar
DEFAULT_SYMBOLS = [
    'BTC/USDT',
    'ETH/USDT', 
    'ADA/USDT',
    'SOL/USDT',
    'MATIC/USDT'
]

# Marcos temporales disponibles
DEFAULT_TIMEFRAMES = [
    '5m',
    '15m',
    '1h', 
    '4h',
    '1d'
]

# Períodos de tiempo para backtesting
DEFAULT_PERIODS = {
    '5m': '2024-01-01T00:00:00Z',
    '15m': '2023-06-01T00:00:00Z',
    '1h': '2023-01-01T00:00:00Z',
    '4h': '2022-01-01T00:00:00Z',
    '1d': '2020-01-01T00:00:00Z'
}

# Configuración de trading
DEFAULT_CAPITAL = 10000  # Capital inicial en USD
DEFAULT_COMMISSION = 0.001  # 0.1% de comisión
DEFAULT_SLIPPAGE = 0.0005  # 0.05% de slippage

# Configuración de datos
DATA_SOURCE = 'binance'  # Exchange por defecto
MAX_CANDLES = 1000  # Máximo número de velas a descargar

# Configuración de resultados
RESULTS_DIR = 'results'
CHARTS_DIR = 'results/charts'
DATA_DIR = 'data'

# Configuración de gráficos
PLOT_SETTINGS = {
    'figsize': (15, 10),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'colors': {
        'entry': 'green',
        'exit': 'red',
        'price': 'blue',
        'sma': 'orange',
        'ema': 'purple'
    }
}
