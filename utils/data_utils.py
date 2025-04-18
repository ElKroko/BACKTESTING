"""
Utilidades para procesamiento y gestión de datos.

Este módulo contiene funciones para obtener datos de APIs,
procesar datos para visualización y crear estructuras HTML dinámicas.
"""
import pandas as pd
import requests
import ta

def get_data(symbol: str, interval: str = '1h') -> pd.DataFrame:
    """
    Obtiene datos OHLCV de Binance para un símbolo y timeframe
    
    Args:
        symbol: Par de trading (ej. 'BTCUSDT')
        interval: Timeframe (ej. '1h', '4h', '1d')
        
    Returns:
        DataFrame con datos de velas
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=500"
    cols = ['Open time','Open','High','Low','Close','Volume','Close time',
            'Quote asset volume','Num trades','Taker buy base','Taker buy quote','Ignore']
    df = pd.DataFrame(requests.get(url).json(), columns=cols)
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    return df[['Open','High','Low','Close','Volume']].astype(float)

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores técnicos comunes para análisis de mercado
    
    Args:
        df: DataFrame con datos OHLCV
        
    Returns:
        DataFrame con indicadores calculados
    """
    df = df.copy()
    df['SMA50']  = df['Close'].rolling(50).mean()
    df['SMA100'] = df['Close'].rolling(100).mean()
    macd = ta.trend.MACD(df['Close'])
    df['MACD']   = macd.macd()
    df['Signal'] = macd.macd_signal()
    df['RSI']    = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
    df['MFI']    = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], 14).money_flow_index()
    return df

def format_backtest_summary(summary: dict, palette: dict) -> str:
    """
    Genera HTML para mostrar un resumen de backtest
    
    Args:
        summary: Diccionario con información del backtest
        palette: Diccionario con colores de la paleta actual
        
    Returns:
        String con HTML formateado
    """
    # Agrega estilos CSS para el resumen
    html = """
    <style>
    .backtest-summary {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        padding: 10px 15px;
        margin-bottom: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .summary-item {
        display: flex;
        align-items: center;
        margin: 5px 10px;
    }
    .summary-icon {
        font-size: 1.2rem;
        margin-right: 8px;
        width: 24px;
        text-align: center;
    }
    .summary-label {
        font-size: 0.8rem;
        opacity: 0.8;
        margin-right: 4px;
    }
    .summary-value {
        font-weight: bold;
        font-size: 0.9rem;
    }
    </style>
    """
    
    # Crea la fila de resumen con los parámetros del último backtest
    html += f"""
    <div class="backtest-summary" style="background-color: {palette['secondary']}; color: {palette['text']}; border: 1px solid {palette['border']};">
        <div class="summary-item">
            <div class="summary-icon">💱</div>
            <div class="summary-label">Symbol:</div>
            <div class="summary-value">{summary['symbol']}</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">⏱️</div>
            <div class="summary-label">Timeframe:</div>
            <div class="summary-value">{summary['interval']}</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">🧠</div>
            <div class="summary-label">Strategy:</div>
            <div class="summary-value">{summary['strategy']}</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">📅</div>
            <div class="summary-label">Period:</div>
            <div class="summary-value">{summary['start_date']} to {summary['end_date']}</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">💰</div>
            <div class="summary-label">Capital:</div>
            <div class="summary-value">${summary['initial_cash']:,.2f}</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">💸</div>
            <div class="summary-label">Commission:</div>
            <div class="summary-value">{summary['commission']}%</div>
        </div>
        <div class="summary-item">
            <div class="summary-icon">🧮</div>
            <div class="summary-label">Slippage:</div>
            <div class="summary-value">{summary['slippage']}%</div>
        </div>
    </div>
    """
    
    return html