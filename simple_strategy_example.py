"""
Ejemplo simple de cómo usar el StrategyTester
"""

from strategy_tester import StrategyTester, ema_crossover_strategy, rsi_strategy
import pandas as pd
import pandas_ta as ta


# ========================= CREAR TU PROPIA ESTRATEGIA =========================

def mi_estrategia_personalizada(df, sma_fast=10, sma_slow=30, rsi_period=14, rsi_threshold=50):
    """
    Ejemplo de estrategia personalizada que combina SMA y RSI
    
    Parámetros:
        sma_fast: Periodo de SMA rápida
        sma_slow: Periodo de SMA lenta
        rsi_period: Periodo del RSI
        rsi_threshold: Umbral del RSI para filtrar señales
    """
    # Calcular indicadores
    df['sma_fast'] = ta.sma(df['close'], length=sma_fast)
    df['sma_slow'] = ta.sma(df['close'], length=sma_slow)
    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
    
    # Condiciones de entrada: SMA rápida cruza por encima de SMA lenta Y RSI > threshold
    df['long_condition'] = (df['sma_fast'] > df['sma_slow']) & (df['rsi'] > rsi_threshold)
    
    # Condiciones de salida: SMA rápida cruza por debajo de SMA lenta O RSI < threshold
    df['short_condition'] = (df['sma_fast'] < df['sma_slow']) | (df['rsi'] < rsi_threshold)
    
    # Generar señales
    df['signal'] = 0
    
    # Detectar cruces
    long_signals = (df['long_condition'] == True) & (df['long_condition'].shift(1) == False)
    short_signals = (df['short_condition'] == True) & (df['short_condition'].shift(1) == False)
    
    df.loc[long_signals, 'signal'] = 1   # Señal de compra
    df.loc[short_signals, 'signal'] = -1  # Señal de venta
    
    return df


def estrategia_volatilidad(df, atr_period=14, volatility_threshold=0.02):
    """
    Estrategia basada en volatilidad usando ATR
    """
    # Calcular ATR (Average True Range)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    df['atr_normalized'] = df['atr'] / df['close']  # ATR normalizado por precio
    
    # Calcular media móvil del precio
    df['sma_20'] = ta.sma(df['close'], length=20)
    
    # Señales basadas en volatilidad alta
    df['high_volatility'] = df['atr_normalized'] > volatility_threshold
    df['price_above_sma'] = df['close'] > df['sma_20']
    
    # Generar señales
    df['signal'] = 0
    
    # Comprar cuando hay alta volatilidad y precio sobre SMA
    buy_condition = df['high_volatility'] & df['price_above_sma']
    buy_signals = buy_condition & (~buy_condition.shift(1).fillna(False))
    
    # Vender cuando volatilidad baja o precio bajo SMA
    sell_condition = (~df['high_volatility']) | (~df['price_above_sma'])
    sell_signals = sell_condition & (~sell_condition.shift(1).fillna(False))
    
    df.loc[buy_signals, 'signal'] = 1
    df.loc[sell_signals, 'signal'] = -1
    
    return df


# ========================= FUNCIÓN PRINCIPAL =========================

def main():
    print("🚀 INICIANDO PRUEBA DE ESTRATEGIAS DE TRADING")
    print("=" * 60)
    
    # Configurar el tester
    tester = StrategyTester(
        symbol='BTC/USDT',
        timeframe='1h',  # Puedes cambiar a '5m', '15m', '1h', '4h', '1d'
        start_date='2023-06-01T00:00:00Z'  # Fecha de inicio
    )
    
    # Descargar datos
    print("📊 Descargando datos...")
    tester.fetch_data()
    
    # Definir las estrategias que quieres probar
    mis_estrategias = {
        
        # Estrategia original (EMA)
        'EMA_Original': {
            'func': ema_crossover_strategy,
            'params': {'fast_ema': 9, 'slow_ema': 21}
        },
        
        # Variaciones de EMA
        'EMA_Rapida': {
            'func': ema_crossover_strategy,
            'params': {'fast_ema': 5, 'slow_ema': 15}
        },
        
        'EMA_Lenta': {
            'func': ema_crossover_strategy,
            'params': {'fast_ema': 12, 'slow_ema': 26}
        },
        
        # Estrategia RSI
        'RSI_Conservador': {
            'func': rsi_strategy,
            'params': {'rsi_period': 14, 'oversold': 30, 'overbought': 70}
        },
        
        'RSI_Agresivo': {
            'func': rsi_strategy,
            'params': {'rsi_period': 14, 'oversold': 20, 'overbought': 80}
        },
        
        # Tu estrategia personalizada
        'Mi_Estrategia_v1': {
            'func': mi_estrategia_personalizada,
            'params': {'sma_fast': 10, 'sma_slow': 30, 'rsi_period': 14, 'rsi_threshold': 50}
        },
        
        'Mi_Estrategia_v2': {
            'func': mi_estrategia_personalizada,
            'params': {'sma_fast': 8, 'sma_slow': 25, 'rsi_period': 10, 'rsi_threshold': 60}
        },
        
        # Estrategia de volatilidad
        'Volatilidad_Standard': {
            'func': estrategia_volatilidad,
            'params': {'atr_period': 14, 'volatility_threshold': 0.02}
        },
        
        'Volatilidad_Agresiva': {
            'func': estrategia_volatilidad,
            'params': {'atr_period': 10, 'volatility_threshold': 0.015}
        }
    }
    
    # Probar todas las estrategias
    print("\n🔄 Probando estrategias...")
    resultados = tester.compare_strategies(mis_estrategias)
    
    # Mostrar resultados ordenados por rentabilidad
    print("\n📈 RANKING DE ESTRATEGIAS (por rentabilidad)")
    print("=" * 80)
    
    ranking = resultados.sort_values('total_return_pct', ascending=False)
    
    for i, (_, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i:2d}. {row['strategy_name']:20} | "
              f"Retorno: {row['total_return_pct']:6.2f}% | "
              f"Operaciones: {row['total_trades']:3d} | "
              f"Éxito: {row['win_rate']:5.1f}% | "
              f"Sharpe: {row['sharpe_ratio']:5.2f}")
    
    # Guardar resultados detallados
    resultados.to_csv('comparacion_estrategias.csv', index=False)
    print(f"\n💾 Resultados guardados en: comparacion_estrategias.csv")
    
    # Graficar las 3 mejores estrategias
    print(f"\n📊 Generando gráficos de las 3 mejores estrategias...")
    top_3 = ranking.head(3)['strategy_name'].values
    
    for i, estrategia in enumerate(top_3, 1):
        print(f"   {i}. Graficando: {estrategia}")
        try:
            tester.plot_strategy_results(estrategia, save_plot=True)
        except Exception as e:
            print(f"      Error al graficar {estrategia}: {e}")
    
    print(f"\n✅ ¡Análisis completado!")
    print(f"🏆 Mejor estrategia: {ranking.iloc[0]['strategy_name']} "
          f"({ranking.iloc[0]['total_return_pct']:.2f}% de retorno)")


# ========================= EJECUTAR =========================

if __name__ == "__main__":
    main()
