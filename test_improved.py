"""
Versión mejorada del archivo test.py original con sistema modular
"""

import ccxt
import pandas as pd
import pandas_ta as ta
from strategy_tester import StrategyTester


def estrategia_ema_original(df, fast_ema=9, slow_ema=21):
    """
    Tu estrategia original de EMA convertida a función modular
    """
    # Calcular EMAs
    df['ema9'] = ta.ema(df['close'], length=fast_ema)
    df['ema21'] = ta.ema(df['close'], length=slow_ema)
    
    # Condiciones para señales
    df['long'] = df['ema9'] > df['ema21']
    
    # Generar señales más claras
    df['signal'] = 0
    
    # Señal de compra: cuando EMA rápida cruza por encima de EMA lenta
    buy_condition = (df['long'] == True) & (df['long'].shift(1) == False)
    df.loc[buy_condition, 'signal'] = 1
    
    # Señal de venta: cuando EMA rápida cruza por debajo de EMA lenta
    sell_condition = (df['long'] == False) & (df['long'].shift(1) == True)
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def main():
    """
    Recrear tu análisis original pero con el nuevo sistema
    """
    print("🔄 Ejecutando análisis original mejorado...")
    
    # Configurar igual que tu código original
    tester = StrategyTester(
        symbol='BTC/USDT',
        timeframe='5m',
        start_date='2020-01-01T00:00:00Z'
    )
    
    # Descargar datos
    tester.fetch_data()
    
    # Aplicar tu estrategia original
    df_with_signals = tester.apply_strategy(
        strategy_func=estrategia_ema_original,
        params={'fast_ema': 9, 'slow_ema': 21},
        strategy_name='EMA_Original'
    )
    
    # Mostrar las mismas métricas que tu código original
    print("\n📊 RESULTADOS ORIGINALES:")
    print("=" * 50)
    
    # Puntos de entrada y salida
    entry_points = df_with_signals[df_with_signals['signal'] == 1]
    exit_points = df_with_signals[df_with_signals['signal'] == -1]
    
    print(f"Entradas: {len(entry_points)} operaciones")
    print(f"Salidas: {len(exit_points)} operaciones")
    
    # Calcular ganancias con el método original
    def calculate_profit_original(entry_points, exit_points, df):
        total_profit = 0
        for entry in entry_points.index:
            # Buscar la siguiente salida después de la entrada
            future_exits = exit_points[exit_points.index > entry]
            if len(future_exits) > 0:
                exit_time = future_exits.index[0]
                profit = df.loc[exit_time, 'close'] - df.loc[entry, 'close']
                total_profit += profit
        return total_profit
    
    ganancia_original = calculate_profit_original(entry_points, exit_points, df_with_signals)
    print(f"Ganancia total (método original): ${ganancia_original:.2f}")
    
    # Calcular métricas avanzadas con el nuevo sistema
    metrics = tester.calculate_performance(df_with_signals)
    
    print(f"\n📈 MÉTRICAS AVANZADAS:")
    print(f"Total de operaciones: {metrics['total_trades']}")
    print(f"Operaciones ganadoras: {metrics['winning_trades']}")
    print(f"Operaciones perdedoras: {metrics['losing_trades']}")
    print(f"Tasa de éxito: {metrics['win_rate']:.2f}%")
    print(f"Ganancia total: ${metrics['total_profit']:.2f}")
    print(f"Retorno total: {metrics['total_return_pct']:.2f}%")
    print(f"Ganancia promedio por operación: ${metrics['avg_profit_per_trade']:.2f}")
    print(f"Mayor ganancia: ${metrics['max_profit']:.2f}")
    print(f"Mayor pérdida: ${metrics['max_loss']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Mostrar los últimos 10 registros como en tu código original
    print(f"\n📋 ÚLTIMOS 10 REGISTROS:")
    print(df_with_signals[['close', 'ema9', 'ema21', 'long', 'signal']].tail(10))
    
    # Guardar resultados
    df_with_signals.to_csv('backtest_results_improved.csv')
    print(f"\n💾 Resultados guardados en: backtest_results_improved.csv")
    
    # Graficar resultados
    try:
        tester.plot_strategy_results('EMA_Original', save_plot=True)
        print("📊 Gráfico generado exitosamente")
    except Exception as e:
        print(f"❌ Error al generar gráfico: {e}")
    
    return tester, df_with_signals, metrics


# Ejemplo de cómo probar variaciones de tu estrategia original
def probar_variaciones():
    """
    Probar diferentes variaciones de tu estrategia EMA
    """
    print("\n🔬 PROBANDO VARIACIONES DE TU ESTRATEGIA...")
    
    tester = StrategyTester(
        symbol='BTC/USDT',
        timeframe='5m',
        start_date='2020-01-01T00:00:00Z'
    )
    
    tester.fetch_data()
    
    # Diferentes combinaciones de EMA
    variaciones = {
        'EMA_9_21_Original': {
            'func': estrategia_ema_original,
            'params': {'fast_ema': 9, 'slow_ema': 21}
        },
        'EMA_5_15_Rapida': {
            'func': estrategia_ema_original,
            'params': {'fast_ema': 5, 'slow_ema': 15}
        },
        'EMA_12_26_Lenta': {
            'func': estrategia_ema_original,
            'params': {'fast_ema': 12, 'slow_ema': 26}
        },
        'EMA_8_20_Media': {
            'func': estrategia_ema_original,
            'params': {'fast_ema': 8, 'slow_ema': 20}
        },
        'EMA_10_30_Conservador': {
            'func': estrategia_ema_original,
            'params': {'fast_ema': 10, 'slow_ema': 30}
        }
    }
    
    resultados = tester.compare_strategies(variaciones)
    
    print("\n🏆 RANKING DE VARIACIONES EMA:")
    ranking = resultados.sort_values('total_return_pct', ascending=False)
    
    for i, (_, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {row['strategy_name']:20} | "
              f"Retorno: {row['total_return_pct']:6.2f}% | "
              f"Operaciones: {row['total_trades']:3d} | "
              f"Éxito: {row['win_rate']:5.1f}%")
    
    # Guardar comparación
    resultados.to_csv('variaciones_ema.csv', index=False)
    print(f"\n💾 Comparación guardada en: variaciones_ema.csv")
    
    return resultados


if __name__ == "__main__":
    # Ejecutar análisis original mejorado
    tester, df, metrics = main()
    
    # Probar variaciones
    variaciones_results = probar_variaciones()
    
    print(f"\n✅ ¡Análisis completado!")
    print(f"🎯 Ahora puedes usar strategy_tester.py para probar cualquier estrategia personalizada")
