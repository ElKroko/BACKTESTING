"""
Optimizador de estrategias - Encuentra los mejores parámetros automáticamente
"""

import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from strategy_tester import StrategyTester
from itertools import product
import warnings
warnings.filterwarnings('ignore')


def estrategia_ema_mejorada(df, fast_ema=9, slow_ema=21, volume_filter=False, rsi_filter=False, rsi_period=14):
    """
    Estrategia EMA mejorada con filtros opcionales
    """
    # Calcular EMAs
    df['ema_fast'] = ta.ema(df['close'], length=fast_ema)
    df['ema_slow'] = ta.ema(df['close'], length=slow_ema)
    
    # Condición básica de EMA
    ema_condition = df['ema_fast'] > df['ema_slow']
    
    # Filtro de volumen (opcional)
    if volume_filter:
        volume_ma = ta.sma(df['vol'], length=20)
        high_volume = df['vol'] > volume_ma * 1.2
    else:
        high_volume = True
    
    # Filtro de RSI (opcional)
    if rsi_filter:
        df['rsi'] = ta.rsi(df['close'], length=rsi_period)
        rsi_ok = (df['rsi'] > 30) & (df['rsi'] < 70)  # Evitar extremos
    else:
        rsi_ok = True
    
    # Combinar condiciones
    df['long'] = ema_condition & high_volume & rsi_ok
    
    # Generar señales
    df['signal'] = 0
    
    # Señal de compra: cuando todas las condiciones se cumplen
    buy_condition = (df['long'] == True) & (df['long'].shift(1) == False)
    df.loc[buy_condition, 'signal'] = 1
    
    # Señal de venta: cuando EMA se cruza hacia abajo
    sell_condition = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def optimizar_ema_parametros(symbol='BTC/USDT', timeframe='1h', start_date='2023-01-01T00:00:00Z'):
    """
    Optimiza automáticamente los parámetros de EMA
    """
    print(f"🔧 Optimizando parámetros EMA para {symbol}...")
    
    # Crear tester
    tester = StrategyTester(symbol=symbol, timeframe=timeframe, start_date=start_date)
    tester.fetch_data()
    
    # Rangos de parámetros a probar
    fast_emas = [5, 8, 9, 12, 15]
    slow_emas = [15, 20, 21, 26, 30, 35]
    
    resultados = []
    total_combinations = len(fast_emas) * len(slow_emas)
    current = 0
    
    print(f"Probando {total_combinations} combinaciones...")
    
    for fast, slow in product(fast_emas, slow_emas):
        if fast >= slow:  # Skip invalid combinations
            continue
            
        current += 1
        print(f"  Progreso: {current}/{total_combinations} - EMA({fast},{slow})", end='\r')
        
        try:
            # Aplicar estrategia
            df_result = tester.apply_strategy(
                estrategia_ema_mejorada,
                {'fast_ema': fast, 'slow_ema': slow},
                f'EMA_{fast}_{slow}'
            )
            
            # Calcular métricas
            metrics = tester.calculate_performance(df_result)
            
            # Agregar parámetros al resultado
            metrics['fast_ema'] = fast
            metrics['slow_ema'] = slow
            metrics['params'] = f"EMA({fast},{slow})"
            
            resultados.append(metrics)
            
        except Exception as e:
            print(f"\n  Error con EMA({fast},{slow}): {e}")
            continue
    
    print("\n")
    
    # Convertir a DataFrame y ordenar
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        df_resultados = df_resultados.sort_values('total_return_pct', ascending=False)
        
        print("🏆 TOP 10 COMBINACIONES EMA:")
        print("=" * 80)
        
        top_10 = df_resultados.head(10)
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            print(f"{i:2d}. {row['params']:12} | "
                  f"Retorno: {row['total_return_pct']:6.2f}% | "
                  f"Trades: {row['total_trades']:3d} | "
                  f"Éxito: {row['win_rate']:5.1f}% | "
                  f"Sharpe: {row['sharpe_ratio']:5.2f}")
        
        # Guardar resultados
        df_resultados.to_csv('ema_optimization_results.csv', index=False)
        print(f"\n💾 Resultados completos guardados en: ema_optimization_results.csv")
        
        return df_resultados
    else:
        print("❌ No se pudieron obtener resultados")
        return None


def comparar_multiples_timeframes(symbol='BTC/USDT', estrategia_params={'fast_ema': 9, 'slow_ema': 21}):
    """
    Compara la misma estrategia en diferentes marcos temporales
    """
    print(f"📊 Comparando estrategia en múltiples timeframes...")
    
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    start_dates = {
        '5m': '2024-01-01T00:00:00Z',   # Más reciente para timeframes cortos
        '15m': '2023-06-01T00:00:00Z',
        '1h': '2023-01-01T00:00:00Z',
        '4h': '2022-01-01T00:00:00Z',
        '1d': '2020-01-01T00:00:00Z'
    }
    
    resultados_tf = []
    
    for tf in timeframes:
        print(f"\n🔄 Probando timeframe: {tf}")
        
        try:
            # Crear tester para cada timeframe
            tester = StrategyTester(
                symbol=symbol, 
                timeframe=tf, 
                start_date=start_dates[tf]
            )
            tester.fetch_data()
            
            # Aplicar estrategia
            df_result = tester.apply_strategy(
                estrategia_ema_mejorada,
                estrategia_params,
                f'EMA_{tf}'
            )
            
            # Calcular métricas
            metrics = tester.calculate_performance(df_result)
            metrics['timeframe'] = tf
            metrics['data_points'] = len(df_result)
            
            resultados_tf.append(metrics)
            
            print(f"  ✅ {tf}: {metrics['total_return_pct']:.2f}% retorno, {metrics['total_trades']} trades")
            
        except Exception as e:
            print(f"  ❌ Error con {tf}: {e}")
            continue
    
    if resultados_tf:
        df_tf = pd.DataFrame(resultados_tf)
        df_tf = df_tf.sort_values('total_return_pct', ascending=False)
        
        print(f"\n🏆 RANKING POR TIMEFRAME:")
        print("=" * 80)
        
        for i, (_, row) in enumerate(df_tf.iterrows(), 1):
            print(f"{i}. {row['timeframe']:4} | "
                  f"Retorno: {row['total_return_pct']:6.2f}% | "
                  f"Trades: {row['total_trades']:3d} | "
                  f"Éxito: {row['win_rate']:5.1f}% | "
                  f"Datos: {row['data_points']:4d}")
        
        # Guardar resultados
        df_tf.to_csv('timeframe_comparison.csv', index=False)
        print(f"\n💾 Comparación guardada en: timeframe_comparison.csv")
        
        return df_tf
    else:
        print("❌ No se pudieron obtener resultados")
        return None


def crear_estrategia_hibrida(df, ema_fast=9, ema_slow=21, rsi_period=14, bb_period=20, bb_std=2):
    """
    Estrategia híbrida que combina EMA, RSI y Bollinger Bands
    """
    # EMAs
    df['ema_fast'] = ta.ema(df['close'], length=ema_fast)
    df['ema_slow'] = ta.ema(df['close'], length=ema_slow)
    
    # RSI
    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
    
    # Bollinger Bands
    bb = ta.bbands(df['close'], length=bb_period, std=bb_std)
    df['bb_upper'] = bb[f'BBU_{bb_period}_{bb_std}']
    df['bb_middle'] = bb[f'BBM_{bb_period}_{bb_std}']
    df['bb_lower'] = bb[f'BBL_{bb_period}_{bb_std}']
    
    # Condiciones de compra (todas deben cumplirse)
    ema_bullish = df['ema_fast'] > df['ema_slow']
    rsi_not_overbought = df['rsi'] < 75
    rsi_not_oversold = df['rsi'] > 25
    price_above_bb_middle = df['close'] > df['bb_middle']
    
    buy_condition = ema_bullish & rsi_not_overbought & rsi_not_oversold & price_above_bb_middle
    
    # Condiciones de venta (cualquiera puede activarse)
    ema_bearish = df['ema_fast'] < df['ema_slow']
    rsi_extreme = (df['rsi'] > 80) | (df['rsi'] < 20)
    price_below_bb_middle = df['close'] < df['bb_middle']
    
    sell_condition = ema_bearish | rsi_extreme | price_below_bb_middle
    
    # Generar señales
    df['signal'] = 0
    
    # Detectar cruces/cambios
    buy_signals = buy_condition & (~buy_condition.shift(1).fillna(False))
    sell_signals = sell_condition & (~sell_condition.shift(1).fillna(False))
    
    df.loc[buy_signals, 'signal'] = 1
    df.loc[sell_signals, 'signal'] = -1
    
    return df


def analisis_completo():
    """
    Análisis completo con múltiples estrategias y optimizaciones
    """
    print("🚀 INICIANDO ANÁLISIS COMPLETO DE ESTRATEGIAS")
    print("=" * 60)
    
    # 1. Optimizar parámetros EMA
    print("\n1️⃣ OPTIMIZANDO PARÁMETROS EMA...")
    ema_optimization = optimizar_ema_parametros(
        symbol='BTC/USDT',
        timeframe='1h',
        start_date='2023-01-01T00:00:00Z'
    )
    
    if ema_optimization is not None and len(ema_optimization) > 0:
        mejor_ema = ema_optimization.iloc[0]
        print(f"🏆 Mejor combinación EMA: EMA({mejor_ema['fast_ema']},{mejor_ema['slow_ema']}) - {mejor_ema['total_return_pct']:.2f}%")
        
        # 2. Probar mejor EMA en diferentes timeframes
        print(f"\n2️⃣ PROBANDO MEJOR EMA EN DIFERENTES TIMEFRAMES...")
        tf_results = comparar_multiples_timeframes(
            symbol='BTC/USDT',
            estrategia_params={
                'fast_ema': int(mejor_ema['fast_ema']),
                'slow_ema': int(mejor_ema['slow_ema'])
            }
        )
    
    # 3. Comparar estrategia híbrida
    print(f"\n3️⃣ PROBANDO ESTRATEGIA HÍBRIDA...")
    
    tester = StrategyTester(symbol='BTC/USDT', timeframe='1h', start_date='2023-01-01T00:00:00Z')
    tester.fetch_data()
    
    estrategias_finales = {
        'EMA_Original': {
            'func': estrategia_ema_mejorada,
            'params': {'fast_ema': 9, 'slow_ema': 21}
        },
        'EMA_Optimizada': {
            'func': estrategia_ema_mejorada,
            'params': {
                'fast_ema': int(mejor_ema['fast_ema']) if ema_optimization is not None and len(ema_optimization) > 0 else 9,
                'slow_ema': int(mejor_ema['slow_ema']) if ema_optimization is not None and len(ema_optimization) > 0 else 21
            }
        },
        'EMA_con_Filtros': {
            'func': estrategia_ema_mejorada,
            'params': {'fast_ema': 9, 'slow_ema': 21, 'volume_filter': True, 'rsi_filter': True}
        },
        'Estrategia_Hibrida': {
            'func': crear_estrategia_hibrida,
            'params': {'ema_fast': 9, 'ema_slow': 21, 'rsi_period': 14}
        }
    }
    
    resultados_finales = tester.compare_strategies(estrategias_finales)
    
    print(f"\n🏆 RANKING FINAL DE ESTRATEGIAS:")
    print("=" * 80)
    
    ranking_final = resultados_finales.sort_values('total_return_pct', ascending=False)
    
    for i, (_, row) in enumerate(ranking_final.iterrows(), 1):
        print(f"{i}. {row['strategy_name']:20} | "
              f"Retorno: {row['total_return_pct']:6.2f}% | "
              f"Trades: {row['total_trades']:3d} | "
              f"Éxito: {row['win_rate']:5.1f}% | "
              f"Sharpe: {row['sharpe_ratio']:5.2f}")
    
    # Guardar resultados finales
    resultados_finales.to_csv('analisis_completo_estrategias.csv', index=False)
    print(f"\n💾 Análisis completo guardado en: analisis_completo_estrategias.csv")
    
    # Graficar la mejor estrategia
    mejor_estrategia = ranking_final.iloc[0]['strategy_name']
    print(f"\n📊 Generando gráfico de la mejor estrategia: {mejor_estrategia}")
    
    try:
        tester.plot_strategy_results(mejor_estrategia, save_plot=True)
        print("✅ Gráfico generado exitosamente")
    except Exception as e:
        print(f"❌ Error al generar gráfico: {e}")
    
    return resultados_finales


if __name__ == "__main__":
    # Ejecutar análisis completo
    resultados = analisis_completo()
    
    print(f"\n🎉 ¡ANÁLISIS COMPLETADO!")
    print(f"📁 Revisa los archivos CSV generados para ver todos los resultados")
    print(f"📈 Los gráficos muestran visualmente el rendimiento de las estrategias")
