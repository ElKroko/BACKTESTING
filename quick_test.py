"""
Prueba rápida de estrategias - Versión simplificada para experimentar
"""

import ccxt
import pandas as pd
import pandas_ta as ta
from strategy_tester import StrategyTester


def probar_diferentes_periodos():
    """
    Prueba la misma estrategia EMA en diferentes períodos
    """
    print("🕐 PROBANDO DIFERENTES PERÍODOS DE TIEMPO")
    print("=" * 50)
    
    # Diferentes períodos a probar
    periodos = {
        '2020_Q1': '2020-01-01T00:00:00Z',
        '2021_Bull': '2021-01-01T00:00:00Z', 
        '2022_Bear': '2022-01-01T00:00:00Z',
        '2023_Recovery': '2023-01-01T00:00:00Z',
        '2024_Current': '2024-01-01T00:00:00Z'
    }
    
    resultados_periodos = []
    
    for nombre, fecha_inicio in periodos.items():
        print(f"\n🔄 Probando período: {nombre}")
        
        try:
            # Crear tester para cada período
            tester = StrategyTester(
                symbol='BTC/USDT',
                timeframe='1h',
                start_date=fecha_inicio
            )
            
            tester.fetch_data()
            
            # Aplicar tu estrategia EMA mejorada
            from strategy_optimizer import estrategia_ema_mejorada
            
            df_result = tester.apply_strategy(
                estrategia_ema_mejorada,
                {'fast_ema': 9, 'slow_ema': 21},
                f'EMA_{nombre}'
            )
            
            # Calcular métricas
            metrics = tester.calculate_performance(df_result)
            metrics['periodo'] = nombre
            metrics['fecha_inicio'] = fecha_inicio
            
            resultados_periodos.append(metrics)
            
            print(f"  📊 Retorno: {metrics['total_return_pct']:.2f}% | "
                  f"Trades: {metrics['total_trades']} | "
                  f"Éxito: {metrics['win_rate']:.1f}%")
            
        except Exception as e:
            print(f"  ❌ Error en {nombre}: {e}")
            continue
    
    if resultados_periodos:
        df_periodos = pd.DataFrame(resultados_periodos)
        df_periodos = df_periodos.sort_values('total_return_pct', ascending=False)
        
        print(f"\n🏆 RANKING POR PERÍODO:")
        print("=" * 70)
        
        for i, (_, row) in enumerate(df_periodos.iterrows(), 1):
            print(f"{i}. {row['periodo']:15} | "
                  f"Retorno: {row['total_return_pct']:6.2f}% | "
                  f"Trades: {row['total_trades']:3d} | "
                  f"Éxito: {row['win_rate']:5.1f}% | "
                  f"Sharpe: {row['sharpe_ratio']:5.2f}")
        
        # Guardar resultados
        df_periodos.to_csv('resultados_por_periodo.csv', index=False)
        print(f"\n💾 Resultados guardados en: resultados_por_periodo.csv")
        
        return df_periodos
    
    return None


def probar_parametros_ema_rapido():
    """
    Prueba rápida de algunos parámetros EMA populares
    """
    print("\n⚡ PRUEBA RÁPIDA DE PARÁMETROS EMA")
    print("=" * 50)
    
    # Configurar tester
    tester = StrategyTester(
        symbol='BTC/USDT',
        timeframe='1h',
        start_date='2023-01-01T00:00:00Z'  # Año más reciente
    )
    
    tester.fetch_data()
    
    # Parámetros EMA populares
    ema_configs = {
        'EMA_5_15': {'fast_ema': 5, 'slow_ema': 15},
        'EMA_9_21_Original': {'fast_ema': 9, 'slow_ema': 21},
        'EMA_12_26_MACD': {'fast_ema': 12, 'slow_ema': 26},
        'EMA_8_21_Balanced': {'fast_ema': 8, 'slow_ema': 21},
        'EMA_10_30_Conservative': {'fast_ema': 10, 'slow_ema': 30}
    }
    
    from strategy_optimizer import estrategia_ema_mejorada
    
    estrategias = {}
    for nombre, params in ema_configs.items():
        estrategias[nombre] = {
            'func': estrategia_ema_mejorada,
            'params': params
        }
    
    # Comparar estrategias
    resultados = tester.compare_strategies(estrategias)
    
    print(f"\n🏆 RANKING PARÁMETROS EMA:")
    print("=" * 70)
    
    ranking = resultados.sort_values('total_return_pct', ascending=False)
    
    for i, (_, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {row['strategy_name']:20} | "
              f"Retorno: {row['total_return_pct']:6.2f}% | "
              f"Trades: {row['total_trades']:3d} | "
              f"Éxito: {row['win_rate']:5.1f}% | "
              f"Sharpe: {row['sharpe_ratio']:5.2f}")
    
    # Guardar y graficar mejor resultado
    resultados.to_csv('comparacion_emas_rapida.csv', index=False)
    
    mejor_estrategia = ranking.iloc[0]['strategy_name']
    print(f"\n🎯 Mejor estrategia: {mejor_estrategia}")
    
    try:
        tester.plot_strategy_results(mejor_estrategia, save_plot=True)
        print("📊 Gráfico generado exitosamente")
    except Exception as e:
        print(f"❌ Error al generar gráfico: {e}")
    
    return resultados


def analizar_tu_estrategia_original():
    """
    Analiza específicamente tu estrategia original en diferentes contextos
    """
    print("\n🔍 ANÁLISIS DETALLADO DE TU ESTRATEGIA ORIGINAL")
    print("=" * 60)
    
    # Tu estrategia original
    def estrategia_original_exacta(df, fast_ema=9, slow_ema=21):
        """Tu estrategia EMA original exacta"""
        df['ema9'] = ta.ema(df['close'], length=fast_ema)
        df['ema21'] = ta.ema(df['close'], length=slow_ema)
        
        df['long'] = df['ema9'] > df['ema21']
        df['signal'] = df['long'].diff().fillna(0)
        
        return df
    
    # Probar en diferentes configuraciones
    configuraciones = [
        {
            'nombre': 'Original_5min_2020',
            'timeframe': '5m',
            'start_date': '2020-01-01T00:00:00Z',
            'descripcion': 'Tu configuración original exacta'
        },
        {
            'nombre': 'Mejorada_1h_2023',
            'timeframe': '1h', 
            'start_date': '2023-01-01T00:00:00Z',
            'descripcion': 'Misma estrategia, mejor timeframe'
        },
        {
            'nombre': 'Bull_Market_2021',
            'timeframe': '1h',
            'start_date': '2021-01-01T00:00:00Z',
            'descripcion': 'Durante mercado alcista'
        }
    ]
    
    resultados_config = []
    
    for config in configuraciones:
        print(f"\n🔄 Probando: {config['descripcion']}")
        
        try:
            tester = StrategyTester(
                symbol='BTC/USDT',
                timeframe=config['timeframe'],
                start_date=config['start_date']
            )
            
            tester.fetch_data()
            
            # Usar tu estrategia original exacta
            df_result = tester.apply_strategy(
                estrategia_original_exacta,
                {'fast_ema': 9, 'slow_ema': 21},
                config['nombre']
            )
            
            metrics = tester.calculate_performance(df_result)
            metrics['configuracion'] = config['nombre']
            metrics['descripcion'] = config['descripcion']
            metrics['timeframe'] = config['timeframe']
            
            resultados_config.append(metrics)
            
            print(f"  📊 {config['nombre']}: {metrics['total_return_pct']:.2f}% | "
                  f"{metrics['total_trades']} trades | {metrics['win_rate']:.1f}% éxito")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    if resultados_config:
        df_configs = pd.DataFrame(resultados_config)
        
        print(f"\n📈 RESUMEN COMPARATIVO:")
        print("=" * 80)
        
        for _, row in df_configs.iterrows():
            print(f"• {row['descripcion']}")
            print(f"  Retorno: {row['total_return_pct']:6.2f}% | "
                  f"Trades: {row['total_trades']:3d} | "
                  f"Éxito: {row['win_rate']:5.1f}% | "
                  f"Sharpe: {row['sharpe_ratio']:5.2f}")
            print()
        
        # Guardar análisis
        df_configs.to_csv('analisis_estrategia_original.csv', index=False)
        print(f"💾 Análisis guardado en: analisis_estrategia_original.csv")
        
        return df_configs
    
    return None


def main():
    """
    Función principal para pruebas rápidas
    """
    print("🚀 SISTEMA DE PRUEBAS RÁPIDAS DE ESTRATEGIAS")
    print("=" * 60)
    
    # 1. Analizar tu estrategia original
    print("\n1️⃣ Analizando tu estrategia original...")
    analisis_original = analizar_tu_estrategia_original()
    
    # 2. Probar diferentes parámetros EMA
    print("\n2️⃣ Probando diferentes parámetros EMA...")
    resultados_ema = probar_parametros_ema_rapido()
    
    # 3. Probar diferentes períodos
    print("\n3️⃣ Probando diferentes períodos de tiempo...")
    resultados_periodos = probar_diferentes_periodos()
    
    print(f"\n✅ ¡PRUEBAS COMPLETADAS!")
    print(f"📁 Revisa los archivos CSV generados para análisis detallado")
    print(f"📊 Los gráficos PNG muestran visualmente el rendimiento")
    
    # Resumen final
    if resultados_ema is not None:
        mejor_ema = resultados_ema.sort_values('total_return_pct', ascending=False).iloc[0]
        print(f"\n🏆 Mejor configuración EMA encontrada:")
        print(f"   {mejor_ema['strategy_name']} - {mejor_ema['total_return_pct']:.2f}% retorno")
    
    if resultados_periodos is not None:
        mejor_periodo = resultados_periodos.sort_values('total_return_pct', ascending=False).iloc[0]
        print(f"🕐 Mejor período para trading:")
        print(f"   {mejor_periodo['periodo']} - {mejor_periodo['total_return_pct']:.2f}% retorno")


if __name__ == "__main__":
    main()
