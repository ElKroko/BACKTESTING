"""
Script simple para probar el sistema de estrategias
"""

import os
import sys

def test_imports():
    """Probar todas las importaciones necesarias"""
    print("🔍 PROBANDO IMPORTACIONES...")
    
    try:
        import ccxt
        print("✅ ccxt OK")
    except ImportError as e:
        print(f"❌ ccxt: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas OK")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
        
    try:
        import pandas_ta as ta
        print("✅ pandas_ta OK")
    except ImportError as e:
        print(f"❌ pandas_ta: {e}")
        return False
        
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib OK")
    except ImportError as e:
        print(f"❌ matplotlib: {e}")
        return False
    
    return True


def test_strategy_tester():
    """Probar el sistema strategy_tester"""
    print("\n🔍 PROBANDO STRATEGY TESTER...")
    
    # Cambiar al directorio correcto
    strategy_tester_dir = os.path.join(os.path.dirname(__file__), 'strategy_tester')
    if not os.path.exists(strategy_tester_dir):
        print(f"❌ Directorio no encontrado: {strategy_tester_dir}")
        return False
    
    sys.path.insert(0, strategy_tester_dir)
    
    try:
        # Probar importaciones del sistema
        from engine.data_manager import DataManager
        print("✅ DataManager OK")
        
        from engine.performance_analyzer import PerformanceAnalyzer  
        print("✅ PerformanceAnalyzer OK")
        
        from engine.backtester import BacktestEngine
        print("✅ BacktestEngine OK")
        
        from strategies import AVAILABLE_STRATEGIES
        print(f"✅ Estrategias disponibles: {len(AVAILABLE_STRATEGIES)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importando strategy_tester: {e}")
        return False


def run_simple_test():
    """Ejecutar una prueba simple del sistema"""
    print("\n🚀 EJECUTANDO PRUEBA SIMPLE...")
    
    try:
        from engine.backtester import BacktestEngine
        from strategies import get_strategy
        
        # Crear engine
        engine = BacktestEngine(results_dir="results")
        print("✅ Engine creado")
        
        # Obtener una estrategia simple
        ema_strategy = get_strategy('EMA_Crossover')
        print("✅ Estrategia EMA obtenida")
        
        # Probar descarga de datos
        print("📊 Descargando datos de prueba...")
        
        # Ejecutar estrategia simple
        result = engine.run_strategy(
            strategy_func=ema_strategy,
            strategy_params={'fast_ema': 9, 'slow_ema': 21},
            symbol='BTC/USDT',
            timeframe='1h',
            strategy_name='EMA_Test'
        )
        
        if result is not None:
            print("✅ Estrategia ejecutada exitosamente")
            print(f"📈 Resultados: {len(result)} registros")
            return True
        else:
            print("❌ La estrategia no devolvió resultados")
            return False
            
    except Exception as e:
        print(f"❌ Error en prueba simple: {e}")
        return False


def main():
    """Función principal"""
    print("🧪 SISTEMA DE PRUEBAS - STRATEGY TESTER")
    print("=" * 50)
    
    # 1. Probar importaciones
    if not test_imports():
        print("\n❌ Faltan dependencias. Instala con: pip install ccxt pandas-ta matplotlib seaborn")
        return
    
    # 2. Probar strategy_tester
    if not test_strategy_tester():
        print("\n❌ Problema con el sistema strategy_tester")
        return
    
    # 3. Ejecutar prueba simple
    if run_simple_test():
        print("\n✅ ¡SISTEMA FUNCIONANDO CORRECTAMENTE!")
        print("📁 Revisa la carpeta 'results' para ver los resultados")
        print("🚀 Puedes ejecutar: python strategy_tester/main.py")
    else:
        print("\n❌ Problema en la ejecución")


if __name__ == "__main__":
    main()
