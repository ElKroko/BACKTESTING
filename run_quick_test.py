"""
Script directo para probar el sistema
"""

print("🚀 PROBANDO STRATEGY TESTER SYSTEM")
print("=" * 40)

# Probar importaciones básicas
try:
    import ccxt
    import pandas as pd
    import pandas_ta as ta
    print("✅ Dependencias básicas OK")
except ImportError as e:
    print(f"❌ Error en dependencias: {e}")
    exit(1)

# Probar acceso a strategy_tester
import os
import sys

strategy_dir = os.path.join(os.getcwd(), 'strategy_tester')
sys.path.insert(0, strategy_dir)

try:
    from strategies import AVAILABLE_STRATEGIES, get_strategy
    print(f"✅ Estrategias cargadas: {len(AVAILABLE_STRATEGIES)}")
    
    # Listar estrategias disponibles
    print("\n📋 ESTRATEGIAS DISPONIBLES:")
    for name in AVAILABLE_STRATEGIES.keys():
        print(f"   • {name}")
    
except ImportError as e:
    print(f"❌ Error cargando estrategias: {e}")
    exit(1)

# Probar una estrategia simple
try:
    from engine.backtester import BacktestEngine
    
    print("\n🔄 Probando motor de backtesting...")
    engine = BacktestEngine()
    
    # Obtener estrategia EMA
    ema_strategy = get_strategy('EMA_Crossover')
    
    print("✅ Sistema listo para usar!")
    print("\n🎯 COMANDOS PARA EJECUTAR:")
    print("1. cd strategy_tester")
    print("2. python main.py")
    print("\nO usar el script principal:")
    print("python run_strategy_tester.py")
    
except Exception as e:
    print(f"❌ Error en motor: {e}")
    
print("\n🏁 Prueba completada")
