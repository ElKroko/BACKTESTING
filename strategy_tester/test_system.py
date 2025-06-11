"""
Script de prueba del sistema strategy_tester
"""

print("🔍 DIAGNÓSTICO DEL SISTEMA")
print("=" * 40)

# 1. Probar importaciones básicas
try:
    import pandas as pd
    import pandas_ta as ta
    import ccxt
    print("✅ Dependencias básicas OK")
except ImportError as e:
    print(f"❌ Dependencias: {e}")
    exit(1)

# 2. Probar configuración
try:
    from config.settings import DEFAULT_SYMBOLS
    print(f"✅ Configuración OK - {len(DEFAULT_SYMBOLS)} símbolos")
except ImportError as e:
    print(f"❌ Configuración: {e}")

# 3. Probar estrategias
try:
    from strategies.trend_following import ema_crossover_strategy
    print("✅ Estrategia EMA OK")
except ImportError as e:
    print(f"❌ Estrategias: {e}")

# 4. Probar engine
try:
    from engine.data_manager import DataManager
    from engine.backtester import BacktestEngine
    print("✅ Motor de backtesting OK")
except ImportError as e:
    print(f"❌ Engine: {e}")

# 5. Prueba completa del sistema
try:
    from strategies import AVAILABLE_STRATEGIES, get_strategy
    print(f"✅ Sistema completo OK - {len(AVAILABLE_STRATEGIES)} estrategias")
    
    # Listar estrategias
    print("\n📋 ESTRATEGIAS DISPONIBLES:")
    for name in list(AVAILABLE_STRATEGIES.keys())[:5]:  # Mostrar solo las primeras 5
        print(f"   • {name}")
    
    if len(AVAILABLE_STRATEGIES) > 5:
        print(f"   ... y {len(AVAILABLE_STRATEGIES) - 5} más")
    
except ImportError as e:
    print(f"❌ Sistema completo: {e}")

print(f"\n🏁 Diagnóstico completado")
