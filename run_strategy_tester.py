"""
Script para ejecutar el Strategy Tester de forma simple
"""

import os
import sys

# Cambiar al directorio del strategy_tester
strategy_tester_dir = os.path.join(os.path.dirname(__file__), 'strategy_tester')
os.chdir(strategy_tester_dir)

# Añadir al path
sys.path.insert(0, strategy_tester_dir)

# Importar y ejecutar
try:
    from main import StrategyTesterApp
    
    print("🚀 INICIANDO STRATEGY TESTER")
    print("=" * 50)
    
    # Crear la aplicación
    app = StrategyTesterApp()
    
    # Mostrar estrategias disponibles
    app.list_available_strategies()
    
    # Ejecutar análisis rápido
    print("\n🔄 EJECUTANDO ANÁLISIS RÁPIDO...")
    results = app.quick_analysis()
    
    if results is not None:
        print("\n✅ ¡Análisis completado!")
        print("📁 Revisa la carpeta 'results' para ver los resultados")
    else:
        print("\n❌ Hubo un problema con el análisis")
        
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("Asegúrate de que todas las dependencias estén instaladas")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Revisa la configuración del sistema")
