"""
Controlador de Cache - Herramientas para gestionar el cache de datos
"""

import os
import sys
from datetime import datetime, timedelta

# Añadir el directorio actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from engine.data_manager import DataManager


class CacheController:
    """
    Controlador para gestionar el cache de datos de manera inteligente
    """
    
    def __init__(self):
        self.data_manager = DataManager()
    
    def show_cache_status(self):
        """Mostrar estado actual del cache"""
        print("📊 ESTADO DEL CACHE")
        print("=" * 50)
        
        cache_info = self.data_manager.get_cache_info()
        
        print(f"📁 Archivos en cache: {cache_info['total_files']}")
        print(f"💾 Tamaño total: {cache_info['total_size_mb']:.2f} MB")
        print(f"🎯 Símbolos cacheados: {len(cache_info['cached_symbols'])}")
        
        if cache_info['cached_symbols']:
            print(f"   • {', '.join(cache_info['cached_symbols'])}")
        
        print("\n📋 ARCHIVOS DETALLADOS:")
        for file_info in cache_info['files']:
            age = datetime.now() - file_info['modified']
            age_str = f"{age.days}d {age.seconds//3600}h" if age.days > 0 else f"{age.seconds//3600}h {(age.seconds//60)%60}m"
            print(f"   • {file_info['file'][:40]:40} | {file_info['size_mb']:.1f}MB | {age_str}")
    
    def show_cached_data_details(self):
        """Mostrar detalles de los datos cacheados"""
        print("\n📈 DETALLES DE DATOS CACHEADOS")
        print("=" * 70)
        
        cached_data = self.data_manager.list_cached_data()
        
        if not cached_data:
            print("   🔍 No hay datos cacheados con metadatos")
            return
        
        for data in cached_data:
            print(f"\n🎯 {data['symbol']} ({data['timeframe']})")
            print(f"   📅 Período: {data['actual_start'][:10]} → {data['actual_end'][:10]}")
            print(f"   📊 Velas: {data['total_candles']:,}")
            print(f"   🕐 Última actualización: {data['last_updated'][:16]}")
            
            # Calcular edad
            last_update = datetime.fromisoformat(data['last_updated'])
            age = datetime.now() - last_update
            
            if age.total_seconds() < 3600:  # Menos de 1 hora
                age_str = f"{age.seconds//60}m"
                status = "🟢 Reciente"
            elif age.total_seconds() < 86400:  # Menos de 1 día
                age_str = f"{age.seconds//3600}h"
                status = "🟡 Algo antiguo"
            else:  # Más de 1 día
                age_str = f"{age.days}d"
                status = "🔴 Antiguo"
            
            print(f"   ⏰ Edad: {age_str} - {status}")
    
    def clean_old_cache(self, days: int = 7):
        """Limpiar cache antiguo"""
        print(f"\n🗑️ Limpiando cache anterior a {days} días...")
        self.data_manager.clean_old_cache(days)
        print("✅ Limpieza completada")
    
    def refresh_symbol_data(self, symbol: str, timeframe: str = '1h', 
                           start_date: str = '2023-01-01T00:00:00Z'):
        """Refrescar datos de un símbolo específico"""
        print(f"\n🔄 Refrescando datos: {symbol} ({timeframe})")
        
        try:
            df = self.data_manager.download_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                force_download=True,  # Forzar descarga
                max_cache_age_hours=0  # Ignorar cache
            )
            
            print(f"✅ Datos refrescados: {len(df)} velas")
            return df
        
        except Exception as e:
            print(f"❌ Error refrescando {symbol}: {e}")
            return None
    
    def validate_cache_integrity(self):
        """Validar integridad de todos los archivos de cache"""
        print("\n🔍 VALIDANDO INTEGRIDAD DEL CACHE")
        print("=" * 50)
        
        cached_data = self.data_manager.list_cached_data()
        valid_files = 0
        corrupted_files = 0
        
        for data in cached_data:
            symbol = data['symbol']
            timeframe = data['timeframe']
            start_date = data['start_date']
            
            is_valid, metadata = self.data_manager._is_cache_valid(
                symbol, timeframe, start_date, max_age_hours=999999  # Ignorar edad
            )
            
            if is_valid:
                print(f"✅ {symbol} ({timeframe}) - OK")
                valid_files += 1
            else:
                print(f"❌ {symbol} ({timeframe}) - CORRUPTO")
                corrupted_files += 1
        
        print(f"\n📊 Resumen:")
        print(f"   ✅ Archivos válidos: {valid_files}")
        print(f"   ❌ Archivos corruptos: {corrupted_files}")
        
        if corrupted_files > 0:
            print(f"\n💡 Ejecuta 'clean_corrupted_cache()' para limpiar archivos corruptos")
    
    def get_cache_recommendations(self):
        """Obtener recomendaciones sobre el cache"""
        print("\n💡 RECOMENDACIONES DE CACHE")
        print("=" * 50)
        
        cache_info = self.data_manager.get_cache_info()
        
        # Recomendación de tamaño
        if cache_info['total_size_mb'] > 500:
            print("⚠️ Cache grande (>500MB)")
            print("   💡 Considera limpiar datos antiguos")
        elif cache_info['total_size_mb'] > 100:
            print("📊 Cache moderado (>100MB)")
            print("   ✅ Tamaño aceptable")
        else:
            print("✅ Cache pequeño (<100MB)")
        
        # Recomendación de archivos antiguos
        old_files = 0
        for file_info in cache_info['files']:
            age = datetime.now() - file_info['modified']
            if age.days > 7:
                old_files += 1
        
        if old_files > 0:
            print(f"\n📅 {old_files} archivos tienen más de 7 días")
            print("   💡 Ejecuta clean_old_cache() para limpiarlos")
        
        # Recomendación de símbolos
        if len(cache_info['cached_symbols']) > 10:
            print(f"\n🎯 Muchos símbolos cacheados ({len(cache_info['cached_symbols'])})")
            print("   💡 Considera mantener solo los más usados")
    
    def interactive_menu(self):
        """Menú interactivo para gestionar cache"""
        while True:
            print("\n" + "="*60)
            print("🗃️  CONTROLADOR DE CACHE - STRATEGY TESTER")
            print("="*60)
            print("1. 📊 Ver estado del cache")
            print("2. 📋 Ver detalles de datos cacheados")
            print("3. 🔄 Refrescar datos de un símbolo")
            print("4. 🗑️ Limpiar cache antiguo")
            print("5. 🔍 Validar integridad del cache")
            print("6. 💡 Ver recomendaciones")
            print("7. 🚮 Limpiar todo el cache")
            print("0. 🚪 Salir")
            print("-" * 60)
            
            try:
                choice = input("👉 Selecciona una opción (0-7): ").strip()
                
                if choice == '0':
                    print("👋 ¡Hasta luego!")
                    break
                elif choice == '1':
                    self.show_cache_status()
                elif choice == '2':
                    self.show_cached_data_details()
                elif choice == '3':
                    symbol = input("🎯 Símbolo (ej: BTC/USDT): ").strip()
                    timeframe = input("⏰ Timeframe (ej: 1h): ").strip() or '1h'
                    self.refresh_symbol_data(symbol, timeframe)
                elif choice == '4':
                    days = input("📅 Días para mantener (default: 7): ").strip()
                    days = int(days) if days.isdigit() else 7
                    self.clean_old_cache(days)
                elif choice == '5':
                    self.validate_cache_integrity()
                elif choice == '6':
                    self.get_cache_recommendations()
                elif choice == '7':
                    confirm = input("⚠️ ¿Seguro que quieres limpiar TODO el cache? (sí/no): ").strip().lower()
                    if confirm in ['sí', 'si', 'yes', 'y', 's']:
                        self.data_manager.clear_cache()
                        print("✅ Cache completamente limpiado")
                    else:
                        print("❌ Operación cancelada")
                else:
                    print("❌ Opción no válida")
                    
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Función principal"""
    controller = CacheController()
    controller.interactive_menu()


if __name__ == "__main__":
    main()
