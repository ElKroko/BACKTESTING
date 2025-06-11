"""
Sistema Avanzado de Testing - Pruebas comprehensivas de estrategias
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import itertools
import json

# Añadir el directorio actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from engine import BacktestEngine
from strategies import AVAILABLE_STRATEGIES, list_strategies, get_strategy
from config.settings import (
    DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES, DEFAULT_PERIODS,
    DEFAULT_CAPITAL, DEFAULT_COMMISSION, DEFAULT_SLIPPAGE
)


class AdvancedStrategyTester:
    """
    Sistema avanzado para testing comprehensivo de estrategias
    """
    
    def __init__(self):
        self.engine = BacktestEngine()
        self.results_dir = "results/advanced_testing"
        self._ensure_results_dir()
        
        # Timeframes disponibles
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '8h', '12h', '1d', '1w']
        
        # Configuración por defecto
        self.default_config = {
            'capital': DEFAULT_CAPITAL,
            'commission': DEFAULT_COMMISSION,
            'slippage': DEFAULT_SLIPPAGE,
            'start_date': '2023-01-01T00:00:00Z',
            'symbol': 'BTC/USDT'
        }
    
    def _ensure_results_dir(self):
        """Crear directorio de resultados si no existe"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/data", exist_ok=True)
        os.makedirs(f"{self.results_dir}/reports", exist_ok=True)
    
    def test_strategy_all_timeframes(self, 
                                   strategy_name: str,
                                   symbol: str = 'BTC/USDT',
                                   start_date: str = '2023-01-01T00:00:00Z',
                                   capital: float = 10000) -> Dict:
        """
        Probar una estrategia en todos los timeframes
        """
        print(f"🎯 TESTING MULTI-TIMEFRAME: {strategy_name}")
        print(f"📊 Símbolo: {symbol}")
        print("=" * 60)
        
        results = {}
        summary_data = []
        
        for timeframe in self.timeframes:
            print(f"\n🔍 Probando timeframe: {timeframe}")
            
            try:
                # Obtener función de estrategia
                strategy_func = get_strategy(strategy_name)
                
                # Ejecutar backtest
                result = self.engine.run_backtest(
                    strategy_func=strategy_func,
                    strategy_params={},  # Parámetros por defecto
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    initial_capital=capital,
                    strategy_name=strategy_name
                )
                
                results[timeframe] = result
                
                # Extraer métricas clave para resumen
                metrics = result['metrics']
                summary_data.append({
                    'timeframe': timeframe,
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'total_return_pct': metrics['total_return_pct'],
                    'total_trades': metrics['total_trades'],
                    'win_rate_pct': metrics['win_rate'],
                    'profit_factor': metrics['profit_factor'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown_pct': metrics['max_drawdown_pct'],
                    'calmar_ratio': metrics['calmar_ratio'],
                    'avg_trade_pct': metrics['avg_profit_per_trade'],
                    'final_balance': result['final_capital']
                })
                
                print(f"   ✅ {timeframe}: {metrics['total_return_pct']:.2f}% | "
                      f"Trades: {metrics['total_trades']} | "
                      f"Win%: {metrics['win_rate']:.1f}")
                
            except Exception as e:
                print(f"   ❌ Error en {timeframe}: {e}")
                results[timeframe] = None
        
        # Crear DataFrame de resumen
        summary_df = pd.DataFrame(summary_data)
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/data/{strategy_name}_all_timeframes_{timestamp}.csv"
        summary_df.to_csv(filename, index=False)
        
        # Mostrar ranking
        self._show_timeframe_ranking(summary_df, strategy_name)
        
        return {
            'strategy': strategy_name,
            'symbol': symbol,
            'results': results,
            'summary': summary_df,
            'filename': filename
        }
    
    def test_all_strategies_single_timeframe(self,
                                           timeframe: str = '1h',
                                           symbol: str = 'BTC/USDT',
                                           start_date: str = '2023-01-01T00:00:00Z',
                                           capital: float = 10000) -> Dict:
        """
        Probar todas las estrategias en un solo timeframe
        """
        print(f"🎯 TESTING TODAS LAS ESTRATEGIAS: {timeframe}")
        print(f"📊 Símbolo: {symbol}")
        print("=" * 60)
        
        results = {}
        summary_data = []
        
        strategies = list(AVAILABLE_STRATEGIES.keys())
        
        for i, strategy_name in enumerate(strategies, 1):
            print(f"\n🔍 ({i}/{len(strategies)}) Probando: {strategy_name}")
            
            try:
                # Obtener función de estrategia
                strategy_func = get_strategy(strategy_name)
                
                # Ejecutar backtest
                result = self.engine.run_backtest(
                    strategy_func=strategy_func,
                    strategy_params={},  # Parámetros por defecto
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    initial_capital=capital,
                    strategy_name=strategy_name
                )
                
                results[strategy_name] = result
                
                # Extraer métricas clave
                metrics = result['metrics']
                summary_data.append({
                    'strategy': strategy_name,
                    'timeframe': timeframe,
                    'symbol': symbol,
                    'total_return_pct': metrics['total_return_pct'],
                    'total_trades': metrics['total_trades'],
                    'win_rate_pct': metrics['win_rate'],
                    'profit_factor': metrics['profit_factor'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown_pct': metrics['max_drawdown_pct'],
                    'calmar_ratio': metrics['calmar_ratio'],
                    'avg_trade_pct': metrics['avg_profit_per_trade'],
                    'final_balance': result['final_capital']
                })
                
                print(f"   ✅ {metrics['total_return_pct']:.2f}% | "
                      f"Trades: {metrics['total_trades']} | "
                      f"Win%: {metrics['win_rate']:.1f} | "
                      f"Sharpe: {metrics['sharpe_ratio']:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[strategy_name] = None
        
        # Crear DataFrame de resumen
        summary_df = pd.DataFrame(summary_data)
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/data/all_strategies_{timeframe}_{timestamp}.csv"
        summary_df.to_csv(filename, index=False)
        
        # Mostrar ranking
        self._show_strategy_ranking(summary_df, timeframe)
        
        return {
            'timeframe': timeframe,
            'symbol': symbol,
            'results': results,
            'summary': summary_df,
            'filename': filename
        }
    
    def grid_search_all_combinations(self,
                                   symbols: List[str] = None,
                                   timeframes: List[str] = None,
                                   strategies: List[str] = None,
                                   start_date: str = '2023-01-01T00:00:00Z',
                                   capital: float = 10000,
                                   max_combinations: int = None) -> Dict:
        """
        Grid search comprehensivo de todas las combinaciones
        """
        # Valores por defecto
        if symbols is None:
            symbols = ['BTC/USDT']
        if timeframes is None:
            timeframes = self.timeframes
        if strategies is None:
            strategies = list(AVAILABLE_STRATEGIES.keys())
        
        # Generar todas las combinaciones
        all_combinations = list(itertools.product(symbols, timeframes, strategies))
        
        # Limitar si es necesario
        if max_combinations and len(all_combinations) > max_combinations:
            print(f"⚠️ Limitando a {max_combinations} de {len(all_combinations)} combinaciones")
            all_combinations = all_combinations[:max_combinations]
        
        total_combinations = len(all_combinations)
        
        print(f"🚀 GRID SEARCH COMPREHENSIVO")
        print(f"📊 {len(symbols)} símbolos × {len(timeframes)} timeframes × {len(strategies)} estrategias")
        print(f"🔢 Total combinaciones: {total_combinations}")
        print("=" * 70)
        
        all_results = []
        failed_combinations = []
        
        for i, (symbol, timeframe, strategy) in enumerate(all_combinations, 1):
            print(f"\n🔍 ({i}/{total_combinations}) {symbol} | {timeframe} | {strategy}")
            
            try:
                # Obtener función de estrategia
                strategy_func = get_strategy(strategy)
                
                # Ejecutar backtest
                result = self.engine.run_backtest(
                    strategy_func=strategy_func,
                    strategy_params={},  # Parámetros por defecto
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    initial_capital=capital,
                    strategy_name=strategy
                )
                
                # Extraer métricas y agregar información de la combinación
                metrics = result['metrics']
                row_data = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'strategy': strategy,
                    'total_return_pct': metrics['total_return_pct'],
                    'total_trades': metrics['total_trades'],
                    'win_rate_pct': metrics['win_rate'],
                    'profit_factor': metrics['profit_factor'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown_pct': metrics['max_drawdown_pct'],
                    'calmar_ratio': metrics['calmar_ratio'],
                    'avg_trade_pct': metrics['avg_profit_per_trade'],
                    'final_balance': result['final_capital'],
                    'start_date': start_date,
                    'capital': capital,
                    'test_timestamp': datetime.now().isoformat()
                }
                
                all_results.append(row_data)
                
                print(f"   ✅ {metrics['total_return_pct']:.2f}% | "
                      f"Trades: {metrics['total_trades']} | "
                      f"Sharpe: {metrics['sharpe_ratio']:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                failed_combinations.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'strategy': strategy,
                    'error': str(e)
                })
        
        # Crear DataFrame con todos los resultados
        results_df = pd.DataFrame(all_results)
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/data/grid_search_complete_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        
        # Guardar errores si los hay
        if failed_combinations:
            error_filename = f"{self.results_dir}/data/grid_search_errors_{timestamp}.csv"
            pd.DataFrame(failed_combinations).to_csv(error_filename, index=False)
        
        # Mostrar resumen final
        self._show_grid_search_summary(results_df, failed_combinations)
        
        return {
            'results': results_df,
            'failed_combinations': failed_combinations,
            'filename': filename,
            'total_tested': len(all_results),
            'total_failed': len(failed_combinations),
            'success_rate': len(all_results) / total_combinations * 100 if total_combinations > 0 else 0
        }
    
    def _show_timeframe_ranking(self, summary_df: pd.DataFrame, strategy_name: str):
        """Mostrar ranking por timeframe"""
        if summary_df.empty:
            return
        
        print(f"\n🏆 RANKING POR TIMEFRAME - {strategy_name}")
        print("=" * 70)
        
        # Ordenar por retorno total
        ranked = summary_df.sort_values('total_return_pct', ascending=False)
        
        for i, (_, row) in enumerate(ranked.iterrows(), 1):
            print(f" {i:2d}. {row['timeframe']:4s} | "
                  f"Retorno: {row['total_return_pct']:8.2f}% | "
                  f"Trades: {row['total_trades']:4.0f} | "
                  f"Win%: {row['win_rate_pct']:5.1f} | "
                  f"Sharpe: {row['sharpe_ratio']:5.2f}")
        
        # Mejores métricas
        best_return = ranked.iloc[0] if len(ranked) > 0 else None
        best_sharpe = ranked.loc[ranked['sharpe_ratio'].idxmax()] if not ranked.empty else None
        
        if best_return is not None:
            print(f"\n🥇 Mejor retorno: {best_return['timeframe']} ({best_return['total_return_pct']:.2f}%)")
        if best_sharpe is not None:
            print(f"📈 Mejor Sharpe: {best_sharpe['timeframe']} ({best_sharpe['sharpe_ratio']:.2f})")
    
    def _show_strategy_ranking(self, summary_df: pd.DataFrame, timeframe: str):
        """Mostrar ranking por estrategia"""
        if summary_df.empty:
            return
        
        print(f"\n🏆 RANKING DE ESTRATEGIAS - {timeframe}")
        print("=" * 80)
        
        # Ordenar por retorno total
        ranked = summary_df.sort_values('total_return_pct', ascending=False)
        
        for i, (_, row) in enumerate(ranked.iterrows(), 1):
            print(f" {i:2d}. {row['strategy']:20s} | "
                  f"Retorno: {row['total_return_pct']:8.2f}% | "
                  f"Trades: {row['total_trades']:4.0f} | "
                  f"Win%: {row['win_rate_pct']:5.1f} | "
                  f"Sharpe: {row['sharpe_ratio']:5.2f}")
        
        # Mejores por diferentes métricas
        if not ranked.empty:
            best_return = ranked.iloc[0]
            best_sharpe = ranked.loc[ranked['sharpe_ratio'].idxmax()]
            best_winrate = ranked.loc[ranked['win_rate_pct'].idxmax()]
            
            print(f"\n🥇 Mejor retorno: {best_return['strategy']} ({best_return['total_return_pct']:.2f}%)")
            print(f"📈 Mejor Sharpe: {best_sharpe['strategy']} ({best_sharpe['sharpe_ratio']:.2f})")
            print(f"🎯 Mejor Win Rate: {best_winrate['strategy']} ({best_winrate['win_rate_pct']:.1f}%)")
    
    def _show_grid_search_summary(self, results_df: pd.DataFrame, failed_combinations: List):
        """Mostrar resumen del grid search"""
        if results_df.empty:
            print("\n❌ No se obtuvieron resultados válidos")
            return
        
        print(f"\n🎊 RESUMEN GRID SEARCH")
        print("=" * 70)
        print(f"✅ Combinaciones exitosas: {len(results_df)}")
        print(f"❌ Combinaciones fallidas: {len(failed_combinations)}")
        
        # Top 10 mejores combinaciones
        print(f"\n🏆 TOP 10 MEJORES COMBINACIONES:")
        print("-" * 80)
        
        top_combinations = results_df.nlargest(10, 'total_return_pct')
        
        for i, (_, row) in enumerate(top_combinations.iterrows(), 1):
            print(f" {i:2d}. {row['symbol']:10s} | {row['timeframe']:4s} | "
                  f"{row['strategy']:15s} | {row['total_return_pct']:8.2f}% | "
                  f"Sharpe: {row['sharpe_ratio']:5.2f}")
        
        # Estadísticas generales
        print(f"\n📊 ESTADÍSTICAS GENERALES:")
        print(f"   • Retorno promedio: {results_df['total_return_pct'].mean():.2f}%")
        print(f"   • Retorno mediano: {results_df['total_return_pct'].median():.2f}%")
        print(f"   • Mejor retorno: {results_df['total_return_pct'].max():.2f}%")
        print(f"   • Peor retorno: {results_df['total_return_pct'].min():.2f}%")
        print(f"   • Sharpe promedio: {results_df['sharpe_ratio'].mean():.2f}")
        
        # Mejores por categoría
        print(f"\n🏅 MEJORES POR CATEGORÍA:")
        
        # Por símbolo
        if 'symbol' in results_df.columns:
            best_by_symbol = results_df.groupby('symbol')['total_return_pct'].max().sort_values(ascending=False)
            print(f"   📈 Mejor símbolo: {best_by_symbol.index[0]} ({best_by_symbol.iloc[0]:.2f}%)")
        
        # Por timeframe
        best_by_timeframe = results_df.groupby('timeframe')['total_return_pct'].max().sort_values(ascending=False)
        print(f"   ⏰ Mejor timeframe: {best_by_timeframe.index[0]} ({best_by_timeframe.iloc[0]:.2f}%)")
        
        # Por estrategia
        best_by_strategy = results_df.groupby('strategy')['total_return_pct'].max().sort_values(ascending=False)
        print(f"   🎯 Mejor estrategia: {best_by_strategy.index[0]} ({best_by_strategy.iloc[0]:.2f}%)")


def main():
    """Función principal con menú interactivo"""
    tester = AdvancedStrategyTester()
    
    while True:
        print("\n" + "="*70)
        print("🚀 SISTEMA AVANZADO DE TESTING DE ESTRATEGIAS")
        print("="*70)
        print("1. 📊 Probar UNA estrategia en TODOS los timeframes")
        print("2. 🎯 Probar TODAS las estrategias en UN timeframe")
        print("3. 🔍 Grid Search COMPLETO (todas × todas)")
        print("4. 📋 Listar estrategias disponibles")
        print("5. ⚙️ Configuración personalizada")
        print("0. 🚪 Salir")
        print("-" * 70)
        
        try:
            choice = input("👉 Selecciona una opción (0-5): ").strip()
            
            if choice == '0':
                print("👋 ¡Hasta luego!")
                break
                
            elif choice == '1':
                # Una estrategia, todos los timeframes
                strategies = list(AVAILABLE_STRATEGIES.keys())
                print(f"\n📋 Estrategias disponibles:")
                for i, strategy in enumerate(strategies, 1):
                    print(f"   {i:2d}. {strategy}")
                
                strategy_idx = input("\n👉 Selecciona estrategia (número): ").strip()
                if strategy_idx.isdigit() and 1 <= int(strategy_idx) <= len(strategies):
                    strategy_name = strategies[int(strategy_idx) - 1]
                    symbol = input("📊 Símbolo (default: BTC/USDT): ").strip() or 'BTC/USDT'
                    
                    result = tester.test_strategy_all_timeframes(
                        strategy_name=strategy_name,
                        symbol=symbol
                    )
                    print(f"\n💾 Resultados guardados en: {result['filename']}")
                else:
                    print("❌ Selección inválida")
            
            elif choice == '2':
                # Todas las estrategias, un timeframe
                print(f"\n⏰ Timeframes disponibles:")
                for i, tf in enumerate(tester.timeframes, 1):
                    print(f"   {i:2d}. {tf}")
                
                tf_idx = input("\n👉 Selecciona timeframe (número): ").strip()
                if tf_idx.isdigit() and 1 <= int(tf_idx) <= len(tester.timeframes):
                    timeframe = tester.timeframes[int(tf_idx) - 1]
                    symbol = input("📊 Símbolo (default: BTC/USDT): ").strip() or 'BTC/USDT'
                    
                    result = tester.test_all_strategies_single_timeframe(
                        timeframe=timeframe,
                        symbol=symbol
                    )
                    print(f"\n💾 Resultados guardados en: {result['filename']}")
                else:
                    print("❌ Selección inválida")
            
            elif choice == '3':
                # Grid search completo
                print("\n⚠️ ADVERTENCIA: Grid search completo puede tomar MUCHO tiempo")
                symbols_input = input("📊 Símbolos (separados por coma, default: BTC/USDT): ").strip()
                symbols = [s.strip() for s in symbols_input.split(',')] if symbols_input else ['BTC/USDT']
                
                max_comb = input("🔢 Máximo combinaciones (default: sin límite): ").strip()
                max_combinations = int(max_comb) if max_comb.isdigit() else None
                
                confirm = input("👉 ¿Continuar? (sí/no): ").strip().lower()
                if confirm in ['sí', 'si', 'yes', 'y', 's']:
                    result = tester.grid_search_all_combinations(
                        symbols=symbols,
                        max_combinations=max_combinations
                    )
                    print(f"\n💾 Resultados guardados en: {result['filename']}")
                    print(f"📊 Éxito: {result['success_rate']:.1f}%")
                else:
                    print("❌ Operación cancelada")
            
            elif choice == '4':
                # Listar estrategias
                strategies = list(AVAILABLE_STRATEGIES.keys())
                print(f"\n📋 ESTRATEGIAS DISPONIBLES ({len(strategies)}):")
                print("-" * 50)
                for i, strategy in enumerate(strategies, 1):
                    print(f"   {i:2d}. {strategy}")
                
                print(f"\n⏰ TIMEFRAMES DISPONIBLES ({len(tester.timeframes)}):")
                print("-" * 50)
                for i, tf in enumerate(tester.timeframes, 1):
                    print(f"   {i:2d}. {tf}")
            
            elif choice == '5':
                # Configuración personalizada
                print(f"\n⚙️ CONFIGURACIÓN ACTUAL:")
                print(f"   💰 Capital inicial: ${tester.default_config['capital']:,.2f}")
                print(f"   📅 Fecha inicio: {tester.default_config['start_date']}")
                print(f"   💱 Comisión: {tester.default_config['commission']*100:.3f}%")
                print(f"   ⚡ Slippage: {tester.default_config['slippage']*100:.3f}%")
                
                # Aquí se podría agregar funcionalidad para cambiar la configuración
                print("\n💡 Funcionalidad de configuración personalizada próximamente...")
            
            else:
                print("❌ Opción no válida")
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
