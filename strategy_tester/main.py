"""
Script principal del Strategy Tester
Permite probar todas las estrategias de forma organizada
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Añadir el directorio actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from engine import BacktestEngine
from strategies import AVAILABLE_STRATEGIES, list_strategies, get_strategy
from config.settings import (
    DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES, DEFAULT_PERIODS,
    DEFAULT_CAPITAL, DEFAULT_COMMISSION, DEFAULT_SLIPPAGE
)


class StrategyTesterApp:
    """
    Aplicación principal para probar estrategias
    """
    
    def __init__(self):
        self.engine = BacktestEngine(results_dir="results")
        self.available_strategies = AVAILABLE_STRATEGIES
        
    def list_available_strategies(self):
        """Listar todas las estrategias disponibles"""
        print("📋 ESTRATEGIAS DISPONIBLES:")
        print("=" * 50)
        
        categories = {
            'Trend Following': ['EMA_Crossover', 'SMA_Crossover', 'MACD', 'ADX_Trend'],
            'Mean Reversion': ['RSI', 'Bollinger_Bands', 'Stochastic', 'Williams_R'],
            'Momentum': ['Momentum', 'Price_Channel', 'Volatility_Breakout'],
            'Hybrid': ['Multi_Indicator', 'Trend_Momentum', 'Scalping']
        }
        
        for category, strategies in categories.items():
            print(f"\n🔸 {category}:")
            for strategy in strategies:
                if strategy in self.available_strategies:
                    print(f"   • {strategy}")
        
        print(f"\nTotal: {len(self.available_strategies)} estrategias")
    
    def test_single_strategy(self, strategy_name: str, symbol: str = 'BTC/USDT',
                           timeframe: str = '1h', days_back: int = 365,
                           **strategy_params):
        """
        Probar una estrategia individual
        
        Args:
            strategy_name: Nombre de la estrategia
            symbol: Símbolo a tradear
            timeframe: Marco temporal
            days_back: Días hacia atrás
            **strategy_params: Parámetros específicos de la estrategia
        """
        if strategy_name not in self.available_strategies:
            print(f"❌ Estrategia '{strategy_name}' no encontrada")
            self.list_available_strategies()
            return None
        
        # Calcular fecha de inicio
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%dT00:00:00Z')
        
        print(f"🚀 PROBANDO ESTRATEGIA: {strategy_name}")
        print(f"   📊 Símbolo: {symbol}")
        print(f"   ⏱️ Timeframe: {timeframe}")
        print(f"   📅 Período: {days_back} días")
        
        # Ejecutar backtest
        strategy_func = get_strategy(strategy_name)
        result = self.engine.run_backtest(
            strategy_func=strategy_func,
            strategy_params=strategy_params,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            strategy_name=strategy_name
        )
        
        if 'error' not in result:
            # Guardar resultados completos
            saved_files = self.engine.save_results(strategy_name)
            print(f"\n📁 Archivos generados:")
            for file_type, file_path in saved_files.items():
                print(f"   • {file_type}: {file_path}")
        
        return result
    
    def compare_strategy_variations(self, base_strategy: str, param_variations: dict,
                                  symbol: str = 'BTC/USDT', timeframe: str = '1h'):
        """
        Comparar variaciones de una estrategia con diferentes parámetros
        
        Args:
            base_strategy: Nombre de la estrategia base
            param_variations: Dict con variaciones {nombre: parámetros}
            symbol: Símbolo a tradear
            timeframe: Marco temporal
        """
        print(f"🔬 COMPARANDO VARIACIONES DE: {base_strategy}")
        print("=" * 60)
        
        if base_strategy not in self.available_strategies:
            print(f"❌ Estrategia '{base_strategy}' no encontrada")
            return None
        
        strategy_func = get_strategy(base_strategy)
        strategies_dict = {}
        
        # Crear dict de estrategias para comparación
        for variation_name, params in param_variations.items():
            full_name = f"{base_strategy}_{variation_name}"
            strategies_dict[full_name] = {
                'func': strategy_func,
                'params': params
            }
        
        # Comparar estrategias
        comparison_df = self.engine.compare_strategies(
            strategies=strategies_dict,
            symbol=symbol,
            timeframe=timeframe
        )
        
        return comparison_df
    
    def test_multiple_strategies(self, strategy_list: list = None, 
                               symbol: str = 'BTC/USDT', timeframe: str = '1h'):
        """
        Probar múltiples estrategias diferentes
        
        Args:
            strategy_list: Lista de estrategias a probar (None = todas)
            symbol: Símbolo a tradear
            timeframe: Marco temporal
        """
        if strategy_list is None:
            strategy_list = list(self.available_strategies.keys())
        
        print(f"🎯 PROBANDO {len(strategy_list)} ESTRATEGIAS")
        print("=" * 60)
        
        strategies_dict = {}
        
        # Parámetros por defecto para cada tipo de estrategia
        default_params = {
            'EMA_Crossover': {'fast_ema': 12, 'slow_ema': 26},
            'SMA_Crossover': {'fast_sma': 10, 'slow_sma': 30},
            'MACD': {'fast': 12, 'slow': 26, 'signal_period': 9},
            'RSI': {'rsi_period': 14, 'oversold': 30, 'overbought': 70},
            'Bollinger_Bands': {'bb_period': 20, 'bb_std': 2},
            'Stochastic': {'k_period': 14, 'oversold': 20, 'overbought': 80},
            'Multi_Indicator': {'ema_fast': 12, 'ema_slow': 26, 'rsi_period': 14},
            'Trend_Momentum': {'sma_period': 50, 'atr_period': 14}
        }
        
        # Crear dict de estrategias
        for strategy_name in strategy_list:
            if strategy_name in self.available_strategies:
                strategies_dict[strategy_name] = {
                    'func': get_strategy(strategy_name),
                    'params': default_params.get(strategy_name, {})
                }
        
        # Comparar todas las estrategias
        comparison_df = self.engine.compare_strategies(
            strategies=strategies_dict,
            symbol=symbol,
            timeframe=timeframe
        )
        
        return comparison_df
    
    def optimize_strategy_parameters(self, strategy_name: str, param_ranges: dict,
                                   symbol: str = 'BTC/USDT', timeframe: str = '1h'):
        """
        Optimizar parámetros de una estrategia
        
        Args:
            strategy_name: Nombre de la estrategia
            param_ranges: Dict con rangos de parámetros {param: [valores]}
            symbol: Símbolo a tradear
            timeframe: Marco temporal
        """
        print(f"⚙️ OPTIMIZANDO PARÁMETROS: {strategy_name}")
        print("=" * 60)
        
        if strategy_name not in self.available_strategies:
            print(f"❌ Estrategia '{strategy_name}' no encontrada")
            return None
        
        strategy_func = get_strategy(strategy_name)
        
        # Generar todas las combinaciones de parámetros
        from itertools import product
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = list(product(*param_values))
        
        print(f"🔄 Probando {len(combinations)} combinaciones...")
        
        optimization_results = []
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            params_str = "_".join([f"{k}_{v}" for k, v in params.items()])
            test_name = f"{strategy_name}_{params_str}"
            
            print(f"  Progreso: {i+1}/{len(combinations)} - {params}", end='\r')
            
            try:
                result = self.engine.run_backtest(
                    strategy_func=strategy_func,
                    strategy_params=params,
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_name=test_name
                )
                
                if 'error' not in result:
                    metrics = result['metrics']
                    optimization_row = {
                        'params': str(params),
                        'total_return_pct': metrics['total_return_pct'],
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'max_drawdown_pct': metrics['max_drawdown_pct'],
                        'win_rate': metrics['win_rate'],
                        'total_trades': metrics['total_trades'],
                        'profit_factor': metrics['profit_factor']
                    }
                    optimization_row.update(params)
                    optimization_results.append(optimization_row)
                    
            except Exception as e:
                print(f"\n  ❌ Error con {params}: {e}")
                continue
        
        print("\n")
        
        if optimization_results:
            optimization_df = pd.DataFrame(optimization_results)
            optimization_df = optimization_df.sort_values('total_return_pct', ascending=False)
            
            print(f"🏆 TOP 10 COMBINACIONES:")
            print("=" * 80)
            
            top_10 = optimization_df.head(10)
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                print(f"{i:2d}. {row['params'][:50]:<50} | "
                      f"Ret: {row['total_return_pct']:6.2f}% | "
                      f"Sharpe: {row['sharpe_ratio']:5.2f}")
            
            # Guardar resultados de optimización
            optimization_file = os.path.join(self.engine.results_dir, 'data', 
                                           f'{strategy_name}_optimization.csv')
            optimization_df.to_csv(optimization_file, index=False)
            print(f"\n💾 Resultados de optimización guardados: {optimization_file}")
            
            return optimization_df
        else:
            print("❌ No se pudieron obtener resultados de optimización")
            return None
    
    def run_comprehensive_analysis(self, symbol: str = 'BTC/USDT'):
        """
        Ejecutar análisis comprehensivo en múltiples timeframes
        
        Args:
            symbol: Símbolo a analizar
        """
        print(f"🔍 ANÁLISIS COMPREHENSIVO: {symbol}")
        print("=" * 60)
        
        timeframes = ['1h', '4h', '1d']
        top_strategies = ['EMA_Crossover', 'RSI', 'MACD', 'Multi_Indicator']
        
        all_results = {}
        
        for tf in timeframes:
            print(f"\n📊 Analizando timeframe: {tf}")
            
            try:
                tf_results = self.test_multiple_strategies(
                    strategy_list=top_strategies,
                    symbol=symbol,
                    timeframe=tf
                )
                
                if len(tf_results) > 0:
                    all_results[tf] = tf_results
                    
                    # Mostrar mejor estrategia para este timeframe
                    best = tf_results.iloc[0]
                    print(f"  🏆 Mejor: {best['strategy_name']} - {best['total_return_pct']:.2f}%")
                
            except Exception as e:
                print(f"  ❌ Error con {tf}: {e}")
                continue
        
        # Resumen final
        if all_results:
            print(f"\n📈 RESUMEN POR TIMEFRAME:")
            print("=" * 60)
            
            for tf, results in all_results.items():
                if len(results) > 0:
                    best = results.iloc[0]
                    print(f"{tf:4} | {best['strategy_name']:20} | "
                          f"{best['total_return_pct']:6.2f}% | "
                          f"{best['win_rate']:5.1f}% | "
                          f"{best['sharpe_ratio']:5.2f}")
        
        return all_results
    
    def quick_analysis(self, symbol: str = 'BTC/USDT', timeframe: str = '1h'):
        """
        Ejecutar un análisis rápido con estrategias populares
        
        Args:
            symbol: Símbolo a analizar
            timeframe: Marco temporal
        """
        print(f"⚡ ANÁLISIS RÁPIDO: {symbol} ({timeframe})")
        print("=" * 50)
        
        # Estrategias populares para probar
        quick_strategies = {
            'EMA_9_21': {
                'func': get_strategy('EMA_Crossover'),
                'params': {'fast_ema': 9, 'slow_ema': 21}
            },
            'EMA_12_26': {
                'func': get_strategy('EMA_Crossover'),
                'params': {'fast_ema': 12, 'slow_ema': 26}
            },
            'RSI_14': {
                'func': get_strategy('RSI'),
                'params': {'rsi_period': 14, 'oversold': 30, 'overbought': 70}
            },
            'Bollinger_Bands': {
                'func': get_strategy('Bollinger_Bands'),
                'params': {'bb_period': 20, 'bb_std': 2}
            },
            'MACD_Default': {
                'func': get_strategy('MACD'),
                'params': {'fast': 12, 'slow': 26, 'signal': 9}
            }
        }
        
        try:
            # Ejecutar comparación
            results = self.engine.compare_strategies(
                strategies=quick_strategies,
                symbol=symbol,
                timeframe=timeframe
            )
            
            if results is not None and len(results) > 0:
                # Mostrar ranking
                print(f"\n🏆 RANKING DE RENDIMIENTO:")
                print("=" * 80)
                
                ranking = results.sort_values('total_return_pct', ascending=False)
                
                for i, (_, row) in enumerate(ranking.head(5).iterrows(), 1):
                    print(f"{i}. {row['strategy_name']:15} | "
                          f"Retorno: {row['total_return_pct']:6.2f}% | "
                          f"Trades: {row['total_trades']:3d} | "
                          f"Éxito: {row['win_rate']:5.1f}%")
                
                return results
            else:
                print("❌ No se pudieron obtener resultados")
                return None
                
        except Exception as e:
            print(f"❌ Error en análisis rápido: {e}")
            return None

def main():
    """
    Función principal con menú interactivo
    """
    app = StrategyTesterApp()
    
    print("🚀 STRATEGY TESTER - SISTEMA DE BACKTESTING")
    print("=" * 60)
    
    while True:
        print("\n📋 OPCIONES DISPONIBLES:")
        print("1. Listar estrategias disponibles")
        print("2. Probar una estrategia específica")
        print("3. Comparar variaciones de estrategia")
        print("4. Probar múltiples estrategias")
        print("5. Optimizar parámetros")
        print("6. Análisis comprehensivo")
        print("7. Ejemplos predefinidos")
        print("0. Salir")
        
        try:
            opcion = input("\n👉 Selecciona una opción (0-7): ").strip()
            
            if opcion == "0":
                print("👋 ¡Hasta luego!")
                break
                
            elif opcion == "1":
                app.list_available_strategies()
                
            elif opcion == "2":
                strategy = input("📝 Nombre de la estrategia: ").strip()
                symbol = input("📊 Símbolo (default: BTC/USDT): ").strip() or "BTC/USDT"
                timeframe = input("⏱️ Timeframe (default: 1h): ").strip() or "1h"
                
                app.test_single_strategy(strategy, symbol, timeframe)
                
            elif opcion == "3":
                strategy = input("📝 Estrategia base: ").strip()
                
                # Ejemplo para EMA
                if strategy.lower() == "ema_crossover":
                    variations = {
                        'Fast': {'fast_ema': 8, 'slow_ema': 21},
                        'Standard': {'fast_ema': 12, 'slow_ema': 26},
                        'Slow': {'fast_ema': 15, 'slow_ema': 35}
                    }
                else:
                    print("Usando parámetros por defecto...")
                    variations = {'Default': {}}
                
                app.compare_strategy_variations(strategy, variations)
                
            elif opcion == "4":
                print("Probando todas las estrategias principales...")
                app.test_multiple_strategies()
                
            elif opcion == "5":
                strategy = input("📝 Estrategia a optimizar: ").strip()
                
                # Ejemplo para EMA
                if strategy.lower() == "ema_crossover":
                    param_ranges = {
                        'fast_ema': [8, 12, 15],
                        'slow_ema': [21, 26, 30]
                    }
                    app.optimize_strategy_parameters(strategy, param_ranges)
                else:
                    print("Define los rangos de parámetros en el código...")
                
            elif opcion == "6":
                symbol = input("📊 Símbolo a analizar (default: BTC/USDT): ").strip() or "BTC/USDT"
                app.run_comprehensive_analysis(symbol)
                
            elif opcion == "7":
                ejemplos_predefinidos(app)
                
            else:
                print("❌ Opción no válida")
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def ejemplos_predefinidos(app):
    """Ejecutar ejemplos predefinidos"""
    print("\n🎯 EJEMPLOS PREDEFINIDOS:")
    print("1. Comparar EMAs vs SMAs")
    print("2. Optimizar RSI")
    print("3. Probar estrategias híbridas")
    
    ejemplo = input("Selecciona ejemplo (1-3): ").strip()
    
    if ejemplo == "1":
        # Comparar EMA vs SMA
        strategies = {
            'EMA_Fast': {'func': get_strategy('EMA_Crossover'), 'params': {'fast_ema': 12, 'slow_ema': 26}},
            'SMA_Fast': {'func': get_strategy('SMA_Crossover'), 'params': {'fast_sma': 12, 'slow_sma': 26}},
            'EMA_Slow': {'func': get_strategy('EMA_Crossover'), 'params': {'fast_ema': 20, 'slow_ema': 50}},
            'SMA_Slow': {'func': get_strategy('SMA_Crossover'), 'params': {'fast_sma': 20, 'slow_sma': 50}}
        }
        app.engine.compare_strategies(strategies)
        
    elif ejemplo == "2":
        # Optimizar RSI
        param_ranges = {
            'rsi_period': [10, 14, 20],
            'oversold': [20, 30, 35],
            'overbought': [65, 70, 80]
        }
        app.optimize_strategy_parameters('RSI', param_ranges)
        
    elif ejemplo == "3":
        # Probar híbridas
        hybrid_strategies = ['Multi_Indicator', 'Trend_Momentum', 'Scalping']
        app.test_multiple_strategies(hybrid_strategies)


if __name__ == "__main__":
    main()
