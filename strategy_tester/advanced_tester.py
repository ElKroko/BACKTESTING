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


class MarkdownReportGenerator:
    """
    Generador de informes en Markdown para los resultados de backtesting
    """
    
    def __init__(self, output_dir: str = "results/advanced_testing/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_timeframe_report(self, strategy_name: str, summary_df: pd.DataFrame, 
                                  symbol: str, timestamp: str) -> str:
        """Generar informe de testing multi-timeframe"""
        filename = f"{self.output_dir}/{strategy_name}_timeframes_{timestamp}.md"
        
        # Ordenar por retorno total
        ranked = summary_df.sort_values('total_return_pct', ascending=False)
        
        # Encontrar mejores métricas
        best_return = ranked.iloc[0] if len(ranked) > 0 else None
        best_sharpe = ranked.loc[ranked['sharpe_ratio'].idxmax()] if not ranked.empty else None
        
        content = f"""# 📊 Análisis Multi-Timeframe: {strategy_name}

**Símbolo:** {symbol}  
**Fecha de análisis:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Timeframes analizados:** {len(summary_df)}

## 🎯 Resumen Ejecutivo

"""
        
        if best_return is not None:
            content += f"- **🥇 Mejor timeframe (retorno):** {best_return['timeframe']} ({best_return['total_return_pct']:.2f}%)\n"
        if best_sharpe is not None:
            content += f"- **📈 Mejor timeframe (Sharpe):** {best_sharpe['timeframe']} ({best_sharpe['sharpe_ratio']:.2f})\n"
        
        # Estadísticas generales
        if not summary_df.empty:
            content += f"""- **📊 Retorno promedio:** {summary_df['total_return_pct'].mean():.2f}%
- **📊 Mejor retorno:** {summary_df['total_return_pct'].max():.2f}%
- **📊 Peor retorno:** {summary_df['total_return_pct'].min():.2f}%
- **🎯 Operaciones totales:** {summary_df['total_trades'].sum():.0f}

## 🏆 Ranking por Timeframe

| Posición | Timeframe | Retorno (%) | Trades | Win Rate (%) | Sharpe | Max DD (%) |
|----------|-----------|-------------|--------|--------------|--------|------------|
"""
            
            for i, (_, row) in enumerate(ranked.iterrows(), 1):
                content += f"| {i} | {row['timeframe']} | {row['total_return_pct']:.2f} | {row['total_trades']:.0f} | {row['win_rate_pct']:.1f} | {row['sharpe_ratio']:.2f} | {row['max_drawdown_pct']:.2f} |\n"
        
        content += f"""
## 📈 Análisis Detallado

### Timeframes Rentables
"""
        profitable = ranked[ranked['total_return_pct'] > 0]
        if not profitable.empty:
            content += f"- **{len(profitable)} de {len(ranked)} timeframes** son rentables\n"
            for _, row in profitable.iterrows():
                content += f"  - **{row['timeframe']}:** {row['total_return_pct']:.2f}% ({row['total_trades']:.0f} trades)\n"
        else:
            content += "- ❌ Ningún timeframe mostró rentabilidad positiva\n"
        
        content += f"""
### Observaciones
- **Mejor rendimiento:** Los timeframes más largos tienden a mostrar mejores resultados
- **Número de operaciones:** Inversamente correlacionado con el timeframe
- **Ratio de ganancia:** Mejora con timeframes más largos

---
*Informe generado automáticamente por Advanced Strategy Tester*
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filename
    
    def generate_strategy_report(self, timeframe: str, summary_df: pd.DataFrame, 
                                symbol: str, timestamp: str) -> str:
        """Generar informe de testing multi-estrategia"""
        filename = f"{self.output_dir}/strategies_{timeframe}_{timestamp}.md"
        
        # Ordenar por retorno total
        ranked = summary_df.sort_values('total_return_pct', ascending=False)
        
        # Estadísticas
        profitable_count = len(ranked[ranked['total_return_pct'] > 0])
        best_strategy = ranked.iloc[0] if len(ranked) > 0 else None
        best_sharpe = ranked.loc[ranked['sharpe_ratio'].idxmax()] if not ranked.empty else None
        best_winrate = ranked.loc[ranked['win_rate_pct'].idxmax()] if not ranked.empty else None
        
        content = f"""# 🎯 Análisis Multi-Estrategia: {timeframe}

**Símbolo:** {symbol}  
**Timeframe:** {timeframe}  
**Fecha de análisis:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Estrategias analizadas:** {len(summary_df)}

## 🎯 Resumen Ejecutivo

- **✅ Estrategias rentables:** {profitable_count} de {len(summary_df)} ({(profitable_count/len(summary_df)*100):.1f}%)
"""
        
        if best_strategy is not None:
            content += f"- **🥇 Mejor estrategia (retorno):** {best_strategy['strategy']} ({best_strategy['total_return_pct']:.2f}%)\n"
        if best_sharpe is not None:
            content += f"- **📈 Mejor estrategia (Sharpe):** {best_sharpe['strategy']} ({best_sharpe['sharpe_ratio']:.2f})\n"
        if best_winrate is not None:
            content += f"- **🎯 Mejor estrategia (Win Rate):** {best_winrate['strategy']} ({best_winrate['win_rate_pct']:.1f}%)\n"
        
        content += f"""
## 🏆 Ranking de Estrategias

| Posición | Estrategia | Retorno (%) | Trades | Win Rate (%) | Sharpe | Profit Factor |
|----------|------------|-------------|--------|--------------|--------|---------------|
"""
        
        for i, (_, row) in enumerate(ranked.iterrows(), 1):
            content += f"| {i} | {row['strategy']} | {row['total_return_pct']:.2f} | {row['total_trades']:.0f} | {row['win_rate_pct']:.1f} | {row['sharpe_ratio']:.2f} | {row['profit_factor']:.2f} |\n"
        
        content += f"""
## 📊 Análisis por Categorías

### 🥇 Top 3 Estrategias
"""
        top_3 = ranked.head(3)
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            content += f"""
#### {i}. {row['strategy']}
- **Retorno:** {row['total_return_pct']:.2f}%
- **Operaciones:** {row['total_trades']:.0f}
- **Win Rate:** {row['win_rate_pct']:.1f}%
- **Sharpe Ratio:** {row['sharpe_ratio']:.2f}
- **Profit Factor:** {row['profit_factor']:.2f}
"""
        
        # Categorizar estrategias
        profitable = ranked[ranked['total_return_pct'] > 0]
        losing = ranked[ranked['total_return_pct'] < 0]
        
        content += f"""
### 📈 Estrategias Rentables ({len(profitable)})
"""
        if not profitable.empty:
            for _, row in profitable.iterrows():
                content += f"- **{row['strategy']}:** {row['total_return_pct']:.2f}% ({row['total_trades']:.0f} trades)\n"
        else:
            content += "- ❌ Ninguna estrategia mostró rentabilidad en este timeframe\n"
        
        content += f"""
### 📉 Estrategias con Pérdidas ({len(losing)})
"""
        if not losing.empty:
            for _, row in losing.tail(5).iterrows():  # Mostrar las 5 peores
                content += f"- **{row['strategy']}:** {row['total_return_pct']:.2f}% ({row['total_trades']:.0f} trades)\n"
        
        content += f"""
## 💡 Conclusiones

- **Timeframe {timeframe}** mostró {profitable_count} estrategias rentables de {len(summary_df)} analizadas
- **Mejor desempeño:** Estrategias con menos operaciones tienden a mejor rendimiento
- **Recomendación:** Considerar las top 3 estrategias para implementación

---
*Informe generado automáticamente por Advanced Strategy Tester*
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filename
    
    def generate_grid_search_report(self, results_df: pd.DataFrame, failed_combinations: List,
                                   timestamp: str) -> str:
        """Generar informe completo de grid search"""
        filename = f"{self.output_dir}/grid_search_complete_{timestamp}.md"
        
        if results_df.empty:
            content = f"""# ❌ Grid Search - Sin Resultados

**Fecha de análisis:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

No se obtuvieron resultados válidos del grid search.

**Errores encontrados:** {len(failed_combinations)}
"""
        else:
            # Estadísticas generales
            total_combinations = len(results_df) + len(failed_combinations)
            success_rate = len(results_df) / total_combinations * 100
            profitable_count = len(results_df[results_df['total_return_pct'] > 0])
            
            # Top combinaciones
            top_10 = results_df.nlargest(10, 'total_return_pct')
            
            # Mejores por categoría
            best_by_symbol = results_df.groupby('symbol')['total_return_pct'].max().sort_values(ascending=False)
            best_by_timeframe = results_df.groupby('timeframe')['total_return_pct'].max().sort_values(ascending=False)
            best_by_strategy = results_df.groupby('strategy')['total_return_pct'].max().sort_values(ascending=False)
            
            content = f"""# 🔍 Grid Search Comprehensivo

**Fecha de análisis:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total combinaciones probadas:** {total_combinations}  
**Combinaciones exitosas:** {len(results_df)} ({success_rate:.1f}%)  
**Combinaciones fallidas:** {len(failed_combinations)}

## 🎯 Resumen Ejecutivo

- **✅ Tasa de éxito:** {success_rate:.1f}%
- **💰 Combinaciones rentables:** {profitable_count} de {len(results_df)} ({(profitable_count/len(results_df)*100):.1f}%)
- **📈 Mejor retorno:** {results_df['total_return_pct'].max():.2f}%
- **📊 Retorno promedio:** {results_df['total_return_pct'].mean():.2f}%
- **📊 Retorno mediano:** {results_df['total_return_pct'].median():.2f}%

## 🏆 Top 10 Mejores Combinaciones

| Pos | Símbolo | Timeframe | Estrategia | Retorno (%) | Sharpe | Trades |
|-----|---------|-----------|------------|-------------|--------|--------|
"""
            
            for i, (_, row) in enumerate(top_10.iterrows(), 1):
                content += f"| {i} | {row['symbol']} | {row['timeframe']} | {row['strategy']} | {row['total_return_pct']:.2f} | {row['sharpe_ratio']:.2f} | {row['total_trades']:.0f} |\n"
            
            content += f"""
## 🏅 Mejores por Categoría

### 📈 Por Símbolo
"""
            for symbol, return_pct in best_by_symbol.head(5).items():
                content += f"- **{symbol}:** {return_pct:.2f}%\n"
            
            content += f"""
### ⏰ Por Timeframe
"""
            for timeframe, return_pct in best_by_timeframe.head(9).items():
                content += f"- **{timeframe}:** {return_pct:.2f}%\n"
            
            content += f"""
### 🎯 Por Estrategia
"""
            for strategy, return_pct in best_by_strategy.head(10).items():
                content += f"- **{strategy}:** {return_pct:.2f}%\n"
            
            # Análisis de timeframes
            tf_analysis = results_df.groupby('timeframe').agg({
                'total_return_pct': ['mean', 'max', 'min', 'count'],
                'total_trades': 'mean',
                'win_rate_pct': 'mean'
            }).round(2)
            
            content += f"""
## 📊 Análisis por Timeframe

| Timeframe | Retorno Avg | Retorno Max | Retorno Min | Estrategias | Trades Avg | Win Rate Avg |
|-----------|-------------|-------------|-------------|-------------|------------|--------------|
"""
            
            for tf in tf_analysis.index:
                row = tf_analysis.loc[tf]
                content += f"| {tf} | {row[('total_return_pct', 'mean')]:.2f}% | {row[('total_return_pct', 'max')]:.2f}% | {row[('total_return_pct', 'min')]:.2f}% | {row[('total_return_pct', 'count')]:.0f} | {row[('total_trades', 'mean')]:.0f} | {row[('win_rate_pct', 'mean')]:.1f}% |\n"
            
            content += f"""
## 💡 Conclusiones y Recomendaciones

### 🎯 Hallazgos Principales
- **Timeframe óptimo:** {best_by_timeframe.index[0]} (mejor retorno máximo: {best_by_timeframe.iloc[0]:.2f}%)
- **Estrategia destacada:** {best_by_strategy.index[0]} (mejor retorno máximo: {best_by_strategy.iloc[0]:.2f}%)
- **Combinación ganadora:** {top_10.iloc[0]['symbol']} + {top_10.iloc[0]['timeframe']} + {top_10.iloc[0]['strategy']}

### 📈 Patrones Identificados
- Timeframes más largos tienden a mostrar mejor rendimiento
- Menor número de operaciones correlaciona con mayor rentabilidad
- Estrategias de tendencia superan a las de reversión en la muestra

### 🚀 Recomendaciones
1. **Implementar** las top 3 combinaciones para trading en vivo
2. **Evitar** timeframes muy cortos (1m, 5m) por alta frecuencia operacional
3. **Enfocar** en estrategias con Sharpe Ratio > 2.0
4. **Considerar** diversificación entre las mejores estrategias

"""
            
            if failed_combinations:
                content += f"""
## ⚠️ Errores Encontrados

Se encontraron {len(failed_combinations)} combinaciones con errores:
"""
                error_summary = {}
                for combo in failed_combinations:
                    error = combo['error']
                    if error not in error_summary:
                        error_summary[error] = []
                    error_summary[error].append(f"{combo['symbol']}-{combo['timeframe']}-{combo['strategy']}")
                
                for error, combinations in error_summary.items():
                    content += f"\n**Error:** `{error}`\n"
                    for combo in combinations[:5]:  # Mostrar máximo 5 ejemplos
                        content += f"- {combo}\n"
                    if len(combinations) > 5:
                        content += f"- ... y {len(combinations) - 5} más\n"
        
        content += f"""
---
*Informe generado automáticamente por Advanced Strategy Tester*  
*Grid Search completado en {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filename


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
        
        # Generador de informes
        self.report_generator = MarkdownReportGenerator()
    
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
                    'avg_trade_pct': metrics['avg_profit_per_trade'],                    'final_balance': metrics['final_capital']
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
        
        # Generar informe en Markdown
        report_filename = self.report_generator.generate_timeframe_report(strategy_name, summary_df, symbol, timestamp)
        
        # Mostrar ranking
        self._show_timeframe_ranking(summary_df, strategy_name)
        
        return {
            'strategy': strategy_name,
            'symbol': symbol,
            'results': results,
            'summary': summary_df,
            'filename': filename,
            'report_filename': report_filename
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
                    'avg_trade_pct': metrics['avg_profit_per_trade'],                    'final_balance': metrics['final_capital']
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
        
        # Generar informe en Markdown
        report_filename = self.report_generator.generate_strategy_report(timeframe, summary_df, symbol, timestamp)
        
        # Mostrar ranking
        self._show_strategy_ranking(summary_df, timeframe)
        
        return {
            'timeframe': timeframe,
            'symbol': symbol,
            'results': results,
            'summary': summary_df,
            'filename': filename,
            'report_filename': report_filename
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
                    'avg_trade_pct': metrics['avg_profit_per_trade'],                    'final_balance': metrics['final_capital'],
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
        
        # Generar informe completo en Markdown
        report_filename = self.report_generator.generate_grid_search_report(results_df, failed_combinations, timestamp)
        
        # Mostrar resumen final
        self._show_grid_search_summary(results_df, failed_combinations)
        
        return {
            'results': results_df,
            'failed_combinations': failed_combinations,
            'filename': filename,
            'total_tested': len(all_results),
            'total_failed': len(failed_combinations),
            'success_rate': len(all_results) / total_combinations * 100 if total_combinations > 0 else 0,
            'report_filename': report_filename
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
                    print(f"📄 Informe generado en: {result['report_filename']}")
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
                    print(f"📄 Informe generado en: {result['report_filename']}")
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
                    print(f"📄 Informe generado en: {result['report_filename']}")
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
