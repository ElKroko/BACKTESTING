"""
Motor principal de backtesting - Ejecuta estrategias y analiza resultados
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from .data_manager import DataManager
from .performance_analyzer import PerformanceAnalyzer


class BacktestEngine:
    """
    Motor principal de backtesting que combina descarga de datos,
    ejecución de estrategias y análisis de rendimiento
    """
    
    def __init__(self, results_dir: str = "../results"):
        self.data_manager = DataManager()
        self.performance_analyzer = PerformanceAnalyzer()
        self.results_dir = results_dir
        self.results = {}
        self._ensure_results_dir()
        
    def _ensure_results_dir(self):
        """Crear directorio de resultados si no existe"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'charts'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'reports'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'data'), exist_ok=True)
    
    def run_backtest(self, 
                    strategy_func: Callable,
                    strategy_params: Dict[str, Any],
                    symbol: str = 'BTC/USDT',
                    timeframe: str = '1h',
                    start_date: str = '2023-01-01T00:00:00Z',
                    end_date: Optional[str] = None,
                    initial_capital: float = 10000,
                    commission: float = 0.001,
                    slippage: float = 0.0005,
                    strategy_name: Optional[str] = None) -> Dict:
        """
        Ejecutar backtest completo de una estrategia
        
        Args:
            strategy_func: Función que implementa la estrategia
            strategy_params: Parámetros para la estrategia
            symbol: Símbolo a tradear
            timeframe: Marco temporal
            start_date: Fecha de inicio
            end_date: Fecha de fin (opcional)
            initial_capital: Capital inicial
            commission: Comisión por trade
            slippage: Slippage por trade
            strategy_name: Nombre de la estrategia
            
        Returns:
            Diccionario con resultados completos
        """
        if strategy_name is None:
            strategy_name = f"{strategy_func.__name__}_{symbol.replace('/', '_')}_{timeframe}"
        
        print(f"🚀 Ejecutando backtest: {strategy_name}")
        print(f"   📊 Símbolo: {symbol}")
        print(f"   ⏱️ Timeframe: {timeframe}")
        print(f"   📅 Período: {start_date} - {end_date or 'presente'}")
        print(f"   💰 Capital: ${initial_capital:,.2f}")
        
        # 1. Descargar datos
        try:
            df = self.data_manager.download_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            print(f"❌ Error descargando datos: {e}")
            return {'error': str(e)}
        
        # 2. Aplicar estrategia
        try:
            print(f"🔄 Aplicando estrategia...")
            df_with_signals = strategy_func(df.copy(), **strategy_params)
            
            if 'signal' not in df_with_signals.columns:
                raise ValueError("La estrategia debe devolver un DataFrame con columna 'signal'")
                
        except Exception as e:
            print(f"❌ Error aplicando estrategia: {e}")
            return {'error': str(e)}
        
        # 3. Analizar rendimiento
        try:
            print(f"📈 Analizando rendimiento...")
            metrics = self.performance_analyzer.calculate_comprehensive_metrics(
                df_with_signals,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage
            )
        except Exception as e:
            print(f"❌ Error analizando rendimiento: {e}")
            return {'error': str(e)}
        
        # 4. Crear resultado completo
        result = {
            'strategy_name': strategy_name,
            'strategy_params': strategy_params,
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'commission': commission,
            'slippage': slippage,
            'data_info': self.data_manager.get_data_info(df),
            'signals_data': df_with_signals,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # 5. Guardar resultado
        self.results[strategy_name] = result
        
        # 6. Mostrar resumen
        self._print_summary(metrics)
        
        print(f"✅ Backtest completado: {strategy_name}")
        
        return result
    
    def compare_strategies(self, 
                          strategies: Dict[str, Dict],
                          symbol: str = 'BTC/USDT',
                          timeframe: str = '1h',
                          start_date: str = '2023-01-01T00:00:00Z',
                          initial_capital: float = 10000) -> pd.DataFrame:
        """
        Comparar múltiples estrategias
        
        Args:
            strategies: Dict con estrategias {nombre: {'func': función, 'params': parámetros}}
            symbol: Símbolo a tradear
            timeframe: Marco temporal
            start_date: Fecha de inicio
            initial_capital: Capital inicial
            
        Returns:
            DataFrame con comparación de resultados
        """
        print(f"🔄 Comparando {len(strategies)} estrategias...")
        print("=" * 60)
        
        comparison_results = []
        
        for strategy_name, strategy_config in strategies.items():
            print(f"\n🔍 Probando: {strategy_name}")
            
            try:
                result = self.run_backtest(
                    strategy_func=strategy_config['func'],
                    strategy_params=strategy_config['params'],
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    initial_capital=initial_capital,
                    strategy_name=strategy_name
                )
                
                if 'error' not in result:
                    # Extraer métricas principales para comparación
                    metrics = result['metrics']
                    comparison_row = {
                        'strategy_name': strategy_name,
                        'total_trades': metrics['total_trades'],
                        'win_rate': metrics['win_rate'],
                        'total_return_pct': metrics['total_return_pct'],
                        'profit_factor': metrics['profit_factor'],
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'max_drawdown_pct': metrics['max_drawdown_pct'],
                        'avg_trade_duration_hours': metrics['avg_trade_duration_hours'],
                        'params': str(strategy_config['params'])
                    }
                    comparison_results.append(comparison_row)
                else:
                    print(f"   ❌ Error: {result['error']}")
                    
            except Exception as e:
                print(f"   ❌ Error inesperado: {e}")
                continue
        
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            comparison_df = comparison_df.sort_values('total_return_pct', ascending=False)
            
            # Mostrar ranking
            print(f"\n🏆 RANKING DE ESTRATEGIAS")
            print("=" * 80)
            
            for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
                print(f"{i:2d}. {row['strategy_name']:25} | "
                      f"Retorno: {row['total_return_pct']:6.2f}% | "
                      f"Trades: {row['total_trades']:3d} | "
                      f"Win%: {row['win_rate']:5.1f} | "
                      f"Sharpe: {row['sharpe_ratio']:5.2f}")
            
            # Guardar comparación
            comparison_file = os.path.join(self.results_dir, 'data', 'strategy_comparison.csv')
            comparison_df.to_csv(comparison_file, index=False)
            print(f"\n💾 Comparación guardada en: {comparison_file}")
            
            return comparison_df
        else:
            print("❌ No se pudieron obtener resultados para ninguna estrategia")
            return pd.DataFrame()
    
    def generate_charts(self, strategy_name: str, save_charts: bool = True) -> Dict[str, str]:
        """
        Generar gráficos para una estrategia
        
        Args:
            strategy_name: Nombre de la estrategia
            save_charts: Si guardar los gráficos
            
        Returns:
            Diccionario con rutas de archivos generados
        """
        if strategy_name not in self.results:
            raise ValueError(f"Estrategia '{strategy_name}' no encontrada")
        
        result = self.results[strategy_name]
        df = result['signals_data']
        metrics = result['metrics']
        
        chart_files = {}
        
        # 1. Gráfico principal de precio y señales
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Precio y señales
        ax1.plot(df.index, df['close'], label='Precio', alpha=0.8, linewidth=1)
        
        # Marcar señales
        entry_points = df[df['signal'] == 1]
        exit_points = df[df['signal'] == -1]
        
        ax1.scatter(entry_points.index, entry_points['close'], 
                   color='green', marker='^', s=60, label='Compra', alpha=0.8, zorder=5)
        ax1.scatter(exit_points.index, exit_points['close'], 
                   color='red', marker='v', s=60, label='Venta', alpha=0.8, zorder=5)
        
        ax1.set_title(f'{strategy_name} - {result["symbol"]} ({result["timeframe"]})', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Precio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Equity curve
        if len(metrics['equity_curve']) > 0:
            ax2.plot(metrics['equity_curve'].index, metrics['equity_curve']['balance'], 
                    color='blue', linewidth=2, label='Balance')
            ax2.axhline(y=result['initial_capital'], color='gray', linestyle='--', alpha=0.7, label='Capital inicial')
            ax2.set_ylabel('Balance ($)')
            ax2.set_title('Curva de Equity')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Drawdown
            ax3.fill_between(metrics['equity_curve'].index, 
                           metrics['equity_curve']['drawdown'], 0,
                           color='red', alpha=0.3, label='Drawdown')
            ax3.set_ylabel('Drawdown (%)')
            ax3.set_xlabel('Tiempo')
            ax3.set_title('Drawdown')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_charts:
            chart_file = os.path.join(self.results_dir, 'charts', f'{strategy_name}_main_chart.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            chart_files['main_chart'] = chart_file
            print(f"📊 Gráfico principal guardado: {chart_file}")
        
        plt.show()
        
        # 2. Gráfico de métricas de rendimiento
        if len(metrics['trades_df']) > 0:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Distribución de returns
            returns = metrics['trades_df']['return_pct']
            ax1.hist(returns, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(returns.mean(), color='red', linestyle='--', label=f'Media: {returns.mean():.2f}%')
            ax1.set_xlabel('Retorno por Trade (%)')
            ax1.set_ylabel('Frecuencia')
            ax1.set_title('Distribución de Retornos')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Profits vs Losses
            profits = metrics['trades_df'][metrics['trades_df']['profit'] > 0]['profit']
            losses = metrics['trades_df'][metrics['trades_df']['profit'] < 0]['profit']
            
            categories = ['Ganancias', 'Pérdidas']
            values = [len(profits), len(losses)]
            colors = ['green', 'red']
            
            ax2.bar(categories, values, color=colors, alpha=0.7)
            ax2.set_ylabel('Número de Trades')
            ax2.set_title(f'Distribución Win/Loss\nWin Rate: {metrics["win_rate"]:.1f}%')
            ax2.grid(True, alpha=0.3)
            
            # Profits acumulados
            cumulative_profits = metrics['trades_df']['profit'].cumsum()
            ax3.plot(range(len(cumulative_profits)), cumulative_profits, linewidth=2, color='blue')
            ax3.set_xlabel('Número de Trade')
            ax3.set_ylabel('Profit Acumulado ($)')
            ax3.set_title('Evolución de Profits')
            ax3.grid(True, alpha=0.3)
            
            # Trade duration
            durations = metrics['trades_df']['duration_hours']
            ax4.hist(durations, bins=15, alpha=0.7, color='orange', edgecolor='black')
            ax4.axvline(durations.mean(), color='red', linestyle='--', 
                       label=f'Media: {durations.mean():.1f}h')
            ax4.set_xlabel('Duración (horas)')
            ax4.set_ylabel('Frecuencia')
            ax4.set_title('Distribución de Duración de Trades')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_charts:
                metrics_file = os.path.join(self.results_dir, 'charts', f'{strategy_name}_metrics_chart.png')
                plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
                chart_files['metrics_chart'] = metrics_file
                print(f"📊 Gráfico de métricas guardado: {metrics_file}")
            
            plt.show()
        
        return chart_files
    
    def generate_report(self, strategy_name: str) -> str:
        """
        Generar reporte completo para una estrategia
        
        Args:
            strategy_name: Nombre de la estrategia
            
        Returns:
            Ruta del archivo de reporte generado
        """
        if strategy_name not in self.results:
            raise ValueError(f"Estrategia '{strategy_name}' no encontrada")
        
        result = self.results[strategy_name]
        
        # Generar reporte de rendimiento
        report_file = os.path.join(self.results_dir, 'reports', f'{strategy_name}_report.txt')
        report_text = self.performance_analyzer.generate_performance_report(
            result['metrics'], 
            save_path=report_file
        )
        
        return report_file
    
    def save_results(self, strategy_name: str) -> Dict[str, str]:
        """
        Guardar todos los resultados de una estrategia
        
        Args:
            strategy_name: Nombre de la estrategia
            
        Returns:
            Diccionario con rutas de archivos guardados
        """
        if strategy_name not in self.results:
            raise ValueError(f"Estrategia '{strategy_name}' no encontrada")
        
        result = self.results[strategy_name]
        saved_files = {}
        
        # 1. Guardar datos con señales
        signals_file = os.path.join(self.results_dir, 'data', f'{strategy_name}_signals.csv')
        result['signals_data'].to_csv(signals_file)
        saved_files['signals'] = signals_file
        
        # 2. Guardar trades individuales
        if len(result['metrics']['trades_df']) > 0:
            trades_file = os.path.join(self.results_dir, 'data', f'{strategy_name}_trades.csv')
            result['metrics']['trades_df'].to_csv(trades_file, index=False)
            saved_files['trades'] = trades_file
        
        # 3. Guardar equity curve
        if len(result['metrics']['equity_curve']) > 0:
            equity_file = os.path.join(self.results_dir, 'data', f'{strategy_name}_equity.csv')
            result['metrics']['equity_curve'].to_csv(equity_file)
            saved_files['equity'] = equity_file
        
        # 4. Generar reporte
        report_file = self.generate_report(strategy_name)
        saved_files['report'] = report_file
        
        # 5. Generar gráficos
        chart_files = self.generate_charts(strategy_name)
        saved_files.update(chart_files)
        
        print(f"💾 Resultados completos guardados para: {strategy_name}")
        
        return saved_files
    
    def _print_summary(self, metrics: Dict):
        """Imprimir resumen de resultados"""
        print(f"\n📊 RESUMEN DE RESULTADOS:")
        print(f"   Operaciones: {metrics['total_trades']}")
        print(f"   Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   Retorno Total: {metrics['total_return_pct']:.2f}%")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
