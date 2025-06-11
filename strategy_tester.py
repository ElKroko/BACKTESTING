import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, List, Callable, Any
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class StrategyTester:
    """
    Sistema modular para probar diferentes estrategias de trading
    """
    
    def __init__(self, symbol: str = 'BTC/USDT', timeframe: str = '5m', 
                 start_date: str = '2020-01-01T00:00:00Z'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.data = None
        self.results = {}
        
    def fetch_data(self):
        """Descargar datos de Binance"""
        print(f"Descargando datos de {self.symbol} ({self.timeframe})...")
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(
            self.symbol, 
            timeframe=self.timeframe, 
            since=exchange.parse8601(self.start_date)
        )
        
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        
        self.data = df
        print(f"Datos descargados: {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")
        return df
    
    def apply_strategy(self, strategy_func: Callable, params: Dict[str, Any], 
                      strategy_name: str = None) -> pd.DataFrame:
        """
        Aplicar una estrategia específica a los datos
        
        Args:
            strategy_func: Función que implementa la estrategia
            params: Diccionario con parámetros de la estrategia
            strategy_name: Nombre de la estrategia
        
        Returns:
            DataFrame con señales aplicadas
        """
        if self.data is None:
            raise ValueError("Primero debes descargar los datos con fetch_data()")
        
        df_copy = self.data.copy()
        df_with_signals = strategy_func(df_copy, **params)
        
        if strategy_name:
            self.results[strategy_name] = df_with_signals
            
        return df_with_signals
    
    def calculate_performance(self, df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """
        Calcular métricas de rendimiento de una estrategia
        """
        if 'signal' not in df.columns:
            raise ValueError("El DataFrame debe contener una columna 'signal'")
        
        # Identificar puntos de entrada y salida
        entry_points = df[df['signal'] == 1].copy()
        exit_points = df[df['signal'] == -1].copy()
        
        trades = []
        capital = initial_capital
        position = 0
        
        for i, row in df.iterrows():
            if row['signal'] == 1 and position == 0:  # Entrada
                position = capital / row['close']
                entry_price = row['close']
                entry_time = i
                
            elif row['signal'] == -1 and position > 0:  # Salida
                exit_price = row['close']
                profit = (exit_price - entry_price) * position
                capital += profit
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': profit,
                    'return_pct': (exit_price - entry_price) / entry_price * 100
                })
                
                position = 0
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_return_pct': 0,
                'avg_profit_per_trade': 0,
                'max_profit': 0,
                'max_loss': 0,
                'sharpe_ratio': 0
            }
        
        trades_df = pd.DataFrame(trades)
        
        total_profit = trades_df['profit'].sum()
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] < 0])
        win_rate = winning_trades / len(trades_df) * 100
        
        returns = trades_df['return_pct'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
        
        metrics = {
            'total_trades': len(trades_df),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_return_pct': (capital - initial_capital) / initial_capital * 100,
            'avg_profit_per_trade': total_profit / len(trades_df),
            'max_profit': trades_df['profit'].max(),
            'max_loss': trades_df['profit'].min(),
            'sharpe_ratio': sharpe_ratio,
            'trades': trades_df
        }
        
        return metrics
    
    def compare_strategies(self, strategies: Dict[str, Dict]) -> pd.DataFrame:
        """
        Comparar múltiples estrategias
        
        Args:
            strategies: Dict con formato {nombre: {'func': función, 'params': parámetros}}
        """
        comparison_results = []
        
        for name, strategy_config in strategies.items():
            print(f"\nProbando estrategia: {name}")
            
            # Aplicar estrategia
            df_with_signals = self.apply_strategy(
                strategy_config['func'], 
                strategy_config['params'],
                name
            )
            
            # Calcular rendimiento
            metrics = self.calculate_performance(df_with_signals)
            metrics['strategy_name'] = name
            metrics['params'] = str(strategy_config['params'])
            
            comparison_results.append(metrics)
            
            # Mostrar resultados básicos
            print(f"  - Operaciones: {metrics['total_trades']}")
            print(f"  - Tasa de éxito: {metrics['win_rate']:.2f}%")
            print(f"  - Beneficio total: ${metrics['total_profit']:.2f}")
            print(f"  - Retorno: {metrics['total_return_pct']:.2f}%")
            print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        # Crear DataFrame de comparación
        comparison_df = pd.DataFrame(comparison_results)
        
        # Remover la columna 'trades' para la comparación
        comparison_summary = comparison_df.drop('trades', axis=1, errors='ignore')
        
        return comparison_summary
    
    def plot_strategy_results(self, strategy_name: str, save_plot: bool = True):
        """
        Graficar resultados de una estrategia específica
        """
        if strategy_name not in self.results:
            raise ValueError(f"Estrategia '{strategy_name}' no encontrada en resultados")
        
        df = self.results[strategy_name]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Gráfico de precios y señales
        ax1.plot(df.index, df['close'], label='Precio', alpha=0.7)
        
        # Marcar puntos de entrada y salida
        entry_points = df[df['signal'] == 1]
        exit_points = df[df['signal'] == -1]
        
        ax1.scatter(entry_points.index, entry_points['close'], 
                   color='green', marker='^', s=100, label='Entrada', alpha=0.8)
        ax1.scatter(exit_points.index, exit_points['close'], 
                   color='red', marker='v', s=100, label='Salida', alpha=0.8)
        
        ax1.set_title(f'Estrategia: {strategy_name} - {self.symbol}')
        ax1.set_ylabel('Precio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de señales
        ax2.plot(df.index, df['signal'], label='Señales', linewidth=2)
        ax2.set_ylabel('Señal')
        ax2.set_xlabel('Tiempo')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'strategy_{strategy_name}_{self.symbol.replace("/", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado como: {filename}")
        
        plt.show()


# ========================= ESTRATEGIAS PREDEFINIDAS =========================

def ema_crossover_strategy(df: pd.DataFrame, fast_ema: int = 9, slow_ema: int = 21) -> pd.DataFrame:
    """
    Estrategia de cruce de EMAs
    """
    df[f'ema_{fast_ema}'] = ta.ema(df['close'], length=fast_ema)
    df[f'ema_{slow_ema}'] = ta.ema(df['close'], length=slow_ema)
    
    df['long'] = df[f'ema_{fast_ema}'] > df[f'ema_{slow_ema}']
    df['signal'] = df['long'].diff().fillna(0)
    
    return df


def rsi_strategy(df: pd.DataFrame, rsi_period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
    """
    Estrategia basada en RSI
    """
    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
    
    # Señales de entrada cuando RSI sale de oversold
    # Señales de salida cuando RSI entra en overbought
    df['long'] = (df['rsi'] > oversold) & (df['rsi'].shift(1) <= oversold)
    df['short'] = (df['rsi'] < overbought) & (df['rsi'].shift(1) >= overbought)
    
    df['signal'] = 0
    df.loc[df['long'], 'signal'] = 1
    df.loc[df['short'], 'signal'] = -1
    
    return df


def bollinger_bands_strategy(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2) -> pd.DataFrame:
    """
    Estrategia de Bandas de Bollinger
    """
    bb = ta.bbands(df['close'], length=bb_period, std=bb_std)
    df['bb_upper'] = bb[f'BBU_{bb_period}_{bb_std}']
    df['bb_middle'] = bb[f'BBM_{bb_period}_{bb_std}']
    df['bb_lower'] = bb[f'BBL_{bb_period}_{bb_std}']
    
    # Comprar cuando el precio toca la banda inferior
    # Vender cuando el precio toca la banda superior
    df['long'] = (df['close'] <= df['bb_lower']) & (df['close'].shift(1) > df['bb_lower'].shift(1))
    df['short'] = (df['close'] >= df['bb_upper']) & (df['close'].shift(1) < df['bb_upper'].shift(1))
    
    df['signal'] = 0
    df.loc[df['long'], 'signal'] = 1
    df.loc[df['short'], 'signal'] = -1
    
    return df


def macd_strategy(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """
    Estrategia MACD
    """
    macd = ta.macd(df['close'], fast=fast, slow=slow, signal=signal_period)
    df['macd'] = macd[f'MACD_{fast}_{slow}_{signal_period}']
    df['macd_signal'] = macd[f'MACDs_{fast}_{slow}_{signal_period}']
    df['macd_hist'] = macd[f'MACDh_{fast}_{slow}_{signal_period}']
    
    # Señal de compra cuando MACD cruza por encima de la línea de señal
    # Señal de venta cuando MACD cruza por debajo de la línea de señal
    df['long'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['short'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    df['signal'] = 0
    df.loc[df['long'], 'signal'] = 1
    df.loc[df['short'], 'signal'] = -1
    
    return df


# ========================= EJEMPLO DE USO =========================

if __name__ == "__main__":
    # Crear instancia del tester
    tester = StrategyTester(symbol='BTC/USDT', timeframe='1h', start_date='2023-01-01T00:00:00Z')
    
    # Descargar datos
    tester.fetch_data()
    
    # Definir estrategias a probar
    strategies_to_test = {
        'EMA_9_21': {
            'func': ema_crossover_strategy,
            'params': {'fast_ema': 9, 'slow_ema': 21}
        },
        'EMA_12_26': {
            'func': ema_crossover_strategy,
            'params': {'fast_ema': 12, 'slow_ema': 26}
        },
        'RSI_Default': {
            'func': rsi_strategy,
            'params': {'rsi_period': 14, 'oversold': 30, 'overbought': 70}
        },
        'RSI_Aggressive': {
            'func': rsi_strategy,
            'params': {'rsi_period': 14, 'oversold': 20, 'overbought': 80}
        },
        'Bollinger_Default': {
            'func': bollinger_bands_strategy,
            'params': {'bb_period': 20, 'bb_std': 2}
        },
        'MACD_Default': {
            'func': macd_strategy,
            'params': {'fast': 12, 'slow': 26, 'signal_period': 9}
        }
    }
    
    # Comparar estrategias
    print("=" * 60)
    print("COMPARACIÓN DE ESTRATEGIAS")
    print("=" * 60)
    
    comparison_results = tester.compare_strategies(strategies_to_test)
    
    # Mostrar tabla de comparación
    print("\n" + "=" * 60)
    print("TABLA DE COMPARACIÓN")
    print("=" * 60)
    
    # Seleccionar columnas importantes para mostrar
    cols_to_show = ['strategy_name', 'total_trades', 'win_rate', 'total_return_pct', 'sharpe_ratio']
    print(comparison_results[cols_to_show].round(2).to_string(index=False))
    
    # Guardar resultados
    comparison_results.to_csv('strategy_comparison.csv', index=False)
    print(f"\nResultados guardados en: strategy_comparison.csv")
    
    # Graficar la mejor estrategia (por retorno total)
    best_strategy = comparison_results.loc[comparison_results['total_return_pct'].idxmax(), 'strategy_name']
    print(f"\nMejor estrategia: {best_strategy}")
    
    # Graficar resultados de la mejor estrategia
    tester.plot_strategy_results(best_strategy)
