"""
Debug script para investigar el problema del Max Drawdown
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from engine import BacktestEngine
from strategies import get_strategy
import pandas as pd

def debug_drawdown():
    """Debuggear el cálculo de drawdown"""
    
    print("🔍 DEBUGGING MAX DRAWDOWN")
    print("=" * 50)
    
    # Crear engine
    engine = BacktestEngine()
    
    # Probar con RSI en 4h (que sabemos que es rentable)
    strategy_func = get_strategy('RSI')
    
    result = engine.run_backtest(
        strategy_func=strategy_func,
        strategy_params={},
        symbol='BTC/USDT',
        timeframe='4h',
        start_date='2023-01-01T00:00:00Z',
        initial_capital=10000,
        strategy_name='RSI_DEBUG'
    )
    
    metrics = result['metrics']
    
    print(f"\n📊 RESULTADOS BÁSICOS:")
    print(f"   Retorno Total: {metrics['total_return_pct']:.2f}%")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"   Final Capital: {metrics['final_capital']:.2f}")
    
    # Revisar la equity curve
    equity_curve = metrics['equity_curve']
    print(f"\n🔍 EQUITY CURVE INFO:")
    print(f"   Filas: {len(equity_curve)}")
    print(f"   Columnas: {list(equity_curve.columns)}")
    
    if len(equity_curve) > 0:
        print(f"   Balance inicial: {equity_curve['balance'].iloc[0]:.2f}")
        print(f"   Balance final: {equity_curve['balance'].iloc[-1]:.2f}")
        print(f"   Balance máximo: {equity_curve['balance'].max():.2f}")
        print(f"   Balance mínimo: {equity_curve['balance'].min():.2f}")
        
        # Calcular drawdown manualmente
        balance = equity_curve['balance']
        peak = balance.expanding().max()
        drawdown = ((balance - peak) / peak * 100)
        max_dd = abs(drawdown.min())
        
        print(f"\n🧮 CÁLCULO MANUAL DE DRAWDOWN:")
        print(f"   Peak máximo: {peak.max():.2f}")
        print(f"   Drawdown mínimo: {drawdown.min():.2f}%")
        print(f"   Max Drawdown calculado: {max_dd:.2f}%")
        
        # Mostrar algunos valores para debug
        print(f"\n📈 MUESTRA DE DATOS:")
        sample_size = min(10, len(equity_curve))
        for i in range(sample_size):
            print(f"   {i}: Balance={balance.iloc[i]:.2f}, Peak={peak.iloc[i]:.2f}, DD={drawdown.iloc[i]:.2f}%")
    
    # Revisar trades
    if 'trades_df' in metrics:
        trades_df = metrics['trades_df']
        print(f"\n💼 TRADES INFO:")
        print(f"   Total trades: {len(trades_df)}")
        if len(trades_df) > 0:
            print(f"   Profit promedio: {trades_df['profit'].mean():.2f}")
            print(f"   Profit total: {trades_df['profit'].sum():.2f}")
            print(f"   Trades ganadores: {len(trades_df[trades_df['profit'] > 0])}")
            print(f"   Trades perdedores: {len(trades_df[trades_df['profit'] < 0])}")

if __name__ == "__main__":
    debug_drawdown()
