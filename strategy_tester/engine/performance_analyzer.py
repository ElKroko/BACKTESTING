"""
Analizador de rendimiento - Calcula métricas detalladas de estrategias
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """
    Analizador completo de rendimiento de estrategias de trading
    """
    
    def __init__(self):
        self.trades_history = []
        
    def calculate_comprehensive_metrics(self, 
                                      df: pd.DataFrame, 
                                      initial_capital: float = 10000,
                                      commission: float = 0.001,
                                      slippage: float = 0.0005) -> Dict:
        """
        Calcular métricas completas de rendimiento
        
        Args:
            df: DataFrame con columna 'signal' y precios
            initial_capital: Capital inicial
            commission: Comisión por trade (0.1% = 0.001)
            slippage: Slippage por trade (0.05% = 0.0005)
            
        Returns:
            Diccionario con métricas completas
        """
        if 'signal' not in df.columns:
            raise ValueError("DataFrame debe contener columna 'signal'")
        
        # Calcular trades
        trades = self._extract_trades(df, initial_capital, commission, slippage)
        
        if not trades:
            return self._empty_metrics()
        
        trades_df = pd.DataFrame(trades)
        
        # Métricas básicas
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] < 0])
        break_even_trades = len(trades_df[trades_df['profit'] == 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        loss_rate = (losing_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Profits y losses
        profits = trades_df[trades_df['profit'] > 0]['profit']
        losses = trades_df[trades_df['profit'] < 0]['profit']
        
        total_profit = trades_df['profit'].sum()
        gross_profit = profits.sum() if len(profits) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        
        # Métricas de riesgo-beneficio
        avg_win = profits.mean() if len(profits) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        reward_risk_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf') if avg_win > 0 else 0
        
        # Métricas de retorno
        final_capital = initial_capital + total_profit
        total_return_pct = (final_capital - initial_capital) / initial_capital * 100
        
        # Calcular equity curve para métricas adicionales
        equity_curve = self._calculate_equity_curve(trades_df, initial_capital)
        
        # Drawdown
        drawdown_info = self._calculate_drawdown(equity_curve)
        
        # Sharpe y Sortino ratios
        returns = trades_df['return_pct'].values
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # Métricas de consistencia
        consecutive_wins = self._calculate_consecutive_trades(trades_df, 'win')
        consecutive_losses = self._calculate_consecutive_trades(trades_df, 'loss')
        
        # Métricas de duración
        durations = trades_df['duration_hours'].values
        avg_trade_duration = np.mean(durations) if len(durations) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = total_return_pct / drawdown_info['max_drawdown_pct'] if drawdown_info['max_drawdown_pct'] > 0 else 0
        
        return {
            # Métricas básicas
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'break_even_trades': break_even_trades,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            
            # Métricas financieras
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_profit': total_profit,
            'total_return_pct': total_return_pct,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            
            # Métricas por trade
            'avg_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'reward_risk_ratio': reward_risk_ratio,
            'max_profit': trades_df['profit'].max() if total_trades > 0 else 0,
            'max_loss': trades_df['profit'].min() if total_trades > 0 else 0,
            
            # Métricas de riesgo
            'max_drawdown_pct': drawdown_info['max_drawdown_pct'],
            'max_drawdown_duration': drawdown_info['max_drawdown_duration'],
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Métricas de consistencia
            'max_consecutive_wins': consecutive_wins['max'],
            'max_consecutive_losses': consecutive_losses['max'],
            'avg_consecutive_wins': consecutive_wins['avg'],
            'avg_consecutive_losses': consecutive_losses['avg'],
            
            # Métricas de tiempo
            'avg_trade_duration_hours': avg_trade_duration,
            'shortest_trade_hours': durations.min() if len(durations) > 0 else 0,
            'longest_trade_hours': durations.max() if len(durations) > 0 else 0,
            
            # Datos adicionales
            'trades_df': trades_df,
            'equity_curve': equity_curve,
            'commission_paid': sum([t['commission'] for t in trades]),
            'slippage_cost': sum([t['slippage'] for t in trades])
        }
    
    def _extract_trades(self, df: pd.DataFrame, initial_capital: float, commission: float, slippage: float) -> List[Dict]:
        """Extraer trades individuales del DataFrame"""
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        capital = initial_capital
        
        for timestamp, row in df.iterrows():
            if row['signal'] == 1 and position == 0:  # Entrada
                position = capital / row['close']
                entry_price = row['close']
                entry_time = timestamp
                
            elif row['signal'] == -1 and position > 0:  # Salida
                exit_price = row['close']
                exit_time = timestamp
                
                # Calcular costos
                trade_commission = (entry_price + exit_price) * position * commission
                trade_slippage = (entry_price + exit_price) * position * slippage
                
                # Calcular profit neto
                gross_profit = (exit_price - entry_price) * position
                net_profit = gross_profit - trade_commission - trade_slippage
                
                capital += net_profit
                
                # Calcular duración
                duration = (exit_time - entry_time).total_seconds() / 3600  # horas
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position,
                    'gross_profit': gross_profit,
                    'commission': trade_commission,
                    'slippage': trade_slippage,
                    'profit': net_profit,
                    'return_pct': (exit_price - entry_price) / entry_price * 100,
                    'duration_hours': duration
                })
                position = 0
        
        return trades
    
    def _calculate_equity_curve(self, trades_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        """Calcular curva de equity"""
        equity_curve = pd.DataFrame(index=trades_df['exit_time'])
        equity_curve['balance'] = initial_capital + trades_df['profit'].cumsum()
        equity_curve['drawdown'] = 0.0
        
        # Calcular drawdown running
        peak = initial_capital
        for i, balance in enumerate(equity_curve['balance']):
            if balance > peak:
                peak = balance
            equity_curve.iloc[i, equity_curve.columns.get_loc('drawdown')] = (balance - peak) / peak * 100
        
        return equity_curve
    
    def _calculate_drawdown(self, equity_curve: pd.DataFrame) -> Dict:
        """Calcular métricas de drawdown mejorado"""
        if len(equity_curve) == 0 or 'balance' not in equity_curve.columns:
            return {'max_drawdown_pct': 0, 'max_drawdown_duration': 0}
        
        # Calcular drawdown correctamente desde la curva de balance
        balance = equity_curve['balance']
        
        # Calcular peak running (máximo histórico)
        peak = balance.expanding().max()
        
        # Calcular drawdown en porcentaje
        drawdown = ((balance - peak) / peak * 100).fillna(0)
        
        # Máximo drawdown (valor más negativo)
        max_drawdown_pct = abs(drawdown.min()) if not drawdown.empty else 0
        
        # Si hay NaN o inf, usar 0
        if pd.isna(max_drawdown_pct) or np.isinf(max_drawdown_pct):
            max_drawdown_pct = 0
        
        # Calcular duración del máximo drawdown
        in_drawdown = drawdown < -0.1  # Considerar drawdown > 0.1%
        if in_drawdown.any():
            drawdown_periods = []
            current_period = 0
            
            for is_dd in in_drawdown:
                if is_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0
            
            if current_period > 0:
                drawdown_periods.append(current_period)
            
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        else:
            max_drawdown_duration = 0
        
        return {
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_duration': max_drawdown_duration
        }
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calcular Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calcular Sortino ratio"""
        if len(returns) == 0:
            return 0
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    def _calculate_consecutive_trades(self, trades_df: pd.DataFrame, trade_type: str) -> Dict:
        """Calcular estadísticas de trades consecutivos"""
        if len(trades_df) == 0:
            return {'max': 0, 'avg': 0}
        
        if trade_type == 'win':
            results = trades_df['profit'] > 0
        else:  # loss
            results = trades_df['profit'] < 0
        
        consecutive_counts = []
        current_count = 0
        
        for result in results:
            if result:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive_counts.append(current_count)
                current_count = 0
        
        if current_count > 0:
            consecutive_counts.append(current_count)
        
        return {
            'max': max(consecutive_counts) if consecutive_counts else 0,
            'avg': np.mean(consecutive_counts) if consecutive_counts else 0
        }
    
    def _empty_metrics(self) -> Dict:
        """Métricas vacías cuando no hay trades"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'break_even_trades': 0,
            'win_rate': 0,
            'loss_rate': 0,
            'initial_capital': 0,
            'final_capital': 0,
            'total_profit': 0,
            'total_return_pct': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'profit_factor': 0,
            'avg_profit_per_trade': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'reward_risk_ratio': 0,
            'max_profit': 0,
            'max_loss': 0,
            'max_drawdown_pct': 0,
            'max_drawdown_duration': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_consecutive_wins': 0,
            'avg_consecutive_losses': 0,
            'avg_trade_duration_hours': 0,
            'shortest_trade_hours': 0,
            'longest_trade_hours': 0,
            'trades_df': pd.DataFrame(),
            'equity_curve': pd.DataFrame(),
            'commission_paid': 0,
            'slippage_cost': 0
        }
    
    def generate_performance_report(self, metrics: Dict, save_path: Optional[str] = None) -> str:
        """
        Generar reporte completo de rendimiento
        
        Args:
            metrics: Métricas calculadas
            save_path: Ruta para guardar el reporte
            
        Returns:
            Reporte en formato texto
        """
        report = []
        report.append("=" * 80)
        report.append("REPORTE COMPLETO DE RENDIMIENTO")
        report.append("=" * 80)
        report.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Resumen ejecutivo
        report.append("📊 RESUMEN EJECUTIVO")
        report.append("-" * 40)
        report.append(f"Capital inicial:          ${metrics['initial_capital']:,.2f}")
        report.append(f"Capital final:            ${metrics['final_capital']:,.2f}")
        report.append(f"Retorno total:            {metrics['total_return_pct']:,.2f}%")
        report.append(f"Profit factor:            {metrics['profit_factor']:,.2f}")
        report.append(f"Sharpe ratio:             {metrics['sharpe_ratio']:,.2f}")
        report.append(f"Máximo drawdown:          {metrics['max_drawdown_pct']:,.2f}%")
        report.append("")
        
        # Estadísticas de trading
        report.append("📈 ESTADÍSTICAS DE TRADING")
        report.append("-" * 40)
        report.append(f"Total de operaciones:     {metrics['total_trades']}")
        report.append(f"Operaciones ganadoras:    {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
        report.append(f"Operaciones perdedoras:   {metrics['losing_trades']} ({metrics['loss_rate']:.1f}%)")
        report.append(f"Empates:                  {metrics['break_even_trades']}")
        report.append("")
        report.append(f"Ganancia promedio:        ${metrics['avg_win']:,.2f}")
        report.append(f"Pérdida promedio:         ${metrics['avg_loss']:,.2f}")
        report.append(f"Ratio R/R:                {metrics['reward_risk_ratio']:,.2f}")
        report.append(f"Mayor ganancia:           ${metrics['max_profit']:,.2f}")
        report.append(f"Mayor pérdida:            ${metrics['max_loss']:,.2f}")
        report.append("")
        
        # Métricas de riesgo
        report.append("⚠️ ANÁLISIS DE RIESGO")
        report.append("-" * 40)
        report.append(f"Máximo drawdown:          {metrics['max_drawdown_pct']:,.2f}%")
        report.append(f"Duración max DD:          {metrics['max_drawdown_duration']} trades")
        report.append(f"Sharpe ratio:             {metrics['sharpe_ratio']:,.2f}")
        report.append(f"Sortino ratio:            {metrics['sortino_ratio']:,.2f}")
        report.append(f"Calmar ratio:             {metrics['calmar_ratio']:,.2f}")
        report.append("")
        
        # Consistencia
        report.append("🔄 ANÁLISIS DE CONSISTENCIA")
        report.append("-" * 40)
        report.append(f"Max wins consecutivos:    {metrics['max_consecutive_wins']}")
        report.append(f"Max losses consecutivos:  {metrics['max_consecutive_losses']}")
        report.append(f"Avg wins consecutivos:    {metrics['avg_consecutive_wins']:.1f}")
        report.append(f"Avg losses consecutivos:  {metrics['avg_consecutive_losses']:.1f}")
        report.append("")
        
        # Costos
        report.append("💰 ANÁLISIS DE COSTOS")
        report.append("-" * 40)
        report.append(f"Comisiones pagadas:       ${metrics['commission_paid']:,.2f}")
        report.append(f"Costos de slippage:       ${metrics['slippage_cost']:,.2f}")
        report.append(f"Costos totales:           ${metrics['commission_paid'] + metrics['slippage_cost']:,.2f}")
        report.append("")
        
        # Tiempo
        report.append("⏱️ ANÁLISIS TEMPORAL")
        report.append("-" * 40)
        report.append(f"Duración promedio:        {metrics['avg_trade_duration_hours']:.1f} horas")
        report.append(f"Trade más corto:          {metrics['shortest_trade_hours']:.1f} horas")
        report.append(f"Trade más largo:          {metrics['longest_trade_hours']:.1f} horas")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Reporte guardado en: {save_path}")
        
        return report_text
