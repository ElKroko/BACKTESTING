"""
Motor de Backtesting - Paquete principal
"""

from .backtester import BacktestEngine
from .data_manager import DataManager
from .performance_analyzer import PerformanceAnalyzer

__all__ = ['BacktestEngine', 'DataManager', 'PerformanceAnalyzer']
