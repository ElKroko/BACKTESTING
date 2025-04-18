"""
Paquete de utilidades para el sistema de backtesting.

Este paquete contiene módulos de utilidades para:
- Procesamiento de datos (data_utils)
- Generación de HTML y componentes UI (html_utils)

Importaciones comunes están disponibles directamente desde el paquete.
"""

# Exposición de funciones comunes para acceso directo
from utils.html_utils import tooltip, create_info_card, create_stats_row, create_tabbed_content
from utils.data_utils import get_data, calculate_indicators, format_backtest_summary

# Versión del paquete
__version__ = '1.0.0'