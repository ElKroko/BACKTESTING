# Sistema de Prueba de Estrategias de Trading

Este sistema te permite probar diferentes estrategias de trading de forma modular y comparar sus resultados fácilmente.

## 📁 Archivos Creados

1. **`strategy_tester.py`** - Framework principal con clase StrategyTester
2. **`simple_strategy_example.py`** - Ejemplo simple de uso
3. **`test_improved.py`** - Tu código original mejorado
4. **`README_estrategias.md`** - Esta guía

## 🚀 Cómo Usar

### 1. Ejecutar tu Estrategia Original Mejorada

```bash
python test_improved.py
```

Esto ejecutará tu estrategia EMA original pero con métricas mejoradas y gráficos.

### 2. Probar Múltiples Estrategias

```bash
python simple_strategy_example.py
```

Esto probará varias estrategias predefinidas y te mostrará una comparación completa.

### 3. Crear tu Propia Estrategia

```python
def mi_estrategia(df, parametro1=10, parametro2=20):
    # Tu lógica aquí
    df['indicador'] = ta.sma(df['close'], length=parametro1)
    
    # Generar señales
    df['signal'] = 0
    df.loc[condicion_compra, 'signal'] = 1   # Señal de compra
    df.loc[condicion_venta, 'signal'] = -1   # Señal de venta
    
    return df

# Usar la estrategia
tester = StrategyTester()
tester.fetch_data()
result = tester.apply_strategy(mi_estrategia, {'parametro1': 15, 'parametro2': 25})
```

## 🛠️ Clase StrategyTester

### Métodos Principales

- **`fetch_data()`** - Descarga datos de Binance
- **`apply_strategy(func, params)`** - Aplica una estrategia
- **`calculate_performance(df)`** - Calcula métricas de rendimiento
- **`compare_strategies(strategies)`** - Compara múltiples estrategias
- **`plot_strategy_results(name)`** - Genera gráficos

### Configuración

```python
tester = StrategyTester(
    symbol='BTC/USDT',      # Par de trading
    timeframe='1h',         # Marco temporal: '5m', '15m', '1h', '4h', '1d'
    start_date='2023-01-01T00:00:00Z'  # Fecha de inicio
)
```

## 📊 Métricas Calculadas

- **Total de operaciones** - Número total de trades
- **Tasa de éxito** - Porcentaje de trades ganadores
- **Retorno total** - Ganancia/pérdida total en porcentaje
- **Sharpe Ratio** - Relación riesgo/beneficio
- **Ganancia por operación** - Promedio de ganancia por trade
- **Mayor ganancia/pérdida** - Mejor y peor trade

## 📈 Estrategias Predefinidas

### 1. EMA Crossover
```python
ema_crossover_strategy(df, fast_ema=9, slow_ema=21)
```

### 2. RSI
```python
rsi_strategy(df, rsi_period=14, oversold=30, overbought=70)
```

### 3. Bandas de Bollinger
```python
bollinger_bands_strategy(df, bb_period=20, bb_std=2)
```

### 4. MACD
```python
macd_strategy(df, fast=12, slow=26, signal_period=9)
```

## 🎯 Ejemplo Completo

```python
from strategy_tester import StrategyTester, ema_crossover_strategy

# 1. Configurar
tester = StrategyTester(symbol='BTC/USDT', timeframe='1h')
tester.fetch_data()

# 2. Definir estrategias a comparar
estrategias = {
    'EMA_Rapida': {
        'func': ema_crossover_strategy,
        'params': {'fast_ema': 5, 'slow_ema': 15}
    },
    'EMA_Lenta': {
        'func': ema_crossover_strategy,
        'params': {'fast_ema': 12, 'slow_ema': 26}
    }
}

# 3. Comparar
resultados = tester.compare_strategies(estrategias)

# 4. Ver resultados
print(resultados[['strategy_name', 'total_return_pct', 'win_rate']])

# 5. Graficar la mejor
mejor = resultados.loc[resultados['total_return_pct'].idxmax(), 'strategy_name']
tester.plot_strategy_results(mejor)
```

## 🔧 Personalización Avanzada

### Crear Estrategia Personalizada

```python
def mi_estrategia_avanzada(df, sma_period=20, rsi_period=14, volume_threshold=1000000):
    """
    Estrategia que combina SMA, RSI y filtro de volumen
    """
    # Calcular indicadores
    df['sma'] = ta.sma(df['close'], length=sma_period)
    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
    
    # Condiciones
    precio_sobre_sma = df['close'] > df['sma']
    rsi_no_sobrecomprado = df['rsi'] < 70
    rsi_no_sobrevendido = df['rsi'] > 30
    volumen_alto = df['vol'] > volume_threshold
    
    # Señales de compra: precio sobre SMA, RSI neutral, volumen alto
    compra = precio_sobre_sma & rsi_no_sobrecomprado & rsi_no_sobrevendido & volumen_alto
    
    # Señales de venta: precio bajo SMA O RSI extremo
    venta = (~precio_sobre_sma) | (df['rsi'] > 70) | (df['rsi'] < 30)
    
    # Generar señales
    df['signal'] = 0
    df.loc[compra & (~compra.shift(1).fillna(False)), 'signal'] = 1
    df.loc[venta & (~venta.shift(1).fillna(False)), 'signal'] = -1
    
    return df
```

## 📋 Tips para Crear Estrategias

1. **Siempre devolver el DataFrame con columna 'signal'**
2. **Usar 1 para compra, -1 para venta, 0 para sin señal**
3. **Evitar señales consecutivas del mismo tipo**
4. **Incluir parámetros configurables en la función**
5. **Probar con diferentes marcos temporales**

## 🎨 Personalizar Gráficos

Los gráficos se guardan automáticamente como PNG. Puedes modificar el método `plot_strategy_results()` en `strategy_tester.py` para personalizar:

- Colores
- Indicadores adicionales
- Tamaño de gráfico
- Formato de guardado

## 🔍 Debugging

Si tienes errores:

1. **Verifica que tu función devuelva el DataFrame con 'signal'**
2. **Asegúrate de que los parámetros sean correctos**
3. **Revisa que las fechas sean válidas**
4. **Confirma que tienes conexión a internet para descargar datos**

## 🚀 Próximos Pasos

1. Ejecuta `simple_strategy_example.py` para ver el sistema en acción
2. Modifica los parámetros para experimentar
3. Crea tus propias estrategias siguiendo los ejemplos
4. Compara resultados en diferentes marcos temporales
5. Optimiza las estrategias más prometedoras

¡Feliz trading! 📈
