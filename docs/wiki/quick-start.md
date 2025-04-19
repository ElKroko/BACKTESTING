# Guía de Inicio Rápido

## Introducción

Esta guía te ayudará a poner en marcha la plataforma de Backtesting de Criptomonedas en minutos, desde la instalación hasta la ejecución de tu primer análisis y backtest.

## Requisitos Previos

- Python 3.8 o superior
- Conexión a Internet (para obtener datos de mercado)
- Conocimientos básicos de análisis técnico y trading

## Instalación en 4 Pasos

### 1. Clonar el Repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd <CARPETA_DEL_REPOSITORIO>
```

### 2. Configurar el Entorno Virtual

Ejecuta el script de configuración incluido:

```bash
# En Windows
setup_venv.bat

# En Linux/Mac
chmod +x setup_venv.sh
./setup_venv.sh
```

### 3. Verificar Dependencias

Las principales dependencias que se instalarán son:
- streamlit (interfaz de usuario)
- pandas (manipulación de datos)
- numpy (cálculos numéricos)
- plotly (visualización)
- ta (indicadores técnicos)

### 4. Iniciar la Aplicación

```bash
streamlit run app_container.py
```

La aplicación se abrirá automáticamente en tu navegador predeterminado.

## Tu Primer Análisis en 3 Minutos

### 1. Accede a la Pestaña de Análisis

Haz clic en la pestaña **"Análisis"** en la parte superior de la aplicación.

### 2. Introduce un Símbolo

En el campo de texto, escribe un símbolo de criptomoneda como `BTCUSDT` o `ETHUSDT`.

### 3. Explora el Análisis

- En el panel izquierdo, verás el gráfico interactivo de TradingView
- En el panel derecho, explora los diferentes timeframes (Semanal, Diario, Horario)
- Desplázate hacia abajo para ver las métricas de mercados de derivados

## Tu Primer Backtest en 5 Minutos

### 1. Accede a la Pestaña de Backtesting

Haz clic en la pestaña **"Backtest"** en la parte superior de la aplicación.

### 2. Configura el Backtest

En el panel lateral:
- Selecciona un símbolo (ej. `BTCUSDT`)
- Elige un intervalo (ej. `1d` para diario)
- Selecciona una estrategia (ej. `SMA Crossover`)
- Establece un capital inicial (ej. `10000`)
- Ajusta los parámetros específicos de la estrategia

### 3. Ejecuta y Analiza

- Haz clic en el botón "Run Backtest"
- Observa los resultados:
  - Métricas de rendimiento (Profit Factor, Win Rate, etc.)
  - Curva de equity
  - Registro de operaciones

## Próximos Pasos

Una vez que te hayas familiarizado con la plataforma, puedes:

1. **Personalizar estrategias existentes**:
   - Ajusta los parámetros para optimizar los resultados
   - Experimenta con diferentes timeframes y símbolos

2. **Crear nuevas estrategias**:
   - Consulta [Implementando Nuevas Estrategias](../examples/implementing_strategies.md)
   - Añade tus propias ideas de trading al sistema

3. **Explorar conceptos avanzados**:
   - Smart Money Concepts en la pestaña correspondiente
   - Análisis de métricas de derivados para una visión más profunda

4. **Exportar y compartir análisis**:
   - Genera informes PDF para compartir tus análisis
   - Guarda configuraciones de backtesting para futuras referencias

## Solución de Problemas Comunes

### Error al obtener datos

Si recibes errores al obtener datos de Binance:

1. Verifica tu conexión a Internet
2. Confirma que el símbolo introducido es válido
3. Espera unos minutos (posible limitación de la API)

### Rendimiento lento

Si la aplicación funciona lentamente:

1. Reduce el rango de datos para el backtest
2. Cierra otras aplicaciones que consuman muchos recursos
3. Considera usar un equipo más potente para backtest extensivos

## Recursos Adicionales

- [Documentación completa](../index.md)
- [Ejemplos de código](../examples/)
- [Guía de contribución](./contributing.md)