# Crypto Backtesting & Analysis Platform

![Platform Banner](https://img.shields.io/badge/Crypto-Backtesting-blue?style=for-the-badge&logo=bitcoin)

Una plataforma avanzada para análisis técnico, backtesting de estrategias y visualización de conceptos Smart Money en criptomonedas.

## 📋 Índice

- [Crypto Backtesting \& Analysis Platform](#crypto-backtesting--analysis-platform)
  - [📋 Índice](#-índice)
  - [✨ Características](#-características)
    - [📊 Análisis Técnico](#-análisis-técnico)
    - [🔄 Backtesting de Estrategias](#-backtesting-de-estrategias)
    - [💡 Smart Money Concepts](#-smart-money-concepts)
  - [🏗️ Arquitectura del Sistema](#️-arquitectura-del-sistema)
  - [🔧 Instalación](#-instalación)
  - [⚙️ Configuración del Entorno](#️-configuración-del-entorno)
  - [🚀 Uso](#-uso)
    - [📊 Análisis](#-análisis)
    - [🔄 Backtests](#-backtests)
    - [💡 Smart Money Concepts](#-smart-money-concepts-1)
  - [📚 Módulos Principales](#-módulos-principales)
    - [analysis\_tab.py](#analysis_tabpy)
    - [backtest\_tab.py](#backtest_tabpy)
    - [smartmoney\_tab.py](#smartmoney_tabpy)
    - [strategies.py](#strategiespy)
  - [🔌 API y Fuentes de Datos](#-api-y-fuentes-de-datos)
  - [📝 Contribución](#-contribución)
  - [🔄 Actualizaciones](#-actualizaciones)
    - [Abril 2025](#abril-2025)
    - [Historial de versiones anteriores](#historial-de-versiones-anteriores)

## ✨ Características

### 📊 Análisis Técnico
- **Visualización Avanzada**: Gráficos interactivos de TradingView y Plotly
- **Indicadores Técnicos**: SMA, MACD, RSI, MFI y más
- **Niveles Clave**: Soporte, resistencia y niveles pivote
- **Métricas de Derivados**: Funding Rate, Open Interest y Order Flow Delta
- **Reportes PDF**: Generación de informes descargables

### 🔄 Backtesting de Estrategias
- **Estrategias Predefinidas**: 
  - Cruce de Medias Móviles
  - Ruptura de Bollinger
  - Reversión RSI
  - Momentum MACD
  - Ruptura de S/R
- **Métricas de Rendimiento**: Profit Factor, Sharpe Ratio, Drawdown
- **Simulación de Capital**: Prueba con diferentes cantidades iniciales
- **Visualización de Trades**: Equity curve, log de operaciones

### 💡 Smart Money Concepts
- **Order Blocks**: Identificación de bloques de órdenes
- **Fair Value Gaps**: Detección de gaps de valor justo
- **Liquidity Pools**: Visualización de zonas de liquidez

## 🏗️ Arquitectura del Sistema

```
├── tabs/                  # Módulos de pestañas principales
│   ├── analysis_tab.py    # Análisis técnico y métricas de derivados
│   ├── backtest_tab.py    # Backtesting de estrategias
│   ├── smartmoney_tab.py  # Smart Money Concepts
│   └── leveraged_backtest.py  # Backtesting con apalancamiento
├── models/                # Lógica de negocio
│   └── strategies.py      # Implementación de estrategias
├── utils/                 # Utilidades comunes
│   ├── data_utils.py      # Obtención y procesamiento de datos
│   └── html_utils.py      # Elementos HTML personalizados
├── static/                # Recursos estáticos
│   ├── css/               # Estilos
│   └── js/                # Scripts
├── templates/             # Plantillas HTML
└── app_container.py       # Punto de entrada principal
```

## 🔧 Instalación

1. Clona este repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <CARPETA_DEL_REPOSITORIO>
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuración del Entorno

Para configurar el entorno virtual y asegurar que todas las dependencias necesarias estén instaladas:

1. Asegúrate de tener Python instalado en tu sistema.
2. Ejecuta el archivo `setup_venv.bat` haciendo doble clic en él o ejecutando el siguiente comando en la terminal:

   ```bash
   setup_venv.bat
   ```

Este script:
- Crea un entorno virtual en la carpeta `venv`
- Activa el entorno virtual
- Instala las dependencias listadas en el archivo `requirements.txt`

## 🚀 Uso

Ejecuta la aplicación con Streamlit:

```bash
streamlit run app_container.py
```

La aplicación se abrirá en tu navegador con tres pestañas principales:

### 📊 Análisis

1. Ingresa un símbolo (ej. BTCUSDT)
2. Visualiza el gráfico interactivo de TradingView
3. Explora indicadores técnicos por timeframe (horario, diario, semanal)
4. Analiza métricas de derivados:
   - Funding Rate: Indicador de sentimiento del mercado perpetuo
   - Open Interest: Volumen de contratos abiertos
   - Order Flow Delta: Presión compradora vs. vendedora
5. Descarga un reporte PDF con el análisis completo

### 🔄 Backtests

1. Selecciona símbolo, timeframe y estrategia
2. Configura el capital inicial y parámetros adicionales
3. Haz clic en "Run Backtest" para obtener:
   - Métricas de rendimiento (Profit Factor, Win Rate, etc.)
   - Curva de equity
   - Registro detallado de operaciones
4. Compara resultados entre diferentes estrategias

### 💡 Smart Money Concepts

1. Visualiza conceptos avanzados de price action
2. Explora Order Blocks, Fair Value Gaps y Liquidity Pools
3. Analiza la estructura del mercado en diferentes timeframes

## 📚 Módulos Principales

### analysis_tab.py
Proporciona análisis técnico completo con indicadores, niveles y métricas de derivados usando TradingView y Plotly.

### backtest_tab.py
Implementa la funcionalidad de backtesting, simulando órdenes y calculando métricas de rendimiento.

### smartmoney_tab.py
Detecta y visualiza conceptos Smart Money en gráficos de precios.

### strategies.py
Define las estrategias de trading que se pueden probar, cada una aceptando un DataFrame OHLC y devolviendo una lista de operaciones.

## 🔌 API y Fuentes de Datos

- **Binance**: Datos OHLC históricos, funding rate, open interest
- **TradingView**: Widgets de gráficos interactivos
- **Plotly**: Visualización de datos avanzada

## 📝 Contribución

Las contribuciones son bienvenidas. Para contribuir:

1. Haz un fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/nueva-caracteristica`)
3. Haz commit de tus cambios (`git commit -m 'Añade nueva característica'`)
4. Haz push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## 🔄 Actualizaciones

### Abril 2025
- Añadidas métricas de mercados de derivados (Funding Rate, Open Interest, Order Flow Delta)
- Nuevo diseño de interfaz con gráficos interactivos de Plotly
- Mejor manejo de errores de API

### Historial de versiones anteriores
- v1.0.0: Lanzamiento inicial con análisis técnico básico y backtesting
- v1.1.0: Añadido módulo de Smart Money Concepts
- v1.2.0: Mejoras en la interfaz de usuario y optimización de rendimiento

---

**Nota**: Esta aplicación utiliza la API pública de Binance para datos OHLC. Asegúrate de tener una conexión a Internet activa al ejecutar la aplicación.