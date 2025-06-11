# Configuración del Entorno Virtual - Strategy Tester

## 🚀 Configuración Inicial

### 1. Activar el Entorno Virtual
```powershell
# En PowerShell
.\venv\Scripts\Activate.ps1

# En CMD
.\venv\Scripts\activate.bat
```

### 2. Verificar que el entorno está activo
Deberías ver `(venv)` al inicio de tu línea de comandos.

### 3. Ejecutar la aplicación
```powershell
python main.py
```

## 📦 Dependencias Instaladas

- **pandas**: Manipulación de datos
- **pandas-ta**: Indicadores técnicos
- **ccxt**: Conexión con exchanges de criptomonedas
- **matplotlib**: Visualización de gráficos
- **seaborn**: Visualizaciones estadísticas avanzadas
- **numpy**: Computación numérica (versión < 2.0 para compatibilidad)

## 🎯 Funcionalidades Probadas

### ✅ Diagnóstico del Sistema
- Todas las dependencias funcionando correctamente
- 14 estrategias disponibles
- Motor de backtesting operativo

### 🗃️ Sistema de Cache Inteligente (NUEVO)
- **Cache automático**: Los datos se guardan automáticamente
- **Detección de obsolescencia**: Actualiza solo cuando es necesario
- **Validación de integridad**: Verifica archivos corruptos
- **Gestión inteligente**: Diferentes intervalos según timeframe
- **Metadatos**: Información detallada de cada archivo

#### Funcionalidades del Cache:
- **Carga desde cache**: Si los datos están recientes, los usa directamente
- **Actualización incremental**: Solo descarga datos nuevos cuando es necesario
- **Gestión de memoria**: Compresión y almacenamiento eficiente
- **Limpieza automática**: Elimina archivos antiguos

### 📊 Controlador de Cache
Ejecuta: `python cache_manager.py`

**Opciones disponibles:**
1. **Ver estado del cache**: Archivos, tamaño, símbolos
2. **Ver detalles**: Metadatos completos de cada dataset
3. **Refrescar datos**: Actualizar un símbolo específico
4. **Limpiar cache antiguo**: Eliminar archivos obsoletos
5. **Validar integridad**: Verificar archivos corruptos
6. **Ver recomendaciones**: Sugerencias de optimización
7. **Limpiar todo**: Reiniciar cache completamente

### 📊 Análisis Comprehensivo Realizado
**Símbolo**: BTC/USDT  
**Timeframes**: 1h, 4h, 1d  
**Estrategias probadas**: EMA_Crossover, RSI, MACD, Multi_Indicator

#### Mejores Resultados por Timeframe:
- **1h**: RSI - 28.44% retorno (60% win rate)
- **4h**: EMA_Crossover - 199.21% retorno (32.5% win rate)  
- **1d**: EMA_Crossover - 167.75% retorno (33.3% win rate)

### 🔧 Optimización de Parámetros RSI
**Mejores combinaciones encontradas**:
1. RSI(10) - Oversold: 35, Overbought: 80 → **191.46% retorno**
2. RSI(14) - Oversold: 35, Overbought: 80 → **165.20% retorno**  
3. RSI(20) - Oversold: 35, Overbought: 80 → **163.22% retorno**

## 🔄 Para futuras sesiones

### Activar el entorno:
```powershell
cd f:\Codes\BACKTESTING\strategy_tester
.\venv\Scripts\Activate.ps1
```

### Instalar nuevas dependencias (si es necesario):
```powershell
pip install nombre_paquete
pip freeze > requirements.txt  # Actualizar requirements
```

### Desactivar el entorno:
```powershell
deactivate
```

## 📁 Archivos Generados

- `requirements.txt`: Lista de todas las dependencias
- `results/data/strategy_comparison.csv`: Comparaciones de estrategias
- `results/data/RSI_optimization.csv`: Resultados de optimización RSI
- Cache de datos en `data/`: Datos descargados de exchanges

## 🎮 Opciones del Menú Principal

1. **Listar estrategias**: Ver todas las estrategias disponibles
2. **Probar estrategia específica**: Backtest individual
3. **Comparar variaciones**: Probar diferentes parámetros
4. **Múltiples estrategias**: Comparar varias estrategias
5. **Optimizar parámetros**: Encontrar mejores parámetros
6. **Análisis comprehensivo**: Análisis completo multi-timeframe
7. **Ejemplos predefinidos**: Casos de uso preconfigurados

## ⚠️ Notas Importantes

- El sistema descarga datos reales de Binance
- Los datos se guardan en cache para evitar descargas repetidas
- Los resultados son solo para backtesting, no garantizan rendimientos futuros
- Siempre verifica los resultados antes de usar en trading real

## 🛠️ Solución de Problemas

Si encuentras errores:

1. **Verificar que el entorno esté activo**: `(venv)` debe aparecer en el prompt
2. **Reinstalar dependencias problemáticas**:
   ```powershell
   pip uninstall pandas-ta
   pip install pandas-ta
   ```
3. **Verificar versiones**:
   ```powershell
   pip list
   ```

¡El sistema está listo para usar! 🚀
