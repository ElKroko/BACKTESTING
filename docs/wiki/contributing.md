# Guía de Contribución

¡Gracias por tu interés en contribuir a la plataforma de Backtesting de Criptomonedas! Esta guía te ayudará a entender el proceso para contribuir de manera efectiva al proyecto.

## Código de Conducta

Por favor, adhiérete a los siguientes principios:

- Sé respetuoso con todos los colaboradores
- Acepta críticas constructivas y retroalimentación
- Enfócate en lo que es mejor para la comunidad
- Muestra empatía hacia otros miembros de la comunidad

## ¿Cómo Puedo Contribuir?

Hay muchas maneras de contribuir al proyecto:

### 1. Reportar Bugs

Si encuentras un bug, por favor crea un informe con la siguiente información:

- Descripción clara y concisa del bug
- Pasos detallados para reproducirlo
- Capturas de pantalla si es posible
- Tu entorno (sistema operativo, versión de Python, etc.)
- Cualquier información adicional relevante

### 2. Sugerir Mejoras

Las sugerencias de mejoras son siempre bienvenidas:

- Describe claramente la mejora y sus beneficios
- Explica cómo esta mejora aportaría valor a los usuarios
- Proporciona ejemplos o mockups si es posible

### 3. Contribuir con Código

Para contribuir con código:

1. **Fork del repositorio**:
   - Crea un fork del repositorio a tu cuenta personal

2. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/TU_USUARIO/NOMBRE_REPOSITORIO.git
   cd NOMBRE_REPOSITORIO
   ```

3. **Crear una rama**:
   ```bash
   git checkout -b feature/nueva-caracteristica
   # o
   git checkout -b fix/arreglo-bug
   ```

4. **Realizar cambios**:
   - Sigue las convenciones de estilo del proyecto
   - Añade pruebas para nuevas funcionalidades
   - Actualiza la documentación relevante

5. **Verificar los cambios**:
   - Ejecuta pruebas si aplica
   - Asegúrate de que todo funciona correctamente

6. **Commit de los cambios**:
   ```bash
   git commit -m "Descripción clara de los cambios"
   ```

7. **Push a tu repositorio**:
   ```bash
   git push origin feature/nueva-caracteristica
   ```

8. **Crear un Pull Request**:
   - Visita tu fork en GitHub y crea un pull request
   - Proporciona una descripción clara de los cambios y su propósito

## Estructura del Proyecto

Para contribuir efectivamente, es útil entender la estructura del proyecto:

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
├── templates/             # Plantillas HTML
└── app_container.py       # Punto de entrada principal
```

## Convenciones de Estilo

Seguimos las convenciones de estilo de PEP 8 para Python:

- Utiliza 4 espacios para la indentación (no tabulaciones)
- Líneas de máximo 79-80 caracteres
- Dos líneas en blanco alrededor de las definiciones de clase y funciones de nivel superior
- Una línea en blanco entre métodos de clase
- Importaciones agrupadas en el siguiente orden:
  1. Librerías estándar
  2. Librerías de terceros
  3. Importaciones locales

## Documentación

Al contribuir, por favor actualiza la documentación relevante:

- Añade docstrings a nuevas funciones y clases
- Actualiza archivos README si es necesario
- Considera crear o actualizar documentos en la carpeta `docs/`

## Pruebas

Aunque actualmente no hay un framework de pruebas establecido, considera:

- Probar manualmente tus cambios en varios escenarios
- En el futuro, podremos implementar pruebas automatizadas

## Proceso de Revisión

Una vez que hayas enviado un pull request:

1. Otros contribuyentes revisarán tu código
2. Pueden sugerir cambios o mejoras
3. Una vez que todo esté en orden, tu contribución será aceptada

## Agradecimientos

¡Todas las contribuciones son valiosas! Los contribuyentes serán reconocidos en nuestro archivo README.

## Preguntas Frecuentes

### ¿Puedo contribuir si soy principiante?

¡Absolutamente! Hay tareas de diferentes niveles de dificultad. Las etiquetas "good first issue" son un buen punto de partida.

### ¿Cómo puedo saber en qué trabajar?

Consulta los issues abiertos o propón nuevas funcionalidades que creas que podrían mejorar la plataforma.

### ¿Dónde puedo pedir ayuda?

No dudes en abrir un issue solicitando orientación o contactar a los mantenedores del proyecto.