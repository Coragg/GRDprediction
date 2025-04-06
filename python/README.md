# Análisis y Predicción de GRD

Este módulo contiene herramientas para el análisis de calidad de datos y la predicción de GRD (Grupo de Riesgo de Desempeño) utilizando técnicas de machine learning.

## Contenido

El archivo principal `grd_analysis_colab.py` implementa las siguientes funcionalidades:

1. **Análisis de Calidad de Datos**:
   - **Completitud**: Detección y análisis de valores faltantes en el dataset.
   - **Correctitud**: Verificación de la validez y precisión de los datos (rangos de edad, distribución de sexo, etc.).
   - **Outliers**: Detección y análisis de valores atípicos en variables numéricas.

2. **Preprocesamiento de Datos**:
   - Creación de rangos de edad.
   - Selección de características relevantes.
   - Codificación de variables categóricas.
   - Normalización de variables numéricas.

3. **Modelado y Evaluación**:
   - Entrenamiento de múltiples modelos (Random Forest, Gradient Boosting, Regresión Logística, SVM, KNN).
   - Evaluación de rendimiento mediante validación cruzada.
   - Comparación de precisión entre modelos.
   - Análisis detallado del mejor modelo.

4. **Visualizaciones**:
   - Gráficos de barras para valores faltantes.
   - Boxplots para detección de outliers.
   - Gráficos de comparación de modelos.
   - Matrices de confusión.
   - Importancia de características.

## Requisitos

Para ejecutar este código se requieren las siguientes bibliotecas:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Uso

1. Asegúrate de tener el archivo de datos `filtered_data_elpino.csv` en el mismo directorio que el script.
2. Ejecuta el script:
   ```
   python grd_analysis_colab.py
   ```
3. Los resultados y gráficos se guardarán en el directorio de trabajo.

## Resultados

El script generará varios archivos de salida:
- `completitud_datos.png`: Gráfico de barras con las columnas con más valores faltantes.
- `distribucion_grd.png`: Distribución de los GRDs más frecuentes.
- `outliers_edad.png`: Boxplot para la detección de outliers en la variable edad.
- `comparacion_modelos.png`: Comparación de precisión entre los diferentes modelos.
- `matriz_confusion.png`: Matriz de confusión del mejor modelo.
- `importancia_caracteristicas.png`: Importancia de las características en el mejor modelo.

## Notas

Este código fue desarrollado originalmente en Google Colab y adaptado para su uso local.
