# Aplicación de CRISP-DM en la Deteccion-Deterioro-Cognitivo

## Introducción
Este proyecto se enfoca en la implementación del proceso CRISP-DM para un problema de detección y clasificación de deterioro cognitivo en adultos mayores. Se trabajará con un conjunto de datos que contiene 24 atributos, obtenidos de respuestas al test ACE-III. El objetivo es entrenar modelos que clasifiquen el deterioro cognitivo en cuatro categorías, cada una con un número variable de grados.

Los algoritmos de machine learning que se utilizarán incluyen regresión logística, máquinas de soporte vectorial (SVM), árboles de decisión y k-Nearest Neighbors (K-NN). Se evaluarán los modelos en términos de tasa de acierto, precisión, recall y f1-score, usando pipelines para simplificar el flujo de procesamiento.

---
## Pasos de CRISP-DM

### 1. Comprensión del Negocio
- **Objetivo**: Clasificar el grado de deterioro cognitivo en adultos mayores basado en 24 atributos de sus respuestas en el test ACE-III.
- **Categorías de Clases**: Cuatro categorías, cada una representando un nivel de deterioro cognitivo. La primera categoría contiene 7 grados, la segunda y tercera contienen 3 cada una, y la cuarta es una categoría binaria.
- **Éxito del Modelo**: Se busca maximizar la métrica `recall` dado que es esencial identificar correctamente los casos positivos de deterioro cognitivo.

### 2. Comprensión de los Datos
- Analizar la distribución de cada uno de los 24 atributos.
- Investigar la correlación entre atributos.
- Visualizar las clases para verificar la distribución de los grados de deterioro cognitivo y observar posibles desbalances.

### 3. Preparación de los Datos
#### Preprocesamiento:
- **Limpieza**: Verificar valores faltantes y decidir cómo manejarlos (p. ej., imputación o eliminación).
- **Codificación**: Convertir los atributos categóricos a variables numéricas si es necesario.
- **Normalización/Estandarización**: Escalar los datos para optimizar el rendimiento de los modelos, en especial para SVM y K-NN.
  
#### Reducción de Características:
- Aplicar al menos una técnica de reducción de características, como SelectKBest o selección basada en importancia, para ver su impacto en el modelo. Se realizará un experimento con los datos completos y otro aplicando reducción de dimensionalidad.

### 4. Modelado
En esta sección se realizarán los experimentos. Se harán 4 experimentos en total, siendo estos experimentos con y sin reducción de dimensionalidad, y experimentos con y sin estandarización de datos.
- **Modelos Seleccionados**:
  - Regresión Logística
  - SVM
  - Árboles de Decisión
  - K-NN
- **Pipelines**: Configurar un pipeline que incluya los pasos de preprocesamiento y entrenamiento del modelo, lo cual asegura un flujo de trabajo replicable y eficiente.
- **Evaluación**: Utilizar la técnica de `shuffle split cross-validation` para evaluar el rendimiento de cada modelo en términos de tasa de acierto, precisión, recall y f1-score.

### 5. Evaluación
- Comparar el rendimiento de los modelos utilizando las métricas mencionadas, con especial énfasis en el `recall`.
- Identificar el modelo con el mejor balance de recall, precisión, y f1-score, tanto en el experimento con todas las características como en el experimento con reducción de características.
- Analizar y documentar el impacto de la reducción de dimensionalidad en el rendimiento de cada modelo.

### 6. Implementación y Documentación de Resultados
- Documentar los modelos con los mejores resultados para cada métrica, destacando los pasos del pipeline y los hiperparámetros seleccionados.
- Incluir gráficos y tablas que muestren el rendimiento de los modelos en las diferentes clases y categorías.
- Discutir posibles mejoras, como probar otras técnicas de reducción de características o añadir validaciones adicionales.

## Librerías Utilizadas

Este proyecto hace uso de las siguientes librerías de Python:

- **pandas**: Para manipulación y análisis de datos.
- **scikit-learn**: Para la implementación de modelos de machine learning y técnicas de preprocesamiento.
  - `RandomizedSearchCV`: Para realizar búsquedas aleatorias de hiperparámetros.
  - `ShuffleSplit`: Para la validación cruzada de los modelos.
  - `LogisticRegression`: Para la regresión logística.
  - `SVC`: Para la máquina de soporte vectorial.
  - `DecisionTreeClassifier`: Para los árboles de decisión.
  - `KNeighborsClassifier`: Para el modelo de k-Nearest Neighbors.
  - `Pipeline`: Para la creación de pipelines que simplifican el flujo de trabajo.
  - `StandardScaler`: Para la estandarización de características.
  - `make_scorer`, `precision_score`, `recall_score`, `f1_score`: Para calcular métricas de evaluación de modelos.
  - `SelectKBest`, `f_classif`: Para la selección de características más relevantes.
- **tabulate**: Para crear tablas bien formateadas en texto.
- **matplotlib**: Para la visualización de resultados mediante gráficos.
- **numpy**: Para operaciones matemáticas y manejo de arreglos numéricos.

Si no tienes las librerías instaladas, puedes ejecutar el siguiente comando para instalarlas:

```bash
pip install pandas scikit-learn tabulate matplotlib numpy
```

---
### Dataset
El dataset original, `det_con.csv`, fue obtenido de una fuente privada. Debido a restricciones de distribución, solo se proporciona una muestra del 50% de los datos en el archivo `det_con_sample.csv`.

**Nota**: La muestra es solo para fines de prueba y no contiene todo el contenido del dataset original.

---

## Resultados

**Nota: Los resultados corresponden solo al análisis de la clase 4**

### Con Estandarización y Sin Reducir Dimensionalidad

| Algoritmo           | Precisión | Recall  | F1-Score | Hiperparámetros                                                |
|---------------------|-----------|---------|----------|---------------------------------------------------------------|
| Regresión Logística | 0.9187    | 0.9205  | 0.9153   | clf__solver: lbfgs, clf__C: 0.1                                |
| SVM                 | 0.9288    | 0.9295  | 0.9239   | clf__kernel: linear, clf__C: 0.1                               |
| Árboles de Decisión | 0.8880    | 0.8911  | 0.8882   | clf__min_samples_split: 10, clf__max_depth: 10                 |
| K-NN                | 0.9178    | 0.9187  | 0.9115   | clf__weights: uniform, clf__n_neighbors: 9                     |

En este caso, el modelo con mejores métricas de evaluación fue el **SVM** con un kernel lineal, destacando especialmente su `recall` más alto que los otros tres modelos.

De esto se puede concluir que el modelo **SVM**, con un kernel seleccionado acorde al tipo de distribución de los datos, y teniendo previamente una estandarización de los mismos, puede dar muy buenos resultados.

### Sin Estandarizar y Con Reducir Dimensionalidad

| Algoritmo           | Precisión | Recall  | F1-Score | Hiperparámetros                                                |
|---------------------|-----------|---------|----------|---------------------------------------------------------------|
| Regresión Logística | 0.9216    | 0.9241  | 0.9199   | clf__solver: lbfgs, clf__C: 0.1                                |
| SVM                 | 0.9208    | 0.9223  | 0.9154   | clf__kernel: linear, clf__C: 0.1                               |
| Árboles de Decisión | 0.9026    | 0.9027  | 0.8919   | clf__min_samples_split: 10, clf__max_depth: 30                 |
| K-NN                | 0.9326    | 0.9357  | 0.9313   | clf__weights: uniform, clf__n_neighbors: 9                     |

En este caso, el modelo con mayor `recall` y métricas generales fue el **K-NN**. El modelo **SVM** fue el peor en rendimiento, lo que resalta la importancia de la estandarización de los datos para ese modelo. Por otro lado, **K-NN** no muestra una gran influencia con o sin estandarización. Sin embargo, la gran diferencia está en la reducción de dimensionalidad, lo cual resultó ser una ventaja considerable para el **K-NN**.

### Con Estandarización y Con Reducir Dimensionalidad

| Algoritmo           | Precisión | Recall  | F1-Score | Hiperparámetros                                                |
|---------------------|-----------|---------|----------|---------------------------------------------------------------|
| Regresión Logística | 0.9184    | 0.9179  | 0.9097   | feature_selection__k: 10, clf__solver: lbfgs, clf__C: 0.01     |
| SVM                 | 0.9198    | 0.9214  | 0.9161   | feature_selection__k: 5, clf__kernel: rbf, clf__C: 1           |
| Árboles de Decisión | 0.9231    | 0.9241  | 0.9194   | feature_selection__k: 5, clf__min_samples_split: 10, clf__max_depth: 30 |
| K-NN                | 0.9313    | 0.9295  | 0.9238   | feature_selection__k: 5, clf__weights: uniform, clf__n_neighbors: 7 |

En este caso, el modelo **K-NN** muestra una gran superioridad tanto en términos de `recall` como de `F1-Score`, especialmente cuando se aplica tanto la reducción de dimensionalidad como la estandarización de los datos.

### Sin Estandarizar y Sin Reducir Dimensionalidad

| Algoritmo           | Precisión | Recall  | F1-Score | Hiperparámetros                                                |
|---------------------|-----------|---------|----------|---------------------------------------------------------------|
| Regresión Logística | 0.9191    | 0.9205  | 0.9129   | clf__solver: lbfgs, clf__C: 0.1                                |
| SVM                 | 0.9267    | 0.9286  | 0.9228   | clf__kernel: linear, clf__C: 0.1                               |
| Árboles de Decisión | 0.8939    | 0.8946  | 0.8921   | clf__min_samples_split: 10, clf__max_depth: None               |
| K-NN                | 0.8839    | 0.8893  | 0.8616   | clf__weights: distance, clf__n_neighbors: 9                    |

En este último caso, se observa una disminución general en las métricas de todos los modelos. El modelo con mayor `recall` y métricas generales fue el **SVM**, y el peor fue el **K-NN**. Esto resalta que, si bien el **SVM** es sensible a la estandarización de los datos, tiende a responder mejor que el **K-NN** cuando no se aplica reducción de dimensionalidad.

### Importancia de Estandarización de Datos y Reducción de Dimensionalidad

Como se puede apreciar en los resultados, la **estandarización de los datos** y la **reducción de dimensionalidad** tienen un impacto significativo en el rendimiento de los modelos. La estandarización mejora el desempeño del modelo **SVM**, mientras que la reducción de dimensionalidad ofrece una ventaja considerable para el modelo **K-NN**, acelerando el proceso de entrenamiento y mejora de las métricas de evaluación.

---
## Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo [LICENSE](./LICENSE) para más detalles.
