# Credit Risk Classification - Proyecto Final

## Introducción
La gestión de riesgos es algo clave para las empresas en todo el mundo, especialmente en la industria bancaria. Los bancos tienen la tarea de asegurarse de que gestionan y controlan eficazmente el riesgo especialmente crucial es en la determinación de qué solicitantes deben recibir tarjetas de crédito.

En este proyecto se busca crear un modelo de _Random Forest_ para clasificar si el cliente tiene buen record crediticio o no y con base en eso tomar una decision informada para la aprobación de tarjetas de crédito. El modelo clasifica a las personas en dos grupos: "buenos" y "malos". 

## Objetivo

El objetivo principal de este proyecto es construir un modelo de aprendizaje automático capaz de predecir si un solicitante de tarjetas de credito es "bueno" o "malo". La distinción entre estas categorías no está definida explícitamente y debe establecerse a través del proceso de entrenamiento y evaluación del modelo de aprendizaje automático.

## Libraries necesarias
Este proyecto utiliza varias bibliotecas de Python para el procesamiento de datos, la construcción de modelos y la evaluación, que incluyen:

- `pandas` para la manipulación de datos.
- `numpy` para operaciones numéricas.
- `matplotlib` para la visualización de datos.
- `RandomForestClassifier` de `sklearn.ensemble` para construir el modelo de Clasificador de Bosques Aleatorios.
- `train_test_split` de `sklearn.model_selection` para dividir el conjunto de datos en conjuntos de entrenamiento y prueba.
- `metrics` de `sklearn` para evaluar el rendimiento del modelo.
- `RandomUnderSampler` de `imblearn.under_sampling` para manejar datos desequilibrados.

Tenga en cuenta que este proyecto puede requerir una versión específica de NumPy y SciPy. Asegúrese de tener instalada una versión compatible de NumPy (>=1.16.5 y <1.23.0) para evitar problemas de compatibilidad.
