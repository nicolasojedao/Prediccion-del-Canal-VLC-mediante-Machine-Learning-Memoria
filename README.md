# Predicción del Canal VLC mediante Machine Learning

Este repositorio contiene la implementación desarrollada para mi trabajo de título de Ingeniería Civil en Informática y Telecomunicaciones de la Universidad Diego Portales.

El objetivo del proyecto es desarrollar modelos de aprendizaje automático capaces de aproximar la respuesta del canal de Comunicaciones por Luz Visible (Visible Light Communications, VLC) en entornos mineros subterráneos, utilizando como referencia un simulador implementado en MATLAB.

---

## Objetivo

Comparar distintos modelos de Machine Learning para predecir la ganancia del canal VLC y evaluar tanto su precisión como su costo computacional.

Los modelos estudiados fueron:

- Random Forest (Baseline)
- Multi-Layer Perceptron (MLP)
- XGBoost

---

## Datasets utilizados

El dataset principal utilizado para este proyecto, del cual se obtienen y recopilan los datos que se pueden observar a lo largo del Capítulo 5 del documento de tesis, fue "dataset_vlc_z_variable_10k_FINAL_V5.csv".

El dataset "dataset_vlc_z_constante_10k_FINAL_V5.csv" fue utilizado para obtener los datos del escenario Z constante y que se pueden observar dentro de la sección Anexo A.2.

El dataset "dataset_vlc_z_variable_10k_FINAL_v6.csv" fue utilizado como un conjunto de datos de validación para la sección 5.3.1 del Capítulo 5.

---


## Ingeniería de características

Los modelos utilizan variables físicas obtenidas a partir de la geometría del sistema, entre ellas:

- Distancia tridimensional
- Cos(α)
- Sin(α)
- Cos(β)
- Inverso del cuadrado de la distancia
- Factor de Lambert (utilizado únicamente por el modelo MLP)

---

## Métricas de evaluación

El desempeño de los modelos fue evaluado mediante:

- Coeficiente de determinación (R²)
- Error Cuadrático Medio (RMSE)
- Error Absoluto Medio (MAE)
- Análisis de residuos
- QQ-Plot
- Mapas espaciales del error
- Función de distribución acumulada (CDF)
- Tiempo de inferencia
- Throughput (predicciones por segundo)

---

## Herramientas utilizadas

### Simulación

- MATLAB

### Machine Learning

- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy

### Visualización

- Matplotlib
- Seaborn

---

## Autor

**Nicolás Ojeda**

Ingeniería Civil en Informática y Telecomunicaciones

Universidad Diego Portales
