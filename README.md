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
