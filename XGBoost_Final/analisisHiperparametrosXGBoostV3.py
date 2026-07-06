import pandas as pd
import matplotlib
matplotlib.use('Agg') # Desactiva Tkinter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


print("ANALISIS HIPERPARAMETROS XGBOOST")





TIPO_DATASET = "CONSTANTE" 

if TIPO_DATASET == "VARIABLE":
    archivo_csv = 'historial_gridsearch_xgboost_z_variable.csv'
    prefijo_img = 'analisis_xgb_z_var_'
else:
    archivo_csv = 'historial_gridsearch_xgboost_z_constante.csv'
    prefijo_img = 'analisis_xgb_z_const_'

try:
    df_cv = pd.read_csv(archivo_csv)
    print(f"Archivo '{archivo_csv}' cargado exitosamente con {len(df_cv)} configuraciones")
except FileNotFoundError:
    print(f"ERROR: No se encontro el archivo '{archivo_csv}'.")
    exit()

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)







# GRAFICO 1: cantidad de arboles y learning rate

plt.figure(figsize=(10, 6))
df_g1 = df_cv.groupby(['param_learning_rate', 'param_n_estimators'])['mean_test_score'].mean().reset_index()

sns.lineplot(data=df_g1, x='param_n_estimators', y='mean_test_score', hue='param_learning_rate', marker='o', palette='Set1', linewidth=2)

plt.title(f'Efecto Promedio de la Cantidad de Árboles y Tasa de Aprendizaje (Z {TIPO_DATASET})', fontsize=14, fontweight='bold')
plt.xlabel('Cantidad de Árboles (n_estimators)', fontsize=12)
plt.ylabel('R² Medio (Comportamiento General)', fontsize=12)
plt.legend(title='Tasa de Aprendizaje (learning_rate)')
plt.tight_layout()
plt.savefig(f'{prefijo_img}1_learning_trees.png', dpi=300, bbox_inches='tight')
plt.close()





# GRAFICO 2: analisis sobre ajuste y profundidad

plt.figure(figsize=(10, 6))
df_g2 = df_cv.groupby(['param_max_depth', 'param_min_child_weight'])['Gap_CV'].mean().reset_index()

sns.lineplot(data=df_g2, x='param_max_depth', y='Gap_CV', hue='param_min_child_weight', marker='s', palette='Set2', linewidth=2)

plt.title(f'Evolución del Sobreajuste según Profundidad y Peso de Hoja (Z {TIPO_DATASET})', fontsize=14, fontweight='bold')
plt.xlabel('Profundidad Máxima del Árbol (max_depth)', fontsize=12)
plt.ylabel('Brecha Promedio (Train R² - Test R²)', fontsize=12)
plt.legend(title='Min Child Weight')
plt.tight_layout()
plt.savefig(f'{prefijo_img}2_gap_vs_depth_child.png', dpi=300, bbox_inches='tight')
plt.close()





# GRAFICO 3: heatmap depth y learning rate

pivot_table = df_cv.groupby(['param_max_depth', 'param_learning_rate'])['mean_test_score'].mean().unstack()

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="viridis_r", cbar_kws={'label': 'R² Medio (Tendencia)'})

plt.title(f'Mapa de Calor Promedio: Profundidad vs Tasa de Aprendizaje (Z {TIPO_DATASET})', fontsize=14, fontweight='bold')
plt.xlabel('Tasa de Aprendizaje (learning_rate)', fontsize=12)
plt.ylabel('Profundidad Máxima (max_depth)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{prefijo_img}3_heatmap_profundidad.png', dpi=300, bbox_inches='tight')
plt.close()





# GRAFICO 4: tiempo de entrenamiento vs cantidad de arboles

plt.figure(figsize=(10, 6))
sns.barplot(data=df_cv, x='param_n_estimators', y='mean_fit_time', hue='param_max_depth', palette='coolwarm', errorbar=None)

plt.title(f'Costo Computacional: Tiempo vs Cantidad de Árboles (Z {TIPO_DATASET})', fontsize=14, fontweight='bold')
plt.xlabel('Cantidad de Árboles (n_estimators)', fontsize=12)
plt.ylabel('Tiempo Medio de Entrenamiento (segundos)', fontsize=12)
plt.legend(title='Profundidad Máxima')
plt.tight_layout()
plt.savefig(f'{prefijo_img}4_tiempo_entrenamiento.png', dpi=300, bbox_inches='tight')
plt.close()


# GRAFICO 5: boxplot

idx_mejores = df_cv.groupby('param_max_depth')['mean_test_score'].idxmax()
df_mejores = df_cv.loc[idx_mejores].copy()

splits_cols = ['split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score']
df_melted = df_mejores.melt(id_vars=['param_max_depth'], value_vars=splits_cols, var_name='Fold', value_name='R2_Score')

plt.figure(figsize=(10, 6))
orden_arq = df_mejores.sort_values('mean_test_score', ascending=False)['param_max_depth']

sns.boxplot(data=df_melted, x='param_max_depth', y='R2_Score', order=orden_arq, hue='param_max_depth', legend=False, palette='Pastel1', showfliers=False)
sns.stripplot(data=df_melted, x='param_max_depth', y='R2_Score', order=orden_arq, color='black', alpha=0.6, jitter=True)

plt.title(f'Estabilidad de la mejor configuración encontrada para cada profundidad (Z {TIPO_DATASET})', fontsize=14, fontweight='bold')
plt.xlabel('Profundidad Máxima (max_depth)', fontsize=12)
plt.ylabel('R² en Test Set (Folds)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{prefijo_img}5_boxplot_folds.png', dpi=300, bbox_inches='tight')
plt.close()


# GRAFICO 6: impacto L1 y L2

plt.figure(figsize=(10, 6))
df_g6 = df_cv.groupby(['param_reg_lambda', 'param_reg_alpha'])['mean_test_score'].mean().reset_index()

sns.lineplot(data=df_g6, x='param_reg_lambda', y='mean_test_score', hue='param_reg_alpha', marker='D', palette='Set1', linewidth=2)

plt.title(f'Impacto de la Regularización en el Rendimiento General (Z {TIPO_DATASET})', fontsize=14, fontweight='bold')
plt.xlabel('Penalización L2 (reg_lambda)', fontsize=12)
plt.ylabel('R² Medio', fontsize=12)
plt.legend(title='Penalización L1 (reg_alpha)')
plt.tight_layout()
plt.savefig(f'{prefijo_img}6_regularizacion.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nLISTO")