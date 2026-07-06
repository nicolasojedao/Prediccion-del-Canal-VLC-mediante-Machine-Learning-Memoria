import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


print("ANALISIS HIPERPARAMETROS")



archivo_csv = 'historial_gridsearch_mlp_z_constante_v2.csv' 
try:
    df_cv = pd.read_csv(archivo_csv)
    print(f"Archivo '{archivo_csv}' cargado exitosamente con {len(df_cv)} configuraciones")
except FileNotFoundError:
    print(f"ERROR: No se encontro el archivo '{archivo_csv}'")
    exit()


sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
df_cv['Arquitectura'] = df_cv['param_hidden_layer_sizes'].astype(str)











# GRAFICO 1: efecto de la Regularizacion sobre el rendimiento

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_cv, x='param_alpha', y='mean_test_score', 
             hue='Arquitectura', marker='o', palette='tab10', linewidth=2)

plt.xscale('log') 
plt.title('Impacto de la Regularización (α) en el Rendimiento (CV Test R²)', fontsize=14, fontweight='bold')
plt.xlabel('Tasa de Regularización α (Escala Logarítmica)', fontsize=12)
plt.ylabel('R² Medio (Validación Cruzada)', fontsize=12)
plt.legend(title='Capas Ocultas', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('analisis_cv_1_alpha_vs_r2.png', dpi=300, bbox_inches='tight')
plt.close()






# GRAFICO 2: analisis de sobre ajuste vs alfa

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_cv, x='param_alpha', y='Gap_CV', hue='Arquitectura', marker='s', palette='Set2', linewidth=2)

plt.xscale('log')
plt.title('Evolución del Sobreajuste (Gap Train-Test) según Regularización', fontsize=14, fontweight='bold')
plt.xlabel('Tasa de Regularización α (Escala Logarítmica)', fontsize=12)
plt.ylabel('Brecha de Generalización (R² Train - R² Test)', fontsize=12)
plt.axhline(0, color='red', linestyle='--', linewidth=1) 
plt.legend(title='Capas Ocultas', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('analisis_cv_2_gap_vs_alpha.png', dpi=300, bbox_inches='tight')
plt.close()







# GRAFICO 3: heatmap de arquitectura vs alfa

pivot_table = df_cv.pivot(index="Arquitectura", columns="param_alpha", values="mean_test_score")
pivot_table = pivot_table.sort_index(axis=1)
pivot_table.columns = [f"{c:.4f}" for c in pivot_table.columns]

plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, annot=True, cmap="viridis_r", cbar_kws={'label': 'R² Medio (CV)'})

plt.title('Mapa de Calor: Rendimiento del Espacio de Búsqueda', fontsize=14, fontweight='bold')
plt.xlabel('Regularización (α)', fontsize=12)
plt.ylabel('Arquitectura de Red', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('analisis_cv_3_heatmap_arquitectura.png', dpi=300, bbox_inches='tight')
plt.close()





# GRAFICO 4: tiempo de entrenamiento por arquitectura

plt.figure(figsize=(10, 6))

orden_tiempo = df_cv.groupby('Arquitectura')['mean_fit_time'].mean().sort_values().index

sns.barplot(data=df_cv, x='Arquitectura', y='mean_fit_time', order=orden_tiempo, palette='coolwarm', errorbar='sd', capsize=0.1)

plt.title('Costo Computacional: Tiempo de Entrenamiento por Arquitectura', fontsize=14, fontweight='bold')
plt.xlabel('Arquitectura de Red', fontsize=12)
plt.ylabel('Tiempo Medio de Entrenamiento (segundos)', fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('analisis_cv_4_tiempo_entrenamiento.png', dpi=300, bbox_inches='tight')
plt.close()


# GRAFICO 5: boxplot

idx_mejores = df_cv.groupby('Arquitectura')['mean_test_score'].idxmax()
df_mejores = df_cv.loc[idx_mejores]

splits_cols = ['split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score']
df_melted = df_mejores.melt(id_vars=['Arquitectura'], value_vars=splits_cols, var_name='Fold', value_name='R2_Score')

plt.figure(figsize=(10, 6))
orden_arq = df_mejores.sort_values('mean_test_score', ascending=False)['Arquitectura']

sns.boxplot(data=df_melted, x='Arquitectura', y='R2_Score', order=orden_arq, palette='Pastel1', showfliers=False)
sns.stripplot(data=df_melted, x='Arquitectura', y='R2_Score', order=orden_arq, color='black', alpha=0.6, jitter=True)

plt.title('Estabilidad de la Validación Cruzada (Los 5 Folds por Arquitectura)', fontsize=14, fontweight='bold')
plt.xlabel('Arquitectura de Red (Mejor Configuración)', fontsize=12)
plt.ylabel('R² en Test Set (Folds)', fontsize=12)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('analisis_cv_5_boxplot_folds.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nLISTO")