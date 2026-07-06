import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 


print("  PIPELINE DIAGNOSTICO: XGBOOST Z VARIABLE ")


# 1 CARGA DE DATOS Y FEATURE ENGINEERING 
archivo_csv = "../dataset_vlc_z_variable_10k_FINAL_V5.csv"
print("CARGANDO DATOS")

columnas = ["X", "Y", "Z", "Alpha", "Beta", "H_LoS", "H_NLoS", "H_Scat", "H0_Total", "P_rx"]
df = pd.read_csv(archivo_csv, header=None, names=columnas)
X_LED, Y_LED, Z_LED = 3.0, 0.5, 4.5
df["Distancia_3D"] = np.sqrt((df["X"] - X_LED)**2 + (df["Y"] - Y_LED)**2 + (df["Z"] - Z_LED)**2)
df["Cos_Alpha"] = np.cos(np.radians(df["Alpha"]))
df["Sin_Alpha"] = np.sin(np.radians(df["Alpha"]))
df["Cos_Beta"]  = np.cos(np.radians(df["Beta"]))
df["Inv_Dist_Sq"] = 1.0 / (df["Distancia_3D"]**2) 

# NO INCLUIR LAMBERT
X = df[["X", "Y", "Z", "Alpha", "Beta", "Distancia_3D", "Cos_Alpha", "Sin_Alpha", "Cos_Beta", "Inv_Dist_Sq"]]
y = np.log10(df["H0_Total"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)






# 2 CARGA DEL MODELO CAMPEÓN
print("CARGANDO MODELO XGBOOST PRE-ENTRENADO Z VARIABLE")
mejor_xgb = joblib.load('modelo_xgboost_z_variable_campeon.joblib')
scaler = joblib.load('escalador_xgboost_z_variable_campeon.joblib')

X_test_scaled = scaler.transform(X_test)
y_pred = mejor_xgb.predict(X_test_scaled)

residuos = y_test - y_pred
errores_absolutos = np.abs(residuos)




# 3 CALCULO Y EXPORTACION
print("CALCULANDO METRICAS FINLAES")
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"R²   : {r2:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")

df_metricas = pd.DataFrame({
    "Modelo": ["XGBoost"],
    "Escenario": ["Z Variable"],
    "R2": [r2],
    "RMSE": [rmse],
    "MAE": [mae]
})

df_metricas.to_csv("metricas_xgb_z_variable.csv", index=False)
print("archivo 'metricas_xgb_z_variable.csv' generado\n")



# 4 GENERACION GRAFICOS
print("GENERANDO LOS 5 GRAFICOS ")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)



# GRAFICO 1: REAL VS PREDICHO
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal', edgecolor='k')
min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
plt.title('Rendimiento Global: XGBoost Real vs. Predicho (Z Variable)', fontsize=14, fontweight='bold')
plt.xlabel('Ganancia Real (Simulada)', fontsize=12)
plt.ylabel('Ganancia Predicha (XGBoost)', fontsize=12)


r2_val = r2_score(y_test, y_pred)



plt.text(0.05, 0.95, f"R² = {r2_val:.4f}", transform=plt.gca().transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
plt.legend()
plt.tight_layout()
plt.savefig('diagnostico_xgb_var_1_real_vs_predicho.png', dpi=300, bbox_inches='tight')
plt.close()




# GRAFICO 2: RESIDUAL PLOT
plt.figure(figsize=(9, 6)) 
sns.scatterplot(x=y_pred, y=residuos, alpha=0.5, color='royalblue', edgecolor='k')
sns.regplot(x=y_pred, y=residuos, scatter=False, lowess=True, color='darkorange', line_kws={'linewidth': 2})
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title('Residual Plot: XGBoost Z Variable', fontsize=14, fontweight='bold')
plt.xlabel('Ganancia Predicha (Log10)', fontsize=12)
plt.ylabel('Residuo (Real - Predicho)', fontsize=12)
plt.tight_layout()
plt.savefig('diagnostico_xgb_var_2_residual_plot.png', dpi=300, bbox_inches='tight')
plt.close()




# GRAFICO 3: MAPA ESPACIAL CON SIGNOS
plt.figure(figsize=(12, 4))
max_error_abs = np.max(errores_absolutos)
scatter = plt.scatter(X_test['X'], X_test['Y'], c=residuos, cmap='coolwarm', vmin=-max_error_abs, vmax=max_error_abs, alpha=0.9, s=40, edgecolors='none')

plt.scatter(X_LED, Y_LED, color='yellow', marker='*', s=300, label='LED Tx', edgecolors='black')
plt.colorbar(scatter, label='Residuo: Subestima (Azul) - Sobreestima (Rojo)')
plt.title('Mapa Espacial del Error: XGBoost Z Variable', fontsize=14, fontweight='bold')
plt.xlabel('Eje X: Longitud del Túnel (metros)', fontsize=12)
plt.ylabel('Eje Y: Ancho (metros)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6) 
plt.tight_layout()
plt.savefig('diagnostico_xgb_var_3_mapa_espacial.png', dpi=300, bbox_inches='tight')
plt.close()





# GRAFICO 4: CDF DEL ERROR ABSOLUTO
plt.figure(figsize=(8, 6))
sns.ecdfplot(data=errores_absolutos, color='darkorange', linewidth=2.5)

p90, p95, p99 = np.percentile(errores_absolutos, [90, 95, 99])
plt.axvline(p90, color='gray', linestyle='--', lw=1, label=f'P90: Error < {p90:.3f}')
plt.axvline(p95, color='blue', linestyle='--', lw=1, label=f'P95: Error < {p95:.3f}')
plt.axvline(p99, color='red',  linestyle='--', lw=1, label=f'P99: Error < {p99:.3f}')

plt.axhline(0.90, color='gray', linestyle=':', lw=1)
plt.axhline(0.95, color='blue', linestyle=':', lw=1)
plt.axhline(0.99, color='red',  linestyle=':', lw=1)

plt.title('CDF: Probabilidad Acumulada del Error XGBoost', fontsize=14, fontweight='bold')
plt.xlabel('Error Absoluto (Log10)', fontsize=12)
plt.ylabel('Proporción Acumulada', fontsize=12)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('diagnostico_xgb_var_4_cdf.png', dpi=300, bbox_inches='tight')
plt.close()

# GRAFICO 5: QQ-PLOT
plt.figure(figsize=(7, 6))
stats.probplot(residuos, dist="norm", plot=plt)
plt.title('QQ-Plot: Normalidad de Residuos XGBoost', fontsize=14, fontweight='bold')
plt.xlabel('Cuantiles Teóricos', fontsize=12)
plt.ylabel('Cuantiles Ordenados de los Residuos', fontsize=12)
plt.tight_layout()
plt.savefig('diagnostico_xgb_var_5_qq_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n LISTO")