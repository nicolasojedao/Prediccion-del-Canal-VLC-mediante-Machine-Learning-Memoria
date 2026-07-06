import pandas as pd
import numpy as np
import time
import joblib
import warnings 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=UserWarning)

print("  TEST INFERENCIA MLP vs XGB vs RF    ")

# 1 CARGAR DATOS 
archivo_csv = "dataset_vlc_z_variable_10k_FINAL_V5.csv"
print(f"Cargando dataset base ({archivo_csv})...")

columnas = ["X", "Y", "Z", "Alpha", "Beta", "H_LoS", "H_NLoS", "H_Scat", "H0_Total", "P_rx"]
df = pd.read_csv(archivo_csv, header=None, names=columnas)

X_LED, Y_LED, Z_LED = 3.0, 0.5, 4.5
ang_rad = 60
m = -np.log(2) / np.log(np.abs(np.cos(np.radians(ang_rad)))) 

df["Distancia_3D"] = np.sqrt((df["X"] - X_LED)**2 + (df["Y"] - Y_LED)**2 + (df["Z"] - Z_LED)**2)
df["Cos_Alpha"] = np.cos(np.radians(df["Alpha"]))
df["Sin_Alpha"] = np.sin(np.radians(df["Alpha"]))
df["Cos_Beta"]  = np.cos(np.radians(df["Beta"]))
df["Inv_Dist_Sq"] = 1.0 / (df["Distancia_3D"]**2) 
df["Lambert_Factor"] = (np.abs(df["Cos_Alpha"])**m) * df["Inv_Dist_Sq"] 

# MLP USA LAMBERT Y LOS ARBOLES NO
X_mlp = df[["X", "Y", "Z", "Alpha", "Beta", "Distancia_3D", "Cos_Alpha", "Sin_Alpha", "Cos_Beta", "Inv_Dist_Sq", "Lambert_Factor"]]
X_trees = df[["X", "Y", "Z", "Alpha", "Beta", "Distancia_3D", "Cos_Alpha", "Sin_Alpha", "Cos_Beta", "Inv_Dist_Sq"]]
y = np.log10(df["H0_Total"])

X_train_mlp, X_test_mlp, y_train, y_test = train_test_split(X_mlp, y, test_size=0.20, random_state=42)
X_test_trees = X_trees.loc[X_test_mlp.index] 





# 2 CREAR DATASETS TEST
print("PREPARANDO DATASET DE PRUEBA")
multiplicador = (1000000 // len(X_test_mlp)) + 1

X_masivo_mlp = pd.concat([X_test_mlp] * multiplicador, ignore_index=True).iloc[:1000000]
X_masivo_trees = pd.concat([X_test_trees] * multiplicador, ignore_index=True).iloc[:1000000]

X_single_mlp = X_test_mlp.iloc[[0]]
X_single_trees = X_test_trees.iloc[[0]]




# 3 CARGAR MODELOS Y ESCALADORES
print("CARGANDO MODELOS YA ENTRENADOS")


mlp_model = joblib.load('./MLP_Final/modelo_mlp_z_variable_campeon.joblib')
mlp_scaler = joblib.load('./MLP_Final/escalador_mlp_z_variable_campeon.joblib')

xgb_model = joblib.load('./XGBoost_Final/modelo_xgboost_z_variable_campeon.joblib')
xgb_scaler = joblib.load('./XGBoost_Final/escalador_xgboost_z_variable_campeon.joblib')
rf_model = joblib.load('./Baseline/modelo_rf_z_variable_baseline.joblib')

X_masivo_mlp_scaled = mlp_scaler.transform(X_masivo_mlp)
X_single_mlp_scaled = mlp_scaler.transform(X_single_mlp)

X_masivo_xgb_scaled = xgb_scaler.transform(X_masivo_trees)
X_single_xgb_scaled = xgb_scaler.transform(X_single_trees)

X_masivo_rf = X_masivo_trees
X_single_rf = X_single_trees





# 4 TIEMPO (10 ITERACIONES)
resultados = []

def medir_tiempo_riguroso(modelo, nombre, datos_masivos, datos_single, iteraciones=10):
    print(f"\n[{nombre}] iniciando prueba ({iteraciones} iteraciones)")
    
    # WARM-UP
    _ = modelo.predict(datos_single)
    



    # PRUEBA (1 Millón)
    tiempos_masivos = []
    for i in range(iteraciones):
        start = time.perf_counter()
        _ = modelo.predict(datos_masivos)
        tiempos_masivos.append(time.perf_counter() - start)
    
    t_masivo_mean = np.mean(tiempos_masivos)
    t_masivo_std = np.std(tiempos_masivos)
    




    # PRUEBA SINGLE
    tiempos_single = []
    for i in range(iteraciones):
        start = time.perf_counter()
        for _ in range(1000):
            _ = modelo.predict(datos_single)
        tiempos_single.append((time.perf_counter() - start) / 1000) 
        
    t_single_mean = np.mean(tiempos_single) * 1000 
    t_single_std = np.std(tiempos_single) * 1000
    


    # THROUGPUT
    throughput = 1000000 / t_masivo_mean
    
    resultados.append({
        "Modelo": nombre,
        "T_Masivo_s": t_masivo_mean,
        "Std_Masivo_s": t_masivo_std,
        "T_Single_ms": t_single_mean,
        "Std_Single_ms": t_single_std,
        "Throughput_pred_s": throughput
    })
    
    print(f"1 millon    : {t_masivo_mean:.4f} s (+- {t_masivo_std:.4f})")
    print(f"1 Punto     : {t_single_mean:.4f} ms (+- {t_single_std:.4f})")
    print(f"Througput  : {throughput:,.0f} predicciones/segundo")

medir_tiempo_riguroso(mlp_model, "MLP", X_masivo_mlp_scaled, X_single_mlp_scaled)
medir_tiempo_riguroso(xgb_model, "XGBoost", X_masivo_xgb_scaled, X_single_xgb_scaled)
medir_tiempo_riguroso(rf_model, "Random Forest", X_masivo_rf, X_single_rf)




# 5. EXPORTAR TABLA Y GRAFICOS
df_res = pd.DataFrame(resultados)
df_res.to_csv("metricas_tiempo_inferencia.csv", index=False)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
fig, axes = plt.subplots(1, 3, figsize=(18, 5)) 

x_pos = np.arange(len(df_res))




# grafico 1: tiempo 
sns.barplot(data=df_res, x="Modelo", y="T_Masivo_s", hue="Modelo", palette="viridis", legend=False, ax=axes[0], alpha=0.9)
axes[0].errorbar(x_pos, df_res["T_Masivo_s"], yerr=df_res["Std_Masivo_s"], fmt='none', c='black', capsize=5, elinewidth=2)
axes[0].set_title("Simulación Masiva (1.000.000 puntos)", fontweight='bold')
axes[0].set_ylabel("Tiempo Medio (Segundos)")
axes[0].set_xlabel("")

for i, v in enumerate(df_res["T_Masivo_s"]):
    axes[0].text(i, v + df_res["Std_Masivo_s"].iloc[i] + (v*0.05), f"{v:.2f} s", ha='center', fontweight='bold')




# grafico 2: tiempo real 
sns.barplot(data=df_res, x="Modelo", y="T_Single_ms", hue="Modelo", palette="magma", legend=False, ax=axes[1], alpha=0.9)
axes[1].errorbar(x_pos, df_res["T_Single_ms"], yerr=df_res["Std_Single_ms"], fmt='none', c='black', capsize=5, elinewidth=2)
axes[1].set_title("Inferencia en Tiempo Real (1 punto)", fontweight='bold')
axes[1].set_ylabel("Tiempo Medio (Milisegundos)")
axes[1].set_xlabel("")

for i, v in enumerate(df_res["T_Single_ms"]):
    axes[1].text(i, v + df_res["Std_Single_ms"].iloc[i] + (v*0.05), f"{v:.4f} ms", ha='center', fontweight='bold')





# grafico 3: Throughput
sns.barplot(data=df_res, x="Modelo", y="Throughput_pred_s", hue="Modelo", palette="crest", legend=False, ax=axes[2], alpha=0.9)
axes[2].set_title("Throughput (Capacidad de Procesamiento)", fontweight='bold')
axes[2].set_ylabel("Predicciones / Segundo")
axes[2].set_xlabel("")
for i, v in enumerate(df_res["Throughput_pred_s"]):
    axes[2].text(i, v + (v*0.02), f"{v/1000:,.0f}k", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("grafico_tiempos_throughput.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n LISTO")
