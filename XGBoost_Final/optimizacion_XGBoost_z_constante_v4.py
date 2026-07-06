import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


print("OPTIMIZACION  XGBOOST: Z CONSTANTE    ")
print("Sin Factor de Lambert")



# 1 CARGA DE DATOS Y PREPROCESAMIENTO

archivo_csv = "../dataset_vlc_z_constante_10k_FINAL_V5.csv"

columnas = ["X", "Y", "Z", "Alpha", "Beta", "H_LoS", "H_NLoS", "H_Scat", "H0_Total", "P_rx"]
df = pd.read_csv(archivo_csv, header=None, names=columnas)

X_LED, Y_LED, Z_LED = 3.0, 0.5, 4.5

df["Distancia_3D"] = np.sqrt((df["X"] - X_LED)**2 + (df["Y"] - Y_LED)**2 + (df["Z"] - Z_LED)**2)
df["Cos_Alpha"] = np.cos(np.radians(df["Alpha"]))
df["Sin_Alpha"] = np.sin(np.radians(df["Alpha"]))
df["Cos_Beta"]  = np.cos(np.radians(df["Beta"]))
df["Inv_Dist_Sq"] = 1.0 / (df["Distancia_3D"]**2) 
X = df[["X", "Y", "Alpha", "Beta", "Distancia_3D", "Cos_Alpha", "Sin_Alpha", "Cos_Beta", "Inv_Dist_Sq"]]
y = np.log10(df["H0_Total"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocesamiento listo\n")


# 2 CONFIGURACION DE SEARCH

xgb_base = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    random_state=42,
    n_jobs=1
)

param_grid = {
    'n_estimators': [400, 500, 700],       
    'max_depth': [7, 9, 11],             
    'learning_rate': [0.03, 0.05, 0.07],    
    'subsample': [0.8],               
    'colsample_bytree': [0.8],
    'min_child_weight': [5, 7, 9],             
    'gamma': [0, 0.25, 0.5],          
    'reg_alpha': [0, 0.1, 0.5],       
    'reg_lambda': [1, 5, 10]          
}

grid_search = GridSearchCV(
    estimator=xgb_base, 
    param_grid=param_grid, 
    scoring='r2', 
    cv=5,         
    verbose=3,    
    n_jobs=-1,    
    return_train_score=True, 
    refit=True
)


# 3 EJECUCION OPTIMIZACION

start_time = time.time()
print("\n iniciando gridsearch xboost")
grid_search.fit(X_train_scaled, y_train)
end_time = time.time()

print(f"\n opimizacion masiva terminada en {(end_time - start_time)/60:.2f} minutos")
print("\n hiperparametros seleccionados")
print(grid_search.best_params_)


# 4 EXTRACCION Y EXPORTACION

mejor_xgb = grid_search.best_estimator_
cv_results_df = pd.DataFrame(grid_search.cv_results_)
cv_results_df["Gap_CV"] = cv_results_df["mean_train_score"] - cv_results_df["mean_test_score"]
cv_results_df["Coef_Variacion"] = cv_results_df["std_test_score"] / cv_results_df["mean_test_score"]

columnas_deseadas = [
    'rank_test_score', 'param_max_depth', 'param_learning_rate', 'param_n_estimators', 
    'param_subsample', 'param_colsample_bytree', 'param_min_child_weight',
    'param_gamma', 'param_reg_alpha', 'param_reg_lambda',
    'mean_train_score', 'std_train_score', 
    'mean_test_score', 'std_test_score', 
    'split0_test_score', 'split1_test_score', 'split2_test_score', 
    'split3_test_score', 'split4_test_score',
    'Gap_CV', 'Coef_Variacion',
    'mean_fit_time', 'std_fit_time'
]
cv_results_df = cv_results_df[columnas_deseadas].sort_values(by='rank_test_score', ascending=True)
nombre_csv = 'historial_gridsearch_xgboost_z_constante.csv'
cv_results_df.to_csv(nombre_csv, index=False)







cv_mean = cv_results_df.iloc[0]['mean_test_score']
cv_std = cv_results_df.iloc[0]['std_test_score']
cv_train_mean = cv_results_df.iloc[0]['mean_train_score']
cv_gap = cv_results_df.iloc[0]['Gap_CV']
cv_coef_var = cv_results_df.iloc[0]['Coef_Variacion']

y_train_pred = mejor_xgb.predict(X_train_scaled)
y_test_pred = mejor_xgb.predict(X_test_scaled)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))






print(f"\nRENDIMIENTO DEL MEJOR MODELO (XGBOOST - Z CONSTANTE)")

print(f"METRICAS VALIDACION CRUZADA K5")
print(f"R2 Train CV: {cv_train_mean:.4f}")
print(f"R2 Test CV: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"Gap Medio CV: {cv_gap:.4f}")
print(f"Coef. de Var: {cv_coef_var * 100:.2f}%")
print(f"----------------------------------------------")
print(f"EVALUACION TEST SET FINAL")
print(f"R2 Train Final: {r2_train:.4f} | R2 Test Final: {r2_test:.4f}")
print(f"RMSE Final: {rmse_test:.4f}")


joblib.dump(mejor_xgb, 'modelo_xgboost_z_constante_campeon.joblib')
joblib.dump(scaler, 'escalador_xgboost_z_constante_campeon.joblib')
print(f"\n LISTO Archivo generado: {nombre_csv}")